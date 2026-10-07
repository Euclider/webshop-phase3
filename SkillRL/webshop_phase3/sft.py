"""Official WebShop SFT adapter; no downloads/training at import time."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from phase3.common import require, write_new

DATA_SHA256 = '2c4f045b18a7ffabf7779f8e0e416913debae6427cb70a323c79e30f34d2d051'


def initialize_output(output, recipe, resume, distributed):
    """Only rank0 inspects/creates output; all ranks receive the same verdict."""
    output=Path(output);error=[None]
    if distributed.get_rank()==0:
        try:
            if output.exists() and any(output.iterdir()) and not resume:
                raise ValueError('SFT output exists; explicit resume required')
            if resume:
                require(json.loads((output/'sft-recipe.json').read_text()) == recipe, 'SFT resume recipe changed')
            write_new(output/'sft-recipe.json', recipe)
        except Exception as exc:error[0]=str(exc)
    distributed.broadcast_object_list(error,src=0)
    require(error[0] is None, 'SFT output initialization failed: '+str(error[0]))


def encode_example(row, tokenizer, max_length=16384):
    require(set(row) >= {'instruction', 'output'}, 'Missing official SFT fields')
    user = {'role': 'user', 'content': row['instruction']}
    prefix = tokenizer.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    if prefix.endswith('<think>\n'): prefix = prefix[:-len('<think>\n')]
    full = tokenizer.apply_chat_template([user, {'role': 'assistant', 'content': row['output']}],
                                         tokenize=False, add_generation_prompt=False)
    require(full.startswith(prefix), 'Incompatible assistant prefix')
    prompt = tokenizer.encode(prefix, add_special_tokens=False)
    ids = tokenizer.encode(full, add_special_tokens=False)
    require(ids[:len(prompt)] == prompt and len(ids) > len(prompt), 'SFT token boundary mismatch')
    require(len(ids) <= max_length, 'SFT overflow; never silently truncate')
    return {'input_ids': ids, 'labels': [-100]*len(prompt) + ids[len(prompt):]}


def prepare(data, model, output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    data, output = Path(data), Path(output)
    require(hashlib.sha256(data.read_bytes()).hexdigest() == DATA_SHA256, 'Changed official WebShop SFT source')
    rows = pq.read_table(data).to_pylist()
    require(len(rows) == 2553, 'Official WebShop release must have 2553 rows')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    encoded = [encode_example(row, tokenizer) for row in rows]
    output.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist(encoded), output/'encoded.parquet')
    write_new(output/'recipe.json', {'schema': 'webshop.phase3.sft.v1', 'model': str(Path(model).resolve()),
        'source_sha256': DATA_SHA256, 'encoded_sha256': hashlib.sha256((output/'encoded.parquet').read_bytes()).hexdigest(),
        'examples': 2553, 'learning_rate': 1e-4, 'batch_size': 16, 'epochs': 3, 'seed': 404,
        'loss': 'global_response_token_mean', 'max_length': 16384, 'full_parameter': True,
        'original_supervision_preserved': True, 'synthetic_history_added': False,
        'scheduler': 'linear', 'warmup_ratio': 0., 'weight_decay': 0.,
        'split_provenance': 'upstream release; no task IDs in published SFT, no claim of verified Eval nonoverlap'})


def train(prepared, output, resume=None):
    import torch
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
    from .numerics import install
    install()
    prepared, output = Path(prepared), Path(output)
    recipe = json.loads((prepared/'recipe.json').read_text())
    raw = prepared/'encoded.parquet'
    require(hashlib.sha256(raw.read_bytes()).hexdigest() == recipe['encoded_sha256'], 'SFT data changed')
    require(int(os.environ.get('WORLD_SIZE', '0')) == 16, 'SFT global batch16 requires sixteen DDP ranks')
    rows = pq.read_table(raw).to_pylist()
    require(len(rows) == 2553, 'SFT count mismatch')
    # Pad the sampler inventory with zero-loss sentinels, not repeated examples.
    rows += [{'input_ids': [1, 1], 'labels': [-100, -100]}] * ((-len(rows)) % 16)
    tokenizer = AutoTokenizer.from_pretrained(recipe['model'], local_files_only=True)
    def collate(examples):
        require(len(examples) == 1, 'One sample per rank gives global batch16')
        row = examples[0]
        ids = torch.tensor([row['input_ids']], dtype=torch.long)
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                'labels': torch.tensor([row['labels']], dtype=torch.long)}
    class ResponseTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop('labels')
            positions = torch.where(labels[0, 1:] != -100)[0]
            if not len(positions): positions = torch.zeros(1, device=labels.device, dtype=torch.long)
            targets = labels[:, positions+1]
            outputs = model(**inputs, use_cache=False, logits_to_keep=positions)
            total = (targets != -100).sum().to(torch.float32)
            torch.distributed.all_reduce(total)
            loss = torch.nn.functional.cross_entropy(outputs.logits.float().reshape(-1, outputs.logits.shape[-1]),
                targets.reshape(-1), ignore_index=-100, reduction='sum') * 16 / total.clamp_min(1)
            return (loss, outputs) if return_outputs else loss
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    if not torch.distributed.is_initialized():torch.distributed.init_process_group(backend='nccl')
    initialize_output(output,recipe,resume,torch.distributed)
    # B200 has room for full FP32 master parameters/Adam; autocast does BF16 math.
    model = AutoModelForCausalLM.from_pretrained(recipe['model'], dtype=torch.float32,
        attn_implementation='sdpa', local_files_only=True)
    args = TrainingArguments(output_dir=str(output), per_device_train_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=1e-4, num_train_epochs=3,
        lr_scheduler_type='linear', warmup_ratio=0., weight_decay=0., bf16=True, tf32=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False, save_strategy='steps', save_steps=160,
        save_total_limit=2, logging_steps=1, seed=404, data_seed=404, report_to=[],
        remove_unused_columns=False, dataloader_num_workers=0, optim='adamw_torch_fused')
    trainer = ResponseTrainer(model=model, args=args, train_dataset=rows, data_collator=collate,
                              processing_class=tokenizer)
    trainer.train(resume_from_checkpoint=resume)
    require(trainer.state.global_step==480, 'Incomplete SFT training; no final completion receipt')
    trainer.save_model(str(output/'final'))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output/'final')
        write_new(output/'complete.json', {'schema': 'webshop.phase3.sft_complete.v1',
            'recipe': recipe, 'optimizer_steps': trainer.state.global_step,
            'expected_optimizer_steps': 480, 'model': str((output/'final').resolve())})


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare', 'train'])
    p.add_argument('--data'); p.add_argument('--model'); p.add_argument('--prepared'); p.add_argument('--output', required=True)
    p.add_argument('--resume'); p.add_argument('--execute', action='store_true'); a = p.parse_args()
    if a.mode == 'prepare':
        require(a.data and a.model, 'prepare requires --data and --model')
        prepare(a.data, a.model, a.output)
    else:
        require(a.execute and a.prepared, 'Training requires --prepared and explicit --execute')
        train(a.prepared, a.output, a.resume)


if __name__ == '__main__': main()
