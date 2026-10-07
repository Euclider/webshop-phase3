"""Four forward scores on the actual old-policy batch; no new rollout inputs."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from phase3.common import require, write_new


def summarize(rows):
    groups = defaultdict(list)
    for row in rows: groups[row['skill_id']].append(row)
    result = []
    for sid, values in sorted(groups.items()):
        n = len(values)
        mean = lambda name: math.fsum(float(r[name]) for r in values)/n
        signs = [(-1 if r['P_int'] > 0 else 1 if r['P_int'] < 0 else 0) if r['direction_valid'] else 0 for r in values]
        result.append({'skill_id': sid, 'n_loss_tokens': n,
            'n_tasks': len({r['task_id'] for r in values}), 'n_trajectories': len({r['trajectory_id'] for r in values}),
            'D_sign_balance': sum(signs)/n, 'D_signed': -mean('P_int'), 'D_original': mean('D_contribution'),
            'D_signed_gate': -math.fsum(r['P_int']*r['gate'] for r in values)/n,
            'C_centered': mean('C_upd_centered'), 'M_delta_centered': mean('delta_centered_norm')})
    return result


def run_shard(spec):
    import torch
    import pandas as pd
    from transformers import AutoTokenizer
    from logicbench_phase12.fixed_state import _load_model
    from webshop_phase12.dense_scoring import recorded_inputs, control_inputs, score_dense
    from webshop_phase12.prompts import remove_guidance
    from phase2.stable_direction import token_signals
    from .bank import Bank
    from .numerics import install, trim_inputs, VERSION
    install()
    out = Path(spec['output']); out.mkdir(parents=True, exist_ok=True)
    write_new(out/'source.json', spec)
    if (out/'complete.json').exists(): return json.loads((out/'complete.json').read_text())
    archive = torch.load(spec['batch'], map_location='cpu', weights_only=False)
    require(archive['schema'] == 'webshop.phase3.direction_batch.v1' and archive['update'] == spec['start']+1
            and archive['bank_sha256'] == spec['bank_sha256'] and archive['temperature'] == 1., 'Foreign direction batch')
    bank = Bank.load(spec['bank'], spec['bank_sha256'])
    tok = AutoTokenizer.from_pretrained(spec['old_model'], local_files_only=True)
    old, new = _load_model(spec['old_model'], 'cuda'), _load_model(spec['new_model'], 'cuda')
    tensors = archive['tensors']; rows = []; parity = 0.; seen = set()
    for i, meta in enumerate(archive['metadata']):
        # Exact duplicate distributed filler rows are not extra natural support.
        if meta['decision_id'] in seen: continue
        seen.add(meta['decision_id'])
        if i % spec['world'] != spec['rank']: continue
        mask = tensors['actual_loss_mask'][i]
        if not mask.any(): continue
        info = meta['info']; skill = bank.get(info['selected_skill_id'])
        require(info['skill_version_sha256'] == skill.payload_sha256 and info['phase2_payload_text'] == skill.payload,
                'Stale readout payload')
        def encode(text):
            chat = tok.apply_chat_template([{'role':'user','content':text}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
            return tok.encode(chat, add_special_tokens=False)
        plen = tensors['prompts'].shape[-1]
        expected = tensors['prompts'][i,tensors['attention_mask'][i,:plen].bool()].tolist()
        require(encode(info['prompt_text']) == expected, 'Prompt replay mismatch')
        response = tensors['responses'].shape[-1]
        original = trim_inputs(recorded_inputs(tensors,i), response)
        control = control_inputs(tensors,i,encode(remove_guidance(info['prompt_text'],skill.payload)),
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        control = trim_inputs(control, response)
        oo, no = score_dense(old,original,response,mask), score_dense(new,original,response,mask)
        actions = tensors['responses'][i,mask].to(oo.device)
        error = (oo.gather(1,actions[:,None]).flatten().cpu()-tensors['old_log_probs'][i,mask]).abs().max().item()
        parity = max(parity,error)
        require(error <= 1e-3, f'Native/readout logprob parity failed: {error}')
        oc, nc = score_dense(old,control,response,mask), score_dense(new,control,response,mask)
        signals = token_signals(oo,no,oc,nc,actions,tensors['advantages'][i,mask].to(oo.device))
        for j in range(len(actions)):
            rows.append({'skill_id':skill.skill_id,'task_id':int(info['task_id']),
                'trajectory_id':meta['trajectory_id'],'decision_id':meta['decision_id'],'token_offset':j,
                **{k:v[j].item() for k,v in signals.items() if v.ndim == 1}})
        del oo,no,oc,nc,signals
    pd.DataFrame(rows).to_parquet(out/'tokens.parquet',index=False)
    record = {'rank':spec['rank'],'tokens':len(rows),'max_logprob_error':parity,'numerics':VERSION,
              'aggregation':'token_mean','gold_outcomes_used':False}
    write_new(out/'complete.json',record)
    return record


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args()
    run_shard(json.loads(Path(a.spec).read_text()))
