"""Non-training target checks. CPU-only mode never publishes GPU admission."""
import argparse
import json
import math
from pathlib import Path
from phase3.common import require, write_new, digest
from webshop_phase12.assets import ROOT
from webshop_phase12.snapshot import file_hash


def verify_sources(manifest):
    for row in manifest['sources']:
        require(file_hash(ROOT/row['path'])==row['sha256'],'Changed source: '+row['path'])
    for name,sha in manifest['data_hashes'].items():
        require(file_hash(Path(manifest['root'])/'prepared'/name)==sha,'Changed task schedule')
    for name in ('frozen-router-model.json','sft-model.json'):
        receipt=json.loads((Path(manifest['root'])/name).read_text())
        for row in receipt['files']:
            stat=(Path(receipt['path'])/row['path']).stat()
            require(stat.st_size==row['size'] and stat.st_mtime_ns==row['mtime_ns'],'Changed model snapshot')


def acceptable(report):
    ranks=report.get('ranks',[])
    return (len(ranks)==16 and {r.get('rank') for r in ranks}==set(range(16)) and
        all('B200' in r.get('device','') and r.get('router_passed') is True and r.get('generation_passed') is True
            and r.get('fast_linear_attention') is True
            and r.get('finite') is True and isinstance(r.get('max_logprob_error'),(int,float))
            and math.isfinite(r['max_logprob_error']) and r['max_logprob_error']<=1e-3 for r in ranks))


def gpu_probe(spec):
    import torch
    from transformers import AutoTokenizer
    from webshop_phase12.llm_service import load_backend
    from webshop_phase12.llm_router import FrozenLLMSkillRouter, build_router_prompt
    from webshop_phase12.visible_state import VisibleMemory
    from webshop_phase12.prompts import build_state_prompt
    from webshop_phase12.dense_scoring import score_dense
    from logicbench_phase12.fixed_state import _load_model
    from .bank import Bank
    from .numerics import install, trim_inputs
    from .policy import Policy
    install();out=Path(spec['output']);out.mkdir(parents=True,exist_ok=True)
    from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available
    require(is_fast_path_available, 'Install compatible flash-linear-attention and causal-conv1d; slow fallback is not admitted')
    require('B200' in torch.cuda.get_device_name(0),'B200 deployment gate requires B200, not CPU/another GPU')
    bank=Bank.load(spec['bank'],spec['bank_sha256'])
    backend=load_backend();policy=None
    try:
        labels=backend.identity['labels']
        memory=VisibleMemory('Find a blue cotton shirt under $30',50)
        memory.observe('Search',['search[<your query>]'],{'page_type':'index'})
        state=memory.state();prompt=build_router_prompt(bank,state,labels)
        one=backend.select_many([prompt])[0]
        batch=backend.select_many([prompt,prompt])
        require(all(x['label']==one['label'] for x in batch),'Router batch parity failed')
        backend.sleep();backend.wake()
        require(backend.select_many([prompt])[0]['label']==one['label'],'Router wake parity failed')
        router=FrozenLLMSkillRouter(out/'probe-router.sqlite3',backend,bank,labels)
        bundle=router.route_many([state])[0]
        sid=bundle['selected_skill_id']
        require(router.route_many([state])[0]['skill_router_api']['cache_hit'],'Router cache failed')
        policy=Policy(spec['model'])
        text=build_state_prompt(state,bank.get(sid).payload)
        generated=policy.generate_batch([{'prompt':text,'seed':404,'temperature':0.,'max_new_tokens':32}])
        require(len(generated)==1 and generated[0][2]>0,'No real policy generation')
        policy.close();policy=None
        backend.sleep()
        model=_load_model(spec['model'],'cuda')
        tok=AutoTokenizer.from_pretrained(spec['model'],local_files_only=True)
        chat=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok.encode(chat,add_special_tokens=False);action=tok.encode('<action>click[search]</action>',add_special_tokens=False)
        pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        inputs=torch.full((1,16384+512),pad,dtype=torch.long)
        mask=torch.zeros_like(inputs);inputs[0,16384-len(ids):16384]=torch.tensor(ids)
        inputs[0,16384:16384+len(action)]=torch.tensor(action)
        mask[0,16384-len(ids):16384+len(action)]=1
        pos=(mask.cumsum(-1)-1).clamp_min(0)
        pos[:,16384:]=len(ids)-1+torch.arange(1,513)
        dense={'input_ids':inputs,'attention_mask':mask,'position_ids':pos}
        loss=torch.arange(512)<len(action)
        left=score_dense(model,dense,512,loss)
        right=score_dense(model,trim_inputs(dense,512),512,loss)
        token_ids=torch.tensor(action,device=left.device)[:,None]
        error=(left.gather(1,token_ids)-right.gather(1,token_ids)).abs().max().item()
        record={'rank':spec['rank'],'device':torch.cuda.get_device_name(0),'router_passed':True,
            'fast_linear_attention':bool(is_fast_path_available),
            'generation_passed':True,'max_logprob_error':error,'finite':bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
            'peak_memory_bytes':torch.cuda.max_memory_allocated(),
            'boundary':'inference and padding parity only; no SFT/RL update, native FSDP restore or editor API call'}
        require(error<=1e-3 and record['finite'],'Padding parity failed')
        write_new(out/'complete.json',record)
    finally:
        if policy:policy.close()
        backend.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--spec');p.add_argument('--gpu',action='store_true');a=p.parse_args()
    if a.spec:return gpu_probe(json.loads(Path(a.spec).read_text()))
    require(a.manifest,'Provide --manifest')
    manifest=json.loads(Path(a.manifest).read_text());verify_sources(manifest)
    if not a.gpu:
        print(json.dumps({'source_checks':'passed','gpu_admission':False,'training_started':False}));return
    from .bank import Bank
    from .run import environment, distributed
    root=Path(manifest['root']);bank=Bank.initial();path=bank.save(root/'initial-bank')
    specs=[{'rank':r,'world':16,'model':manifest['sft_model'],'bank':str(path),'bank_sha256':bank.manifest_sha256,
            'output':str(root/'preflight'/f'rank{r}')} for r in range(16)]
    ranks=distributed('webshop_phase3.preflight',specs,environment(manifest,path,bank.manifest_sha256,root/'preflight/router.sqlite3'),manifest)
    result={'passed':acceptable({'ranks':ranks}),'ranks':ranks,'manifest_sha256':digest(manifest),
            'formal_training_started':False,'native_training_acceptance':False}
    require(result['passed'],'Incomplete GPU preflight')
    write_new(root/'preflight/complete.json',result)


if __name__=='__main__':main()
