"""Bind local assets and freeze the common task stream; never starts training."""
import argparse
import json
import os
from pathlib import Path
from phase3.common import require, write_new
from webshop_phase12.assets import ROOT
from webshop_phase12.snapshot import model_receipt, file_hash
from .bank import Bank
from .protocol import schedule


def validate_paths(base, sft):
    require(Path(base).resolve() != Path(sft).resolve(), 'Router and SFT policy must be distinct snapshots')


def source_receipt():
    directories = ('webshop_phase12','webshop_phase3','phase3','phase2','skillnet_cohort',
                   'verl','agent_system','scripts')
    paths = {p.relative_to(ROOT) for name in directories for p in (ROOT/name).rglob('*.py')}
    paths.update(p.relative_to(ROOT) for p in (ROOT/'verl/trainer/config').glob('*.yaml'))
    paths.update([Path('configs/webshop_phase3_b200_vllm.json'), Path('memory_data/webshop/claude_style_skills.json')])
    return [{'path':str(p),'sha256':file_hash(ROOT/p)} for p in sorted(paths)]


def prepare(args):
    import pandas as pd
    validate_paths(args.base_model,args.sft_model)
    require(args.nnodes*args.gpus_per_node == 16, 'Exactly sixteen GPUs required')
    require(args.ray_address != 'local' or args.nnodes == 1, 'Multi-node training needs a Ray cluster address')
    completion=json.loads(Path(args.sft_complete).read_text())
    require(completion['schema']=='webshop.phase3.sft_complete.v1' and completion['optimizer_steps']==480,
            'Expected completed shared cold-start SFT')
    require(Path(completion['model']).resolve()==Path(args.sft_model).resolve(), 'SFT model/receipt mismatch')
    require(Path(completion['recipe']['model']).resolve()==Path(args.base_model).resolve(), 'SFT started from a different base')
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=False)
    from webshop_phase12.envs import ShopWorld, SHOP
    world=ShopWorld(1)
    plan=schedule(len(world.server.goals))
    rows=[{'data_source':'webshop','prompt':[{'role':'user','content':'WebShop task'}],
        'env_kwargs':{'task_id':task},'extra_info':{'index':index,'task_id':task,'split':'train'},
        'reward_model':{'style':'rule','ground_truth':''},'ability':'shopping'}
        for index,task in enumerate(sum(plan['updates'],[]))]
    prepared=root/'prepared';prepared.mkdir()
    pd.DataFrame(rows).to_parquet(prepared/'train.parquet',index=False)
    dev=[{**rows[0],'env_kwargs':{'task_id':task},'extra_info':{'index':task,'task_id':task,'split':'dev'}} for task in plan['dev_ids']]
    pd.DataFrame(dev).to_parquet(prepared/'dev.parquet',index=False)
    write_new(root/'frozen-router-model.json',model_receipt(args.base_model))
    write_new(root/'sft-model.json',{**model_receipt(args.sft_model),'source':'shared completed WebShop cold-start SFT'})
    bank=Bank.initial();bank.save(root/'initial-bank')
    manifest={'schema':'webshop.phase3.run.v1','root':str(root),'base_model':str(Path(args.base_model).resolve()),
        'sft_model':str(Path(args.sft_model).resolve()),'sft_complete':completion,'env_root':str(SHOP),
        'nnodes':args.nnodes,'gpus_per_node':args.gpus_per_node,'ray_address':args.ray_address,
        'router_gpus':list(range(max(0,args.gpus_per_node-2),args.gpus_per_node)),
        'schedule':plan,'initial_bank_sha256':bank.manifest_sha256,'sources':source_receipt(),
        'data_hashes':{p.name:file_hash(p) for p in prepared.glob('*.parquet')},
        'archive_contract':'compact-phase3-no-phase12-utility',
        'numerics':'webshop.phase3.qwen35-mask-and-trim.v1'}
    write_new(root/'manifest.json',manifest)
    print(json.dumps({'manifest':str(root/'manifest.json'),'training_started':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('base-model','sft-model','sft-complete','root'):p.add_argument('--'+key,required=True)
    p.add_argument('--nnodes',type=int,default=2);p.add_argument('--gpus-per-node',type=int,default=8)
    p.add_argument('--ray-address',default='auto');prepare(p.parse_args())
