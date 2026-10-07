"""Explicit, resumable three-arm driver. Default CLI only prints the plan."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from phase3.common import require, write_new, digest
from webshop_phase12.assets import ROOT
from .bank import Bank
from .protocol import ARMS, UPDATES, WINDOW


def completed_stage(root, name, inputs, action):
    path = Path(root)/name
    if (path/'inputs.json').exists():
        require(json.loads((path/'inputs.json').read_text()) == inputs, 'Changed completed-stage inputs')
    write_new(path/'inputs.json', inputs)
    if (path/'complete.json').exists():
        saved = json.loads((path/'complete.json').read_text())
        require(saved['inputs_sha256'] == digest(inputs), 'Changed completed-stage inputs')
        return saved['result']
    started = time.monotonic()
    result = action()
    write_new(path/'complete.json', {'inputs_sha256': digest(inputs), 'result': result,
                                   'wall_seconds': time.monotonic()-started})
    return result


def environment(manifest, bank_path, bank_sha, cache):
    # Dedicated subprocesses, never mutate another running experiment's env.
    public_env={k:v for k,v in os.environ.items() if not k.endswith('_API_KEY')
                and k not in ('GH_TOKEN','GITHUB_TOKEN','OPENAI_API_KEY','HF_TOKEN')}
    return dict(public_env, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1',
        WEBSHOP_RUN_ROOT=manifest['root'], WEBSHOP_BASE_MODEL=manifest['base_model'],
        WEBSHOP_ENV_ROOT=manifest['env_root'], WEBSHOP_PHASE3_BANK=str(bank_path),
        WEBSHOP_PHASE3_BANK_SHA256=bank_sha, WEBSHOP_ROUTER_CACHE=str(cache),
        WEBSHOP_ROUTER_GPU=str(manifest['router_gpus'][0]),
        WEBSHOP_ROUTER_GPUS=','.join(map(str,manifest['router_gpus'])), WEBSHOP_ROUTER_SHARED='1')


def command(argv, log_root, env):
    log_root = Path(log_root); log_root.mkdir(parents=True, exist_ok=True)
    index = len(list(log_root.glob('attempt-*.log')))
    with (log_root/f'attempt-{index:03d}.log').open('x') as log:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0, f'Stage failed; see {log_root}/attempt-{index:03d}.log')


def gpu_job(module, spec, env):
    """Ray assigns one physical GPU; child router shares exactly that device."""
    allocated = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(allocated and ',' not in allocated, 'One Ray-assigned GPU required')
    child = dict(env, CUDA_VISIBLE_DEVICES=allocated, WEBSHOP_ROUTER_GPU=allocated,
        WEBSHOP_ROUTER_GPUS=allocated, WEBSHOP_ROUTER_SHARED='0')
    spec = dict(spec)
    child['WEBSHOP_ROUTER_CACHE'] = str(Path(spec['output'])/'router.sqlite3')
    write_new(Path(spec['output'])/'job.json', spec)
    command([sys.executable,'-B','-u','-m',module,'--spec',str(Path(spec['output'])/'job.json')],
            Path(spec['output'])/'logs', child)
    return json.loads((Path(spec['output'])/'complete.json').read_text())


def distributed(module, specs, env, manifest):
    import ray
    require(not ray.is_initialized(), 'Driver owns its temporary Ray connection')
    ray.init(address=manifest['ray_address'], ignore_reinit_error=False,
             runtime_env={'env_vars': {'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1'}})
    try:
        require(ray.cluster_resources().get('GPU',0) >= 16, 'Target needs sixteen registered Ray GPUs')
        task = ray.remote(num_gpus=1, num_cpus=2, max_calls=1, max_retries=0)(gpu_job)
        futures = [task.remote(module, spec, env) for spec in specs]
        try: return ray.get(futures)
        except BaseException:
            for future in futures: ray.cancel(future, force=True)
            raise
    finally: ray.shutdown()


def evaluation(manifest, model, bank, output, tasks):
    from .evaluation import summarize
    output = Path(output)
    bank_path = bank.save(output/'banks')
    env = environment(manifest, bank_path, bank.manifest_sha256, output/'router.sqlite3')
    inputs = {'model': str(model), 'bank_sha256': bank.manifest_sha256,
              'tasks': tasks, 'seeds': manifest['schedule']['decoding_seeds']}
    def work():
        specs = [{**inputs,'rank':rank,'world':16,'output':str(output/f'shard{rank}')} for rank in range(16)]
        shards = distributed('webshop_phase3.evaluation', specs, env, manifest)
        rows = sorted([r for shard in shards for r in shard['rows']], key=lambda r:(r['task_id'],r['eval_seed']))
        write_new(output/'rows.json', rows)
        summary = summarize(rows,tasks,inputs['seeds']); write_new(output/'summary.json',summary)
        return {'rows_path':str(output/'rows.json'),'summary':summary}
    result = completed_stage(output,'stage',inputs,work)
    return json.loads(Path(result['rows_path']).read_text())


def predict(manifest, arm_root, start, old_model, new_model, bank, output):
    import pandas as pd
    from .readout import summarize
    batch = arm_root/'direction_batches'/f'u{start+1:04d}.pt'
    output=Path(output); bank_path=bank.save(output/'banks')
    inputs={'batch':str(batch),'start':start,'end':start+5,'old_model':str(old_model),'new_model':str(new_model),
            'bank':str(bank_path),'bank_sha256':bank.manifest_sha256}
    def work():
        env=environment(manifest,bank_path,bank.manifest_sha256,output/'unused-router.sqlite3')
        specs=[{**inputs,'rank':rank,'world':16,'output':str(output/f'shard{rank}')} for rank in range(16)]
        distributed('webshop_phase3.readout',specs,env,manifest)
        frames=[pd.read_parquet(output/f'shard{rank}/tokens.parquet') for rank in range(16)]
        tokens=pd.concat(frames,ignore_index=True)
        require(len(tokens)>0,'Empty prediction batch')
        require(not tokens.duplicated(['decision_id','token_offset']).any(),'Duplicate natural token')
        tokens.to_parquet(output/'token_signals.parquet',index=False)
        result={'bank_sha256':bank.manifest_sha256,'start':start,'end':start+5,'aggregation':'token_mean',
                'score':'D_sign_balance','gold_outcomes_used':False,'rows':summarize(tokens.to_dict('records'))}
        write_new(output/'prediction-locked.json',result)
        return result
    return completed_stage(output,'stage',inputs,work)


def checkpoint(root, update):
    path=Path(root)/'checkpoints'/f'global_step_{update}'
    require((path/'data.pt').is_file(),'Missing native dataloader checkpoint')
    for kind in ('model','optim','extra_state'):
        require(all((path/'actor'/f'{kind}_world_size_16_rank_{r}.pt').is_file() for r in range(16)),
                'Incomplete native '+kind+' shards')
    return path


def export_model(root, native, update, env):
    from webshop_phase12.snapshot import model_receipt
    root = Path(root)
    merged = root/'models'/f'u{update:04d}'
    receipt_path = root/'model-receipts'/f'u{update:04d}.json'
    def receipt():
        return {**model_receipt(merged), 'source':'native RL endpoint export', 'update':update}
    if receipt_path.exists():
        require(merged.is_dir(), 'Missing completed export')
        require(json.loads(receipt_path.read_text()) == receipt(), 'Changed completed export')
        return merged
    if merged.exists():
        # Preserve incomplete export rather than overwriting or deleting evidence.
        parent = root/'incomplete-exports'
        parent.mkdir(parents=True, exist_ok=True)
        index = 0
        while (parent/f'u{update:04d}-attempt{index:03d}').exists(): index += 1
        merged.rename(parent/f'u{update:04d}-attempt{index:03d}')
    command([sys.executable,'-B','scripts/model_merger.py','merge','--backend','fsdp','--local_dir',str(native/'actor'),
             '--target_dir',str(merged)],root/'logs'/f'export-u{update:04d}',dict(env,CUDA_VISIBLE_DEVICES=''))
    write_new(receipt_path, receipt())
    return merged


def quarantine_uncommitted(root, resumed, end):
    """Keep crash evidence outside the records of the restored optimizer state."""
    root=Path(root).resolve();paths=[]
    require(0<=resumed<end<=UPDATES, 'Invalid recovery interval')
    for update in range(resumed+1,end+1):
        for relative in (f'episodes/u{update:04d}',f'metrics/u{update:04d}.json',
                         f'direction_batches/u{update:04d}.pt',f'direction_batches/u{update:04d}.json',
                         f'checkpoints/global_step_{update}'):
            path=root/relative
            require(not path.is_symlink(), 'Unsafe recovery artifact')
            if path.exists():paths.append(Path(relative))
    if not paths:return
    index=0
    while (root/'interrupted-updates'/f'recovery{index:03d}').exists():index+=1
    target=root/'interrupted-updates'/f'recovery{index:03d}'
    write_new(target/'recovery.json',{'restored_update':resumed,'window_end':end,
        'paths':[str(p) for p in paths],'reason':'preserved incomplete update; not a completed optimizer result'})
    for relative in paths:
        (target/relative).parent.mkdir(parents=True,exist_ok=True)
        (root/relative).rename(target/relative)


def train_window(manifest, arm, start, bank):
    root=Path(manifest['root'])/'runs'/arm
    bank_path=bank.save(root/'banks')
    env=environment(manifest,bank_path,bank.manifest_sha256,root/'router'/f'{bank.manifest_sha256}.sqlite3')
    Path(env['WEBSHOP_ROUTER_CACHE']).parent.mkdir(parents=True,exist_ok=True)
    latest=root/'checkpoints/latest_checkpointed_iteration.txt'
    resumed=int(latest.read_text()) if latest.exists() else 0
    require(start <= resumed <= start+5,'Native checkpoint outside current window')
    if resumed < start+5:
        if resumed:checkpoint(root,resumed)
        quarantine_uncommitted(root,resumed,start+5)
        argv=[sys.executable,'-B','-u','-m','webshop_phase3.training','--model',manifest['sft_model'],
            '--root',str(root),'--arm',arm,'--bank-hash',bank.manifest_sha256,'--start',str(start),
            '--train-file',str(Path(manifest['root'])/'prepared/train.parquet'),
            '--dev-file',str(Path(manifest['root'])/'prepared/dev.parquet'),
            '--nnodes',str(manifest['nnodes']),'--gpus-per-node',str(manifest['gpus_per_node']),
            '--ray-address',manifest['ray_address'],'--execute']
        if resumed > start: argv += ['--resume-update',str(resumed)]
        command(argv,root/'logs'/f'train-u{start:04d}-u{start+5:04d}',env)
    native=checkpoint(root,start+5)
    merged=export_model(root,native,start+5,env)
    return {'model':str(merged),'native':str(native),'update':start+5}


def execute(manifest, arms):
    from .editor import Editor
    from .evolution import evolve
    from .report import report
    from .preflight import verify_sources
    verify_sources(manifest)
    root=Path(manifest['root'])
    gate=root/'preflight/complete.json'
    require(gate.exists(),'Run target GPU preflight before training')
    acceptance=json.loads(gate.read_text())
    require(acceptance.get('passed') is True and acceptance['manifest_sha256']==digest(manifest), 'Foreign/failed preflight')
    # Shared initial evaluation is an identical model/bank, not a repeated arm result.
    initial=Bank.initial()
    evaluation(manifest,manifest['sft_model'],initial,root/'initial-eval',manifest['schedule']['eval_ids'])
    for arm in arms:
        current=initial; old_model=manifest['sft_model']; arm_root=root/'runs'/arm
        editor=Editor(arm_root/'editor.sqlite3',allow_live=True) if arm!='frozen_bank_grpo' else None
        try:
            for start in range(0,UPDATES,WINDOW):
                window=arm_root/'windows'/f'u{start:04d}-u{start+5:04d}'
                trained=completed_stage(window,'training',{'start':start,'bank':current.manifest_sha256},
                    lambda:train_window(manifest,arm,start,current))
                new_model=trained['model']
                readout=predict(manifest,arm_root,start,old_model,new_model,current,window/'readout') if arm=='reward' else None
                evidence=[json.loads(p.read_text()) for p in sorted((arm_root/'episodes'/f'u{start+1:04d}'/'train').glob('*.json'))]
                if arm!='frozen_bank_grpo':require(len(evidence)==128,'Missing/duplicate OLD batch episodes')
                # The same unchanged endpoint policy compares old/candidate banks.
                def dev(bank): return evaluation(manifest,new_model,bank,window/'dev'/bank.manifest_sha256,manifest['schedule']['dev_ids'])
                dev(current)
                current,event=evolve(current,arm=arm,start=start,output=window/'evolution',episodes=evidence,
                    readout=readout,editor=editor,evaluate=dev)
                write_new(window/'complete.json',{'update':start+5,'model':new_model,'bank_sha256':current.manifest_sha256,'event':event})
                report(root,arm)
                from .retention import collect
                collect(arm_root,start+5)
                old_model=new_model
            evaluation(manifest,old_model,current,arm_root/'final-eval',manifest['schedule']['eval_ids'])
            report(root,arm)
        finally:
            if editor:editor.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    manifest=json.loads(Path(a.manifest).read_text())
    if not a.execute:
        print(json.dumps({'will_train':False,'arms':a.arms,'seed':404,'updates_per_arm':UPDATES,
                          'trajectories_per_update':128,'gpus':16,'manifest':a.manifest},indent=2));return
    lock=Path(manifest['root'])/'queue.lock'
    with lock.open('a') as file:
        fcntl.flock(file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        execute(manifest,a.arms)


if __name__=='__main__':main()
