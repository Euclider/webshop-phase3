"""Batched full-episode evaluation; native score and binary success stay distinct."""
import argparse
import json
import math
import time
from pathlib import Path
from phase3.common import require, write_new


def shard_pairs(tasks, seeds, *, rank, world):
    require(0 <= rank < world and len(set(tasks)) == len(tasks) and len(set(seeds)) == len(seeds), 'Invalid evaluation grid')
    return [(task, seed) for task in tasks for seed in seeds][rank::world]


def summarize(rows, tasks, seeds):
    keys = [(r['task_id'], r['eval_seed']) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == {(t,s) for t in tasks for s in seeds}, 'Incomplete/duplicate evaluation')
    require(rows and all(math.isfinite(r['task_score']) and 0 <= r['task_score'] <= 1
                         and type(r['success']) is bool for r in rows), 'Invalid evaluation outcomes')
    return {'episodes': len(rows), 'tasks': len(tasks), 'decoding_seeds': list(seeds),
        'success_rate': sum(r['success'] for r in rows)/len(rows),
        'mean_score': math.fsum(r['task_score'] for r in rows)/len(rows),
        'mean_steps': math.fsum(r['steps'] for r in rows)/len(rows),
        'prompt_tokens': sum(r['prompt_tokens'] for r in rows),
        'completion_tokens': sum(r['completion_tokens'] for r in rows)}


def run_shard(spec):
    from omegaconf import OmegaConf
    from webshop_phase12.envs import ShopWorld, Manager
    from .policy import Policy
    output = Path(spec['output']); output.mkdir(parents=True, exist_ok=True)
    pairs = shard_pairs(spec['tasks'], spec['seeds'], rank=spec['rank'], world=spec['world'])
    write_new(output/'source.json', spec)
    if (output/'complete.json').exists(): return json.loads((output/'complete.json').read_text())
    count = min(8, len(pairs))
    if not count:
        write_new(output/'complete.json', {'rows': [], 'seconds': 0.}); return {'rows': [], 'seconds': 0.}
    world = ShopWorld(count)
    manager = Manager(world, count, OmegaConf.create({'env': {'max_steps': 50}}), f"eval-{spec['rank']}")
    policy = None; started = time.monotonic(); rows = []
    try:
        policy = Policy(spec['model'])
        for offset in range(0, len(pairs), count):
            chunk = pairs[offset:offset+count]
            pending = [(t,s) for t,s in chunk if not (output/f'task{t}-seed{s}.json').exists()]
            if not pending:
                rows.extend(json.loads((output/f'task{t}-seed{s}.json').read_text()) for t,s in chunk); continue
            # A partial tail gets its own correctly sized Manager, sharing catalog/router.
            previous = manager
            manager = Manager(world, len(pending), OmegaConf.create({'env': {'max_steps': 50}}), 'unused', empty=True)
            manager.router = previous.router
            previous.router = None
            obs, _ = manager.reset([{'task_id': t} for t, _ in pending])
            traces = [[] for _ in pending]; done_rows = {}
            for step in range(50):
                active = [i for i in range(len(pending)) if i not in done_rows]
                if not active: break
                generated = policy.generate_batch([{'prompt': obs['text'][i],
                    'seed': pending[i][1]*100000 + pending[i][0]*100 + step,
                    'temperature': .4, 'max_new_tokens': 512} for i in active])
                actions = ['']*len(pending)
                for i, (text, pt, ct) in zip(active, generated): actions[i] = text
                obs, _, done, infos = manager.step(actions)
                for i, (text, pt, ct) in zip(active, generated):
                    info = infos[i]
                    traces[i].append({'step_index': step, 'observation': info['observation'],
                        'next_observation': info['next_observation'], 'action': info['projected_action'],
                        'raw_model_output': text, 'skill_id': info['selected_skill_id'],
                        'skill_version_sha256': info['skill_version_sha256'], 'visible_state': info['visible_state'],
                        'router': info['skill_router_api'], 'prompt_tokens': pt, 'completion_tokens': ct,
                        'valid_action': info['is_action_valid']})
                    if done[i] or step == 49:
                        task, seed = pending[i]
                        row = {'task_id': task, 'eval_seed': seed, 'task_score': float(info['task_score']),
                            'success': bool(info['won']), 'steps': len(traces[i]),
                            'prompt_tokens': sum(x['prompt_tokens'] for x in traces[i]),
                            'completion_tokens': sum(x['completion_tokens'] for x in traces[i]),
                            'bank_sha256': manager.bank.manifest_sha256, 'model': spec['model']}
                        write_new(output/f'trace{task}-seed{seed}.json', traces[i])
                        write_new(output/f'task{task}-seed{seed}.json', row); done_rows[i] = row
            rows.extend(json.loads((output/f'task{t}-seed{s}.json').read_text()) for t,s in chunk)
    finally:
        manager.close()
        if policy is not None: policy.close()
    result = {'rows': rows, 'seconds': time.monotonic()-started}
    write_new(output/'complete.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--spec', required=True); a = p.parse_args()
    run_shard(json.loads(Path(a.spec).read_text()))
