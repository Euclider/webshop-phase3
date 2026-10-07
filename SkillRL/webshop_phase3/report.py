"""Versioned, append-only Phase3 reports; no claim of Phase1/2 utility gold."""
import argparse
import json
import sqlite3
from pathlib import Path
from phase3.common import digest, write_new
from .protocol import UPDATES


def report(root, arm):
    root=Path(root);run=root/'runs'/arm
    events=[json.loads(p.read_text()) for p in sorted(run.glob('windows/*/evolution/complete.json'))]
    updates=[json.loads(p.read_text()) for p in sorted(run.glob('metrics/u*.json'))]
    episodes=[json.loads(p.read_text()) for p in sorted(run.glob('episodes/*/train/*.json'))]
    steps=[s for e in episodes for s in e['steps']]
    result={'arm':arm,'completed_rl_updates':len(updates),'training_trajectories':len(episodes),
        'editing_events':len(events),'accepted_events':sum(e['accepted'] for e in events),
        'candidate_rejections':sum(e['candidate_rejected'] for e in events),'rollbacks':sum(e['rollback_count'] for e in events),
        'proposed_mutation_units':sum(e.get('proposed_operations',{}).get('mutation_units',0) for e in events),
        'accepted_mutation_units':sum(e.get('proposed_operations',{}).get('mutation_units',0) for e in events if e['accepted']),
        'training_prompt_tokens':sum(s['prompt_tokens'] for s in steps),
        'training_completion_tokens':sum(s['completion_tokens'] for s in steps),
        'training_router_calls':sum(s.get('skill_router_api',{}).get('local_calls',0) for s in steps),
        'training_router_cache_hits':sum(bool(s.get('skill_router_api',{}).get('cache_hit')) for s in steps),
        'invalid_actions':sum(not s['is_action_valid'] for s in steps),'events':events,
        'training_curve':[{'update':u,'success_rate':sum(e['success'] for e in es)/len(es),
                          'mean_score':sum(e['task_score'] for e in es)/len(es),'trajectories':len(es)}
                         for u in sorted({e['global_update'] for e in episodes})
                         if (es:=[e for e in episodes if e['global_update']==u])],
        'optimizer_metrics':updates,
        'dev':[{'path':str(p.relative_to(root)),'metrics':json.loads(p.read_text())}
               for p in sorted(run.glob('windows/*/dev/*/summary.json'))],
        'test':{label:json.loads(p.read_text()) for label,p in (
            ('U0',root/'initial-eval/summary.json'),(f'U{UPDATES}',run/'final-eval/summary.json')) if p.exists()}}
    ledger=run/'editor.sqlite3'
    if ledger.exists():
        with sqlite3.connect(f'file:{ledger}?mode=ro',uri=True) as db:
            attempts=db.execute('SELECT result FROM attempts').fetchall()
        accounted=[json.loads(r[0])['accounting'] for r in attempts if r[0] is not None]
        result['editor_accounting']={'reserved_calls':len(attempts),'ambiguous_calls':sum(r[0] is None for r in attempts),
            'prompt_tokens':sum(a['usage'].get('prompt_tokens') or 0 for a in accounted),
            'completion_tokens':sum(a['usage'].get('completion_tokens') or 0 for a in accounted),
            'latency_seconds':sum(a['latency_seconds'] for a in accounted),
            'provider_cost':None,'cost_note':'No invented monetary price; provider bill not available'}
    version=digest(result)[:16]
    write_new(run/'reports'/f'report-{version}.json',result)
    text=f"# WebShop Phase3: {arm}\n\nCompleted RL updates: {len(updates)} / {UPDATES}.\n\n"
    text+=f"Accepted edits: {result['accepted_events']}; candidate rejections: {result['candidate_rejections']}; rollbacks: {result['rollbacks']}.\n\n"
    text+='| Endpoint | Success rate | Mean native score | Episodes |\n|---|---:|---:|---:|\n'
    for label,value in result['test'].items():
        text+=f"| {label} | {value['success_rate']:.4f} | {value['mean_score']:.4f} | {value['episodes']} |\n"
    text+='\nFull optimizer curves, token counts, proposals and gate outcomes are in the same-version JSON. Training success is not held-out success. Decoding seeds are repeats, not independent RL seeds.\n'
    path=run/'reports'/f'report-{version}.md'
    if not path.exists():path.write_text(text)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--arm',required=True)
    a=p.parse_args();report(a.root,a.arm)
