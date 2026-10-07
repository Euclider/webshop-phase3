"""Old-batch selection -> shared editor -> paired Dev gate -> immutable bank."""
import json
import math
from pathlib import Path

from phase3.common import require, write_new, digest
from .bank import Bank
from .protocol import ARMS, UPDATES, WINDOW


def select_evidence(arm, bank, episodes, readout, *, start):
    require(arm in ('reward', 'skillrl'), 'Nonediting arm')
    require(len({e['trajectory_id'] for e in episodes}) == len(episodes), 'Duplicate evidence trajectory')
    for e in episodes:
        require(e['global_update'] == start+1 and e['sampling_policy_update'] == start and e['split'] == 'train'
                and e['task_id'] >= 1500 and e['bank_sha256'] == bank.manifest_sha256, 'Foreign/future evidence')
        require(type(e['success']) is bool, 'Success must be boolean')
        for step in e['steps']:
            require(step['selected_skill_id'] in bank.skill_ids and
                    step['skill_version_sha256'] == bank.get(step['selected_skill_id']).payload_sha256, 'Stale skill evidence')
    failures = sorted((e for e in episodes if not e['success']), key=lambda e: e['trajectory_id'])
    eligible = {step['selected_skill_id'] for e in failures for step in e['steps']}
    priority = []
    if arm == 'reward':
        require(readout is not None and readout['bank_sha256'] == bank.manifest_sha256 and
                readout['start'] == start and readout['end'] == start+5, 'Foreign readout')
        require(len({r['skill_id'] for r in readout['rows']}) == len(readout['rows']), 'Duplicate skill score')
        rows = [r for r in readout['rows'] if r['skill_id'] in eligible and r['n_loss_tokens'] > 0]
        require(all(math.isfinite(r['D_sign_balance']) for r in rows), 'Nonfinite score')
        priority = [r['skill_id'] for r in sorted(rows, key=lambda r: (-r['D_sign_balance'], r['skill_id']))[:5]]
        failures = [e for e in failures if any(s['selected_skill_id'] in priority for s in e['steps'])]
    visible = bank.skill_ids if arm == 'skillrl' else priority
    evidence = [{'evidence_id': e['trajectory_id'], 'task_id': e['task_id'], 'task': e['task'],
        'success': False, 'task_score': e['task_score'], 'sampling_policy_update': start,
        'steps': [{k: s.get(k) for k in ('step_index', 'observation', 'action', 'next_observation',
                   'selected_skill_id', 'skill_version_sha256', 'is_action_valid')} for s in e['steps']]} for e in failures]
    return {'priority_ids': priority, 'evidence': evidence, 'mutation_budget': 3,
            'evidence_policy_update': start, 'active_bank_size': len(bank),
            'bank': [{'skill_id': sid, 'version_sha256': bank.get(sid).payload_sha256,
                      'name': bank.get(sid).name, 'description': bank.get(sid).description,
                      'body': bank.get(sid).payload} for sid in visible]}


def paired_gate(before, after):
    def index(rows):
        result = {}
        for row in rows:
            key = row['task_id'], row['eval_seed']
            require(key not in result and 500 <= key[0] < 1500, 'Duplicate/non-Dev gate observation')
            require(type(row['success']) is bool and math.isfinite(row['task_score'])
                    and 0 <= row['task_score'] <= 1, 'Invalid gate outcome')
            result[key] = row
        return result
    a, b = index(before), index(after)
    require(a and a.keys() == b.keys(), 'Unpaired task/seed gate')
    tasks = sorted({key[0] for key in a})
    def effect(field):
        return math.fsum(math.fsum(float(b[k][field])-float(a[k][field]) for k in a if k[0] == task)
                         / sum(k[0] == task for k in a) for task in tasks) / len(tasks)
    delta = effect('task_score')
    return {'accepted': delta >= 0., 'delta_score': delta, 'delta_success_rate': effect('success'),
            'tasks': len(tasks), 'episodes': len(a), 'tolerance': 0., 'criterion': 'native_score_non_decreasing',
            'repairs': sum(not a[k]['success'] and b[k]['success'] for k in a),
            'regressions': sum(a[k]['success'] and not b[k]['success'] for k in a)}


def evolve(bank, *, arm, start, output, episodes, readout, editor, evaluate):
    require(arm in ARMS and start in range(0, UPDATES, WINDOW), 'Invalid window/arm')
    output = Path(output)
    source = {'arm': arm, 'start': start, 'end': start+5, 'bank_sha256': bank.manifest_sha256,
              'episodes_sha256': digest(episodes), 'readout_sha256': digest(readout)}
    write_new(output/'source.json', source)
    if (output/'complete.json').exists():
        record = json.loads((output/'complete.json').read_text())
        require(record['source_sha256'] == digest(source), 'Changed completed edit event')
        sha = record['selected_bank_sha256']
        return Bank.load(output/'banks'/f'{sha}.json', sha), record
    record = {'source_sha256': digest(source), 'arm': arm, 'update': start+5,
              'editor_calls': 0, 'accepted': False, 'candidate_rejected': False, 'rollback_count': 0,
              'bank_size_before': len(bank), 'candidate_skills': 0, 'editor_trajectories': 0}
    selected = bank
    if arm == 'frozen_bank_grpo':
        record['outcome'] = 'bank_frozen'
    else:
        payload = select_evidence(arm, bank, episodes, readout, start=start)
        write_new(output/'editor-input.json', payload)
        record.update(candidate_skills=len(payload['bank']), editor_trajectories=len(payload['evidence']))
        if not payload['evidence'] or not payload['bank']:
            record['outcome'] = 'abstain_no_evidence'
        else:
            evidence_ids = {e['evidence_id'] for e in payload['evidence']}
            allowed = {s['skill_id'] for s in payload['bank']}
            def validate(patch):
                bank.apply(patch, event_id=f'u{start+5}', evidence_ids=evidence_ids, allowed_ids=allowed)
            proposal_path = output/'proposal.json'
            if proposal_path.exists():
                proposal = json.loads(proposal_path.read_text())
                require(proposal['payload_sha256'] == digest(payload), 'Changed editor input')
            else:
                patch, accounting = editor(payload, validate)
                validate(patch)
                proposal = {'patch': patch, 'accounting': accounting, 'payload_sha256': digest(payload)}
                write_new(proposal_path, proposal)
            candidate, operations = bank.apply(proposal['patch'], event_id=f'u{start+5}',
                                               evidence_ids=evidence_ids, allowed_ids=allowed)
            candidate.save(output/'banks')
            record.update(editor_calls=proposal['accounting'].get('api_calls', 0),
                editor_accounting=proposal['accounting'], proposed_operations=operations,
                candidate_bank_sha256=candidate.manifest_sha256)
            if operations['NOOP']:
                record['outcome'] = 'editor_noop'
            else:
                gate_path = output/'gate.json'
                if gate_path.exists():
                    gate_record = json.loads(gate_path.read_text())
                    require(gate_record['candidate_sha256'] == candidate.manifest_sha256, 'Changed gate candidate')
                else:
                    before, after = evaluate(bank), evaluate(candidate)
                    gate_record = {'candidate_sha256': candidate.manifest_sha256, 'before': before, 'after': after,
                                   'effect': paired_gate(before, after)}
                    write_new(gate_path, gate_record)
                gate = paired_gate(gate_record['before'], gate_record['after'])
                selected = candidate if gate['accepted'] else bank
                record.update(accepted=gate['accepted'], candidate_rejected=not gate['accepted'], gate=gate,
                              outcome='accepted' if gate['accepted'] else 'candidate_rejected')
    selected.save(output/'banks')
    record.update(selected_bank_sha256=selected.manifest_sha256, bank_size_after=len(selected))
    write_new(output/'complete.json', record)
    return selected, record
