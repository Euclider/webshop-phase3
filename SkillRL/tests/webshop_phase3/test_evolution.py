import json
import pytest
from test_bank_protocol import module


def episodes(bank):
    ids = bank.skill_ids
    return [{'trajectory_id': f'e{i}', 'bank_sha256': bank.manifest_sha256, 'global_update': 1,
        'sampling_policy_update': 0, 'split': 'train', 'task_id': 1500+i, 'task': 'Buy item',
        'success': i == 6, 'task_score': .4, 'steps': [{'selected_skill_id': ids[i],
          'skill_version_sha256': bank.get(ids[i]).payload_sha256, 'observation': 'page', 'action': 'click[x]',
          'next_observation': 'next', 'step_index': 0, 'is_action_valid': True}]} for i in range(7)]


def bundle(bank):
    return {'bank_sha256': bank.manifest_sha256, 'start': 0, 'end': 5,
            'rows': [{'skill_id': sid, 'D_sign_balance': i, 'n_loss_tokens': 1}
                     for i, sid in enumerate(bank.skill_ids[:7])]}


def test_reward_ranks_within_old_failed_pool_and_selects_complete_corresponding_trajectories():
    bank = module('bank').Bank.initial()
    selected = module('evolution').select_evidence('reward', bank, episodes(bank), bundle(bank), start=0)
    assert selected['priority_ids'] == list(reversed(bank.skill_ids[1:6]))
    assert {e['evidence_id'] for e in selected['evidence']} == {'e1','e2','e3','e4','e5'}
    assert len(selected['bank']) == 5 and 'D_sign_balance' not in json.dumps(selected['evidence'])
    other = module('evolution').select_evidence('skillrl', bank, episodes(bank), None, start=0)
    assert len(other['bank']) == 54 and len(other['evidence']) == 6


def test_future_episodes_and_changed_skill_versions_are_rejected():
    bank = module('bank').Bank.initial(); evidence = episodes(bank)
    evidence[0]['global_update'] = 5
    with pytest.raises(ValueError): module('evolution').select_evidence('reward', bank, evidence, bundle(bank), start=0)
    evidence = episodes(bank); evidence[0]['steps'][0]['skill_version_sha256'] = '0'*64
    with pytest.raises(ValueError): module('evolution').select_evidence('reward', bank, evidence, bundle(bank), start=0)


def test_gate_pairs_task_seed_and_uses_native_score_not_binary_success():
    evaluate = module('evolution').paired_gate
    before = [{'task_id': 500, 'eval_seed': 1, 'task_score': .3, 'success': False},
              {'task_id': 500, 'eval_seed': 2, 'task_score': .5, 'success': False}]
    after = [{**before[1], 'task_score': .6}, {**before[0], 'task_score': .4}]
    result = evaluate(before, after)
    assert result['accepted'] and result['delta_score'] == pytest.approx(.1)
    with pytest.raises(ValueError): evaluate(before, after[:1])
    with pytest.raises(ValueError): evaluate(before+before, after+after)


def test_frozen_arm_never_calls_readout_editor_or_gate_and_completed_event_is_reused(tmp_path):
    evolution = module('evolution'); bank = module('bank').Bank.initial()
    def forbidden(*args, **kwargs): pytest.fail('Frozen arm incurred editing cost')
    result, record = evolution.evolve(bank, arm='frozen_bank_grpo', start=0, output=tmp_path,
        episodes=[], readout=None, editor=forbidden, evaluate=forbidden)
    assert result.manifest_sha256 == bank.manifest_sha256 and record['editor_calls'] == 0
    assert record == evolution.evolve(bank, arm='frozen_bank_grpo', start=0, output=tmp_path,
        episodes=[], readout=None, editor=forbidden, evaluate=forbidden)[1]


def test_rejected_candidate_does_not_change_bank_or_repeat_editor_on_resume(tmp_path):
    bank = module('bank').Bank.initial(); calls = []
    def editor(payload, validate):
        calls.append(payload)
        patch = {'operations': [{'op': 'ADD', 'targets': [], 'skill': {'name': 'check',
            'description': 'Before purchase', 'body': 'Read price'}, 'rationale': 'A failure', 'evidence_ids': ['e1']}]}
        validate(patch)
        return patch, {'api_calls': 1, 'usage': {'prompt_tokens': 100, 'completion_tokens': 10}}
    def evaluate(candidate):
        return [{'task_id': 500, 'eval_seed': 1, 'task_score': .7 if len(candidate)==54 else .5,
                 'success': False}]
    kwargs = dict(arm='reward', start=0, output=tmp_path, episodes=episodes(bank),
                  readout=bundle(bank), editor=editor, evaluate=evaluate)
    selected, record = module('evolution').evolve(bank, **kwargs)
    assert selected.manifest_sha256 == bank.manifest_sha256 and record['candidate_rejected']
    assert record['rollback_count'] == 0 and record['proposed_operations']['ADD'] == 1
    module('evolution').evolve(bank, **kwargs)
    assert len(calls) == 1
