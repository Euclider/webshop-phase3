import importlib
import pytest


def module(name):
    try:
        return importlib.import_module('webshop_phase3.' + name)
    except ModuleNotFoundError as error:
        pytest.fail('Missing Phase3 implementation: ' + str(error))


def test_bank_preserves_original_payload_and_order_and_updates_transactionally(tmp_path):
    from webshop_phase12.assets import WebshopBank
    original = WebshopBank()
    Bank = module('bank').Bank
    bank = Bank.initial()
    assert bank.skill_ids == original.skill_ids
    assert [bank.get(s).payload for s in bank.skill_ids] == [original.get(s).payload for s in original.skill_ids]
    sid = bank.skill_ids[0]
    proposal = {'operations': [{'op': 'MODIFY', 'targets': [{'skill_id': sid, 'version_sha256': bank.get(sid).payload_sha256}],
        'skill': {'name': 'Check price', 'description': 'When a price is visible', 'body': 'Read the visible price.'},
        'rationale': 'Observed mismatch', 'evidence_ids': ['trajectory1']}]}
    changed, counts = bank.apply(proposal, event_id='u5', evidence_ids={'trajectory1'}, allowed_ids={sid})
    assert bank.get(sid).payload == original.get(sid).payload
    assert changed.get(sid).payload != bank.get(sid).payload
    assert changed.parent == bank.manifest_sha256
    assert counts['MODIFY'] == 1 and counts['mutation_units'] == 1
    path = changed.save(tmp_path)
    assert Bank.load(path, changed.manifest_sha256).skill_ids == bank.skill_ids
    with pytest.raises(ValueError):
        changed.apply(proposal, event_id='u10', evidence_ids={'trajectory1'}, allowed_ids={sid})


def test_bank_growth_is_in_router_catalog_not_silently_zipped_away():
    from webshop_phase12.llm_router import fixed_prefix
    bank = module('bank').Bank.initial()
    patch = {'operations': [{'op': 'ADD', 'targets': [], 'skill': {'name': 'New procedure',
        'description': 'When revisiting', 'body': 'Compare observed evidence.'},
        'rationale': 'Generalized failure', 'evidence_ids': ['a']}]}
    grown, _ = bank.apply(patch, event_id='u5', evidence_ids={'a'})
    labels = module('bank').labels_for_count(55)
    prompt = fixed_prefix(grown, labels=labels)
    assert len(grown) == 55 and grown.skill_ids[:54] == bank.skill_ids
    assert grown.skill_ids[-1] in prompt and '[BC]' in prompt
    with pytest.raises(ValueError):
        fixed_prefix(grown, labels=labels[:54])


def test_reject_foreign_evidence_and_non_candidate_edits():
    bank = module('bank').Bank.initial()
    sid = bank.skill_ids[0]
    op = {'op': 'DELETE', 'targets': [{'skill_id': sid, 'version_sha256': bank.get(sid).payload_sha256}],
          'skill': None, 'rationale': 'bad', 'evidence_ids': ['foreign']}
    with pytest.raises(ValueError):
        bank.apply({'operations': [op]}, event_id='u5', evidence_ids={'ok'})
    op['evidence_ids'] = ['ok']
    with pytest.raises(ValueError):
        bank.apply({'operations': [op]}, event_id='u5', evidence_ids={'ok'}, allowed_ids=set())


def test_schedule_separates_splits_and_preserves_per_update_grpo_groups():
    plan = module('protocol').schedule(12000, seed=404)
    assert len(plan['updates']) == 150
    assert all(len(x) == len(set(x)) == 16 for x in plan['updates'])
    assert min(sum(plan['updates'], [])) >= 1500
    assert len(plan['dev_ids']) == 64 and all(500 <= x < 1500 for x in plan['dev_ids'])
    assert plan['eval_ids'] == list(range(500))
    assert plan == module('protocol').schedule(12000, seed=404)


def test_bank_loader_rejects_wrong_hash(tmp_path, monkeypatch):
    bank = module('bank').Bank.initial()
    path = bank.save(tmp_path)
    monkeypatch.setenv('WEBSHOP_PHASE3_BANK', str(path))
    monkeypatch.setenv('WEBSHOP_PHASE3_BANK_SHA256', '0' * 64)
    from webshop_phase12.assets import load_runtime_bank
    with pytest.raises(ValueError):
        load_runtime_bank()
