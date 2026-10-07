import json
import pytest
from test_bank_protocol import module


def test_export_receipt_reuses_finished_export_and_preserves_partial(tmp_path, monkeypatch):
    run = module('run')
    model = tmp_path/'models/u0005'
    model.mkdir(parents=True)
    (model/'unfinished.txt').write_text('interrupted export evidence')
    calls = []

    def export(argv, logs, env):
        calls.append(argv)
        model.mkdir(parents=True)
        (model/'model.safetensors').write_bytes(b'synthetic test weights')

    monkeypatch.setattr(run, 'command', export)
    run.export_model(tmp_path, tmp_path/'checkpoints/global_step_5', 5, {})
    assert (tmp_path/'incomplete-exports/u0005-attempt000/unfinished.txt').read_text() == 'interrupted export evidence'
    assert len(calls) == 1
    receipt = json.loads((tmp_path/'model-receipts/u0005.json').read_text())
    assert receipt['source'] == 'native RL endpoint export'
    run.export_model(tmp_path, tmp_path/'checkpoints/global_step_5', 5, {})
    assert len(calls) == 1
    (model/'model.safetensors').write_bytes(b'changed')
    with pytest.raises(Exception, match='export'):
        run.export_model(tmp_path, tmp_path/'checkpoints/global_step_5', 5, {})


def test_completed_stage_recovery_reuses_result_and_records_duration(tmp_path):
    stage = module('run').completed_stage
    result = stage(tmp_path, 'test', {'seed':404}, lambda:{'ok':True})
    assert result == {'ok':True}
    assert json.loads((tmp_path/'test/complete.json').read_text())['wall_seconds'] >= 0
    assert stage(tmp_path, 'test', {'seed':404}, lambda:pytest.fail('rerun')) == result
    with pytest.raises(Exception, match='Changed'):
        stage(tmp_path, 'test', {'seed':505}, lambda:{})


def test_resume_preserves_committed_records_and_quarantines_only_uncommitted_update(tmp_path):
    from phase3.common import write_new
    for update in (4,5):
        write_new(tmp_path/'episodes'/f'u{update:04d}'/'train/e.json',{'update':update})
    write_new(tmp_path/'direction_batches/u0005.json',{'partial':True})
    module('run').quarantine_uncommitted(tmp_path,4,5)
    assert (tmp_path/'episodes/u0004/train/e.json').exists()
    assert not (tmp_path/'episodes/u0005').exists()
    preserved=tmp_path/'interrupted-updates/recovery000'
    assert json.loads((preserved/'episodes/u0005/train/e.json').read_text())['update']==5
    assert (preserved/'direction_batches/u0005.json').exists()
    module('run').quarantine_uncommitted(tmp_path,4,5)
    assert len(list((tmp_path/'interrupted-updates').iterdir()))==1
