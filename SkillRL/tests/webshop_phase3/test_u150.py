import json
import pytest
from test_bank_protocol import module


@pytest.mark.parametrize('arm',['reward','skillrl','frozen_bank_grpo'])
def test_last_training_window_keeps_official_batch_and_can_resume_u149(tmp_path,arm):
    cfg=module('training').configuration(model='/sft',root=tmp_path,arm=arm,bank_hash='a'*64,
        start=145,resume_update=149,train_file='/train',dev_file='/dev',nnodes=2,gpus_per_node=8,ray_address='auto')
    assert cfg.phase3.segment_end==150 and cfg.trainer.resume_from_path.endswith('global_step_149')
    assert cfg.trainer.total_training_steps==cfg.actor_rollout_ref.actor.optim.total_training_steps==150
    assert cfg.data.train_batch_size*cfg.env.rollout.n==128 and cfg.env.max_steps==50


def test_sampling_seeds_continue_beyond_u50_but_episode_horizon_does_not_change():
    seed=module('policy').rollout_seed
    assert seed(404,150,49,127)==405506399
    with pytest.raises(ValueError):seed(404,151,0,0)
    with pytest.raises(ValueError):seed(404,150,50,0)


def test_u150_recovery_and_retention_preserve_latest_model(tmp_path):
    from phase3.common import write_new
    write_new(tmp_path/'episodes/u0150/train/e.json',{'partial':True})
    module('run').quarantine_uncommitted(tmp_path,149,150)
    assert (tmp_path/'interrupted-updates/recovery000/episodes/u0150/train/e.json').exists()
    (tmp_path/'models/u0145').mkdir(parents=True)
    (tmp_path/'models/u0150').mkdir()
    write_new(tmp_path/'windows/u0145-u0150/complete.json',{'update':150,'model':str(tmp_path/'models/u0150')})
    module('retention').collect(tmp_path,150)
    assert (tmp_path/'models/u0150').exists() and not (tmp_path/'models/u0145').exists()


def test_editor_opportunities_and_report_end_at_u150(tmp_path):
    from phase3.common import write_new
    editor=module('editor').Editor(tmp_path/'editor.sqlite3')
    try:assert editor.api.config.max_api_calls==30
    finally:editor.close()
    summary={'success_rate':.5,'mean_score':.6,'episodes':1000}
    write_new(tmp_path/'runs/reward/final-eval/summary.json',summary)
    result=module('report').report(tmp_path,'reward')
    assert result['test']=={'U150':summary}
    report=next((tmp_path/'runs/reward/reports').glob('*.md')).read_text()
    assert '/ 150' in report and '| U150 |' in report
