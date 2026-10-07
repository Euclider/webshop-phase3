import pytest
from test_bank_protocol import module


def test_readout_token_mean_preserves_zero_advantage_positions_and_signed_balance():
    summary = module('readout').summarize([
        {'skill_id':'a','task_id':1500,'trajectory_id':'t1','P_int':-2.,'direction_valid':True,
         'C_upd_centered':.3,'delta_centered_norm':3.,'gate':True,'D_contribution':2.},
        {'skill_id':'a','task_id':1501,'trajectory_id':'t2','P_int':1.,'direction_valid':True,
         'C_upd_centered':.5,'delta_centered_norm':5.,'gate':True,'D_contribution':0.},
        {'skill_id':'a','task_id':1500,'trajectory_id':'t1','P_int':0.,'direction_valid':False,
         'C_upd_centered':0.,'delta_centered_norm':4.,'gate':False,'D_contribution':0.}])
    assert summary[0]['D_sign_balance'] == 0.
    assert summary[0]['D_signed'] == pytest.approx(1/3)
    assert summary[0]['M_delta_centered'] == 4. and summary[0]['n_loss_tokens'] == 3


def test_evaluation_shards_cover_every_task_seed_once_and_merge_rejects_gaps():
    evaluation = module('evaluation')
    shards = [evaluation.shard_pairs([1,2,3], [11,12], rank=i, world=4) for i in range(4)]
    assert sorted(sum(shards, [])) == [(1,11),(1,12),(2,11),(2,12),(3,11),(3,12)]
    rows = [{'task_id':t,'eval_seed':s,'task_score':.5,'success':False,'steps':2,
             'prompt_tokens':100,'completion_tokens':10} for t,s in sum(shards,[])]
    assert evaluation.summarize(rows, [1,2,3], [11,12])['mean_score'] == .5
    with pytest.raises(ValueError): evaluation.summarize(rows[:-1], [1,2,3], [11,12])
    with pytest.raises(ValueError): evaluation.summarize(rows+[rows[0]], [1,2,3], [11,12])


def test_boundary_driver_reuses_completed_training_and_never_trains_on_default_cli(tmp_path):
    orchestration = module('run')
    calls = []
    def stage(): calls.append(1); return {'checkpoint':'u5'}
    assert orchestration.completed_stage(tmp_path, 'train', {'start':0}, stage) == {'checkpoint':'u5'}
    assert orchestration.completed_stage(tmp_path, 'train', {'start':0}, stage) == {'checkpoint':'u5'}
    assert calls == [1]
    with pytest.raises(ValueError): orchestration.completed_stage(tmp_path, 'train', {'start':5}, stage)


def test_policy_request_keeps_all_visible_text_and_records_request_seeds():
    policy = module('policy')
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs): return messages[0]['content']
        def encode(self, text, **kwargs): return list(range(len(text)))
    request = {'prompt':'full state', 'seed':5, 'temperature':.4, 'max_new_tokens':512}
    prompt, sampling = policy.prepare_request(Tokenizer(), request)
    assert prompt['prompt_token_ids'] == list(range(10)) and sampling['seed'] == 5
    with pytest.raises(ValueError): policy.prepare_request(Tokenizer(), {**request,'prompt':'x'*16385})
