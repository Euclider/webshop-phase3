import json
import pytest
from test_bank_protocol import module


def test_multinode_configuration_keeps_global_minibatch(tmp_path):
    cfg = module('training').configuration(model='/sft',root=tmp_path,arm='skillrl',bank_hash='a'*64,
        start=0,train_file='/train',dev_file='/dev',nnodes=2,gpus_per_node=8,ray_address='auto')
    assert cfg.trainer.nnodes*cfg.trainer.n_gpus_per_node == 16
    assert cfg.ray_init.address == 'auto' and 'num_cpus' not in cfg.ray_init
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
    assert not cfg.actor_rollout_ref.ref.fsdp_config.cpu_offload


def test_preflight_never_admits_cpu_receipt_or_missing_parity():
    acceptable = module('preflight').acceptable
    assert not acceptable({'ranks':[]})
    ranks=[{'rank':i,'device':'NVIDIA B200','router_passed':True,'max_logprob_error':0.,
            'generation_passed':True,'finite':True,'fast_linear_attention':True} for i in range(16)]
    assert acceptable({'ranks':ranks})
    ranks[4]['max_logprob_error']=.02
    assert not acceptable({'ranks':ranks})
    ranks[4]['max_logprob_error']=0.
    ranks[4]['fast_linear_attention']=False
    assert not acceptable({'ranks':ranks})


def test_report_counts_candidate_rejections_separately_from_rollbacks(tmp_path):
    root=tmp_path/'runs/reward'
    p=root/'windows/u0000-u0005/evolution';p.mkdir(parents=True)
    (p/'complete.json').write_text(json.dumps({'update':5,'accepted':False,'candidate_rejected':True,
        'rollback_count':0,'editor_calls':1,'proposed_operations':{'ADD':1,'mutation_units':1},
        'candidate_skills':5,'editor_trajectories':12,'bank_size_after':54}))
    result=module('report').report(tmp_path,'reward')
    assert result['candidate_rejections']==1 and result['rollbacks']==0
    assert result['accepted_mutation_units']==0 and result['proposed_mutation_units']==1
    assert result['editing_events']==1 and result['completed_rl_updates']==0


def test_prepare_rejects_same_sft_and_raw_router_snapshot_before_any_environment_load(tmp_path):
    with pytest.raises(ValueError, match='distinct'):
        module('prepare').validate_paths('/same','/same')
