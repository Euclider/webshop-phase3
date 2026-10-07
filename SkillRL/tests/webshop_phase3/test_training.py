from pathlib import Path
import pytest
import torch
from test_bank_protocol import module


def test_training_recipe_preserves_official_groups_and_local_minibatch(tmp_path):
    cfg = module('training').configuration(model='/model/sft', root=tmp_path, arm='reward',
        bank_hash='a'*64, start=5, train_file='/data/train.parquet', dev_file='/data/dev.parquet')
    assert cfg.data.train_batch_size * cfg.env.rollout.n == 128
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size // cfg.trainer.n_gpus_per_node == 4
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 4
    assert cfg.actor_rollout_ref.actor.optim.lr == 1e-6
    assert cfg.actor_rollout_ref.actor.entropy_coeff == 0.
    assert cfg.trainer.resume_from_path.endswith('global_step_5')
    assert cfg.phase3.segment_end == 10 and cfg.trainer.total_training_steps == 150
    assert cfg.actor_rollout_ref.rollout.name == 'vllm_v1'
    assert cfg.actor_rollout_ref.rollout.inference_profile.settings.max_model_len == 16896
    assert cfg.actor_rollout_ref.actor.trim_common_padding
    assert not cfg.env.phase1_archive.enabled and not cfg.phase2.enabled
    assert cfg.trainer.test_freq == 0  # External Dev gate, never dummy native monitor.
    assert cfg.env.webshop.human_goals and not cfg.env.webshop.use_small


def test_padding_contract_masks_single_row_without_mutating_record():
    speed = module('numerics')
    hidden = torch.ones(1, 4, 2)
    masked = speed.mask_padding(hidden, torch.tensor([[0, 0, 1, 1]]))
    assert masked[0, :2].sum() == 0 and hidden.sum() == 8
    batch = {'input_ids': torch.tensor([[0, 0, 5, 6, 7, 0]]),
             'attention_mask': torch.tensor([[0, 0, 1, 1, 1, 0]]),
             'position_ids': torch.tensor([[0, 0, 0, 1, 2, 3]])}
    short = speed.trim_inputs(batch, 2)
    assert short['input_ids'].tolist() == [[5, 6, 7, 0]]
    assert batch['input_ids'].shape == (1, 6)


def test_sft_response_mask_preserves_official_output_and_no_truncation():
    class Tokenizer:
        eos_token_id = 2
        def apply_chat_template(self, messages, **kwargs):
            return 'USER:' + messages[0]['content'] + '\nASSISTANT:' + (messages[1]['content'] + '~' if len(messages) == 2 else '')
        def encode(self, text, **kwargs): return [ord(c)+10 for c in text]
    encode = module('sft').encode_example
    sample = {'instruction': 'state', 'output': '<think>x</think><action>click[a]</action>'}
    row = encode(sample, Tokenizer(), max_length=128)
    n = len('USER:state\nASSISTANT:')
    assert row['labels'][:n] == [-100] * n
    assert row['labels'][n:] == [ord(c)+10 for c in sample['output'] + '~']
    with pytest.raises(ValueError): encode(sample, Tokenizer(), max_length=8)


def test_webshop_episode_capture_uses_task_ids_not_alfworld_paths(tmp_path):
    from omegaconf import OmegaConf
    from phase3.capture import archive_episodes
    config = OmegaConf.create({'domain': 'webshop', 'root': str(tmp_path), 'bank_sha256': 'a'*64, 'branch_id': 'reward'})
    info = {'bank_sha256': 'a'*64, 'task_id': 1510, 'task_description': 'Buy item', 'won': False,
        'task_score': .5, 'projected_action': 'click[buy now]', 'is_action_valid': True,
        'selected_skill_id': 'gen_001', 'skill_version_sha256': 'b'*64, 'visible_state': {'step_index': 0}}
    batch = {'active_masks': True, 'attention_mask': [1, 1, 1], 'responses': [12], 'rewards': 0}
    archive_episodes(config=config, is_train=True, update=1, batches=[[batch]], infos=[[info]],
                     trajectory_ids=['traj1'], episode_rewards=[0])
    import json
    record = json.loads((tmp_path/'episodes/u0001/train/traj1.json').read_text())
    assert record['task_id'] == 1510 and record['task_score'] == .5
    assert record['sampling_policy_update'] == 0
    assert record['steps'][0]['visible_state'] == {'step_index': 0}
