"""Compact Phase3 records: no vocabulary tensors or independent utility gold."""
import io
import json
from pathlib import Path

from phase1.archive import jsonable
from phase3.common import require, write_new


def archive_batch(batch, *, update, config, tokenizer=None):
    if config.phase3.branch_id != 'reward' or update != config.phase3.segment_start + 1:
        return
    import torch
    from skillnet_cohort.common import file_hash, write_new_bytes
    keys = ('prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids', 'advantages', 'old_log_probs')
    tensors = {key: batch.batch[key].detach().cpu().clone() for key in keys}
    n = tensors['responses'].shape[-1]
    src = 'loss_mask' if config.actor_rollout_ref.rollout.multi_turn.enable else 'attention_mask'
    tensors['actual_loss_mask'] = batch.batch[src][:, -n:].detach().cpu().bool()
    metadata = [json.loads(str(x)) for x in batch.non_tensor_batch['phase3_metadata']]
    require(all(m['global_update'] == update and m['info']['bank_sha256'] == config.phase3.bank_sha256 for m in metadata), 'Foreign batch')
    require(torch.isfinite(tensors['advantages'][tensors['actual_loss_mask']]).all(), 'Nonfinite advantage')
    record = {'schema': 'webshop.phase3.direction_batch.v1', 'tensors': tensors, 'metadata': metadata,
              'bank_sha256': config.phase3.bank_sha256, 'update': update,
              'temperature': float(batch.meta_info['temperature'])}
    path = Path(config.phase3.root)/'direction_batches'/f'u{update:04d}.pt'
    buffer = io.BytesIO(); torch.save(record, buffer)
    write_new_bytes(path, buffer.getvalue())
    write_new(path.with_suffix('.json'), {'sha256': file_hash(path), 'rows': len(metadata),
              'loss_tokens': int(tensors['actual_loss_mask'].sum()), 'full_vocab_saved': False})


def archive_episodes(*, config, is_train, update, batches, infos, trajectory_ids, episode_rewards):
    require(is_train, 'WebShop Phase3 validation is external, not editor evidence')
    for index, (batch_steps, info_steps) in enumerate(zip(batches, infos)):
        steps, active = [], []
        for step, (batch, info) in enumerate(zip(batch_steps, info_steps)):
            if not bool(jsonable(batch['active_masks'])): continue
            require(info['bank_sha256'] == config.bank_sha256, 'Foreign episode bank')
            mask = jsonable(batch['attention_mask']); n = len(jsonable(batch['responses']))
            fields = ('observation', 'next_observation', 'raw_model_output', 'admissible_actions', 'selected_skill_id',
                      'skill_version_sha256', 'skill_router_api', 'visible_state', 'visible_state_sha256', 'prompt_text')
            steps.append({**{k: jsonable(info.get(k)) for k in fields}, 'step_index': step,
                'action': info['projected_action'], 'is_action_valid': bool(jsonable(info['is_action_valid'])),
                'prompt_tokens': sum(mask[:-n]), 'completion_tokens': sum(mask[-n:]), 'reward': jsonable(batch['rewards'])})
            active.append(info)
        require(active and active[0]['task_id'] >= 1500, 'Nontraining/empty editor evidence')
        trajectory = str(trajectory_ids[index])
        require('/' not in trajectory and '..' not in trajectory, 'Unsafe trajectory ID')
        record = {'trajectory_id': trajectory, 'branch_id': config.branch_id, 'bank_sha256': config.bank_sha256,
            'global_update': update, 'sampling_policy_update': update-1, 'split': 'train',
            'task_id': int(active[0]['task_id']), 'game_id': f"webshop:{active[0]['task_id']}",
            'task': active[0]['task_description'], 'success': bool(active[-1]['won']),
            'task_score': float(active[-1]['task_score']), 'episode_return': jsonable(episode_rewards[index]), 'steps': steps}
        write_new(Path(config.root)/'episodes'/f'u{update:04d}'/'train'/f'{trajectory}.json', record)
