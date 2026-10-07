"""Native verl training; official WebShop GRPO recipe with declared adaptations."""
import argparse
from pathlib import Path

from phase3.common import require
from webshop_phase12.assets import ROOT
from .protocol import ARMS, UPDATES, WINDOW


def configuration(*, model, root, arm, bank_hash, start, train_file, dev_file, resume_update=None,
                  nnodes=1, gpus_per_node=16, ray_address='local'):
    from omegaconf import OmegaConf
    from scripts.inspect_skillrl_alignment import load_config
    from skillnet_cohort.inference import registration, apply_training, WEBSHOP_B200_PROFILE
    require(arm in ARMS and start in range(0, UPDATES, WINDOW), 'Invalid arm/window')
    require(nnodes*gpus_per_node == 16, 'This recipe requires sixteen total GPUs')
    root = Path(root).resolve()
    cfg = load_config()
    cfg.alignment_run = dict(seed=404, model_path=str(model), train_files=str(train_file), val_files=str(dev_file),
        run_id=f'webshop-phase3-{arm}-s404', output_dir=str(root/'checkpoints'), archive_dir=str(root/'unused'),
        router_cache=str(root/'router.sqlite3'), ray_temp_dir=f'/tmp/ws3-{arm}-{start}')
    cfg.webshop_phase12 = {'enabled': True, 'active_only_generation': True}
    cfg.webshop_run = {'seed': 404}
    resumed = start if resume_update is None else resume_update
    require(start <= resumed < start + 5, 'Resume outside current window')
    cfg.phase3 = dict(enabled=True, domain='webshop', root=str(root), branch_id=arm, bank_sha256=bank_hash,
        segment_start=start, segment_end=start+5, resume_update=resumed, penultimate_recovery_checkpoint=False)
    cfg.phase2 = {'enabled': False}
    cfg.data.train_batch_size = 16
    cfg.data.val_batch_size = 64
    cfg.data.max_prompt_length = 16384
    cfg.data.max_response_length = 512
    cfg.data.filter_overlong_prompts = False
    cfg.data.truncation = 'error'
    cfg.data.shuffle = False
    cfg.data.seed = 404
    cfg.env.env_name = 'webshop'
    # Match the actual native ShopWorld implementation, not inherited ALF defaults.
    cfg.env.webshop.use_small = False
    cfg.env.webshop.human_goals = True
    cfg.env.max_steps = 50
    cfg.env.rollout.n = 8
    cfg.env.use_skills_only_memory = False
    cfg.env.skills_only_memory.step_routing.enabled = False
    cfg.env.phase1_archive.enabled = False
    cfg.actor_rollout_ref.model.external_lib = 'webshop_phase3.model_patch'
    cfg.actor_rollout_ref.actor.ppo_mini_batch_size = 64
    cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 4
    cfg.actor_rollout_ref.actor.entropy_coeff = 0.
    cfg.actor_rollout_ref.actor.optim.total_training_steps = UPDATES
    cfg.actor_rollout_ref.actor.fsdp_config.optimizer_offload = False
    cfg.actor_rollout_ref.actor.fsdp_config.cpu_shard_init = True
    cfg.actor_rollout_ref.ref.fsdp_config.cpu_shard_init = True
    cfg.actor_rollout_ref.ref.fsdp_config.param_offload = False
    cfg.actor_rollout_ref.ref.fsdp_config.cpu_offload = False
    for section in (cfg.actor_rollout_ref.actor, cfg.actor_rollout_ref.ref):
        section.response_logits_only = True
        section.trim_common_padding = True
    apply_training(cfg, registration(WEBSHOP_B200_PROFILE))
    cfg.actor_rollout_ref.rollout.prompt_length = 16384
    cfg.actor_rollout_ref.rollout.response_length = 512
    cfg.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 8
    cfg.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = 4
    cfg.trainer.n_gpus_per_node = gpus_per_node
    cfg.trainer.nnodes = nnodes
    cfg.trainer.total_training_steps = UPDATES
    cfg.trainer.total_epochs = 1
    cfg.trainer.test_freq = 0  # External paired Dev gate; no dummy monitor env.
    cfg.trainer.save_freq = 1  # Latest native optimizer/RNG; small sharded model.
    cfg.trainer.max_actor_ckpt_to_keep = 2
    cfg.trainer.max_critic_ckpt_to_keep = 2
    cfg.trainer.val_before_train = False
    cfg.trainer.resume_mode = 'resume_path' if resumed else 'disable'
    cfg.trainer.resume_from_path = str(root/'checkpoints'/f'global_step_{resumed}') if resumed else None
    cfg.ray_init.num_cpus = 64
    cfg.ray_init.object_store_memory = 16 * 2**30
    cfg.ray_init.include_dashboard = False
    if ray_address != 'local':
        # Resource declarations are forbidden when attaching to an existing cluster.
        cfg.ray_init = {'address': ray_address}
    import os
    cfg.ray_init.runtime_env = {'env_vars': {k:v for k,v in os.environ.items()
        if k.startswith('WEBSHOP_') or k in ('JAVA_HOME','JVM_PATH','PYTHONPATH')}}
    OmegaConf.resolve(cfg)
    return cfg


def main():
    p = argparse.ArgumentParser()
    for key in ('model', 'root', 'arm', 'bank-hash', 'train-file', 'dev-file'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--resume-update', type=int)
    p.add_argument('--nnodes', type=int, default=1)
    p.add_argument('--gpus-per-node', type=int, default=16)
    p.add_argument('--ray-address', default='local')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    cfg = configuration(**{k: v for k, v in vars(a).items() if k != 'execute'})
    if not a.execute:
        from omegaconf import OmegaConf
        print(OmegaConf.to_yaml(cfg)); return
    from verl.trainer.main_ppo import run_ppo
    run_ppo(cfg)


if __name__ == '__main__': main()
