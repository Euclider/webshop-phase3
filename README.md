# WebShop Phase3 — SkillScope

Cold-start **Qwen3.5-4B SFT → three independent GRPO arms**, targeting **16×NVIDIA B200** (2 nodes ×8 GPUs or 1 node ×16 GPUs).

**Start here: [environment and experiment guide](WEBSHOP_PHASE3_B200_START.md).** The code is prepared and CPU-tested; no training results or target-B200 throughput claims are supplied.

## Included

| Component | Entry |
|---|---|
| Cold-start SFT, official-data adapter | [webshop_phase3/sft.py](SkillRL/webshop_phase3/sft.py) |
| Official WebShop SFT data: 2,553 original rows | [data + provenance](SkillRL/data/webshop/skillrl-sft/README.md) |
| Original SkillRL bank: 54 skills | [claude_style_skills.json](SkillRL/memory_data/webshop/claude_style_skills.json) |
| Three-arm runner, resume, editor, paired gate, reports | [webshop_phase3](SkillRL/webshop_phase3) |
| Per-step visible state and frozen local top-1 LLM router | [webshop_phase12](SkillRL/webshop_phase12) |
| 16×B200 environment requirements | [requirements](SkillRL/requirements-webshop-phase3-b200.txt) |
| Settings, evidence selection, cost and retention rules | [Phase3 protocol](SkillRL/docs/webshop/PHASE3_PROTOCOL.md) |
| State/bank/router prompt definition | [state/router protocol](SkillRL/docs/webshop/PROTOCOL.md) |
| Test coverage and unverified runtime boundaries | [validation](SkillRL/docs/webshop/PHASE3_VALIDATION.md) |

### Three arms

- **reward**: reward-directed `D_sign_balance`; top-5 naturally supported skills involved in old-batch failures and their corresponding complete failed trajectories.
- **skillrl**: failure-driven **adapted SkillRL baseline**; same old batch, all failures, complete bank, shared editor and validation gate. Not claimed to reproduce every native SkillRL evolution detail.
- **frozen_bank_grpo**: identical initialization/training/state/router, no bank editing or candidate gate.

All arms share the cold-start SFT model, initial bank and seed404 training task schedule, then run independently for **150 outer RL updates with 30 five-update windows**. Final full evaluation is at **U150**; each evolving arm allows up to30 editor calls, while the frozen arm allows none. Router weights stay at the original pre-SFT Qwen3.5-4B; changing a bank refreshes its catalog/cache identity, not router weights.

GRPO uses 16 tasks ×8 trajectories/update, learning rate1e-6, PPO minibatch64, microbatch4/GPU and KL0.01. Declared differences from the official SkillRL recipe are in the protocol. Acceleration is enabled: vLLM rollout, active-episode batching, frozen-router batching/prefix caching, response-only scoring and common-padding trimming. Actual B200 speed still needs measurement.

## Minimal sequence

```bash
git clone https://github.com/Euclider/webshop-phase3.git
cd webshop-phase3
python3 deploy/webshop/package_phase3.py verify
cd SkillRL
# Follow the environment guide; configure the original model and WebShop products/index.
python -m webshop_phase3.sft prepare \
  --data data/webshop/skillrl-sft/train-00000-of-00001.parquet \
  --model "$WEBSHOP_BASE_MODEL" --output /shared/ws3/sft-prepared
```

The guide then covers distributed SFT, task preparation, CPU/GPU preflight, Ray startup, three-arm execution and recovery. Training requires explicit `--execute`. Preparing/tokenizing SFT data is not training.

## Not included

No model weights/checkpoints, editor keys, historical run outputs, or full WebShop product data/search indexes. Obtain weights/products independently. The original SFT parquet and initial bank **are ordinary Git files**, not missing Git-LFS pointers. They are checked by SHA256 before use.

CPU tests cover the software contracts, not 16-rank FSDP training/restore, full product-data rollout or the real editor gateway. Those require target-server acceptance. Only WebShop-specific tests are the release gate; optional-engine and unrelated benchmark tests have documented upstream failures.

Based on [SkillScope f2dd4a1](https://github.com/Euclider/SkillScope/commit/f2dd4a14a15d1751c81da3fa64e4c4c7cbb205e9), using SkillRL/verl and native WebShop. Shared backend code is retained so the repository is self-contained; other benchmark scripts are not this experiment's entry points. See [third-party notices](THIRD_PARTY_NOTICES.md).
