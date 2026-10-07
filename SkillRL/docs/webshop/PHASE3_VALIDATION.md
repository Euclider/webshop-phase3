# WebShop Phase3 implementation validation

Date: 2026-10-07. Upstream: `Euclider/SkillScope@f2dd4a14a15d1751c81da3fa64e4c4c7cbb205e9`.
Implementation is in an isolated archive copy. No main-thread code/process changes, SFT/RL runs, paid editor calls, git commit or push.

## CPU evidence

Environment: Python3.12, torch2.11.0, Transformers5.10.4, vLLM0.22.0, Ray2.43.0. GPUs hidden for tests. Missing native-WebShop dependencies and JDK21 were supplied through an isolated overlay, without modifying the shared venv.

- Final 150-update release: `tests/webshop_phase12` + `tests/webshop_phase3`: **68 passed** (29 existing state/router tests +39 Phase3 tests), including last-window U145→U150 training/resume, sampling, checkpoint retention, editor budget and report labels. SFT and episode horizon are unchanged.
- Historical broader run before the 150-update extension, adding `tests/phase3`: **313 passed, 1 skipped, 3 failed**. The three failures also reproduce against an unmodified upstream archive in the same environment:
  - `test_router_readiness_probe_does_not_inherit_editor_secret` (LogicBench-specific environment expectation).
  - `test_capture_single_step_actual_batch_and_versions` (hardcoded absent `/home/wangyifan/model/Qwen3.5-4B`).
  - `test_banked_window_parquet_resume_preserves_numpy_chat_cells` (LogicBench parquet/JSON equality).
- Bare full-repository pytest stops at **13 collection errors**, including missing optional flash_attn/pyext/Megatron/SGLang dependencies, duplicate test import names and legacy vLLM imports. This is not a green full-project test result.

New coverage includes bank growth/retirement/version checks, dynamic all-candidate router/cache isolation, actual OLD-batch capture, official-aligned single/multinode GRPO configuration, reward selection vs all-failure baseline, paired native-score gate, strict frozen arm, report accounting, missing GPU acceptance rejection, checkpoint retention, export/partial-update preservation, SFT directory synchronization, deterministic per-request seeds and a simulated three-arm **150-update** schedule with resume. It seals30 windows per arm and does not repeat completed updates/editor calls on restart. The simulation uses synthetic training/editor/evaluation backends; it is **not** a training result.

The bare full-repository command was repeated after the150-update extension and still stopped at the following13 collection errors; no claim is made that it passed:

```
tests/gpu_utility/test_torch_functional.py
tests/models/test_transformer.py
tests/models/test_transformers_ulysses.py
tests/sandbox/test_sandbox.py
tests/single_controller/base/test_decorator.py
tests/skill_router/test_skillrl_embedding_router.py
tests/utils/gpu_tests/megatron/test_pipeline_parallel.py
tests/workers/rollout/test_sglang_async_rollout_search_tools.py
tests/workers/rollout/test_sglang_async_rollout_sf_tools.py
tests/workers/rollout/test_sglang_async_rollout_w_tools.py
tests/workers/rollout/test_sglang_spmd.py
tests/workers/rollout/test_vllm_multi_turn.py
tests/workers/rollout/test_vllm_tool_calling.py
```

An actual randomly initialized tiny Qwen3.5 CPU forward verifies masked-padding versus trimmed-input output parity. This does not establish BF16 GPU kernel or native FSDP parity.

## Actual official SFT data adaptation (no training)

- All **2,553** official rows processed with the real local Qwen3.5 tokenizer.
- Maximum encoded length: **1,975**; minimum supervised tokens per row: **20**; no truncation.
- Original parquet SHA256: `2c4f045b18a7ffabf7779f8e0e416913debae6427cb70a323c79e30f34d2d051`.
- Encoded parquet SHA256 in this environment: `08f5ae37d722d4da82cbd0fb929cf81d02d374ca7a73854452c4e0711f5a31b9`.
- Original multi-skill instructions and think/action output retained; duplicate Qwen think-prefix regression checked. No invented task IDs/history, and no claim that published SFT has verified zero test overlap.

## Target-server checks still required

The current machine is not the requested idle 16×B200 cluster. Therefore no claimed target throughput/speedup, GPU admission receipt, real 16-rank optimizer/checkpoint restore test, full product-index rollout or editor API compatibility result is supplied.

`webshop_phase3.preflight --gpu` is implemented for the target server: real frozen-router batch/cache/sleep-wake, policy vLLM generation, Qwen fast linear-attention kernel availability and dense/trimmed chosen-logprob error ≤1e-3. CPU-only preflight cannot authorize the run. Even GPU inference preflight explicitly marks `native_training_acceptance=false`; it does not claim an optimizer/restore test happened.

Formal execution is an explicit future action. First native update and first complete edit window still need real runtime acceptance. Any protocol/parity failure stops rather than silently changing the inference backend or dropping evidence. API failures/ambiguous requests require manual reconciliation, without automatic paid retry.

## Delivery and scope

GitHub publication check (2026-10-07): the independent release tree passes all **68 WebShop tests** for the final150-update setting. Its environment dependency file resolves to **305 packages**; this is dependency resolution, not a clean-room GPU installation result. The original **2,553-row SFT parquet** and **54-skill bank** are included with hashes and source/licensing notes. Historical experiment reports and unrelated benchmark datasets are omitted; historical broader-suite results above refer to the earlier full upstream workspace, not a claim that every optional test in this reduced release passes.

The source bundle excludes model weights/checkpoints, experiment artifacts and full product data/indexes. It contains the original skill bank, environment/source code, three-arm implementation, tests, environment requirements and startup/protocol documentation. SHA256 manifest verification checks the actual delivered files; it is separate from the old 47-file Phase12 handoff receipt.

Relevant integration changes are limited to WebShop bank/LLM-router hooks, explicit WebShop dispatch in Phase3 capture/resume, registered B200 vLLM profile, and WebShop-only request seeds. Other benchmark behavior is not intentionally changed. Review was performed in-session, not by a separate reviewer.
