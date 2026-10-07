# Official SkillRL WebShop cold-start SFT data

This directory includes the **unchanged original parquet**, not tokenized/model-specific training outputs:

- File: `train-00000-of-00001.parquet`
- Examples: **2,553**; fields: `instruction`, `output`
- SHA256: `2c4f045b18a7ffabf7779f8e0e416913debae6427cb70a323c79e30f34d2d051`
- Source: [Jianwen/SkillRL-SFT-Data](https://huggingface.co/datasets/Jianwen/SkillRL-SFT-Data)
- Pinned [original file](https://huggingface.co/datasets/Jianwen/SkillRL-SFT-Data/resolve/bd3996c4e863ac59e6e4ab35549cf3741faf5e4f/webshop/train-00000-of-00001.parquet)
- Upstream-declared license: MIT; preserved [dataset card](UPSTREAM_DATASET_CARD.md).

The paper's approximate 2,400 count is not used to silently subsample the 2,553-row release. The instructions include upstream retrieved skills; think/action supervision is preserved. Our adapter only applies the Qwen3.5 chat template and response loss mask, rejecting truncation or changed source hashes.

From `SkillRL/`:

```bash
python -m webshop_phase3.sft prepare \
  --data data/webshop/skillrl-sft/train-00000-of-00001.parquet \
  --model "$WEBSHOP_BASE_MODEL" --output /shared/ws3/sft-prepared
```

Preparation does not start SFT. Distributed training uses LR1e-4, global batch16 and3epochs; see the root startup guide. All three RL arms share one completed SFT model, while the router remains the original model.

The associated initial 54-skill bank is `memory_data/webshop/claude_style_skills.json`, SHA256 `79c6c60b6757b6b730e7471b537781936ce0e9cdbd8df6188b57c535893aec20`. No new skills are generated as part of data packaging.
