# Third-party sources and retained notices

- SkillScope base: `Euclider/SkillScope@f2dd4a14a15d1751c81da3fa64e4c4c7cbb205e9`. This repository adds a WebShop-specific Phase3 driver and deployment package; it is not an official SkillRL release.
- SkillRL code and initial bank: [aiming-lab/SkillRL](https://github.com/aiming-lab/SkillRL), MIT, Copyright (c)2026 AIMING Lab. The original notice is retained in [SkillRL/LICENSE](SkillRL/LICENSE).
- SkillRL SFT data: [Jianwen/SkillRL-SFT-Data](https://huggingface.co/datasets/Jianwen/SkillRL-SFT-Data), WebShop subset at revision `bd3996c4e863ac59e6e4ab35549cf3741faf5e4f`; the upstream dataset card declares MIT. Its original card, unchanged parquet and provenance are included in `SkillRL/data/webshop/skillrl-sft/`. No authorship of that dataset is claimed here.
- verl and verl-agent/GiGPO components retain their Apache-2.0 source headers and upstream notices. They are not relicensed by the SkillRL MIT notice.
- Native WebShop retains its [Princeton license](SkillRL/agent_system/environments/env_package/webshop/webshop/LICENSE.md) and [upstream README](SkillRL/agent_system/environments/env_package/webshop/webshop/README.md). Full product data/indexes are not redistributed here; follow upstream data terms when obtaining them.

The SFT release has no original task IDs sufficient to verify zero overlap with the evaluation split; provenance is not a non-contamination guarantee. Cite the relevant upstream work and use the adapted-baseline wording in the experiment protocol.
