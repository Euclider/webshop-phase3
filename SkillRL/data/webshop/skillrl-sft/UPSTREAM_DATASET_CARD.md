---
dataset_info:
- config_name: alfworld
  features:
  - name: instruction
    dtype: string
  - name: output
    dtype: string
  splits:
  - name: train
    num_bytes: 42191393
    num_examples: 7486
  download_size: 5617790
  dataset_size: 42191393
- config_name: search
  features:
  - name: instruction
    dtype: string
  - name: output
    dtype: string
  splits:
  - name: train
    num_bytes: 6659585
    num_examples: 1214
  download_size: 1050795
  dataset_size: 6659585
- config_name: webshop
  features:
  - name: instruction
    dtype: string
  - name: output
    dtype: string
  splits:
  - name: train
    num_bytes: 12351072
    num_examples: 2553
  download_size: 1652500
  dataset_size: 12351072
configs:
- config_name: alfworld
  data_files:
  - split: train
    path: alfworld/train-*
- config_name: search
  data_files:
  - split: train
    path: search/train-*
- config_name: webshop
  data_files:
  - split: train
    path: webshop/train-*
license: mit
language:
- en
tags:
- reinforcement-learning
- embodied-ai
- instruction-following
- SFT
- agent
size_categories:
- 10K<n<100K

---

# SkillRL-SFT-Data

This is the **Supervised Fine-Tuning (SFT) dataset** used in the paper [SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning](https://arxiv.org/abs/2602.08234).

SkillRL-SFT-Data provides instruction-output pairs for training base agent policies on three interactive decision-making environments: **ALFWorld**, **WebShop**, and **Search**. Each example contains a structured instruction with retrieved skill context from the hierarchical SkillBank and the corresponding expert action output.

## Dataset Summary

| Config     | Environment                                         | Examples | Description                                                  |
| ---------- | --------------------------------------------------- | -------- | ------------------------------------------------------------ |
| `alfworld` | [ALFWorld](https://github.com/alfworld/alfworld)    | 7,486    | Embodied household tasks (pick & place, clean, heat, cool, examine, etc.) |
| `webshop`  | [WebShop](https://github.com/princeton-nlp/WebShop) | 2,553    | Web-based shopping navigation tasks                          |
| `search`   | Search                                              | 1,214    | Multi-step web search QA tasks (NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle) |

**Total**: 11,253 instruction-output pairs.

## Data Format

Each example contains two fields:

- **`instruction`**: A detailed prompt including the task goal, retrieved relevant experience from the hierarchical SkillBank (general principles, task-specific skills, and mistakes to avoid), current environment observation, and admissible actions.
- **`output`**: The expected agent response, consisting of a step-by-step reasoning process wrapped in `<think>` tags followed by the selected action in `<action>` tags.

## Usage

```python
from datasets import load_dataset

# Load a specific config
alfworld_data = load_dataset("Jianwen/SkillRL-SFT-Data", "alfworld")
webshop_data = load_dataset("Jianwen/SkillRL-SFT-Data", "webshop")
search_data = load_dataset("Jianwen/SkillRL-SFT-Data", "search")
```

## Related Models

| Environment | SFT Checkpoint                                               | RL Checkpoint                                                |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ALFWorld    | [Alfworld-7B-SFT](https://huggingface.co/Jianwen/Alfworld-7B-SFT) | [Alfworld-7B-RL](https://huggingface.co/Jianwen/Alfworld-7B-RL) |
| WebShop     | [Webshop-7B-SFT](https://huggingface.co/Jianwen/Webshop-7B-SFT) | [Webshop-7B-RL](https://huggingface.co/Jianwen/Webshop-7B-RL) |
| Search      | [Search-7B-SFT](https://huggingface.co/Jianwen/Search-7B-SFT) | [Search-7B-RL](https://huggingface.co/Jianwen/Search-7B-RL)  |

All models are fine-tuned from [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). SFT checkpoints are trained on this dataset; RL checkpoints are further optimized via recursive skill-augmented reinforcement learning.

## Related Resources

- **Paper**: [SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning](https://arxiv.org/abs/2602.08234)
- **Code**: [https://github.com/aiming-lab/SkillRL](https://github.com/aiming-lab/SkillRL)

## Citation

```bibtex
@article{xia2026skillrl,
  title={SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning},
  author={Xia, Peng and Chen, Jianwen and Wang, Hanyang and Liu, Jiaqi and Zeng, Kaide and Wang, Yu and Han, Siwei and Zhou, Yiyang and Zhao, Xujiang and Chen, Haifeng and others},
  journal={arXiv preprint arXiv:2602.08234},
  year={2026}
}
```