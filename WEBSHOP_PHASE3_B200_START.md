# WebShop Phase3：16×B200启动说明

本交付只准备代码、数据适配和验证；**没有启动SFT或RL**。完整协议在 [PHASE3_PROTOCOL.md](SkillRL/docs/webshop/PHASE3_PROTOCOL.md)。默认入口不会训练，必须显式提供`--execute`。

部署包解压后，先在源码根目录运行：

```bash
python3 deploy/webshop/package_phase3.py verify
```

本包的`WEBSHOP_PHASE3_BUNDLE.json`校验新增Phase3和适配后的全部交付文件。旧`deploy/webshop/handoff-manifest.json`仅证明原始Phase12发布，不适用于本次有意改动后的文件；不要用旧校验冒充新版本验收。

## 1. 新环境与数据

两节点必须使用相同环境和相同的共享绝对路径；模型、运行目录、商品数据/搜索索引和源码都要可访问。默认2节点×8卡，也支持1节点×16卡。不要修改现有实验的site-packages。

```bash
cd SkillRL
python3.12 -m venv /path/to/webshop-p3-venv
source /path/to/webshop-p3-venv/bin/activate
pip install -r requirements-webshop-phase3-b200.lock
pip install --no-deps -e .
pip install flash-linear-attention==0.5.2
pip install causal-conv1d==1.7.0 --no-build-isolation
python -m spacy download en_core_web_sm
export PYTHONPATH="$PWD"
export JAVA_HOME=/path/to/jdk21
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export WEBSHOP_ENV_ROOT="$PWD/agent_system/environments/env_package/webshop/webshop"
export WEBSHOP_BASE_MODEL=/shared/models/Qwen3.5-4B
```

FLA/causal-conv1d是GPU快路径要求；wheel须与B200/CUDA/PyTorch匹配，不能把本机5090环境当成B200已验收。完整WebShop商品数据和Pyserini索引见下面的数据准备步骤；仓库不包含它们。实际代码依赖`en_core_web_sm`。需要足够CPU内存供各GPU评估进程加载商品目录；具体峰值仍需目标机实测。

`.txt`列出直接依赖，`.lock`固定解析到的305个依赖版本（2026-10-07、Python3.12）。解析成功不等于已完成该新环境的B200安装验收；CUDA驱动、编译器和单独构建的FLA/causal-conv1d仍需目标机核验。不要直接使用上游README中的旧vLLM安装版本。

### 商品数据与搜索索引（新目录）

不要在固定训练环境里直接执行原生`setup.sh`：它包含旧Python依赖及conda安装命令。下面沿用其公开下载ID，只准备数据，不更改环境包。也可以把另一台服务器的同版完整数据和索引复制到同样位置。Google Drive不可达时请手动下载，不用small子集替代正式评估。

```bash
cd "$WEBSHOP_ENV_ROOT"
mkdir -p data
gdown 'https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB' -O data/items_shuffle.json
gdown 'https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi' -O data/items_ins_v2.json
gdown 'https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O' -O data/items_human_ins.json
cd search_engine
mkdir -p resources resources_100 resources_1k resources_100k
python - <<'PY'
import sys, runpy
from pathlib import Path
sys.path.insert(0, '..')
from web_agent_site import utils
# Conversion must use the full files; original utils defaults are the 1k preview.
utils.DEFAULT_FILE_PATH = str(Path('../data/items_shuffle.json').resolve())
utils.DEFAULT_ATTR_PATH = str(Path('../data/items_ins_v2.json').resolve())
runpy.run_path('convert_product_file_format.py', run_name='__main__')
PY
python -m pyserini.index.lucene --collection JsonCollection --input resources \
  --index indexes --generator DefaultLuceneDocumentGenerator --threads 16 \
  --storePositions --storeDocvectors --storeRaw
cd "$PYTHONPATH"
```

`PYTHONPATH`在上面的安装命令中设置为`SkillRL`源码根目录。原生商品数据受上游条款约束，与本仓库已附的SkillRL SFT监督数据不是同一份数据。

CPU测试（不加载4B权重、不训练）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -B -m pytest -q tests/webshop_phase12 tests/webshop_phase3
```

## 2. SFT数据准备及未来冷启动

本仓库已包含官方HF WebShop parquet，无需再次下载；代码校验固定SHA和2,553条，不按“约2,400”静默采样。来源与许可证见`SkillRL/data/webshop/skillrl-sft/README.md`。

```bash
python -m webshop_phase3.sft prepare \
  --data data/webshop/skillrl-sft/train-00000-of-00001.parquet \
  --model "$WEBSHOP_BASE_MODEL" \
  --output /shared/ws3/sft-prepared
```

下面命令**会训练**，本次未执行。在两节点分别设置`NODE_RANK=0/1`后运行；如果单机16卡，改为`--nnodes=1 --nproc-per-node=16`。

```bash
torchrun --nnodes=2 --nproc-per-node=8 --node_rank="$NODE_RANK" \
  --master_addr="$HEAD_IP" --master_port=29500 \
  -m webshop_phase3.sft train --prepared /shared/ws3/sft-prepared \
  --output /shared/ws3/sft --execute
```

恢复SFT需明确指定同一output及`--resume /shared/ws3/sft/checkpoint-N`。只有`complete.json`证明480次optimizer steps全部结束；最终checkpoint为`/shared/ws3/sft/final`。

## 3. Ray集群与三臂准备

在专用于本实验的空闲资源上建立集群；不要停止或重用其他实验的Ray作业。两节点资源要各自申明8卡，源码/环境路径一致：

```bash
# head
ray start --head --port=6379 --num-gpus=8 --object-store-memory=17179869184
# worker
ray start --address="$HEAD_IP:6379" --num-gpus=8 --object-store-memory=17179869184
```

SFT完成后在head准备新root：

```bash
python -m webshop_phase3.prepare \
  --base-model "$WEBSHOP_BASE_MODEL" --sft-model /shared/ws3/sft/final \
  --sft-complete /shared/ws3/sft/complete.json --root /shared/ws3/phase3 \
  --nnodes 2 --gpus-per-node 8 --ray-address auto

python -m webshop_phase3.preflight --manifest /shared/ws3/phase3/manifest.json
python -m webshop_phase3.run --manifest /shared/ws3/phase3/manifest.json
```

以上两条默认只检查/列计划，不训练。`prepare`需要实际商品数据初始化环境，但不会采样policy轨迹。router权重固定为原始模型；请勿把`WEBSHOP_BASE_MODEL`设成SFT结果。

## 4. 目标GPU预检与未来正式执行

GPU预检会占用16张已分配B200，进行真实推理，不执行optimizer更新或付费API：

```bash
python -m webshop_phase3.preflight --manifest /shared/ws3/phase3/manifest.json --gpu
```

它校验FLA快路径、router批处理/缓存/sleep-wake、vLLM策略生成和dense/trimmed chosen-logprob误差≤1e-3。
这**不是**16卡原生训练恢复或真实编辑API的闭环证明。初次上线应检查首个真实训练update和首个编辑窗口的日志；失败会停止并保留证据，不自动换HF或重试付费API。

未来明确决定运行时，在head安全注入`SKILLRL_PHASE3_EDITOR_API_KEY`，不要写进源码/manifest。正式顺序reward→skillrl→frozen_bank_grpo；**每臂150轮RL，每5轮一个窗口，共30窗**。两个演化臂各最多30次编辑API调用，冻结臂为0；完整失败证据可能很长，请核对网关上下文及余额。

```bash
python -m webshop_phase3.run --manifest /shared/ws3/phase3/manifest.json --execute
```

只执行reward可加`--arms reward`；默认无`--execute`绝不会开跑。队列有独占锁；已完成阶段按输入hash复用，原生checkpoint恢复model/optimizer/RNG/dataloader；不重跑完成窗口。
编辑账本遇到不确定/失败调用会停机等待人工核对，不会盲目重复计费。完整证据超网关上下文时也不会自动删轨迹。
未提交到有效checkpoint的更新记录移入`interrupted-updates/recoveryNNN`保留后，才从最新有效checkpoint重做该未完成更新；已完成更新不重跑。未完成HF导出保存在`incomplete-exports`；已校验的导出复用。

## 5. 输出与存储

- `runs/<arm>/metrics`：逐update训练指标；`episodes`：实际动作、状态、技能版本、reward/token/router记录。
- `windows/uXXXX-uYYYY/readout`：逐token标量、全部可评分skill和锁定排名；只reward臂计算。
- `windows/.../evolution`：编辑输入、proposal、bank、配对gate和接受/拒绝记录。
- `initial-eval`、`runs/<arm>/final-eval`：U0与U150各500 tasks×2 seeds的完整端点评估。
- `runs/<arm>/reports/report-<hash>.json/.md`：版本化报告，不覆盖已有报告。统计SR、平均score、训练曲线、proposal/accepted mutation units、拒绝数、真实rollback、tokens和API使用。
- `retention`：模型副本释放审计。窗口封存后只留最新native与HF，原始SFT、轨迹、训练批次、bank及报告不删。
- `logs/attempt-NNN.log`：每次尝试新文件；不会覆盖旧错误日志。

全仓还有其他benchmark和可选引擎测试，不等同于这组WebShop CPU回归。验收记录见 `SkillRL/docs/webshop/PHASE3_VALIDATION.md`。
