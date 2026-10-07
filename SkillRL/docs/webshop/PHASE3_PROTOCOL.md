# WebShop Phase3：冻结状态路由、冷启动和三臂演化

协议日期：2026-10-07。基线源码：Euclider/SkillScope `f2dd4a14a15d1751c81da3fa64e4c4c7cbb205e9`。
用户确认：三臂、seed404、每臂150次RL更新、5更新窗口（共30窗）、16×B200；仅实现和测试，本机不训练。2026-10-07发布前按用户追加要求由50轮扩展至150轮；SFT与其余RL/路由参数不变。

## 数据、初始模型和状态

- 官方初始 bank：`memory_data/webshop/claude_style_skills.json`，54 条（15 general +39 task-specific）；原始 SHA256 `79c6c60b6757b6b730e7471b537781936ce0e9cdbd8df6188b57c535893aec20`。保持 ID、顺序、初始内容；common_mistakes 不作为独立技能注入。
- 官方 SFT：`Jianwen/SkillRL-SFT-Data/webshop/train-00000-of-00001.parquet`，**2,553 条**，不是论文约数2,400；SHA256 `2c4f045b18a7ffabf7779f8e0e416913debae6427cb70a323c79e30f34d2d051`。
- SFT LR=1e-4、global batch=16、epochs=3；全参数，response-only global token mean。AdamW、线性 LR、零 warmup/weight decay 是本适配的明确实现选择，不宣称这些未公开细节逐项复现官方 LLaMA-Factory。
- 官方原始 instruction/output 保留，包括多技能提示与 think/action 监督；只适配 Qwen chat template/loss mask，不伪造缺失历史。三个 RL 臂共用同一个完成的 SFT checkpoint，KL reference 始终是该 SFT 模型。
- 16 ranks 每卡1样本；2,553补7个零loss sampler占位，480 optimizer steps；不重复监督尾部样本。占位不是额外SFT样本。
- 发布数据没有原始任务ID，因此未证明与 Eval 零重叠。不得将“官方公开数据”写成“已核实无泄漏”。
- RL沿用 [PROTOCOL.md](PROTOCOL.md) 的当前观察、全动作历史、最近两轮反馈、已访问商品证据缓存、合法动作和步数。只用可见事实；policy与router共享状态主体。

## 冻结路由

router 是**原始未SFT、未RL的 Qwen3.5-4B**；本地vLLM、BF16、greedy、non-thinking、top1、单token合法标签。不同bank版本只更新目录，不训练router。
固定提示直接复用 `webshop_phase12.llm_router.INSTRUCTION`；完整catalog前缀 + 当前可见状态 + 单标签输出。

初始54标签保持A–Z、AA–AZ、BA/BB。库增长后按确定字母序扩展，只保留tokenizer支持的不同单token标签；不允许zip静默丢失尾部技能。若标签或上下文容量不足，明确报错，不回退字符匹配。
bank版本同时传给Manager、路由cache协议和vLLM目录预热；每个窗口固定bank。新增/删除/修改/合并后新建cache身份、刷新目录KV；退役ID不复用。

## GRPO 与官方脚本的对齐

参考 `aiming-lab/SkillRL@8e66726/examples/grpo_trainer/run_webshop_skills.sh`。

| 参数 | 本实现 | 官方公开脚本 |
|---|---|---|
| task×group / update | 16×8=128 | 相同 |
| 学习率 | 1e-6 | 相同 |
| global PPO minibatch / per-GPU microbatch | 64 / 4 | 相同 |
| PPO epochs / clip / grad clip | 1 / 0.2 / 1.0 | 官方base默认 |
| KL | low_var_kl，0.01，loss中 | 相同 |
| entropy coefficient | 0 | 官方base默认；不是ALFWorld适配的0.001 |
| 无效动作惩罚 | 0.1 | 相同 |
| 环境训练reward | 完全成功10，否则0 | 相同；native 0–1 score另记 |
| 训练温度 / 验证温度 | 1.0 / 0.4 | 相同 |
| 更新数 | 150 | 原脚本150 total_epochs；此处明确150个外层更新，不宣称与150个数据epoch等价 |
| 边界评估 | 每5更新，外部配对Dev | 官方内置每5轮；本实现避免验证router与actor共驻OOM |
| 步数 / prompt / response上限 | 50 /16384 /512 | 原脚本15 /6000 /768；保留已确认状态协议 |
| 执行 | 16 GPU，vLLM V1 TP1×16 | 原脚本8 GPU，TP4 |
| offload | B200 GPU驻留，无optimizer/reference CPU offload | 官方脚本开启；仅资源策略差异 |
| 技能注入 | 逐状态top1，独立冻结LLM | 原生template/embedding top-k；研究适配 |

外层update不是一次Adam step：多步轨迹展平成决策行，按global PPO minibatch64优化。逐update保留原生loss/KL/gradient/optimizer指标，不把150轮写成150个Adam step。任务计划包含2,400个任务位，每位8条轨迹，即每臂19,200条训练轨迹（不含评估）。

Train任务ID≥1500；Dev500–1499中预登记64个；Eval0–499全量。训练任务流由seed404预先生成、每update16个不同任务，三臂共用；组内8次采样。窗口内不依结果换任务。
vLLM训练请求seed按(404,update,环境步,原trajectory行)确定，不依赖GPU编号或微批分组；重启不靠未保存的vLLM内部随机流。

## 三臂与编辑证据

1. **reward**：稳定数值版 `D_sign_balance=mean(-sign(P_int) * direction_valid)`；分母含该skill全部实际loss token，零advantage不被偷偷删除。沿用WebShop token mean，不混用ALFWorld game-equal聚合。
2. **skillrl**：失败驱动**适配版**，并非原生验证失败输入的逐项复现。
3. **frozen_bank_grpo**：同样SFT、GRPO、状态和router，初始bank始终不变。不开编辑器、readout或候选gate；照常做性能监控。

U0→U5在U0采集的U1首批上readout；U5→U10在U5采集的U6首批上；以此类推。每窗old/new四条件只前向，不采新rollout作readout输入。
reward候选池=首批失败轨迹涉及的自然可评分skills；降序D取top5，分数并列按ID；编辑器收到5个skill全文及调用它们的全部完整失败轨迹，不设8条截断。SkillRL收到同一首批全部失败轨迹和全库全文。
编辑器不看readout数值、Eval标签或Dev gate结果；只看候选ID和旧失败证据。不给action-bias额外阈值，避免新增未冻结超参；无证据/候选则abstain。

## 编辑器、gate、成本

- 两个演化臂均 `gpt-5.5`，网关 `https://api.zhizengzeng.com/v1`，仅从 `SKILLRL_PHASE3_EDITOR_API_KEY` 读取密钥。
- 相同schema和prompt、medium reasoning、8192最大完成tokens、600秒timeout、两个演化臂各最多30次调用（150/5）；错误/不确定请求不会自动重复付费。SQLite记录实际usage、响应模型、requestID和耗时，不记录密钥。
- ADD/MODIFY/DELETE各1 mutation unit；MERGE=目标数+1；最多3 units；NOOP单独0units。reward只能改/删/合并已给候选，ADD允许库增长。所有操作生成新版本，不能原地改历史正文。
- 不静默截断完整编辑证据。长失败轨迹可能超过网关上下文；届时明确失败并保留账本，不能为跑通而自动删证据。真实网关兼容性尚需目标服务器验收。
- 每窗固定当前endpoint policy、相同64 Dev tasks×seeds40401/40402，配对比较旧/候选bank。**平均native score不下降**才接受；成功率、修复/退化数也记录。阈值0，不由结果调参。
- 被拒绝候选未部署，记candidate_rejection，不冒充rollback；目前没有已部署库的自动回滚机制，真实rollback计数因此为0。
- U0与U150各全量500 Eval tasks×2 seeds。U0完全相同，可共享一次评估并明确记共同初始化成本；Dev用于选择，不能当独立测试。

## 加速与数据保留

- vLLM策略rollout，16个TP1副本，chunked prefill、CUDA graphs；只对活动episode生成。权重仅在更新/恢复后同步，环境步间保持引擎，不每步FSDP全参聚合/empty_cache。
- router两副本按长度分派、完整输入去重、目录prefix cache，训练GPU共享时rollout后sleep；评估每个GPU shard自己的本地冻结router。
- HF仅用于原生训练/概率评分，不用于policy rollout。启用response-only logits、去共同左padding；Qwen单条样本mask修正也用于actor、reference和readout。新cohort，不复用旧dense口径读出。
- 目标GPU预检要求Qwen线性注意力快路径实际可用（FLA+causal-conv1d），不允许悄悄落回慢PyTorch实现。
- 每update保存native model/optimizer/RNG/dataloader，保留最新两套作交接；窗口完成readout、gate、报告封存后，仅保留最新native checkpoint和HF endpoint，回收旧可再生副本。所有轨迹、batch、指标、bank版本、编辑/评估记录保留。
- 不保存全词表张量、不跑Phase12独立O/P/N效用续跑。readout只存逐token标量和每skill汇总。

## 验收边界

CPU回归、真实tokenizer/SFT数据适配与模拟闭环不等于B200训练验收。目标预检覆盖真实router、vLLM生成和padding概率一致性；**仍不声称验收过16卡FSDP optimizer/恢复、完整商品数据运行和真实editor API**。这些必须在目标资源可用时获得实际证据；本次没有启动训练。
