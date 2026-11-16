# 14 复现审计, 原始实现里会改变结论的七处问题
# 14 AUDIT, seven implementation issues that change conclusions

> 上一章 / previous：[13 全部结果](../13-results/README.md) · 返回目录 / index：[README](../README.md)

## 0. 中英摘要 / Abstract

前面 13 章在回答"EAT 在这套环境里到底表现如何"；这一章换一个问题：**这份实现本身是否足以支撑论文的结论？**
结论是**不足以**。我们在 `code/eat-train/` 快照里核到七处问题：没有全局种子（A1）、critic1/critic2 结构不对称（A2）、D2SAC 的熵项被硬置零（A3）、七个 trainer 的 actor 都不做 tanh 压缩（A4）、隐藏层激活用 ReLU 而论文用 Mish（A5）、快照按现有代码直接 `TypeError` 且评估脚本名与 trainer 错位（A6）、脚本写死 64 节点 + 3e-4 与主扫描并非同一配置（A7）。
其中 A1、A2、A6 有**代码之外的独立证据**（权重张量、逐位相同的评估点、真实报错），A3、A4、A5、A7 是**代码级证据**、影响尚未用受控实验量化。
本章不给"原作者错了"的结论，只给"哪些结论目前**无法**从这份实现推出"，以及要补哪几个实验才能推出来。

The previous 13 chapters ask how EAT performs; this chapter asks whether the implementation can support the paper's claims at all. It cannot, as published. Seven issues are documented, three of them with evidence independent of the source text (checkpoint tensors, a bit-identical evaluation point, a real `TypeError`). Nothing here claims the original authors are wrong; it claims which conclusions **cannot** be drawn from this snapshot, and what experiments would be needed to draw them.

---

## 1. 审计范围与证据等级 / Scope and evidence levels

| 等级 | 含义 | 本章用到的 |
|---|---|---|
| **E3 独立证据** | 代码之外还有可复核的物证（权重张量、曲线、运行时报错） | A1、A2、A6 |
| **E2 代码证据** | 代码行本身可复核，但影响未测量 | A3、A4、A5、A7 |
| **E1 转述** | 来自作者笔记/汇报，本仓库无法复核 | 论文超参、D2SAC 的重写来源 |

所有 E2 结论都必须配一次"修复前后"的对照才能升级为 E3；本章第 5 节列出这些对照。

---

## 2. 上游对照 / Against the upstream repository

上游仓库：<https://github.com/zzf1955/EAT>（论文 "EAT: QoS-Aware Edge-Collaborative AIGC Task Scheduling via Attention-Guided Diffusion Reinforcement Learning"）。

2026 年内可访问到的状态（逐条探测记录）：

| 探测 | 结果 |
|---|---|
| `https://github.com/zzf1955/EAT` | 200，可访问 |
| `raw .../main/README.md` | **200**，内容与本仓库 `code/EAT-master-README.md` 一致 |
| `raw .../main/EAT-Train/README.md` | 404 |
| `raw .../main/EAT-Train/trainer/ppo/trainer.py` | 404 |
| `raw .../main/EAT-Server/app.py` | 404 |
| `raw .../main/demo.png` | 404 |
| `raw .../main/requirements.txt` | 404 |
| `raw .../main/EAT-Train.zip` | 404 |

**能下的结论**：公开仓库目前只能读到 `README.md`，训练/服务代码不在公开可访问路径上。
**不能下的结论**：不能说"作者没上传代码"，本地 `code/` 里的 97 个源文件来源无法从公开仓库核对（可能是早期的公开版本、私有交付或压缩包）。本仓库因此把"原始实现"一律定义为 **`code/` 快照**，而不是"作者现在仓库里的代码"。

A further reason to be careful: the snapshot is **not** the revision that trained the checkpoints. `trainer/dcasac/trainer.py#L116` passes `schedule_timesteps=` while `model/diffusion/rl_diffusion.py#L34` only accepts `n_timesteps`, so that trainer raises `TypeError` as printed. Any claim about the runs must therefore be checked against the checkpoints and curves, not against the source alone, which is what A1–A3 do.

---

## 3. 七处问题 / The seven issues

### A1 没有全局种子，E3

**证据**
- 全树（`code/eat-train/` 97 个 `.py`）检索 `manual_seed` / `np.random.seed` / `random.seed` / `set_seed`：**0 处调用**。
- 环境侧**有**种子：`EdgeEnv(seed=...)` → `np.random.default_rng(seed)`（`env/env.py#L42-L46`、`#L127-L129`）；七个训练脚本都传 `cfg_train.seed`，默认 42（`scripts/config.py#L51`）。
- 物证：DPPO 的两次运行（`2026-08-14_16-17-08` 与 `16-33-03`）在 step 0 的 `test/reward` 都是 **148.589691**，逐位相同；第三次（`15-32-08`）是 151.780380。

**这意味着什么**：环境流是（可能）共享的，而策略初始化、动作采样、replay 采样顺序全部来自未设种子的 torch 全局 RNG。所以"每个算法跑 3 次"**不等于**"3 个受控种子"：既无法保证三次是独立重复，也无法保证别人能重现其中任何一次。

**修复**：`torch.manual_seed` / `np.random.seed` / `random.seed` 三件套 + 环境 `reset(seed=...)` + 把 seed 写进运行目录名与 `run.json`。

### A2 critic1 / critic2 结构不对称，E3

**证据**
- 代码：`trainer/dicasac/trainer.py#L132-L134` 取 `first_dim = _crit[0]`、`tail = tuple(_crit[1:])`，然后
  `L148 critic1 = Critic(preprocess_net=... hidden_sizes=critic_hidden_sizes)`，
  `L169 critic2 = Critic(preprocess_net=... hidden_sizes=tail)`。
- 权重（`tools/ckpt_inventory.py` 直接读 `policy_best.pth`，共 20 个 EAT checkpoint）：

| EAT 版本 | checkpoint 数 | critic1 | critic2 | 差值 |
|---|---|---|---|---|
| 主扫描 `fig6-main`（709,560，08-19） | 3 | 155,809 | 155,809 | **0**（对称） |
| `trials` / `learning-rate` / `fallback-rerun`（841,144，08-12 ~ 08-17） | **17** | **221,601** | **155,809** | **65,792** |

65,792 = 256×256 + 256，**正好是 critic1 多出来的一整层**。17 个 checkpoint 里确实少一层，这已经超出理论上的可能。08-19 的 3 次正式运行反而**对称**。

**这个分布本身是个线索**：`AAAresult.docx` 里作者自己写过"两个 critic 要统一用 `tail`"，而最终主扫描的权重确实是对称的，推断：主扫描之前已经修过这个问题，A2 影响的是 `trials` / 3e-4 支路 / fallback 重跑这三批，**不影响第 13 章那张主表**。这条修正很重要：它把 A2 从"推翻主结果"降级为"让旁支对比失去可比性"。

**影响**：SAC 族的目标是 `min(Q1,Q2) − α·logπ`，两个 Q 应当同分布、同容量；容量不同会让 `min` 系统性偏向弱的那一个，同时"双 Q 抑制高估"的前提（两个独立同分布估计）被破坏。所以凡是用 841,144 版 EAT 与其它臂比较的地方（第 [10](../10-tuning/README.md) 章的 fallback 与 3e-4 两节），结论都要打折扣。

**修复**：两者都用 `critic_hidden_sizes`（或都用 `tail`），然后跑对照。

> EN: 3 main-sweep checkpoints are symmetric; all **17** checkpoints of the 841,144-parameter variant (trials, 3e-4, fallback) are asymmetric by exactly one 256×256+256 layer. The author's own note asked for symmetric critics and the final sweep complies, so A2 weakens the side comparisons (fallback, 3e-4), not the headline table.

### A3 D2SAC 的熵项被硬置零，E2

**证据**：`policy/sac_diffusion.py#L159-L161`

```python
entropy = self.actor.action_entropy(batch_size=act.shape[0]).flatten()
entropy = 0                                     # ← 硬置零
actor_loss = (-q_min - self._alpha * entropy).mean()
```

自动温度分支（`#L167-L173`）的 surrogate 是 `(-entropy).detach() + target_entropy`，`entropy` 恒为 0 时它是常数，梯度只与 `log_alpha` 线性相关，方向只会把 α 推向 0。
对照：同文件里 EAT 用的 `SACDDIMPolicy`（`#L393-L394`）**保留**了这一项。

**这意味着什么**：这个 D2SAC 更像一个无熵正则的扩散 actor 加双 Q。叫它扩散版 SAC 会误导。它全场最低（159.5 ± 4.0）与这一点一致，但**不能**据此说论文的 D2SAC 差。

**重要限定（E1）**：据复现者说明，D2SAC 臂是在原作者"环境不兼容、未上传"后由 AI 重建的。因此 A3 只对 **本仓库 `code/` 里的这一份实现**成立，不能外推到论文的 D2SAC。

**修复**：恢复熵项 + 打开自动温度，跑对照。

### A4 七个 trainer 的 actor 不做 tanh 压缩，E2

**证据**：以下 trainer 的 `ActorProb(...)` 都传了 `unbounded=True`：

| trainer | 行号 |
|---|---|
| `trainer/ppo/trainer.py` | L52 |
| `trainer/appo/trainer.py` | L63 |
| `trainer/cappo/trainer.py` | L63 |
| `trainer/sac/trainer.py` | L70 |
| `trainer/asac/trainer.py` | L72 |
| `trainer/casac/trainer.py` | L66 |
| `trainer/dppo/trainer.py` | L96 |

tianshou 的 `ActorProb(unbounded=True)` 表示输出**不做 tanh**、`log_prob` 也**不做 tanh 修正**；而动作空间是 `Box(-1,1)`（`env/env.py#L88-L90`）。越界的动作由环境 `np.clip` 静默截断（`env/env.py#L181`、`#L187`）。

**影响**（结构性，未量化）：(a) 采样越界被静默截断 → 同一动作可能对应不同概率；(b) SAC 的熵项在无界空间估计，与真实有界动作不一致；(c) 该环境"选任务/选步数"本质上是分档决策，越界截断会把不同档位折叠到一起。

**修复**：`unbounded=False`（标准 tanh-squashed Gaussian + 修正项）或显式 squash。

### A5 隐藏层激活用 ReLU，而非论文的 Mish，E2（论文侧为间接证据）

**证据**
- 代码：上述七个 trainer 的 `ActorProb(...)` / `Critic(...)` **都没有传 `activation=`**，走 tianshou 默认的 ReLU；而同一仓库自己的注意力模块用 Mish（`model/attention/cross_attention.py#L46`、`#L171`，`self_attention.py#L41`），EAT 的 `DiffusionMLP` 也用 Mish（`trainer/dicasac/trainer.py#L44`）。
- 论文侧（间接）：对论文 PDF 的文本层做检索（解压 82 个内容流），`ReLU` 出现 **0 次**，`Mish` 出现 1 次且位于参考文献 `[49] Misra, "Mish: A self regularized non-monotonic activation function"`。

**影响**：论文的 baseline 与 EAT 主干应当同一激活族；实现里 baseline 是 ReLU、EAT 主干是 Mish，等于让 baseline 换了一套非线性，**对 EAT 有利的比较不再公平**。

**修复**：所有 Actor/Critic 显式 `activation=nn.Mish`。

### A6 快照不可直接运行 + 评估脚本名与 trainer 错位，E3

**证据**
- `trainer/dcasac/trainer.py#L116` 传 `schedule_timesteps=`，`model/diffusion/rl_diffusion.py#L34` 只接受 `n_timesteps` → 按现有代码实例化即 `TypeError`。
- `eval_dcasac.py` 实际 `import trainer.dicasac`、`algo_prefix="sac_dica"`（L11、L211、L258），是 **EAT** 的评估脚本；`eval_dsac.py` 才是 `trainer.dsac`（D2SAC-SA）的。
- 快照里**没有**训练/评估 `trainer.dcasac` 的任何脚本（见 [docs/REPRODUCE.md](../docs/REPRODUCE.md) §3 的核实表）。

**影响**：别人拿到这份代码**无法一键复现论文的任何一条曲线**；名字错位还会让评估取到错误的 checkpoint 前缀。

**修复**：修关键字、统一命名，或在 README 里给出"脚本 ↔ trainer ↔ 算法"对照表（本仓库已补）。

### A7 配置漂移：脚本常量与论文、与主扫描都不一致，E2

| 项 | 脚本常量 | 作者笔记/论文 | 后果 |
|---|---|---|---|
| 节点数 | 写死 64（7 个训练脚本） | 32 / 64 均有 | 复现者需改常量 |
| 学习率 | 写死 3e-4 | 主扫描 1e-4 | 直接跑 ≠ 论文配置 |
| 网络 | — | — | `fig6-main` 的 EAT 709,560 与 `learning-rate`/`fallback` 的 841,144 是**两套网络** |
| 扩散步数 | code 默认 5 / 10 | 论文提到 4 / 10 | 未记录实际值 |

**影响**：任何"只改了一个超参"的结论都要先确认网络没变，本仓库 [10-tuning](../10-tuning/README.md) 就是因为这个差点把处（EAT 的 3e-4 支路确实换了网络）。

---

## 4. 和"我们自己的复现"的关系 / How this relates to this repository

| 事实 | 含义 |
|---|---|
| 本地算法代码**已被复现者修改过**（调参、修 bug、换底座计划） | 本审计的对象是 `code/` 快照，**不是**复现者当前的工作代码；两者不能混为一谈 |
| D2SAC 臂是 AI 重建 | A3 的"熵项置零"只说明**这份重建**非标准 SAC，不能用来否定论文的 D2SAC |
| 公开仓库只有 README | 无法对外证明"某一行代码属于原作者"；所以本章一律说"快照里"，不说"作者写的" |
| 复现者的 Fig6（归一化） | 见 [`figures/original/teacher-report/image1.png`](../figures/original/teacher-report/image1.png)；7 条曲线 n=3，EAT 的种子带最宽，D2SAC 最低 |

---

## 5. 要把 E2 升级成 E3 需要的对照 / Experiments that would upgrade the evidence

| 编号 | 对照 | 目的 | 成本（估） |
|---|---|---|---|
| A2 | critic 对称化前 / 后，各 5 种子 | 证明"少一层"是否改变 EAT 的均值与方差 | 10 次训练 |
| A3 | 恢复熵项 + 自动温度前 / 后，各 5 种子 | 证明 D2SAC 的垫底有多少来自实现 | 10 次训练 |
| A4 | `unbounded=False` 前 / 后，SAC 与 PPO 各 5 种子 | 证明越界截断的影响 | 20 次训练 |
| A5 | ReLU / Mish 前 / 后，全部 baseline | 证明激活函数决定了排序的一部分 | 30+ 次训练 |
| A1 | 固定种子跑 2 次，逐位比较曲线 | 证明"可复现"（这是**最低成本、最高说服力**的一项） | 2 次训练 |

> 成本口径：本仓库**没有** EAT-Train 侧的 wall-time 记录（未记录 / not recorded）。同一台机器上的 AMARL 三集群实验记录过 30,000 agent steps ≈ 152 s（RTX 4080 SUPER），只能作为量级参考，不能直接换算。

---

## 6. 汇报要点整理 / The report, reorganised

以下是复现者向导师汇报的要点，按证据等级重排（原文见其汇报与 `EAT复现logs.docx`）：

1. **baseline 之外还有实现问题**：EAT 仓库的 PPO 也不是标准 PPO，`ActorProb(unbounded=True)` 没有 tanh（A4）、隐藏层 ReLU 而非 Mish（A5）、全仓库没有全局种子（A1）。→ 本章第 3 节逐条给了代码位置与物证。
2. **因此不适合拿它做 MAPPO 的底座**：在这个 PPO 上叠多智能体会把 A4/A5/A1 一起放大。→ 结论成立，且理由充分。
3. **做法：算法底座换 CleanRL 标准 PPO/MAPPO，只保留 EAT 的环境模型与复现分析**。→ 这是本章支持的技术判断：环境（`env/`）与调度语义是本仓库里最扎实的部分（第 [00](../00-problem/README.md) 章），而 baseline 实现是最不可信的部分（本章）。
4. **衍生路线（待导师判断）**：换全局种子再跑 EAT，以"EAT 实现修复 + 可复现性分析"作为副投稿。→ 见第 7 节的独立评估。

---

## 7. 副投稿路线的可行性判断（独立意见）

**结论：可行，但必须先补两件事，且不适合作为主线。**

理由：

- **有利**：这类"复现 + 实现审计"的稿件有明确读者（reproducibility / empirical track）；你手上已有 88 次运行、可重算曲线、权重级证据，材料密度高于一般复现报告；A1/A2/A6 是 E3 级证据，本身就能撑起一节。
- **不利**：目前 A3/A4/A5/A7 都停在 E2（"代码这样写"但"影响多少"未知），而审稿人会问的正是"so what"。没有 repair-vs-original 对照，这类稿件容易被判为"bug 报告"。
- **最小可投稿组合**（只要 3 个实验）：
  1. `set_seed` 后同配置跑两次，曲线逐位一致 → 建立"可复现"的地基（2 次训练）；
  2. critic 对称化前后各 5 种子 → 一个明确的"实现细节改变结论"的例子（10 次训练）；
  3. D2SAC 恢复熵项前后各 5 种子 → 第二个例子，且能量化"D2SAC 垫底有多少是实现的锅"（10 次训练）。
- **不建议**做主投稿：算法本身没有新增贡献，篇幅与创新性都撑不起一篇正式论文；作为副线/技术报告或会议短文更合适。

Routes and reasons in English: the audit is worth writing up, but only after (1) a determinism check, (2) a symmetric-critic A/B, and (3) an entropy-restored D2SAC A/B, about 22 short runs. Without them the work reads as a bug list; with them it reads as evidence that implementation details flip conclusions. It should stay a secondary route, not the main paper.

---

## 8. 修复清单 / Repair checklist

| # | 修复 | 文件 | 改动量 | 做完能说什么 |
|---|---|---|---|---|
| 1 | 三件套种子 + 记录 seed | `scripts/*_train.py`、运行目录 | 小 | "本实现可逐位复现" |
| 2 | critic2 改 `hidden_sizes=critic_hidden_sizes` | `trainer/dicasac/trainer.py#L169` | 一行 | "双 Q 对称后 EAT 的方差变化" |
| 3 | 恢复熵项 + auto-α | `policy/sac_diffusion.py#L159-L173` | 小 | "D2SAC 的垫底有多少来自实现" |
| 4 | `unbounded=False` | 7 个 trainer 的 `ActorProb` | 小 | "有界动作 + tanh 修正的影响" |
| 5 | 显式 `activation=nn.Mish` | 7 个 trainer | 小 | "激活函数对 baseline 排序的影响" |
| 6 | 修 `schedule_timesteps` / 统一 eval 命名 | `trainer/dcasac`、`eval_*.py` | 小 | "代码可运行"（复现的入场券） |
| 7 | 每 run 落 `config.json`（含 seed、lr、节点数、网络参数量） | 训练脚本 | 小 | 所有数字可追溯 |

前 6 项都是小改动；真正花时间的是第 5 节的对照重跑。

---

## 9. 训练数据导航 / Training data

本章是审计章，**没有新的训练数据**。

| 想核对的 | 去哪 |
|---|---|
| 参数量（critic1/critic2 不对称） | [`tools/ckpt_inventory.py`](../tools/ckpt_inventory.py)；数值见 [`docs/FACTS.md`](../docs/FACTS.md) §9 |
| 逐次运行的终点/最高点（含 DPPO 的两次 step-0） | [`data/runs_index.csv`](../data/runs_index.csv) |
| 曲线本体 | `data/fig6-main/<algo>/<run>/curves.csv` |
| 代码位置 | `code/eat-train/...`（本章每条都给了行号） |
| 复现者的 Fig6 原图 | [`figures/original/teacher-report/image1.png`](../figures/original/teacher-report/image1.png) |

---
上一章 / previous：[13 全部结果](../13-results/README.md) · 返回目录 / index：[README](../README.md)
