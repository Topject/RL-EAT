# 01 非学习基线, 不训练能做到什么程度
# 01 Heuristics, how far do you get without learning

> 上一章 / previous：[00 问题与舞台](../00-problem/README.md) · 下一章 / next：[02 DDPG](../02-ddpg/README.md)

## 0. 中英摘要 / Abstract

这一章没有训练、没有曲线、没有参数：它只把不学习的几种做法放到同一块评估面板上，量出后面每一章都必须越过的参照线。
面板来自 `docs/FACTS.md` §7：64 节点、λ=0.3、quality_ratio=0.6、单一种子 42；数字由作者手工抄录（`EATlogs/Reading/AlgoAndData.xlsx`），本仓库**没有**可重算的原始评估日志。
结果是一个很干净的权衡：**Random 最快（平均响应 25.02 s）但质量只有 0.403；Greedy 最慢（39.61 s）但质量最高 0.752；FIFO 夹在中间（35.54 s / 0.707）**。
同一张表里 SAC 是 28.72 s / 0.556，正好说明学习型策略不必两列都赢，只要在质量上明显高过随机、在响应上快过 FIFO 就够。
本章的用处就是给出下界与上界：02 章之后的每一条算法都要回答"我比这几个规则基线好在哪里"。

No learning, no curves, no parameters: this chapter puts the non-learning baselines on one evaluation panel and measures the reference line every later chapter is judged against.
The panel comes from `docs/FACTS.md` §7, 64 nodes, λ=0.3, quality_ratio=0.6, a single seed 42, hand-transcribed by the author (`EATlogs/Reading/AlgoAndData.xlsx`); this repository has **no** replayable evaluation log to recompute it from.
The result is a clean trade-off: **random is fastest (mean response 25.02 s) but scores only 0.403 quality; greedy is slowest (39.61 s) but tops quality at 0.752; FIFO sits in between (35.54 s / 0.707)**.
SAC appears on the same table at 28.72 s / 0.556, which is the point: a learned policy does not have to win both columns, it has to beat random on quality and beat FIFO on latency.
This chapter supplies the floor and the ceiling that later chapters are measured against.

## 1. 为什么会有这一步 / Why this step

- [00 章](../00-problem/README.md) 把决策拆成两件事：从可见队列里挑一个任务，再给它指定推理步数。环境里的动作空间因此是三维连续向量 `[exec_flag, step, task]`（`code/eat-train/env/env.py#L158-L196`）。本章问的是：**完全不学，只靠规则，这两件事能做到多好？**
- 规则基线不需要训练，所以它们没有任何"预算不够"的借口：同样 64 节点、同样 λ、同样 reward 权重，跑一次就有一个数（`docs/FACTS.md` §7）。
- 本章要验证的假设：**在规则策略里，"最快"和"质量最高"是互斥的两个角，中间没有便宜可占。**
- 假设如果成立，就解释了为什么需要一个能同时优化两列的东西，也就是第 02 章开始的 actor-critic 谱系。

## 2. 机制 / Mechanism

| 组件 | 实现 | 代码位置 |
|---|---|---|
| FIFO（论文名 `FIFO_MAX_FREE`） | 固定动作 `[1, 0.8, -1]`：执行、推理步数钉在 `step` 维的 0.8、任务维 −1 解码成队首 | `code/eat-train/scripts/eval_FIFO.py#L70-L73` |
| RANDOM | `np.random.uniform(-1.0, 1.0, size=3)`，三个维度独立均匀采样 | `code/eat-train/scripts/eval_rand_action.py#L71-L73` |
| GREEDY（论文名 `GREEDY_LOCAL`） | 在 10×8 的 (step, task) 网格上枚举全部候选，用 `env.preview_reward()` 取一步 reward 最大的那个 | `code/eat-train/scripts/eval_greedy.py#L26-L79` |
| 动作解码（三种方法共用） | `a_exec > 0` 才执行；`step` 线性映射到 `1..max_step`；`task` 线性映射到 `1..visible_len` | `code/eat-train/env/env.py#L167-L196` |
| 奖励 | `-w_t1·tanh(t1/s) - w_t2·tanh(t2/s) + w_q·tanh(steps/s_q) + bias`，权重 q=6 / t1=3 / t2=1 | `code/eat-train/env/env.py#L509-L514`，`code/eat-train/env/config.py#L85-L89` |
| 节点选择 | 不在动作里：`NodeScheduler.select_nodes()` 先找大小正好匹配的闲置模型组，再按索引取闲置节点 | `code/eat-train/env/scheduler.py#L10-L34` |
| 真实集群版本 | 同一套动作，但把环境换成 `mode="real"` 的 8 个节点 IP | `code/eat-train/scripts/eval_real_FIFO.py#L22-L59`，`eval_real_random.py` |

- **动作怎么产生**：三个脚本都没有网络。FIFO 是常量，RANDOM 是 `np.random.uniform`，GREEDY 是 80 个候选里的 argmax。没有任何一步用到梯度。
- **质量从哪来**：`_compute_reward` 里质量项是 `tanh(steps / reward_q_scale)`（`env/env.py#L503`），只跟推理步数单调相关；`reward_q_scale` 默认 30（`env/config.py#L88`），`max_step` 默认 30（`env/config.py#L19`）。所以"质量"这一列本质上就是"平均用了几步"的单调映射。
- **和后面章节的差别**：这里没有 replay buffer、没有目标网络、也没有任何从错误里恢复的机制。FIFO 一旦锁死在队首就永远锁死；GREEDY 每一步只看一步 reward，看不见自己的动作会把队列推长。
- **一个必须说清的缺口**：论文里这一臂叫 `FIFO_MAX_FREE`，但快照里的 `eval_FIFO.py` 只实现了"队首任务 + 固定步数"，没有"挑最空闲节点"的逻辑（节点由 `scheduler.py` 决定）。名称与快照实现是否一一对应，未记录 / not recorded（推断 / inferred：名字可能来自更早的一版实现）。论文里另有 `GREEDY_ORACLE` 臂，本快照没有对应脚本，`docs/FACTS.md` 里也没有它的数值，未记录 / not recorded。

## 3. 实验设置 / Setup

| 项 | 值 | 出处 |
|---|---|---|
| 环境规模 | 64 节点 | `docs/FACTS.md` §7 |
| 到达率 λ | 0.3 | `docs/FACTS.md` §7、§8 |
| 质量比 quality_ratio | 0.6 | `docs/FACTS.md` §7 |
| 种子 | 单一种子 42 | `docs/FACTS.md` §7 |
| 训练步数 | 无（不训练）；每次评估跑到环境 done/truncated | `MAX_STEPS = 0`，`eval_FIFO.py#L18` |
| 可训练参数 | 无 / none | — |
| 学习率 / batch size / 扩散步数 | 不适用 / n.a. | — |
| 脚本默认节点数 | 32（要复现 §7 的 64 节点得手动改 `NUM_NODES`） | `eval_FIFO.py#L26`，`eval_rand_action.py#L27`，`eval_greedy.py#L87` |
| 脚本默认重复次数 | `REPEAT_TIMES = 8`（种子 42 起递增），而 §7 只报了一个种子 | `eval_FIFO.py#L20`，`eval_greedy.py#L84` |
| 原始评估日志 | 未入库，不可重算 | `docs/FACTS.md` §7 |

表里的取值凡出现在代码里的都标了行号，`docs/FACTS.md` 只记录 §7 那一组面板参数。脚本常量与 §7 面板并不完全一致，这一点本身就是复现风险（见 §6）。

## 4. 收敛过程 / Convergence

这一节**没有图，也没有曲线**。非学习基线没有训练过程，`figures/curves/` 下没有它们的图，`data/` 下也没有 `fifo` / `rand` / `greedy` 的运行目录。这是事实，不是漏写。下面这张表就是本章的全部结果。

| 方法 | 平均响应时间 (s) | 平均质量 | 反解的平均推理步数（推断 / inferred） |
|---|---:|---:|---:|
| FIFO | 35.54 | 0.707 | ≈26 |
| Greedy | 39.61 | 0.752 | ≈29 |
| Random | 25.02 | 0.403 | ≈13 |
| SAC（参照，学习型） | 28.72 | 0.556 | ≈19 |

"反解步数"不是实测值：由 `steps ≈ 30 · atanh(quality)` 算出（`atanh` 是 `tanh(steps/30)` 的反函数，见 `env/env.py#L503` 与 `env/config.py#L88`）。它只用来把质量列翻译成人能读的量，属于算术 + 推断。

- 三角形态很清楚：质量最高的代价是响应最慢，质量最低的回报是响应最快，中间那一档（FIFO）两边都不占优。
- 质量 0.033 这种数字属于另一个类别（见 §5），反解出来正好是"1 步"，即推理步数被压到下限。这种情形已经算退化，谈不上权衡。

## 5. 结果与对比 / Results

把 §7 的整块面板放在一起看，规则基线只是其中三行，另外两行是后面章节的伏笔：

| 方法 | 平均响应时间 (s) | 平均质量 | 备注 |
|---|---:|---:|---|
| Random | 25.02 | 0.403 | 全场最快，质量最低 |
| SAC | 28.72 | 0.556 | 学习型参照点 |
| FIFO | 35.54 | 0.707 | 质量第二 |
| Greedy | 39.61 | 0.752 | 质量最高，响应最慢 |
| DDPG | 1.28 / 20.26 | 0.033 / 0.400 | 两个数，前一个是策略崩塌 |
| TD3 | 2.44 | 0.033 / 0.110 | 同上 |

- **权衡本身**：Random 用平均约 13 步换到 0.403 的质量，Greedy 用约 29 步换到 0.752，FIFO 用约 26 步换到 0.707。质量随步数上升，响应时间也随步数上升。这是同一个物理事实的两面：每个任务占机越久，后面排队的人等得越久。
- **"快"可以是症状**：表里响应时间最短的 1.28 s 不是冠军，是崩塌。`docs/FACTS.md` §7 已经写明它的特征，只挑最短任务、永远用最低推理步数，质量因此被压到 0.033。这条线索留给第 02 章。
- **学习型策略的位置**：SAC 在 28.72 s 上拿到 0.556，比 Random 慢 3.7 s（算术），质量高 0.153（算术）；比 FIFO 快 6.8 s（算术），质量低 0.151（算术）。它真正做到的是在同一个响应时间档位上把质量抬高了一截，这才是后面每章要复现的东西。

## 6. 简要分析 / Analysis

1. **质量列就是步数列**。`reward` 的质量项是 `tanh(steps/30)`（`env/env.py#L503`，`env/config.py#L88`），所以 §7 的 0.752 / 0.707 / 0.556 / 0.403 / 0.033 可以单调地反解成约 29 / 26 / 19 / 13 / 1 步（算术 + 推断）。这解释了为什么"质量崩了"和"永远用最低步数"在 `docs/FACTS.md` §7 里是同一句话。
2. **Greedy 的慢可以从 reward 权重推出来**。质量权重 6、时延权重 3 与 1，`quality_ratio` 因此是 0.600（`docs/FACTS.md` §8）。一步贪心最大化的正是"质量增益 − 时延代价"，质量项权重更大，所以它会往高步数靠（推断 / inferred，依据 `env/env.py#L509-L514` 的权重与单调性）。步数越高，每个任务占机越久，队列越长，39.61 s 就是这么来的。
3. **FIFO 落在"质量不错但速度一般"的位置，是可以算出来的**。`step` 维 0.8 在 `max_step=30` 下解码成 27 步（`env/env.py#L181-L182`），对应质量 `tanh(27/30) ≈ 0.716`，而实测是 0.707（算术）。两者只差 0.009，推断 / inferred：这点差距来自"逐步平均"的统计口径（见第 5 条），不是动作不同。所以 0.707 更像是巧合，0.8 这个常量刚好停在高步数一侧。
4. **两个"慢"要分开**。39.61 s 是环境里的任务平均响应时间，不是脚本的墙钟耗时。Greedy 的墙钟开销在另一处：每个决策步最多要跑 10×8 = 80 次 `preview_reward()`（`eval_greedy.py#L36-L38`、`#L62-L73`；队列空时直接返回），这部分成本不进入响应时间，但会进入复现时的实际耗时。
5. **测量口径的提醒**。评估脚本对 `info["quality"]` 逐步取平均（`eval_FIFO.py#L77-L79`），而跳过执行的步会重复上一步的值（`env/env.py#L223` 与 `#L506`）。推断 / inferred：这里的"平均质量"是每步平均而非每任务平均，跳过的步会把上一次的质量重复计入。
6. **复现到了什么、没复现到什么**。复现到的是三角形态（随机最快最低质量、贪心最慢最高质量、FIFO 居中）；**没复现到**的是一个可重算的评估：§7 来自作者手工抄录、单种子、没有原始日志，所以表里每一位小数在本仓库都无法复算，只能引用。脚本默认 32 节点、8 次重复，与 §7 的 64 节点、单种子也不一致。这些数字对应的是哪一次配置，未记录 / not recorded。

## 7. 训练数据导航 / Training data

**这一章没有训练数据。** 非学习基线不产生训练曲线，也不产生 checkpoint，所以 `data/` 下没有 `fifo` / `rand` / `greedy` 目录，`data/by_algorithm.csv` 里也没有它们的行（可以直接核）。能链接的只有下面这些：

| 内容 | 路径 |
|---|---|
| 本算法唯一的数字来源 | [`docs/FACTS.md`](../docs/FACTS.md) §7（64 节点单种子面板，手工抄录） |
| 全部运行索引（**不含**非学习基线） | [`data/runs_index.csv`](../data/runs_index.csv)，88 次运行全是学习型算法 |
| 按算法聚合（**不含**非学习基线） | [`data/by_algorithm.csv`](../data/by_algorithm.csv)，没有 fifo / rand / greedy 行 |
| 评估脚本运行后写出的 CSV | 写到 `data/<NUM_NODES>/<ALGO_NAME>/<timestamp>/eval_run_seed*.csv`（`eval_FIFO.py#L101-L127`），**未入库** |
| 原始评估记录 | `EATlogs/Reading/AlgoAndData.xlsx`（未入库，且是手工抄录，不是本仓库重算） |
| 曲线图 | 无。`figures/curves/` 下没有非学习基线的图 |

## 8. 复现命令 / Reproduce

下面三个脚本把参数写在文件顶部的常量里（`NUM_NODES`、`SEED`、`REPEAT_TIMES`、`STEP_GRID`），没有命令行参数；要贴 FACTS §7 的面板，得先把 `NUM_NODES` 改成 64。入口在快照里存在，但本仓库没有重新执行过（需要原环境的依赖）。

```powershell
cd code\eat-train
$env:PYTHONPATH = ".;scripts"   # env/ 在 eat-train 下，config.py / utils.py 在 scripts/ 下

python scripts\eval_FIFO.py         # FIFO：固定 [1, 0.8, -1]
python scripts\eval_rand_action.py  # RANDOM：三维均匀采样
python scripts\eval_greedy.py       # GREEDY：10x8 网格 + preview_reward
```

- 每个脚本跑完会把逐次结果写进 `data/<NUM_NODES>/<ALGO_NAME>/<timestamp>/`（`eval_FIFO.py#L126-L127`），但**这些 CSV 没有进仓库**，所以本章的数字无法从仓库内重算。
- `eval_real_FIFO.py` / `eval_real_random.py` 会连真实集群（`node_ips` 在文件里写死，`eval_real_FIFO.py#L26-L35`），需要 EAT-Server 在跑；本仓库没有这两个脚本的任何输出，未记录 / not recorded。
- 作者当时产生 §7 那张表的精确命令不在快照里，与训练侧一样，原始调用方式 **未保留 / not preserved**。

---

上一章 / previous：[00 问题与舞台](../00-problem/README.md) · 下一章 / next：[02 DDPG](../02-ddpg/README.md) · 返回目录 / index: [README](../README.md)
