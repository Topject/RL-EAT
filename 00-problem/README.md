# 00 问题与舞台, 三集群 AIGC 调度到底在决策什么
# 00 The Problem and the Stage, what a scheduler actually decides

> 本章没有训练，只有定义。下一步 / next：[01 非学习基线](../01-heuristics/README.md) · 返回目录 / index：[README](../README.md)

## 0. 中英摘要 / Abstract

EAT 要解决的问题是：一堆 AIGC 请求（每个请求是一次 Stable Diffusion 推理）到达边缘侧的三个集群，调度器每一步只能做**一个**决定，要不要开工、开工哪个任务、用多少推理步数。
这个决定同时被三件事拉扯：单次延迟、队列等待、生成质量。本章把这三者的代价函数、动作空间、观测空间固定下来，后面 13 章的所有数字都在这套定义上产生。
**这一章不含任何实验结果，它的作用是让"180.4 分"这种数字有意义。**

The problem EAT attacks: AIGC requests (each one a Stable Diffusion inference) arrive at three edge clusters, and at every step the scheduler makes exactly **one** decision, execute or skip, which waiting task, and how many denoising steps to spend.
Three forces pull on that decision at once: per-task latency, queue waiting time, and generation quality. This chapter freezes the cost function, action space and observation space that every later number is measured against.
**There are no experimental results here. The chapter exists so that a number like "180.4" means something.**

---

## 1. 场景 / The setting

| 元素 | 取值 | 出处 |
|---|---|---|
| 节点数 | 32（默认；实验里出现过 32 / 64） | `code/eat-train/env/config.py#L15`、`code/eat-train/scripts/config.py` |
| 可见任务槽位 | 8 | `code/eat-train/env/config.py#L14` |
| 每回合任务总数 | 32 | `code/eat-train/env/config.py#L28` |
| 时间步长 | 0.5 s | `#L23` |
| 回合上限 | 300 s 模拟时间 / 30 步最大推理步数 | `#L24`、`#L19` |
| 任务并行度候选 | 1 / 2 / 4 / 8 个节点 | `#L39` |
| 到达过程 | 泊松到达，λ 由节点数推出（32 节点 → 0.3 任务/秒） | `code/eat-train/scripts/config.py` |

**"任务"是什么**：一次图片生成 = 在若干节点上跑一个模型并行组（exact-size process group），推理步数可调。
所以这种请求没法塞进任意机器，**要凑齐一组节点**才能开工。这是整个调度问题的硬约束，也是论文里 "edge-collaborative" 的含义。

**What a task is**: one image generation run as a model-parallel group across a specific number of nodes, with a tunable number of inference steps. A task cannot start until an exactly-sized group is available, which is the hard constraint behind the phrase "edge-collaborative".

---

## 2. 动作空间 / Action space

动作是一个三维连续向量，全部落在 `[-1, 1]`：

| 维度 | 含义 | 解码方式 | 出处 |
|---|---|---|---|
| `action[0]` | 执行门 / execute gate | `> 0` 开工，否则空转一个时间步 | `code/eat-train/env/env.py#L168`、`#L173` |
| `action[1]` | 推理步数 / quality step | `round(1 + ((a+1)/2)·(max_step-1))`，即 1..30 | `#L181-L182` |
| `action[2]` | 任务选择 / task index | `round(1 + ((a+1)/2)·(visible_len-1))`，即队首前 8 个之一 | `#L187-L190` |

三个维度都压缩到同一个 `[-1, 1]` 区间，这是后面所有算法（包括扩散动作头）能够共用同一个 actor 接口的前提。
一个容易忽略的细节：`action[2]` 选到的任务如果不可执行（凑不齐节点组），环境不会报错，只是空转一个 `dt`，**"想开工但开不了" 和 "决定不干" 在奖励上完全一样**。这条设计直接决定了后面几章里"退化策略"长什么样。

All three dimensions live in the same `[-1, 1]` box, which is why a Gaussian head and a diffusion head can share one actor interface.
A detail that matters later: if the chosen task cannot be executed (no exact-size node group), the environment silently advances one `dt`. **"Tried and failed" and "chose to wait" are indistinguishable in the reward.** That design choice is exactly what the degenerate policies in later chapters exploit.

---

## 3. 观测空间 / Observation space

观测是一个扁平向量：`num_nodes × 2`（每节点）+ `8 × 2`（每个可见任务），32 节点时共 80 维。

- 节点侧：可用算力/剩余时间一类的两个特征（`node_state_dim = 2`，`config.py#L93`）
- 队列侧：任务的两个特征（`task_state_dim = 2`，`config.py#L94`）
- 定义位置：`code/eat-train/env/env.py#L88-L111`

把"节点集合"和"任务队列"放在同一个扁平向量里，是后面**交叉注意力**能成立的结构基础：任务向量当 Query，节点向量当 Key/Value，形状天然对得上（`code/eat-train/model/attention/cross_attention.py#L69-L134`）。

The flat 80-dim vector holds node features and task features side by side. That layout is what makes **cross-attention** natural later: task tokens query node tokens, with `(B, L, D)` and `(B, N, D)` shapes lining up without any reshaping tricks.

---

## 4. 奖励：三个代价项，一个正数 / The reward: three costs, one positive number

一次成功开工的奖励（`code/eat-train/env/env.py#L453-L521`）：

```text
t1 = 任务占用节点的最长剩余时间 + 镜像/权重传输时间        # 单次延迟
t2 = 当前队列里所有任务的等待时间均值                       # 队列压力
q  = task.steps                                          # 推理步数 = 质量原始值

cost_t1 = tanh(t1 / 50)      cost_t2 = tanh(t2 / 50)      quality = tanh(q / 30)
reward  = -(3·cost_t1 + 1·cost_t2) + 6·quality + 5
```

四个常数来自 `code/eat-train/env/config.py#L84-L90`：`reward_t1_scale=50, reward_t1_weight=3, reward_t2_scale=50, reward_t2_weight=1, reward_q_scale=30, reward_q_weight=6, reward_bais=5`。

三个设计要点：

1. **质量与延迟被放在同一量纲**：三项都过 `tanh` 压到 (0,1)，再乘权重相加，所以 reward 的差异可以直接读成"用多少延迟换多少质量"。
2. **权重比就是质量比**：`quality_ratio = 6/(6+3+1) = 0.600`。换成 `q=10, t1=2` 就变成 `10/(10+2+1) = 0.769`。这正是第 11 章控制任务密度的那两个配置（`EATlogs/Reading/coding.docx` 记录了这个改动）。**改权重比 = 改奖励函数 = 换一把尺子**，跨配置比 reward 没有意义。
3. **代码里有一条硬断言**：`assert reward > 0`（`env.py#L519`）。也就是说任何让 `-(3·cost_t1 + cost_t2) + 6·quality + 5 ≤ 0` 的决策都会直接崩训练，而不是被温柔地惩罚，这让"永远等"这种策略在数值上不可行。

Three things to notice:

1. All three terms pass through `tanh`, so reward differences read directly as "latency paid for quality gained".
2. The weight ratio **is** the quality ratio: `6/(6+3+1)=0.600`, and switching to `q=10, t1=2` gives `0.769`, the two configurations used in chapter 11. **Changing the weights changes the reward function, so returns are not comparable across ratios.**
3. There is a hard `assert reward > 0` at `env.py#L519`. A decision that would make the reward non-positive aborts training rather than being softly penalised, which rules out a "wait forever" policy numerically.

---

## 5. 为什么是"扩散 + 注意力" / Why diffusion plus attention

论文给的两个理由，在这一章的代码里能对上：

1. **动作分布是多峰的**：`action[2]` 决定选哪个任务，实际只有 8 个合法值；`action[1]` 决定步数，质量项让"高步数"局部最优，延迟项让"低步数"局部最优。一个高斯策略只能覆盖单峰，扩散策略可以从多峰分布里采样。这是 D2SAC/DPPO/EAT 三条路线共同的前提（第 06–08 章）。
2. **任务与节点是**非对称匹配**：任务少（8 个）、节点多（32 个），且每个任务需要不同数量的节点组。交叉注意力正好是"少量 query 对齐大量 key"的结构，比把 80 维直接拼进 MLP 更省参数（第 08、09 章会用参数量验证这一点）。

The paper's two arguments map onto the code:

1. **The action distribution is multi-modal**: `action[2]` really has 8 legal values, and `action[1]` is pulled both ways by the quality and latency terms. A Gaussian head covers one mode; a diffusion head can sample several. This is the shared premise of the D2SAC / DPPO / EAT line (chapters 06–08).
2. **Tasks and nodes are an asymmetric match**: 8 queries against 32 keys, with variable group sizes. Cross-attention is exactly the "few queries against many keys" structure, and chapter 09 checks whether it actually buys parameter efficiency.

---

## 6. 出场名单 / The cast

| 算法 | 一句话定位 | 参数量（总/在线） | 章节 |
|---|---|---|---|
| DDPG | 确定性 actor-critic，无熵 | 349,704 / 174,852 | [02](../algorithms/DDPG-README.md) |
| TD3 | 双 critic + 延迟更新 | 524,810 / 262,405 | [03](../algorithms/TD3-README.md) |
| SAC | 最大熵正则 | 437,511 / 262,405 | [04](../algorithms/SAC-README.md) |
| PPO | on-policy 裁剪 | 611,336 / 611,336 | [05](../algorithms/PPO-README.md) |
| D2SAC | DDPM 动作头 | 422,135 / 254,229 | [06](../algorithms/D2SAC-README.md) |
| DPPO | 扩散特征 + PPO | 519,238 / 519,238 | [07](../algorithms/DPPO-README.md) |
| EAT | 交叉注意力 + DDIM + 可学习 η | 709,560 / 397,942 | [08](../algorithms/EAT-README.md) |

参数量由 `tools/ckpt_inventory.py` 直接读 `policy_best.pth` 的张量名与元素数得到，不依赖 torch。

Parameter counts come from `tools/ckpt_inventory.py`, which reads tensor names and element counts straight out of `policy_best.pth` without torch.

---

## 7. 物理侧的一端 / The physical end

仿真里"任务的执行时间"是 `_estimate_exec_seconds(steps, parallelism)` 估出来的。真实系统里那是真的跑一次 Stable Diffusion。这部分实现在 `code/eat-server/`（FastAPI 服务 + distrifuser 后端 + SDXL 脚本），第 [12](../12-server/README.md) 章单独讲。
本仓库**没有**物理执行的日志或结果，所以第 12 章只写实现路径，不写数字。

In simulation, task execution time is estimated by `_estimate_exec_seconds(steps, parallelism)`. On real hardware it is an actual Stable Diffusion run, that lives in `code/eat-server/` and is covered in chapter [12](../12-server/README.md). This repository has **no** physical execution logs or results, so chapter 12 documents the code path and no numbers.

---

## 8. 本章的代码位置 / Code map

| 内容 | 文件 |
|---|---|
| 环境主循环与动作解码 | `code/eat-train/env/env.py`（`step` L157、`_compute_reward` L453、`preview_reward` L529） |
| 配置（规模、奖励权重） | `code/eat-train/env/config.py` |
| 全局实验配置（节点数、λ）| `code/eat-train/scripts/config.py` |
| 节点与进程组管理 | `code/eat-train/env/node.py`、`node_manager.py`、`scheduler.py` |
| 任务与队列 | `code/eat-train/env/task.py`、`taskmanager.py` |
| 交叉/自注意力块 | `code/eat-train/model/attention/cross_attention.py`、`self_attention.py` |

---

## 9. 训练数据导航 / Training data

本章是定义章，**没有训练数据**。

| 想核对的东西 | 去哪 |
|---|---|
| 奖励公式常数 | `code/eat-train/env/config.py#L84-L90` |
| 动作解码 | `code/eat-train/env/env.py#L167-L190` |
| 全部实验的数字 | [`docs/FACTS.md`](../docs/FACTS.md) |
| 全部运行的索引 | [`data/runs_index.csv`](../data/runs_index.csv) |

---
上一章 / previous：无（起点） · 下一章 / next：[01 非学习基线](../01-heuristics/README.md) · [返回目录](../README.md)
