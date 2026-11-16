# 多智能体部分：跑了多少，能报什么
# The multi-agent side: how many runs, and what can be claimed

本文件汇总 EAT 复现之外的第二块工作，也就是把单集群调度扩展成多智能体连续动作版本之后，实际跑过的实验数量与规模。
数字都来自磁盘上的 campaign 清单与日志目录，路径写在每张表下面，可以逐个核对。

This page collects the second piece of work besides the EAT reproduction: the multi-agent continuous-action extension, and the number of experiments actually run. Every count comes from a campaign manifest or a log directory on disk, and the path is listed so it can be checked.

---

## 1. 这一块做了什么

原来的 EAT 是单智能体、离散动作、单集群。扩展后的 AMARL 换了协议，要点有四条：

- 动作连续化：每个 agent 输出 `target`（集群）+ 三个 `float32` 连续量（是否开工、质量步数、任务槽位），落在 `[-1,1]`。
- 一个协议只服务一套语义：`THREE_CLUSTER_EAT_CONTINUOUS_V1` 是三集群 24 节点，`FIVE_CLUSTER_EAT_CONTINUOUS_V1` 是五集群 40 节点 160 任务。
- 训练后端统一到 Tianshou 0.5.1，`Collector` 与 `OnpolicyTrainer` / `OffpolicyTrainer` 接管采样与更新，算法只写策略适配层。
- 评估只走一个入口，所有臂共用同一批评估回合种子，避免"各评各的"。

The original EAT is single-agent, discrete, single-cluster. The extension changes the protocol on four points: continuous actions per agent, one protocol per topology, a single Tianshou training backend, and one evaluator shared by every arm.

---

## 2. 跑了多少

三个层级，加起来 **170 次训练**。另外还有调参与冒烟运行，不计入。

| 组 | 内容 | 训练次数 | 每臂步数 | 每次评估回合 | 证据位置 |
|---|---|---:|---:|---:|---|
| 正式三集群 `formal_a` | 8 个算法 × 5 个种子 | 40 | 30,000 agent steps | 40 | `Result/new_a.tgz`、`new_a_checkpoints.tgz` |
| 正式三集群 `formal_b` | 6 个算法 × 5 个种子 | 30 | 30,000 | 40 | `Result/new_b_details.tgz`、`new_b_check.tgz` |
| 五算法扫描（三集群） | 5 算法 × 4 组配置 × 3 种子，加 2 个追加种子 × 5 算法，加 MA-EAT × 5 种子 | 75 | 30,000 | 5 | `A5Ten/`、`AAAATen/`、`AATen/`、`A63/`、`A5Best/`、`A5MAEAT/` |
| 五集群扩展 | 5 个算法 × 5 个种子 | 25 | 150,000 | 5 | `runs/five_agent_150000_*`、`.audit_a75_*` |
| **合计** | | **170** | | | |

补充两个数字，写材料时可能用得上：

- **正式三集群那 70 次，每个叶节点留了 3 个 checkpoint**（`best.pt`、`final.pt`、`latest.pt`），合计 210 个权重文件，`new_a_checkpoints.tgz` 与 `new_b_check.tgz` 里可以按 `training/<ALGO>/seed_<N>/` 数出来。
- **五算法扫描那 75 次跨 6 个目录**，原因是学习率选择与硬件更换：`AAAATen` 与 `A5Ten` 是两档学习率的对照，`A63` 是在 3090 上重跑，`AATen` 是最早一批，`A5Best` 只放追加的 102 与 210 两个种子。

Two extra numbers worth keeping: the 70 formal runs leave 210 checkpoint files (three per leaf), and the 75 sweep runs are spread over six directories because of learning-rate selection and a GPU change, not because they are duplicates.

---

## 3. 协议与规模

| 项 | 三集群 | 五集群 |
|---|---|---|
| `protocol_id` | `THREE_CLUSTER_EAT_CONTINUOUS_V1` | `FIVE_CLUSTER_EAT_CONTINUOUS_V1` |
| 集群 × 节点 | 3 × 8 = 24 | 5 × 8 = 40 |
| 智能体数 | 3 | 5 |
| 每回合任务上限 | 32 | 160 |
| 动作维度（每 agent） | 目标 3 类 + 连续 3 维 | 目标 5 类 + 连续 3 维 |
| 联合动作特征 | 3 × 6 = 18 | 5 × 8 = 40 |
| 种子 | 43、57、85、102、210 | 43、57、85、102、210 |
| 训练步数 | 30,000 agent steps（约 10,000 环境步） | 150,000 agent steps（约 30,000 环境步） |

环境哈希在 `campaign_manifest.json` 里写着，同一个哈希才允许横比。三集群那批是 `sha256:17c9fafba8d5e7032135e24862d8cff7`。

Each manifest records an environment hash, and only runs with the same hash may be compared. The three-cluster batches all share `sha256:17c9fafba8d5e7032135e24862d8cff7`.

---

## 4. 算法清单

**正式三集群 14 个学习算法**：MAPPO、IPPO、MASAC、MATD3、MADDPG、MA-DPPO、MA-D2SAC、MA-D2SAC-SA、MA-D2SAC-REF、MA-EAT、MA-EAT-PAPER、MA-EAT-SELFQ、MA-EAT-CA、MA-EAT-D。

**四个非学习基线**：FIFO_MAX_FREE、RANDOM_LEGAL、GREEDY_LOCAL（公平），以及 GREEDY_ORACLE（有特权，单独成表）。

**五集群 5 个算法**：MAPPO、MASAC、MATD3、MADDPG、MA-DPPO。另有 MA-EAT 的五种子单独一批。

The formal three-cluster comparison covers fourteen learning algorithms plus four heuristics, with the privileged oracle reported separately. The five-cluster extension covers five algorithms plus a separate five-seed MA-EAT batch.

---

## 5. 写材料时可以这么报

简历一行：

> 多智能体扩展（AMARL）：把单智能体调度改写为三集群与五集群的连续动作 MARL 协议，实现并对比 14 个学习算法与 4 个启发式基线，累计 **170 次正式训练**（每臂 3 万到 15 万智能体步，5 种子，统一评估面板），产出 210 个 checkpoint 与逐叶点的训练日志。

邮件里一句：

> 除 EAT 复现外，我还做了一版多智能体连续动作扩展，三集群有 14 个算法、5 个种子的正式对比（70 次），五集群有 5 个算法的扩展（25 次），另外 75 次是学习率选择与硬件重跑。

A resume line and a one-sentence version are given above, both numbers-first.

---

## 6. 不能报的部分

- **调参与冒烟运行**：工作区里还有约 386 个带清单的 campaign 叶节点，其中大部分是 900 步到 30 步的冒烟，或者调参候选。这些不进任何表。
- **五集群的 5 回合评估**：每次只评 5 回合，样本量小于三集群的 40 回合，做显著性讨论没有意义。
- **跨协议比较**：三集群与五集群的回报不可直接比，拓扑不同，任务数也不同。
- **`不用/` 目录**：那是废弃版本，保留只为追溯。

What cannot be claimed: the smoke and tuning campaigns, the five-episode evaluations on the five-cluster side, comparisons across protocols, and anything in the discarded folder.

---

## 7. 怎么复核

```powershell
# 正式三集群：按算法与种子数叶节点
tar -tzf Result\new_a.tgz | Select-String "training/.+/seed_\d+/status.json"
tar -tzf Result\new_b_details.tgz | Select-String "training/.+/seed_\d+/status.json"

# 五集群：每个叶节点一个清单
Get-ChildItem runs\five_agent_150000_* -Filter campaign_manifest.json
```

三集群的汇总表在 `Result/formal_b_small.tgz` 里，解压后是 `summary.csv`、`fair_table.csv`、`campaign_manifest.json`，可直接读。

The three-cluster summary tables live in `Result/formal_b_small.tgz` as `summary.csv`, `fair_table.csv` and `campaign_manifest.json`.
