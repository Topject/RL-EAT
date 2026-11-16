# `data/`, 训练数据总目录 / training data

这是本仓库**唯一**的训练数据目录。每章结尾的"训练数据导航"都指回这里。
This is the repository's single training-data root. Every chapter's "training data" section links back here.

- 目录结构与字段说明：[`../docs/DATA.md`](../docs/DATA.md)
- 每个数字的出处：[`../docs/FACTS.md`](../docs/FACTS.md)
- 一张表看全部 88 次运行：[`runs_index.csv`](runs_index.csv) · 按算法聚合：[`by_algorithm.csv`](by_algorithm.csv)

---

## 1. 按算法找数据 / Find data by algorithm

| 算法 | 主对比（Figure 6）| 其他实验 | 对应章节 |
|---|---|---|---|
| **DDPG** | [`fig6-main/ddpg/`](fig6-main/ddpg/) | [`fallback-rerun/ddpg/`](fallback-rerun/ddpg/) | [02](../algorithms/DDPG-README.md) |
| **TD3** | [`fig6-main/td3/`](fig6-main/td3/) | [`fallback-rerun/td3/`](fallback-rerun/td3/) · [`learning-rate/td33e/`](learning-rate/td33e/) | [03](../algorithms/TD3-README.md) |
| **SAC** | [`fig6-main/sac/`](fig6-main/sac/) | [`fallback-rerun/sac/`](fallback-rerun/sac/) · [`scale-density-z/sac/`](scale-density-z/sac/) | [04](../algorithms/SAC-README.md) |
| **PPO** | [`fig6-main/ppo/`](fig6-main/ppo/) | [`fallback-rerun/ppo/`](fallback-rerun/ppo/) · [`scale-density-z/ppo/`](scale-density-z/ppo/) | [05](../algorithms/PPO-README.md) |
| **D2SAC** | [`fig6-main/d2sac/`](fig6-main/d2sac/) | [`fallback-rerun/d2sac/`](fallback-rerun/d2sac/) · [`learning-rate/d2sac3e/`](learning-rate/d2sac3e/) · [`trials/d2sac/`](trials/d2sac/) | [06](../algorithms/D2SAC-README.md) |
| **DPPO** | [`fig6-main/dppo/`](fig6-main/dppo/) | [`fallback-rerun/dppo/`](fallback-rerun/dppo/) | [07](../algorithms/DPPO-README.md) |
| **EAT** | [`fig6-main/eat/`](fig6-main/eat/) | [`fallback-rerun/eat/`](fallback-rerun/eat/) · [`learning-rate/eat3e/`](learning-rate/eat3e/) · [`trials/eat/`](trials/eat/) · [`trials/tryeat/`](trials/tryeat/) · [`scale-density-z/eat/`](scale-density-z/eat/) | [08](../algorithms/EAT-README.md) |

消融臂 / ablation arms（[09](../09-ablation/README.md)）：[`ablation/sac_ca/`](ablation/sac_ca/) · [`ablation/d2sac_sa/`](ablation/d2sac_sa/) · [`ablation/eat_ca/`](ablation/eat_ca/)

零散 / leftovers：[`misc/mixed/`](misc/mixed/)（单次运行）· [`model-copy/reference/`](model-copy/reference/)（保留的参考模型）· [`scale-density/`](scale-density/)（32_0.556 的两次短跑）

---

## 2. 一次运行里有什么 / Inside one run

```
data/fig6-main/eat/2026-08-19_20-25-52/
├── curves.csv     tag,step,value，最多 600 点/标签
└── run.json       实验、算法、env_steps、终点/最高回报、η 终点、保留的 tag 列表
```

`run.json` 里最关键的两个字段：

| 字段 | 用法 |
|---|---|
| `test_reward.last` | 该次运行的"终点回报"，`by_algorithm.csv` 的 `final_mean` 就是它的均值 |
| `test_reward.best` | 训练过程中的最高评估回报（不是最终模型的表现） |

---

## 3. 每个实验的用途 / What each campaign is for

| 目录 | 目的 | 运行数 | 章节 |
|---|---|---|---|
| `fig6-main/` | 复现论文 Figure 6：7 算法 × 3 次 | 21 | 02–08 |
| `trials/` | 命名与结构稳定之前的探索跑（多为 67k 步短跑） | 8 | 08 附录 |
| `ablation/` | 拆机制：加/去交叉注意力、加/去自注意力 | 6 | 09 |
| `learning-rate/` | 3e-4 vs 1e-4 | 10 | 10 |
| `fallback-rerun/` | 环境 task-index fallback 修正后的 7 算法全量重跑 | 21 | 10 |
| `scale-density/`、`scale-density-z/` | 32 节点 × 四档 quality_ratio | 15 | 11 |
| `misc/`、`model-copy/` | 零散运行与保留模型 | 2 | — |

合计 88 次运行。

---

## 4. 使用约束 / Usage constraints

1. **只有 3 次运行/算法**（两个消融臂只有 1 次）。1–2 分的差距在噪声内。
2. **训练预算不同**：off-policy 臂 336k 步，EAT 416k，DPPO ~810k，PPO ~849k。横比前先看 `by_algorithm.csv` 的 `env_steps_max`。
3. **同名不同网络**：`fig6-main/eat`（709,560 参数）与 `fallback-rerun/eat`（841,144）不是同一个模型；`d2sac` 同理。
4. **跨 `quality_ratio` 不可比**：奖励函数不同。
5. `curves.csv` 里第一个 `test/reward` 常常是 0.0（更新前评估），画图时该丢掉，但不要修改文件。

1. Only 3 runs per algorithm (two ablation arms have 1). A 1–2 point gap is noise.
2. Budgets differ: 336k (off-policy), 416k (EAT), ~810k (DPPO), ~849k (PPO).
3. Same name, different network: `fig6-main/eat` (709,560 params) vs `fallback-rerun/eat` (841,144).
4. Returns are not comparable across `quality_ratio` values, the reward function changes.
5. The first `test/reward` point is often 0.0 (pre-update evaluation). Drop it when plotting; do not edit the file.
