# 训练数据说明 / Training data

本目录是**本仓库唯一的训练数据来源**。所有章节里引用的曲线、终点回报、η 轨迹，都能在这里按 `tag` 重新取出来。

This directory is the repository's single source of training data. Every curve, final return and η trace quoted in any chapter can be re-derived from here by `tag`.

---

## 1. 目录结构 / Layout

```
data/
├── runs_index.csv        88 次运行，一行一条（含终点/最高回报、路径）
├── by_algorithm.csv      按「实验 × 算法」聚合：n_runs、final_mean、final_std、best_mean、per_seed_final
├── fig6-main/            主对比（论文 Figure 6）：7 个算法 × 3 次
├── trials/               早期探索运行（命名还没稳定时期）
├── ablation/             SAC-CA / D2SAC-SA / EAT-CA 消融
├── learning-rate/        3e-4 学习率支路
├── fallback-rerun/       fallback 环境修正后的第二次全量扫描
├── scale-density/        32 节点 × 不同质量比（少量运行）
├── scale-density-z/      更早一版的规模/密度快照
├── misc/                 零散运行
└── model-copy/           作者保留的单个参考模型
```

叶子目录的命名是 `<variant>__<时间戳>`（没有 variant 时只有时间戳），例如
`data/scale-density-z/eat/32_0.556__2026-08-18_19-19-00/`。
每个叶子目录里固定两个文件：

| 文件 | 内容 |
|---|---|
| `curves.csv` | 三列 `tag,step,value`；最多 600 个点/标签（均匀抽稀，首尾必留） |
| `run.json` | 该次运行的元数据：实验、算法、`env_steps`、终点/最高回报、`ddim_eta` 终点值、保留的 tag 列表、原始文件路径与大小 |

---

## 2. 保留的 tag / Retained tags

原始日志有 15–180 个标签，其中 EAT 那几次每次参数都记一条 `actor_norm/<param>` 和 `actor_sparsity/<param>`（70 个参数 × 2），单个 event 文件因此到 1 GB。
入库时只保留有学习信号的标签：

| tag | 含义 |
|---|---|
| `test/reward`, `test/reward_std` | 固定测试面板上的回报（**Figure 6 的纵轴**） |
| `test/length`, `test/env_step` | 测试回合长度、对应的环境步数 |
| `train/reward`, `train/length`, `train/episode` | 训练回合的回报/长度 |
| `train/avg_quality`, `train/avg_response_time`, `train/completed_tasks_per_epoch` | 只有部分算法（EAT 系）记录 |
| `train/ddim_eta` | EAT 的可学习 η 轨迹 |
| `train/avg_min_q` | EAT 的最小 Q（用于看 critic 是否发散） |
| `train/lr_actor`, `train/lr_critic`, `train/action{0,1,2}_{mean,std}` | 学习率与动作统计 |
| `update/loss/actor`, `update/loss/critic1`, `update/loss/critic2`, `update/loss`, `update/loss/vf`, `update/loss/clip`, `update/loss/ent` | 损失 |
| `save/epoch`, `save/env_step`, `save/gradient_step` | 训练进度 |

被丢弃的：`actor_norm/*`、`actor_sparsity/*`、`critic*_norm/*`、`critic*_sparsity/*`。需要这些的话用原始 event 文件重跑 `tools/extract_runs.py`。

Dropped: the per-parameter `*_norm/*` and `*_sparsity/*` dumps. Re-run `tools/extract_runs.py` against the original event files if you need them.

---

## 3. 三列 CSV 怎么读 / Reading a curve

```python
import csv
from pathlib import Path

rows = [r for r in csv.reader(open(Path("data/fig6-main/eat") / RUN / "curves.csv"))
        if r and r[0] == "test/reward"]
points = [(float(r[1]), float(r[2])) for r in rows]      # (env_step, return)
```

三条约定，看数时不要踩：

1. `step` 是该标签自己的横轴。`test/*` 的 step 就是环境步数，所以不同算法可以直接横比；但每个算法的训练预算不同（336k / 416k / 810k+），曲线在图上会在各自预算处结束。
2. 第一个点常常是 `0.0`：那是"还没更新就评估"的冷启动点，画图时应当丢掉（`tools/plot_curves.py` 的 `drop_cold_start` 就是这么做的），但它**保留在 CSV 里**，不做任何篡改。
3. `test/reward` 是三次/多次评估的平均，单点噪声很大。仓库里的图统一用 `EMA(0.25)` 平滑后再比较，原始点仍可复算。

1. `step` is the tag's own x axis. For `test/*` it is environment steps, so algorithms can be compared horizontally — but each arm ends at its own budget (336k / 416k / 810k+).
2. The first point is often `0.0`: that is the cold-start evaluation before any update. Plots drop it (`drop_cold_start`), the CSV keeps it. Nothing is overwritten.
3. `test/reward` is already an average over evaluation episodes and is noisy point to point. The figures apply `EMA(0.25)` before comparing; raw points remain recomputable.

---

## 4. 重新生成 / Regenerate

原始日志不在仓库里（7.4 GB）。若你手上有 `EATlogs/EAT基线对比/**/events.out.tfevents.*`：

```powershell
python tools\extract_runs.py --log-root <EATlogs目录> --out tmp\extracted --max-points 1200
python tools\curate.py        --in-root tmp\extracted --out data       --max-points 600
python tools\flatten_data.py
python tools\plot_curves.py   --data data --out figures\curves
```

- `tools/safe_tfreader.py`：不依赖 tensorboard / protobuf 的 TFRecord + Event 解析器（约 120 行）
- `tools/extract_runs.py`：遍历 + 抽稀 + 写 `run.json`
- `tools/curate.py`：过滤 tag、平铺目录、写两张索引表
- `tools/flatten_data.py`：把旧的分层目录改成 `<variant>__<run>` 并补齐索引
- `tools/plot_curves.py`：只靠 Pillow 出图
- `tools/ckpt_inventory.py`：读 `.pth` 的张量名/元素数，得到参数量表

---

## 5. 数据与结论的边界 / What the data can and cannot show

| 能证明 / can show | 不能证明 / cannot show |
|---|---|
| 每条曲线的终点、最高点、跨种子方差 | 论文的某一次具体运行（原始日志没有保存运行配置） |
| 参数量与网络结构（来自 checkpoint 张量名） | 作者当时的启动命令（`scripts/fig6/train.py` 不在快照里） |
| 同一协议下算法间的相对位置 | 跨 `quality_ratio` 的回报比较（奖励函数变了） |
| EAT 的 η 轨迹 | 物理执行的延迟/吞吐（没有物理日志） |

> 只有 **3 次运行/算法**（消融里有两个臂只有 1 次）。任何"差 1–2 分"的结论都在噪声范围内，只有 D2SAC 与 EAT 的方差这种量级差异才值得解释。
> There are only **3 runs per algorithm** (two ablation arms have a single run). Any 1–2 point gap is inside the noise; only differences of the size seen for D2SAC and for EAT's variance deserve an explanation.
