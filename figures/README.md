# 图目录 / Figure index

两套图，用途不同：

- **`curves/`**, 本仓库用 `tools/plot_curves.py` 从 `data/` **重新绘制**的图。数字与 `docs/FACTS.md` 一一对应，可以复算。
- **`original/`**, 作者当时自己画的图，**原样保留**。用来对照"当时看到的形状"和"现在重画的形状"。

Two sets. `curves/` is redrawn here from `data/` and every value is traceable to `docs/FACTS.md`. `original/` keeps the author's own renders untouched, for comparison.

重新生成 / regenerate:

```powershell
python tools\plot_curves.py --data data --out figures\curves
```

---

## curves/, 本仓库重绘 / redrawn here

| 文件 | 内容 | 用在 |
|---|---|---|
| `fig6_convergence.png` | 七个算法，3 个种子的均值 ± 种子带，横轴环境步数 | [README](../README.md)、[13](../13-results/README.md) |
| `fig6_final_bar.png` | 终点回报柱状 + 标准差须（纵轴从 140 起，注意） | [13](../13-results/README.md) |
| `fig6_eta.png` | EAT 学到的 DDIM η 轨迹（3 个种子） | [08](../08-eat/README.md)、[13](../13-results/README.md) |
| `ddpg_convergence.png` | DDPG 三次运行 + 均值 | [02](../02-ddpg/README.md) |
| `td3_convergence.png` | TD3 | [03](../03-td3/README.md) |
| `sac_convergence.png` | SAC | [04](../04-sac/README.md) |
| `ppo_convergence.png` | PPO | [05](../05-ppo/README.md) |
| `d2sac_convergence.png` | D2SAC | [06](../06-d2sac/README.md) |
| `dppo_convergence.png` | DPPO | [07](../07-dppo/README.md) |
| `eat_convergence.png` | EAT | [08](../08-eat/README.md) |
| `ablation.png` | SAC / SAC-CA、D2SAC / D2SAC-SA、EAT / EAT-CA 三对 | [09](../09-ablation/README.md) |
| `learning_rate.png` | 3e-4 支路 vs 主对比（1e-4） | [10](../10-tuning/README.md) |
| `fallback_rerun.png` | fallback 环境修正后的第二次全量扫描 | [10](../10-tuning/README.md) |
| `scale_density.png` | 32 节点、四档质量比的 SAC / EAT | [11](../11-scale-density/README.md) |

画图约定：`test/reward` 先丢掉"更新前评估"的冷启动点，再做 `EMA(0.25)`；阴影带是该算法各种子的 min/max；每条曲线在它自己的步数预算处结束（不同算法的预算不同，见 FACTS §1）。

Plotting conventions: cold-start point dropped, then `EMA(0.25)`; the shaded band is the seed min/max; each curve ends at that arm's own step budget.

---

## original/, 作者原图 / the author's own figures

| 目录 | 内容 | 说明 |
|---|---|---|
| `per-algorithm-curves/` | `zfig4/`：`ddpg/td3/sac/ppo/d2sac/dppo/eat_mean_curve.png` + `fig6_mean_normalized.png` | 每个算法一张平滑均值曲线，外加归一化的 Figure 6 |
| `fig6-normalized-v1/` | `Z0/`：`fig6.png`、`fig6_mean_normalized.png` | 早期归一化版本 |
| `fig6-normalized-v2/` | `Z1/`：`fig6_per_algo_normalized.png`、`seeds_test_reward.png` | 按算法归一化 + 种子散点 |
| `paper-style/` | `Figure/NewFigure/`：`Figure1_EAT-DDIM`、`Figure2_DDPG`、`Figure3_SAC`、`Figure4_ppo`、`Figure6_td3`、`Figure7_EAT-CA`、`Figure8_EAT`、`Figure9_*` | 按论文图号命名的看图版本 |
| `early-plots/` | `Figure/LLLLLLLLOVE/` | 最早的两张探索图 |
| `misc/` | `eat_mean_curve.png`、`fig6_mean_normalized.png` | 散落在实验目录根部的两张 |

**对照提示**：作者的原图是把多个种子的 `test/reward` 直接平均后画的，没有做 EMA；`curves/` 里的图做了 EMA(0.25)，所以同一条曲线的抖动幅度会更小，但终点位置一致。如果形状差异很大，先查是不是拿了不同实验目录的数据。
