# 算法 README 目录：七个强化学习算法
# Algorithm READMEs: the seven RL arms

这里放七个算法的单篇 README。每一篇讲一个算法，机制、设置、收敛曲线、结果、分析和数据链接都在同一页。代码在上层的 `code/`，数据在上层的 `data/`，两个目录与本目录同级。

This folder holds one README per algorithm. Code lives in `../code/`, data in `../data/`; both sit next to this folder.

## 目录

| 文件 | 算法 | 训练次数 | 终点回报均值 ± 标准差 | 曲线 | 数据 |
|---|---|---:|---:|---|---|
| [DDPG-README](DDPG-README.md) | DDPG | 3 | 180.4 ± 8.6 | [图](../figures/curves/ddpg_convergence.png) | [`data/fig6-main/ddpg/`](../data/fig6-main/ddpg/) |
| [TD3-README](TD3-README.md) | TD3 | 3 | 181.3 ± 8.3 | [图](../figures/curves/td3_convergence.png) | [`data/fig6-main/td3/`](../data/fig6-main/td3/) |
| [SAC-README](SAC-README.md) | SAC | 3 | 184.3 ± 2.5 | [图](../figures/curves/sac_convergence.png) | [`data/fig6-main/sac/`](../data/fig6-main/sac/) |
| [PPO-README](PPO-README.md) | PPO | 3 | 185.2 ± 2.0 | [图](../figures/curves/ppo_convergence.png) | [`data/fig6-main/ppo/`](../data/fig6-main/ppo/) |
| [D2SAC-README](D2SAC-README.md) | D2SAC | 3 | 159.5 ± 4.0 | [图](../figures/curves/d2sac_convergence.png) | [`data/fig6-main/d2sac/`](../data/fig6-main/d2sac/) |
| [DPPO-README](DPPO-README.md) | DPPO | 3 | 185.6 ± 1.0 | [图](../figures/curves/dppo_convergence.png) | [`data/fig6-main/dppo/`](../data/fig6-main/dppo/) |
| [EAT-README](EAT-README.md) | EAT | 3 | 177.7 ± 16.5 | [图](../figures/curves/eat_convergence.png) | [`data/fig6-main/eat/`](../data/fig6-main/eat/) |

七篇的读法建议按这个顺序：DDPG 是起点，TD3 修偏差，SAC 加熵，PPO 换到 on-policy，D2SAC 把动作头换成扩散，DPPO 把扩散放到特征层，EAT 再加上交叉注意力与可学习 η。

The intended order is the one above: DDPG as the starting point, TD3 fixing bias, SAC adding entropy, PPO moving on-policy, then the three diffusion arms.

## 图的约定

七张曲线图用同一个蓝色，方便并排比较。横轴是环境步数，纵轴是验证回报，阴影带是该算法四次以内各种子的最小到最大范围。图题与图例都写了训练次数，例如 `SAC · 3 runs` 与 `SAC (n=3)`。

All seven curves share one blue so they can be compared side by side. The band is the seed min to max, and the run count appears in both the title and the legend.

重绘命令：

```powershell
python tools\plot_curves.py --data data --out figures\curves
```

## 其他实验去哪了

消融、学习率、规模与密度三类实验不在这个目录，按原来的编号放在上层：`../09-ablation/`、`../10-tuning/`、`../11-scale-density/`。它们比较的是同一算法族内部的两个版本，读之前先看 [../09-ablation/README.md](../09-ablation/README.md) 的说明。

The ablation, learning-rate and scale experiments stay in `../09-ablation/`, `../10-tuning/` and `../11-scale-density/`, because they compare two variants of one family rather than standing on their own.

---

返回总目录：[../README.md](../README.md)　背景与定义：[../00-problem/README.md](../00-problem/README.md)　数字出处：[../docs/FACTS.md](../docs/FACTS.md)
