# 复现步骤 / Reproduce

> **先读这一条 / read this first**：本仓库**没有**在这里重新执行过训练。下面的命令来自代码入口和作者笔记，逐条对照过源码，但没有在干净环境里端到端验证过；任何标着 未验证 的地方都不要当成"已验证可跑"。
> **This repository did not re-run training here.** The commands below are taken from the code entry points and the author's notes and checked against the source, but were not verified end-to-end on a clean machine. Anything marked 未验证 / unverified is not a promise.

---

## 1. 环境 / Environment

作者笔记记录的依赖（`EATlogs/Reading/AAAresult.docx`）：

```powershell
pip install tianshou==0.5.1 gymnasium==1.1.1 tensorboard numpy scipy matplotlib einops pyyaml requests
pip install torch torchvision    # 版本未记录；代码用到 torch 的 register_buffer / nn.Mish
```

| 项 | 值 | 来源 |
|---|---|---|
| Tianshou | 0.5.1 | 作者笔记；代码 import 路径与之匹配（`tianshou.policy.PPOPolicy`、`tianshou.trainer.OffpolicyTrainer`） |
| Gymnasium | 1.1.1 | 作者笔记 |
| Python | 未记录（代码有 `from __future__ import annotations`，需要 ≥3.7） | — |
| GPU | 未记录（作者笔记里出现过 RTX 4080 SUPER 和笔记本两种环境） | — |

## 2. 让 import 跑通 / Make imports resolve

训练脚本以 `env`、`model`、`policy`、`trainer` 为顶层包导入，因此必须把 `EAT-Train` 及其 `scripts` 都放进 `PYTHONPATH`（作者的写法）：

```powershell
$env:PYTHONPATH = "$PWD\code\eat-train;$PWD\code\eat-train\scripts"
```

`scripts\config.py` 里有两层配置：`GlobalConfig`（节点数、可见任务数、泊松到达率）和 `TrainConfig`（网络宽度、扩散步数、保存间隔）。改节点数/密度就是改 `GlobalConfig`。

## 3. 训练 / Training

每个算法有独立的入口脚本（`code/eat-train/scripts/`）：

| 算法 | 训练脚本 | 评估脚本 |
|---|---|---|
| DDPG | `ddpg_train.py` | `eval_ddpg.py` |
| TD3 | `td3_train.py` | `eval_td3.py` |
| SAC | `sac_train.py` | `eval_sac.py` |
| PPO | `ppo_train.py` | `eval_ppo.py` |
| SAC-CA（交叉注意力） | `sac_ca_train.py` | `eval_casac.py` |
| D2SAC（DDPM） | `sac_d_train.py` | `eval_dcasac.py` |
| EAT（DDIM + η） | `sac_dica_train.py` | `eval_dsac.py` |
| 非学习基线 | — | `eval_FIFO.py`、`eval_rand_action.py`、`eval_greedy.py` |

```powershell
cd code\eat-train
python scripts\sac_dica_train.py        # 未验证：脚本自身没有 argparse，参数在文件顶部常量里
```

**注意 / caveat**：这些脚本目前是"改常量再运行"的风格，没有命令行参数（`sac_dica_train.py` 只有 `main()`）。作者当时用的是另一个驱动脚本 `scripts/fig6/train.py`（支持 `--algo eat --epoch 500 --device cuda --seed 43`），**它不在本快照里**，所以本仓库无法复现"一行命令跑一个算法"。

These scripts are "edit the constants, then run" style and take no CLI arguments; the author's actual driver (`scripts/fig6/train.py`) is not in this snapshot, so the original one-line invocation cannot be reproduced here.

## 4. 输出去哪 / Where outputs land

```
EAT-Train/runs/<节点数>_<质量比>/<算法>/<时间戳>/
    events.out.tfevents.*      TensorBoard 曲线（本仓库 data/ 的来源）
    policy.pth                 最后一次保存
    policy_best.pth            验证回报最好的一次
    policy_epoch<N>.pth        save_interval_epochs 的周期快照
```

评估脚本通过 `scripts/utils.py#L144` 的 `find_latest_ckpt(node_num, quality_ratio, algo_prefix)` 自动找最新一次运行的 checkpoint，然后把逐回合 CSV 写到 `data/<节点数>_<质量比>/<算法>/<时间戳>/eval_*.csv`。

## 5. 从 curve 到图 / From curves to figures

不用跑训练，只重绘本仓库的图（只需 Pillow）：

```powershell
python tools\plot_curves.py --data data --out figures\curves
```

## 6. 你手上如果有原始日志 / If you have the original logs

```powershell
python tools\extract_runs.py --log-root <EATlogs目录> --out tmp\extracted --max-points 1200
python tools\curate.py        --in-root tmp\extracted --out data       --max-points 600
python tools\flatten_data.py
python tools\plot_curves.py   --data data --out figures\curves
```

88 次运行、7.4 GB 的 event 文件，解析大约 10 分钟（其中 EAT 的十几份各 ~1 GB）。

## 7. 想验证参数量 / To verify a parameter count

不需要装 torch：

```powershell
python tools\ckpt_inventory.py tmp\ckpt <label>=<path\to\policy_best.pth>
```

它按 `pickletools` 遍历 `data.pkl`，把每个张量名和元素数配起来，输出按子模块聚合的参数量。本仓库 [docs/FACTS.md](FACTS.md) 里的参数量都是这么来的。

## 8. 未验证清单 / What is unverified

| 项 | 状态 |
|---|---|
| 训练脚本端到端跑通 | 未验证（无干净环境测试） |
| 原始启动命令 | 丢失（`scripts/fig6/train.py` 不在快照） |
| 评估指标（响应时间/质量）重算 | 不可能（评估 CSV 未保留） |
| 物理 SD 执行 | 未验证（无日志） |
| CUDA 版本/驱动 | 未记录 |
