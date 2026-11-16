# 命名对照表 / Naming map

这套代码里同一个算法有四套名字：**运行目录名**（TensorBoard 日志里）、**trainer 目录名**（源码里）、**论文里的名字**、**作者笔记里的名字**。
它们不完全一致，看数据前先对一遍。标 推断 的行是由 checkpoint 的张量名与参数量反推的，不是来自保存的配置。

The same algorithm carries four names here: the run-directory name (inside the logs), the trainer directory name (in the source), the paper's name, and the name used in the author's notes. They do not always agree; this table is the Rosetta stone. Rows marked 推断 / inferred were derived from checkpoint tensor names and parameter counts, not from a saved config.

---

## 1. 主对比里的七个算法 / The seven Figure-6 arms

| 运行目录 | trainer 目录 | 论文名 | 关键证据 / evidence |
|---|---|---|---|
| `ddpg` | `trainer/ddpg` | DDPG | `DDPGPolicy`；checkpoint 只有 `actor/critic` + `*_old`，无扩散 buffer |
| `td3` | `trainer/td3` | TD3 | `TD3Policy`；`critic1/critic2` + `actor_old` |
| `sac` | `trainer/sac` | SAC | `SACPolicy`；`critic1/critic2` + `*_old`，actor 是 `ActorProb`（`mu`/`sigma` 张量） |
| `ppo` | `trainer/ppo` | PPO | `PPOPolicy`；`_actor_critic.*` 前缀 |
| `d2sac` | `trainer/dcasac` | D2SAC | 推断：422,135 参数；critic 带 `preprocess.obs_net.*` + `preprocess.action_embed`（= `CrossAttentionQPreprocessNet`）；actor 含 `betas`/`alphas_cumprod`（DDPM）但**没有** `eta_param` |
| `dppo` | `trainer/dppo` | DPPO | 推断：519,238 参数；`actor.preprocess.betas`/`alphas_cumprod` 说明扩散块在 actor 的 preprocess 里，`Diffusion` + `ActorProb` 组合 |
| `eat` | `trainer/dicasac` | EAT | 推断：709,560 参数；actor 含 `eta_param` + `betas`/`alphas_cumprod`（DDIM），critic 为 `preprocess.ln_q/ln_k/ln_v` 自注意力结构 |

## 2. 消融臂 / The ablation arms

| 运行目录 | trainer 目录 | 含义 | 关键证据 / evidence |
|---|---|---|---|
| `sac_ca` | `trainer/casac` | 高斯 SAC + 交叉注意力 | 推断：419,239 参数；actor 有 `cond_preprocess`（交叉注意力），无 `betas` |
| `d2sac_sa` | `trainer/dsac` | DDPM + 自注意力 critic | 推断：709,559 参数；critic 为 `preprocess.ln_q/ln_k/ln_v` 自注意力（每侧 155,809），无 `eta_param` |
| `eat_ca` | 无独立 trainer | EAT 去掉交叉注意力（actor 直接吃原始观测） | 推断：775,608 参数；有 `eta_param`（DDIM），但 actor 的 `model.mlp.model.0` 是 84,992（输入 331 维原始状态），而不是 19,456（输入 75 维注意力特征） |

## 3. 学习率支路 / The 3e-4 arms

| 运行目录 | 对应主对比臂 | 说明 |
|---|---|---|
| `d2sac3e` | `d2sac` | 学习率 3e-4；参数量与主对比相同（422,135） |
| `eat3e` | `eat` | 学习率 3e-4；参数量是 **841,144**，即这一支用的是更大的 critic 版本，不是主对比的 709,560 |
| `td33e` | `td3` | 学习率 3e-4；参数量相同（524,810） |

## 4. 早期草稿 / Early drafts

| 运行目录 | 含义 |
|---|---|
| `tryeat` | EAT 的早期草稿（67,200 步的短跑，参数量 841,144） |
| `trials/eat` | 短跑试验，67,200 步 |
| `trials/d2sac` | 短跑试验 |

## 5. 缩写 / Abbreviations

| 缩写 | 展开 | 在本仓库里对应什么 |
|---|---|---|
| D2SAC | Diffusion + SAC | DDPM 动作头的 SAC（`SACDDPMPolicy`） |
| DPPO | Diffusion + PPO | 扩散块做特征/PV、PPO 做更新的组合 |
| EAT | 论文名（注意力引导的扩散 RL） | DDIM + 交叉注意力 + 可学习 η |
| CA | Cross Attention | 任务 token 做 Query、节点 token 做 Key/Value（`cross_attention.py#L12`） |
| SA | Self Attention | 观测/动作 token 自注意力（`self_attention.py#L11`、`#L127`） |
| DICA | Diffusion + cross attention 的旧名 | 作者笔记里 `sac_dica_train.py` 对应的算法，即后来的 EAT |
| η | DDIM 的随机性系数 | `rl_diffusion_ddim.py#L75` 的 `eta_param`，经 `sigmoid` 映射到 [0,1]，同时充当策略熵代理 |
| DDPM / DDIM | 两种扩散采样 | `model/diffusion/rl_diffusion.py`（DDPM）/ `rl_diffusion_ddim.py`（DDIM） |

## 6. 容易被搞混的三件事 / Three easy confusions

1. **`d2sac` 不等于 `dsac`**：运行目录的 `d2sac` 反推到 `trainer/dcasac`（交叉注意力 critic），而 `trainer/dsac` 对应的是消融里的 `d2sac_sa`。两者参数量差了一倍（422,135 vs 709,559）。
2. **同名不同模型**：`fig6-main` 与 `fallback-rerun` 里的 `eat` 是两个网络（709,560 vs 841,144），`d2sac` 也是（422,135 vs 713,431）。跨扫描比数字之前先看参数量。
3. **`quality_ratio` 不是超参数而是奖励权重的比值**：0.600 与 0.769 对应两套不同的奖励函数，跨这两个值的回报不可比（见 [11 章](../11-scale-density/README.md)）。
