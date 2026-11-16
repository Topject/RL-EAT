# 可核对数字表 / Verifiable numbers

本文件汇总仓库里**所有被章节引用的数字**及其出处。任何章节中出现的数值都必须能在这里找到对应行。
Every number quoted in a chapter must appear here with its source path.

- 原始日志：`EATlogs/EAT基线对比/**/events.out.tfevents.*`（TensorBoard scalar，已由 `tools/extract_runs.py` 解析）
- 清洗后数据：`data/<campaign>/<algo>/<run>/curves.csv`
- 逐次运行汇总：`data/runs_index.csv`
- 按算法聚合：`data/by_algorithm.csv`
- 参数量：`tools/ckpt_inventory.py` 读取 `policy_best.pth` 得到（含目标网络的“总参数量”，以及去掉 `*_old` 的“在线参数量”）

---

## 1. 主对比 `fig6-main`（论文 Fig. 6 复现）

每个算法 3 次独立训练。作者笔记记录的随机种子为 42 / 120 / 245；运行目录本身不保存种子。

| 算法 | 目录 | 终点回报（每次运行） | 均值±样本标准差 | 最高回报 | 训练步数上限 | 参数量（总 / 在线） |
|---|---|---|---|---|---|---|
| DDPG | `ddpg` | 186.18 / 184.51 / 170.44 | **180.4 ± 8.6** | 187.3 | 336,000 | 349,704 / 174,852 |
| TD3 | `td3` | 183.04 / 172.23 / 188.59 | **181.3 ± 8.3** | 191.1 | 336,000 | 524,810 / 262,405 |
| SAC | `sac` | 183.04 / 182.66 / 187.25 | **184.3 ± 2.5** | 189.9 | 336,000 | 437,511 / 262,405 |
| PPO | `ppo` | 182.89 / 186.73 / 185.96 | **185.2 ± 2.0** | 193.0 | 849,149 / 848,668 / 854,041 | 611,336 / 611,336 |
| D2SAC | `d2sac` | 158.63 / 156.04 / 163.83 | **159.5 ± 4.0** | 166.8 | 336,000 | 422,135 / 254,229 |
| DPPO | `dppo` | 186.13 / 184.45 / 186.28 | **185.6 ± 1.0** | 193.0 | 804,141 / 816,547 / 817,408 | 519,238 / 519,238 |
| EAT | `eat` | 187.17 / 187.38 / 158.68 | **177.7 ± 16.5** | 185.3 | 416,000 | 709,560 / 397,942 |

- 复现排序（按终点均值）：**PPO 185.2 ≈ DPPO 185.6 > SAC 184.3 > TD3 181.3 > DDPG 180.4 > EAT 177.7 > D2SAC 159.5**
- 论文声明排序：**EAT > DPPO ≈ D2SAC > PPO ≈ SAC > TD3 > DDPG**（`EATlogs/Reading/复现.docx`）

**「最高回报」这一列的口径**：它是三次运行**各自最高点**的均值（`by_algorithm.csv` 的 `best_mean`），不是单次最高点。
逐次最高点（`runs_index.csv` 的 `test_reward_best`）：

| 算法 | 三次运行各自的最高点 | 单次最高 |
|---|---|---|
| DDPG | 187.84 / 187.00 / 187.09 | 187.84 |
| TD3 | 192.06 / 188.30 / 193.05 | 193.05 |
| SAC | 189.79 / 190.35 / 189.61 | 190.35 |
| PPO | 193.41 / 192.97 / 192.68 | 193.41 |
| D2SAC | 166.68 / 167.44 / 166.22 | 167.44 |
| DPPO | 193.11 / 192.07 / 193.84 | **193.84**（全场最高单点） |
| EAT | 193.54 / 189.41 / 172.83 | 193.54 |

终点回报的**单次最高**是 TD3 的 **188.59**（见上表逐次终点值），不是任何算法的均值。

- EAT 只有 3 次训练中的 1 次掉到 158.68；另外两次（187.17 / 187.38）平均 **187.28**，高于任何算法的三种子均值（最高的 DPPO 是 185.62）
- EAT 学到的 η 终点：0.0940 / 0.0899 / 0.0941（`data/fig6-main/eat/*/curves.csv` 的 `train/ddim_eta`）

## 2. 消融 `ablation`（B 组）

| 算法 | 目录 | 终点回报 | 均值 | 参数量（总 / 在线） | 说明（由 checkpoint 结构判定） |
|---|---|---|---|---|---|
| SAC-CA | `sac_ca` | 185.50（1 次） | 185.5 | 419,239 / 251,333 | 高斯 SAC + 交叉注意力，未跑满 3 次 |
| D2SAC-SA | `d2sac_sa` | 148.75（1 次） | 148.7 | 709,559 / 397,941 | DDPM + 自注意力 Q，未跑满 3 次 |
| EAT-CA | `eat_ca` | 157.41 / 165.31 / 164.47 / 163.83（4 次） | **162.8 ± 3.6** | 775,608 / 463,990 | DDIM + η，但**没有交叉注意力**（actor 直接吃原始观测） |

对照：主对比中的 EAT（`fig6-main/eat`，709,560）带交叉注意力；去掉交叉注意力后参数量反而涨到 775,608，回报掉到 162.8。

## 3. 学习率 `learning-rate`（3e-4 支路）

| 算法 | 目录 | 终点回报（各次） | 均值±标准差 | 最高 | 参数量（总 / 在线） | 与主扫描同名臂是否同一网络 |
|---|---|---|---|---|---|---|
| D2SAC @3e-4 | `d2sac3e` | 162.00 / 158.89 / 155.62 | 158.8 ± 3.2 | 168.1 | 422,135 / 254,229 | 是（主扫描同样 422,135） |
| EAT @3e-4 | `eat3e` | 160.55 / 163.13 / 158.73 / 157.29 | 159.9 ± 2.5 | 166.1 | **841,144 / 463,734** | **否**（主扫描 EAT 是 709,560） |
| TD3 @3e-4 | `td33e` | 176.11 / 176.64 / 182.02 | 178.3 ± 3.3 | 188.7 | 524,810 / 262,405 | 是（主扫描同样 524,810） |

**这张表最容易误读的一行是 EAT**：3e-4 支路的 EAT 参数量是 841,144，比主扫描的 709,560 多 131,584。也就是说"把学习率从 1e-4 抬到 3e-4，EAT 从 177.7 掉到 159.9"这句话里，**变的不只是学习率**。D2SAC 与 TD3 两条支路的参数量与主扫描一致，那两条才可以归因于学习率。

## 4. fallback 重跑 `fallback-rerun`（3 次/算法）

| 算法 | 终点均值±标准差 | 最高回报 | 训练步数 | 参数量（总 / 在线） |
|---|---|---|---|---|
| DDPG | 177.2 ± 5.8 | 185.5 | 336,000 | 349,704 / 174,852 |
| TD3 | 185.1 ± 5.3 | 192.2 | 336,000 | 524,810 / 262,405 |
| SAC | 184.0 ± 2.4 | 191.1 | 336,000 | 437,511 / 262,405 |
| PPO | **188.7 ± 2.0** | 191.6 | 831,355–841,400 | 611,336 / 611,336 |
| D2SAC | 180.7 ± 3.4 | 187.0 | 336,000 | 713,431 / 401,813 |
| DPPO | 187.6 ± 2.8 | 190.9 | 784,803–810,293 | 519,238 / 519,238 |
| EAT | 182.9 ± 2.1 | 187.9 | 336,000 | 841,144 / 463,734 |

注意：fallback 组的 EAT/D2SAC 与 `fig6-main` 的同名算法**参数量不同**（841,144 vs 709,560；713,431 vs 422,135），即两次扫描用的不是同一版网络。

## 5. 早期探索 `trials`（每个 run 只有 67,200 步）

| 算法 | 终点均值±标准差 | 次数 |
|---|---|---|
| D2SAC | 167.7 ± 20.9 | 2 |
| EAT | 166.2 ± 13.1 | 3 |
| tryeat（EAT 早期草稿） | 159.5 ± 6.4 | 3 |

## 6. 节点数与任务密度 `scale-density-z`

`quality_ratio` 决定奖励权重，**不同质量比之间的回报不可直接比较**。

| 配置 | 算法 | 终点回报（各次） | 备注 |
|---|---|---|---|
| 32_0.333 | SAC | 23.99 / 23.92 / 24.89 | 质量权重极低，回报量级不同 |
| 32_0.444 | SAC | 21.84 / −8.53 / 28.47 | 同上；PPO 40.63（43k–167k 步） |
| 32_0.556 | EAT | 155.32 / 161.65 / 154.85 / 0.00 / 144.82 / 186.32 / 162.69 | 7 次；0.00 是中途中断的 run |
| 32_0.600 | SAC | 177.47 / 166.88 / 101.79 | 3 次，336k 步 |
| 32_0.600 | EAT | 181.25 | 只跑到 32k 步 |

## 7. 早期评估指标（64 节点、λ=0.3、quality_ratio=0.6、单个种子 42）

来源：`EATlogs/Reading/AlgoAndData.xlsx`，由作者手工抄录，**不是本仓库重算**。

| 方法 | 平均响应时间 (s) | 平均质量 | 方法 | 平均响应时间 (s) | 平均质量 |
|---|---|---|---|---|---|
| SAC | 28.72 | 0.556 | PPO | 35.32 | 0.540 |
| FIFO | 35.54 | 0.707 | TD3 | 2.44 | 0.033 / 0.110 |
| Greedy | 39.61 | 0.752 | DDPG | 1.28 / 20.26 | 0.033 / 0.400 |
| Random | 25.02 | 0.403 | SAC-CA | 40.35 | 0.739 |
| D2SAC | 29.45 / 32.94 | 0.638 / 0.670 | SAC-DICA | 35.43 | 0.708 |

DDPG 1.28 s / 质量 0.033、TD3 2.44 s / 质量 0.033 是**策略崩塌**的特征：只挑最短任务、永远用最低推理步数。

**质量列可以反解成推理步数**（算术 + 推断）：`_compute_reward` 里 `quality = tanh(steps / reward_q_scale)`，`reward_q_scale = 30`（`env/env.py#L503`、`env/config.py#L88`），`max_step = 30`（`#L19`），所以 `steps = 30·atanh(quality)`：

| 方法 | 质量 | 反解步数 |
|---|---|---|
| Greedy | 0.752 | ≈ 29.6 |
| FIFO | 0.707 | ≈ 26.5 |
| SAC-DICA（EAT 旧名） | 0.708 | ≈ 26.5 |
| SAC-CA | 0.739 | ≈ 28.4 |
| D2SAC | 0.638 / 0.670 | ≈ 22.6 / 24.3 |
| SAC | 0.556 | ≈ 18.8 |
| PPO | 0.540 | ≈ 18.0 |
| Random | 0.403 | ≈ 12.8 |
| DDPG / TD3 | 0.033 | ≈ 1.0 |

"质量即步数"这件事说明：FACTS §7 这张表衡量的是**策略愿意花多少推理步**，不是图像质量的客观评分。

## 8. 超参数与规模（来自代码与作者笔记）

| 项 | 值 | 出处 |
|---|---|---|
| 节点数 | 32 / 64（默认 32） | `code/eat-train/scripts/config.py` `GlobalConfig.node_num` |
| 到达率 λ | 0.3 | 同上 `poisson_lambda` |
| 可见任务数 | 8 | `max_visible_tasks` |
| 网络宽度 | actor/critic 256×256；UNet 256×256 | `TrainConfig` |
| 状态嵌入维度 | 8 | `state_embed_dim` |
| 训练预算 | 500 epochs（图 6 主要配置） | 作者笔记 `EATlogs/Reading/AAAresult.docx` |
| 学习率 | EAT 1e-4（主）｜3e-4（DatA 对照） | 同上 |
| 扩散步数 | 4 / 5 / 10 各版本不同 | `trainer/*/trainer.py` 的 `diffusion_timesteps` |
| 奖励权重 | q=6, t1=3 → quality_ratio 0.600；q=10, t1=2 → 0.769 | `EATlogs/Reading/coding.docx` |
| 硬件 | 单机单卡（作者笔记中出现 RTX 4080 SUPER / 笔记本） | 运行日志与笔记 |

---

## 9. 代码层面的两个硬事实 / Two hard facts found in the code

这两条不是推断，是逐行核对过的源码行为，被第 06、08、09 章引用。

**(1) D2SAC 一族的策略把熵项置零，EAT 一族没有。**

| 文件 / 位置 | 代码 | 后果 |
|---|---|---|
| `code/eat-train/policy/sac_diffusion.py#L159-L161` | `entropy = self.actor.action_entropy(...)` 之后紧跟 `entropy = 0`，然后 `actor_loss = (-q_min - self._alpha * entropy).mean()` | `SACDDPMPolicy` 的 actor 损失退化成纯 `-Q_min` |
| 同文件 `#L167-L173` | 自动温度用 `log_prob_surrogate = (-entropy).detach() + target_entropy` | 因为 entropy 恒为 0，surrogate 变成常数，`alpha` 只会被往 0 推 |
| 同文件 `#L393-L394` | `entropy = self.actor.action_entropy(...)` 后直接用于 `actor_loss = (-q_min - self._alpha * entropy).mean()` | `SACDDIMPolicy`（EAT）保留了熵项 |

**(2) 快照里的 `trainer/dcasac` 按现在的代码会直接报 `TypeError`。**

`code/eat-train/trainer/dcasac/trainer.py#L116` 向 `RLDiffusionActor` 传 `schedule_timesteps=diffusion_timesteps`，但 `code/eat-train/model/diffusion/rl_diffusion.py#L28-L39` 的签名里只有 `n_timesteps`。推断：快照代码与真正训练出 checkpoint 的版本不是同一份；这也是本仓库把"运行目录 ↔ trainer"的对应标注为推断的依据之一。

**(3) 日志里到底有哪些温度/随机性轨迹（逐算法核实）。**

| 算法 | 相关 tag | 覆盖 |
|---|---|---|
| SAC | `update/alpha`、`update/loss/alpha` | 只有 33 个点，覆盖约前 33,000 步（预算 336,000）；值从 0.7575 降到 0.0126 |
| D2SAC | 无 | 该策略把 entropy 置零，也没有 alpha 日志 |
| EAT | `train/ddim_eta` | 500 个点，覆盖整段预算；0.0505（第 1 步）→ 0.0940（第 500 步） |
| PPO | `update/loss/{ent,clip,vf}` | 有熵/裁剪/价值损失，无温度 |

"温度轨迹未记录"这句话**只对 D2SAC 成立**；SAC 有但只覆盖前 1/10 的预算，EAT 用 η 代替温度且全程有记录。

**(4) 审计用的代码事实（[14-audit](../14-audit/README.md) 的 A1–A7 出处）。**

| 编号 | 事实 | 位置 |
|---|---|---|
| A1 | 全树 0 处 `manual_seed` / `np.random.seed` / `random.seed`；只有环境种子（默认 42） | `env/env.py#L42-L46`、`#L127-L129`；`scripts/config.py#L51`；七个 `*_train.py` 传 `cfg_train.seed` |
| A2 | `critic1` 用 `critic_hidden_sizes`，`critic2` 用 `tail = critic_hidden_sizes[1:]`；主扫描 3 个 checkpoint 对称（155,809 / 155,809），841,144 版的 **17 个** checkpoint 不对称（221,601 / 155,809，差 65,792 = 一层） | `trainer/dicasac/trainer.py#L132-L134`、`#L148`、`#L169`；权重见 `tools/ckpt_inventory.py` |
| A3 | `SACDDPMPolicy` 把 `entropy` 硬置零；`SACDDIMPolicy` 保留 | `policy/sac_diffusion.py#L159-L161`、`#L167-L173`、`#L393-L394` |
| A4 | 七个 trainer 的 `ActorProb(unbounded=True)`（无 tanh 压缩） | `ppo#L52`、`appo#L63`、`cappo#L63`、`sac#L70`、`asac#L72`、`casac#L66`、`dppo#L96` |
| A5 | 上述 trainer 均未传 `activation=` → tianshou 默认 ReLU；注意力/扩散模块用 Mish | `cross_attention.py#L46`、`self_attention.py#L41`、`trainer/dicasac/trainer.py#L44` |
| A6 | 传 `schedule_timesteps=` 但签名只接受 `n_timesteps`；`eval_dcasac.py` 实为 EAT 的评估脚本 | `trainer/dcasac/trainer.py#L116`、`rl_diffusion.py#L34`、`eval_dcasac.py#L11` |
| A7 | 七个训练脚本写死 `node_num=64`、`lr = 3e-4` | 各 `*_train.py`（例：`sac_train.py#L36`、`#L46`） |

证据等级：A1 / A2 / A6 为 **E3**（另有物证：DPPO 两次 step-0 均为 148.589691；841,144 版 critic1 = 221,601 对 critic2 = 155,809，差恰为一层 65,792；按现有代码实例化即 `TypeError`）；A3 / A4 / A5 / A7 为 **E2**（代码可核，影响未量化）。
