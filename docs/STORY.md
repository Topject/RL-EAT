# 这段复现到底发生了什么 / What actually happened during the reproduction

这是一份按时间顺序的记录，材料来自 88 次运行的目录时间戳（`data/runs_index.csv`）和作者当时的笔记（`EATlogs/Reading/AAAresult.docx`、`coding.docx`、`AlgoAndData.xlsx`，未随仓库分发）。
它记的是先跑歪、发现哪里歪、再修的过程，没有照着论文的顺序重走一遍。

A chronological record built from the timestamps of all 88 runs plus the author's own notes. It is not the story of the paper; it is the story of the reproduction, including the parts that went sideways.

---

## 第一幕｜先把尺子搞对（08-04 ~ 08-10）

**08-04 12:28** 第一个能跑通的算法是 SAC，只有 336,000 步，终点回报 **177.47**。
同一天 18:52 跑 EAT，32,000 步就到 **181.24**，看起来"扩散策略收敛更快"，但这是假象：两次运行的 `quality_ratio` 不同，奖励函数根本不是同一把尺子。

**08-05 05:37** 第三次 SAC 只有 **101.79**。同一个算法、同一份配置，三次运行差 75 分。这是第一次真正被提醒：**单次运行说明不了任何事**。

**08-10** 一整天在做节点数与任务密度的扫描：32 节点 × quality_ratio 0.333 / 0.444。SAC 的终点回报分别是 **23.99 / 23.92 / 24.89**（0.333）和 **21.84 / −8.53 / 28.47**（0.444），量级只有 20 出头。这跟学得好坏关系不大。`quality_ratio` 从 0.600 降到 0.444 或 0.333 时，奖励里的质量项权重被砍掉，同样的决策只能拿到 20 分。

**这一幕的结论**：`quality_ratio = w_q/(w_q+w_t1+w_t2)` 是奖励函数的一部分，跨它的回报不能比。后来第 11 章就是专门写这件事的。

---

## 第二幕｜主扫描：一步只加一个机制（08-11 ~ 08-15）

从 **08-11 17:42** 到 **08-15 23:36**，主扫描把七个算法每个跑三次，顺序是被 GPU 排队而不是被设计决定的：

| 时间 | 算法 | 备注 |
|---|---|---|
| 08-11 17:42 / 08-13 22:05、22:24 | PPO | 每次要到 **848k–854k** 环境步才走完同样的 500 epoch |
| 08-11 18:51 / 08-13 20:27、21:17 | SAC | 336k 步 |
| 08-12 02:10 / 08-14 14:03、15:30 | DDPG | 336k 步 |
| 08-14 15:32、16:17、16:33 | DPPO | **804k–817k** 步 |
| 08-15 06:50、09:33、17:49 | D2SAC | 336k 步，三次都在 156–164 |
| 08-15 18:19、23:35、23:36 | TD3 | 336k 步 |

**第一个意外**：D2SAC 三次全部停在 **156.04 / 158.62 / 163.82**。扩散动作头本来是用来解决多峰分布的，结果它成了全场最低，比不带熵的 DDPG 还低 20 分。
**第二个意外**：PPO 和 DPPO 看起来最好（185.2 / 185.6），但它们跑的步数是其他算法的 2.4–2.5 倍。这两件事必须在同一段话里说，否则就变成"on-policy 赢了"。

First surprise: all three D2SAC runs stall at 156–164, the lowest arm in the sweep, the diffusion head, added to model multi-modal actions, ends up worse than entropy-free DDPG.
Second surprise: PPO and DPPO look best, but they consumed 2.4–2.5× the environment steps of every other arm.

**中间的插曲**（08-11 20:01 ~ 08-14 23:33）：
- **3e-4 支路**：`d2sac3e` 161.99 / 158.89 / 155.61、`eat3e` 160.54 / 163.12 / 158.73 / 157.28、`td33e` 176.10 / 176.64 / 182.01。把学习率从 1e-4 抬到 3e-4 没有救回扩散臂，EAT 反而更差。
- **消融臂**：`eat_ca` 四次（157.40 / 165.31 / 164.46 / 163.82）、`sac_ca` 一次（185.49）、`d2sac_sa` 一次（148.74）。去掉交叉注意力的 EAT 掉到 ~163，而它的参数量反而更大（775,608 vs 709,560）。

---

## 第三幕｜发现环境里有个坑，然后重跑一遍（08-16 ~ 08-18）

**08-16 23:50 ~ 08-17 13:54** 出现了一批 67,200 步的短跑（`trials/`），EAT 只有 158–181。这时作者在环境里改了一处 **task-index fallback**：当 actor 选中的任务下标越界时，回退到队首任务，而不是空转一步（笔记里保留了两版代码的对照）。
改动之后跑了第二批完整扫描（`fallback-rerun`，08-17 18:14 ~ 08-18 08:42）：

| 算法 | fig6-main（修正前） | fallback-rerun（修正后） |
|---|---|---|
| DDPG | 180.4 ± 8.6 | 177.2 ± 5.8 |
| TD3 | 181.3 ± 8.3 | 185.1 ± 5.3 |
| SAC | 184.3 ± 2.5 | 184.0 ± 2.4 |
| PPO | 185.2 ± 2.0 | **188.7 ± 2.0** |
| D2SAC | 159.5 ± 4.0 | **180.7 ± 3.4** |
| DPPO | 185.6 ± 1.0 | 187.6 ± 2.8 |
| EAT | 177.7 ± 16.5 | **182.9 ± 2.1** |

**最值得注意的是 D2SAC：从 159.5 跳到 180.7。** 这说明差距的大头落在选中的任务常常不可执行上，fallback 让那些动作变成了有意义的决策。

**但必须同时说清楚**：fallback 这一批里 EAT 的参数量是 **841,144**（fig6-main 是 709,560），D2SAC 是 **713,431**（fig6-main 是 422,135）。也就是说，**两次扫描用的不是同一批网络**，两列数字不能当"A/B 测试"来读。这是整个复现里最容易被误读的一处。

The fallback change moves D2SAC from 159.5 to 180.7, which says a large part of the gap was "the chosen task was often not executable" rather than the algorithm. But the fallback sweep also used different networks (EAT 841,144 vs 709,560 parameters; D2SAC 713,431 vs 422,135), so the two columns are not an A/B test.

---

## 第四幕｜EAT 的七个短跑与三次正式运行（08-18 ~ 08-19）

**08-18 13:24 ~ 23:14**，EAT 在 32_0.556 上被反复试了七次：153k、78k、48k、67k、106k、**416k**、300k 步。前六次里有五次在 144–162 之间徘徊，只有跑满 416k 的那次到 **186.31**。这一次就是线索：**EAT 需要更长的预算才显出形状**。

**08-19 10:42、14:43、20:25**，三次正式运行，都是 416,000 步：

| 运行 | 终点回报 | 学到的 η 终点 |
|---|---|---|
| 08-19 10:42 | **187.17** | 0.0940 |
| 08-19 14:43 | **187.38** | 0.0899 |
| 08-19 20:25 | 158.67 | 0.0941 |

前两次的平均 **187.28**，高于任何算法的三种子均值（最高的 DPPO 是 185.62）；第三次掉了 28 分，把均值拖到 177.7。
于是这一章的结论变成了一个必须两种读法都给的句子：

> 在两个种子上，EAT 的平均回报（187.28）高于任何算法；在三个种子上，EAT 的均值排第六。

> Averaged over its two good seeds EAT reaches 187.28, above every algorithm's 3-seed mean; averaged over all three seeds it ranks sixth.

---

## 第五幕｜作者笔记里还没做完的事（未验证）

作者在 `AAAresult.docx` 里记录了针对 EAT/D2SAC 的下一步改动，**这些改动没有对应的运行数据入库**，所以它们只能记为"当时的计划"，不能当作结论：

| 改动 | 目的 |
|---|---|
| 给 η 加 `eta_max` 截断（建议 0.5，再试 0.3 / 0.2） | 防止 η 一路往 1 跑，探索失控 |
| 把 `alpha` 从固定值改成自动温度（传 `alpha_tuple`） | 让探索强度自适应，而不是手猜 0.05 |
| `step_per_collect` 208 + `batch_size` 4096 | 提高每环境步的更新样本量（原 128 → 约 1024） |
| 两个 critic 结构对称（`critic1` 也用 `tail`） | 当时 critic1/critic2 不对称，target Q 不稳 |
| D2SAC 的 `sigma` 加上限 `clamp(max=0.3)` | 防止熵项把动作噪声越推越大 |
| 检查 D2SAC 是否把噪声加了两遍（`_p_mean_variance` + `_p_sample`） | 采样语义应当只在输出时加一次噪声 |

这些正是第 [10](../10-tuning/README.md) 章的"下一步"，也是本仓库最想补充实验的地方。

These are exactly the "next steps" listed in chapter 10, and the experiments this repository would most like to add.

---

## 一句话版本 / The one-paragraph version

先把尺子校准（quality_ratio 决定奖励量级），再一个一个加机制（DDPG → TD3 → SAC → PPO → D2SAC → DPPO → EAT），途中发现环境里的 task-index fallback 会吞掉大量决策，修完之后扩散臂集体上移，最后 EAT 需要 416k 步才显形。它的两个种子是全表最高，第三个种子崩了，于是论文的名次没有复现出来，但每一个机制的作用方向都能在数据里找到对应。

Calibrate the ruler first, then add one mechanism at a time. Along the way, an environment-level task-index fallback turns out to swallow a large share of decisions; fixing it lifts every diffusion arm. EAT needs 416k steps to show its shape: two of its seeds top the whole table, the third collapses, so the paper's ranking does not reproduce, but the direction of every mechanism can be found in the data.
