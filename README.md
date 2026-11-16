# EAT 复现手记：从四个基线到注意力引导的扩散调度
# EAT Reproduction Notes: from four baselines to an attention-guided diffusion scheduler

> 论文：**EAT: QoS-Aware Edge-Collaborative AIGC Task Scheduling via Attention-Guided Diffusion Reinforcement Learning**，投稿 IEEE TMC。
> 仓库内容：把这篇论文的对比实验按算法拆成 15 个顺序项目。每章带训练曲线、结果、简要分析和可以下载的训练数据。

> Paper: **EAT — QoS-Aware Edge-Collaborative AIGC Task Scheduling via Attention-Guided Diffusion Reinforcement Learning** (submitted to IEEE TMC).
> This repository splits the paper's comparison into 15 ordered mini-projects. Each chapter ships its curves, results, a short analysis and the data behind it.

![EAT 系统示意 / EAT system overview](00-problem/demo.png)

---

## 1. 一句话概括

七个算法在同一个三集群边缘调度环境里训练，用同一套评估面板打分。
EAT 有两个种子跑到 187.17 和 187.38，这两个种子的平均值是 187.28，高于任何算法的三种子均值。第三个种子掉到 158.68，于是三种子均值只有 177.7，在七个算法里排第六。
论文写的排序是 `EAT > DPPO ≈ D2SAC > PPO ≈ SAC > TD3 > DDPG`，这个排序我没能复现出来。
这篇手记想说明的是为什么没复现，不是我又赢了什么。差别写在后面 15 章里，每一章都能单独核对。

Seven algorithms, one three-cluster edge-scheduling environment, one shared evaluation panel.
Two of EAT's seeds reach 187.17 and 187.38 and average 187.28, above every algorithm's three-seed mean. The third seed falls to 158.68, so the three-seed mean is 177.7, sixth of seven.
The paper's ordering, `EAT > DPPO ≈ D2SAC > PPO ≈ SAC > TD3 > DDPG`, did not reproduce here.
These notes try to explain why it did not. Each of the 15 chapters can be checked on its own.

---

## 2. 主结果

![Figure 6 复现 / Figure 6 reproduction](figures/curves/fig6_convergence.png)

| 算法 / algorithm | 角色 / role | 终点验证回报 (3 种子) | 最高 | 环境步数 | 参数量（总/在线） |
|---|---|---:|---:|---:|---|
| DDPG | 经典 off-policy 基线 | 180.4 ± 8.6 | 187.3 | 336k | 349,704 / 174,852 |
| TD3 | DDPG 加双 critic | 181.3 ± 8.3 | 191.1 | 336k | 524,810 / 262,405 |
| SAC | 最大熵 off-policy | 184.3 ± 2.5 | 189.9 | 336k | 437,511 / 262,405 |
| PPO | on-policy 基线 | 185.2 ± 2.0 | 193.0 | 849k | 611,336 / 611,336 |
| D2SAC | 扩散动作头的 SAC | 159.5 ± 4.0 | 166.8 | 336k | 422,135 / 254,229 |
| DPPO | 扩散特征加 PPO | 185.6 ± 1.0 | 193.0 | 810k | 519,238 / 519,238 |
| **EAT** | **论文方法** | **177.7 ± 16.5** | 185.3 | 416k | 709,560 / 397,942 |

数字来自 `data/` 里重新解析的 TensorBoard 曲线，逐条可查。出处见 [docs/FACTS.md](docs/FACTS.md)。

读这张表要带三个限定。

1. **步数不齐。** PPO 与 DPPO 是 on-policy，要跑到 81 万到 85 万环境步才用完同样的 epoch 预算，其余算法停在 336k。PPO 的 185.2 大概不能算同预算下的胜利。
2. **EAT 的方差最大。** 两个好种子的平均是 187.28，第三个种子只有 158.68，种子标准差 16.5。次大的 DDPG 也只有 8.6。
3. **最高点不在 EAT 身上。** 单次终点最高是 TD3 的 188.59，单次过程最高是 DPPO 的 193.84。EAT 的形态是两个种子强、一个种子崩。

1. **Budgets differ.** PPO and DPPO are on-policy and needed 810k to 854k environment steps for the same epoch budget; everything else stopped at 336k.
2. **EAT has the widest spread.** Its two good seeds average 187.28, the third falls to 158.68, seed std 16.5. The next widest is DDPG at 8.6.
3. **EAT does not hold the peak.** The highest single final return is TD3 at 188.59, the highest single best-of-run point is DPPO at 193.84.

---

## 3. 故事线

四部分，15 个顺序项目。每一章都是一个能独立打开的小项目。

### 第一部分：先把舞台搭好

| 章 | 内容 | 这一章回答的问题 |
|---|---|---|
| [00](00-problem/README.md) | **问题与舞台** | 三集群 AIGC 调度在决策什么，奖励怎么算，扩散策略为什么合适 |
| [01](01-heuristics/README.md) | **非学习基线** FIFO / Random / Greedy | 不学习能做到什么程度，最快的那个为什么质量最差 |

### 第二部分：一步只加一个机制

| 章 | 算法 | 相比上一章多了什么 | 结果 |
|---|---|---|---|
| [02](02-ddpg/README.md) | DDPG | 起点，确定性 actor-critic | 180.4 ± 8.6 |
| [03](03-td3/README.md) | TD3 | 双 critic 与延迟更新 | 181.3 ± 8.3 |
| [04](04-sac/README.md) | SAC | 熵正则，随机策略 | 184.3 ± 2.5 |
| [05](05-ppo/README.md) | PPO | on-policy 裁剪目标 | 185.2 ± 2.0，用了 2.5 倍步数 |
| [06](06-d2sac/README.md) | D2SAC | 动作头换成 DDPM 去噪 | 159.5 ± 4.0 |
| [07](07-dppo/README.md) | DPPO | 扩散特征加 PPO | 185.6 ± 1.0，2.4 倍步数 |
| [08](08-eat/README.md) | **EAT** | 交叉注意力、DDIM、可学习 η | 177.7 ± 16.5 |

### 第三部分：把机制拆开看

| 章 | 内容 | 一句话结论 |
|---|---|---|
| [09](09-ablation/README.md) | 消融：SAC-CA、D2SAC-SA、**EAT-CA** | 去掉交叉注意力之后，参数从 709,560 涨到 775,608，回报从 177.7 掉到 162.8。方向与论文一致，但两个消融臂只有 1 次运行 |
| [10](10-tuning/README.md) | 调参：学习率 3e-4、fallback 重跑 | 3e-4 救不了 D2SAC 和 TD3。fallback 把 EAT 的方差从 16.5 压到 2.1，可那一次网络也换了 |
| [11](11-scale-density/README.md) | 规模与密度：32 节点，四档质量比 | 跨 quality_ratio 的回报不能比。24 和 180 的差别来自奖励函数，不是学得更好 |

### 第四部分：物理侧、结论与审计

| 章 | 内容 | 一句话结论 |
|---|---|---|
| [12](12-server/README.md) | EAT-Server 与 Stable Diffusion 物理执行 | 仿真里任务等于一段固定执行时间，物理侧才是真实的 SD 推理。本仓库没有物理日志，只记录实现路径 |
| [13](13-results/README.md) | 汇总、论文排序对比、下一步 | 复现到的是机制方向，没复现到的是名次 |
| [14](14-audit/README.md) | **实现审计**：七处会改变结论的问题 | 没有全局种子、critic 不对称、熵项被置零、actor 不做 tanh、ReLU 与 Mish 不符、快照跑不起来、配置漂移。其中三处有代码之外的物证 |

---

## 4. 怎么读这个仓库

```
00-problem/ … 14-audit/       15 个按顺序读的章节，中英双语
code/eat-train/               训练侧源码快照，含 14 个 trainer
code/eat-server/              推理服务侧源码快照
data/<campaign>/<algo>/<run>/ 每次训练一条：curves.csv 加 run.json
data/runs_index.csv           88 次运行一行一条
data/by_algorithm.csv         按实验与算法聚合的均值、标准差
figures/curves/               从 data/ 重绘的图，只用 Pillow
figures/original/             当时手画的图，含交给导师的 Fig6 归一化图
docs/STORY.md                 按时间顺序的复现过程
docs/FACTS.md                 所有被引用数字的出处
docs/DATA.md                  数据字段与自解析方法
docs/REPRODUCE.md             环境安装与复现步骤
docs/GLOSSARY.md              命名对照
tools/                        解析、清洗、绘图、取证、链接检查
```

按目的挑入口。

1. 只想看结论，去第 2 节和 [13-results](13-results/README.md)。
2. 想学机制，从 [00-problem](00-problem/README.md) 按编号读到 [09](09-ablation/README.md)。想知道这份实现能不能信，直接看 [14-audit](14-audit/README.md)。
3. 想核数字，看 [docs/FACTS.md](docs/FACTS.md)，数据本体在 `data/`。想看这段复现真实发生了什么，读 [docs/STORY.md](docs/STORY.md)。
4. 想问这段工作证明了什么，也就是导师或面试官视角，读本文件第 7 到 9 节。

---

## 5. 数据从哪来

原始 TensorBoard 日志有 88 次运行，合计 7.4 GB，没有入库。本仓库把它们解析成 `data/` 下的小 CSV。

- `tools/safe_tfreader.py`：不依赖 tensorboard 与 protobuf 的 TFRecord 与 Event 解析器
- `tools/extract_runs.py`：遍历日志树，每次运行产出 `curves.csv` 与 `run.json`
- `tools/curate.py`：丢掉按参数记录的 `actor_norm/*` 与 `*_sparsity/*`，EAT 那几次单个文件就接近 1 GB
- `tools/plot_curves.py`：只靠 Pillow 重绘全部图

The original TensorBoard logs (88 runs, 7.4 GB) stay in the source workspace and are not committed. Four scripts convert them into the small CSVs under `data/`.

```powershell
python tools\extract_runs.py --log-root <原EATlogs目录> --out tmp\extracted
python tools\curate.py        --in-root tmp\extracted --out data --max-points 600
python tools\plot_curves.py   --data data --out figures\curves
```

---

## 6. 复现状态

| 维度 | 状态 |
|---|---|
| 代码快照 | 可读，97 个源文件。它不是原始运行时的精确版本，驱动脚本 `scripts/fig6/train.py` 不在快照里 |
| 超参数 | 以代码默认为准，作者笔记里另有一批手改值，两者不完全一致 |
| 曲线 | 88 次运行全部重算，可以直接重绘 |
| 评估指标 | 只有手抄的 64 节点单种子表，没有可重算的原始评估日志 |
| 物理 SD 验证 | 代码在，日志与结果缺失 |
| 论文名次 | 未复现 |
| 实现能否支撑论文结论 | 大概不能。七处问题里三处有物证 |

| dimension | status |
|---|---|
| code snapshot | readable, 97 source files, but not the exact revision behind the runs; the driver `scripts/fig6/train.py` is absent |
| hyper-parameters | code defaults are authoritative; the notes record hand-set values that do not always match |
| curves | all 88 runs re-derived and re-plottable |
| evaluation metrics | only a hand-transcribed 64-node single-seed table; no replayable evaluation logs |
| physical SD validation | code present, logs and results missing |
| the paper's ranking | not reproduced |
| can this implementation support the paper | probably not; three of the seven issues have evidence beyond the source |

---

## 7. 这段工作证明了什么

它大概不能证明我跑通了代码，更像一次独立的实现审计，附带对下一步该验证什么的判断。下面这张表给看结果的人准备，每一行都能点开核对。

It is less a demo that the code runs and more an independent audit, with an opinion about what to verify next. Every row below is checkable inside this repository.

| 能力 | 这里的证据 | 去哪看 |
|---|---|---|
| 读懂论文并落到实现 | 7 个算法，3 个种子，88 次训练。自建的 TensorBoard 解析器不依赖 tensorboard、protobuf 和 matplotlib | [`13`](13-results/README.md)、`tools/safe_tfreader.py` |
| 工程化与可复算 | 数据管线从解析到清洗到绘图，checkpoint 取证不装 torch 也能读张量名与参数量，双语 15 章，链接自检脚本 | `tools/`、[`docs/DATA.md`](docs/DATA.md) |
| 怀疑与定位 | 名次不复现时没有停在辩解上，定位到七处实现层原因，按证据强弱分成 E3、E2、E1 三级 | [`14`](14-audit/README.md) |
| 边界意识 | 主动标注 D2SAC 是重建实现，本地代码已改过，物理侧没有数据，快照与训练版本不一致 | [`14`](14-audit/README.md) 第 1、2、4 节 |
| 成本与风险判断 | 下一步给的是 22 次训练的最小闭环，附停止条件 | 本文件第 9 节 |

---

## 8. 简历段落（模板，改成自己的话再用）

措辞原则：带数字，带限定，不说满。复现了论文这句话人人都会写，定位到七处实现缺陷、其中三处有物证，才有信息量。

**中文**

- **EAT 论文复现与实现审计**（独立完成）
  复现 TMC 投稿算法 EAT 与六个基线，7 算法 × 3 种子共 88 次训练。自建 TensorBoard 日志解析、清洗、绘图管线，零第三方绘图依赖，曲线可逐点复算。
- 发现论文排名不可复现，定位到七处实现层原因，包括全局种子缺失、双 critic 结构不对称、扩散策略熵项被置零、actor 越界无约束。证据强度分级为权重级、代码级与转述级。
- 设计三组受控实验（种子确定性、critic 对称化、熵项恢复，约 22 次训练），把代码级证据升级为因果结论，并给出停止条件。产物是开源双语仓库，15 章加数据管线与校验工具。

**English**

- **Reproduction and implementation audit of the EAT paper** (independent)
  Reproduced EAT and six baselines, 7 algorithms by 3 seeds, 88 training runs, and built the TensorBoard parsing, curation and plotting pipeline with no third-party plotting dependency.
- Found that the paper's ranking does not reproduce and traced it to seven implementation-level causes, among them no global RNG seed, asymmetric twin critics, a zeroed entropy term in the diffusion-SAC policy and an unbounded actor. Each claim carries an evidence level.
- Designed three controlled experiments (about 22 short runs) to turn code-level claims into causal ones, with an explicit stop criterion. Shipped as a bilingual repository with 15 chapters, a data pipeline and self-checks.

有三句话建议避开。

- 避开我发现了原作者的 bug 这类说法，改成定位到实现层原因。
- 避开没有数字的复现声明。那句话不带信息。
- 避开 D2SAC 最后一名说明论文有问题这类推论。D2SAC 是重建实现，这条必须和结论写在一起。

---

## 9. 下一步：两周最小闭环

先修 `trainer/dcasac` 的 `schedule_timesteps` 笔误，让快照能跑，见 [`14`](14-audit/README.md) 的 A6。之后做三组对照。

| # | 对照 | 产出 | 成本 |
|---|---|---|---|
| 1 | 设种子后同配置跑两次，逐位比较曲线 | 一个可复现的地基，成本最低，说服力最高 | 2 次训练 |
| 2 | critic 对称化前后各 5 种子，EAT | 少一层是否改变均值与方差 | 10 次训练 |
| 3 | D2SAC 恢复熵项与自动温度前后各 5 种子 | D2SAC 垫底有多少来自实现 | 10 次训练 |

停止条件也先写下来。若第 2 组的均值差小于 3 分，方差也没有明显变化，这条线就不再追加实验，只作为实现细节不影响主结论的负结果保留。

不建议做的两件事。第一是把这条线扩成正式投稿，算法本身没有新增贡献。第二是拿现有 PPO 实现去搭 MAPPO，那大概会把 A4、A5、A1 三处问题一起放大，见 [`14`](14-audit/README.md) 与 [docs/REPRODUCE.md](docs/REPRODUCE.md) 第 3 节。

---

## 10. 出处与致谢

- 代码来源：作者工作区的 `EAT-master/EAT-Train` 与 `EAT-master/EAT-Server`，原样拷入 `code/`，没有改动逻辑。
- 训练日志：`EAT-master/EATlogs/EAT基线对比/**` 下的 TensorBoard event 文件。
- 结果图：`figures/curves/` 是本仓库重绘的，`figures/original/` 保留原图。
- 论文与阅读笔记：`EATlogs/Reading/` 下的文档只作为事实核对来源引用，不随仓库分发。

- Code: the author's `EAT-master/EAT-Train` and `EAT-master/EAT-Server`, copied verbatim into `code/`.
- Training logs: the TensorBoard event files under `EAT-master/EATlogs/EAT基线对比/**`.
- Figures: `figures/curves/` is redrawn here, `figures/original/` keeps the originals.
- Paper and reading notes: cited as cross-check sources only, not redistributed.

---

## 11. 上传到 GitHub

本目录已经是一个独立 git 仓库，并且有初始提交，共 365 个文件、约 20 MB。

```powershell
cd EAT-Story
git remote add origin https://github.com/<你的账号>/<仓库名>.git
git branch -M main
git push -u origin main
```

几点提醒。

- 仓库里没有 `.pth` 权重、原始 `events.out.tfevents.*` 和 `runs/` 输出，见 `.gitignore`。想要完整可复现包，把这批文件作为 Release 附件，再在 `docs/DATA.md` 里补一条下载链接。
- 原始 TensorBoard 日志合计 7.4 GB。要公开就走 Release 或 Git LFS，不要直接 `git add`。
- `data/` 是最大的一块，14 MB 全是文本 CSV，压缩率高，正常 push 没有压力。

This directory is already its own git repository with one initial commit, 365 files and about 20 MB. Weights, raw event files and `runs/` output are excluded on purpose; publish those as Release assets if you want a fully reproducible bundle.

---

## 12. 一句话结论

机制的方向大致复现出来了，名次没有。交叉注意力、DDIM 与可学习 η 的组合在两个种子上拿到 187.28 的平均终点回报，高于任何算法的三种子均值。它对随机种子比较敏感，第三个种子落在 158.68。论文的排序还需要更多种子和同预算训练才能验证。

The mechanisms roughly reproduce, the ranking does not. The cross-attention plus DDIM plus learnable-eta combination averages 187.28 over its two good seeds, above every algorithm's three-seed mean. It is fairly seed-sensitive, and the third seed lands at 158.68. Verifying the paper's ordering needs more seeds and budget-matched training.

---

<sub>本仓库是学习与复现记录，不是官方实现。论文方法的所有功劳属于原作者。/ This repository is a reproduction log, not an official implementation. Credit for the method belongs to the original authors.</sub>
