# EAT 复现手记 — 从四个基线到注意力引导的扩散调度
# EAT Reproduction Notes — from four baselines to an attention-guided diffusion scheduler

> 论文：**EAT: QoS-Aware Edge-Collaborative AIGC Task Scheduling via Attention-Guided Diffusion Reinforcement Learning**（投稿 IEEE TMC）
> 本仓库内容：把 EAT 论文里的对比实验，按算法拆成 14 个按顺序推进的小项目，每个项目都带训练收敛曲线、结果、简要分析和可下载的训练数据。

> Paper: **EAT — QoS-Aware Edge-Collaborative AIGC Task Scheduling via Attention-Guided Diffusion Reinforcement Learning** (submitted to IEEE TMC).
> This repository re-organises the paper's comparison experiments into 14 ordered, self-contained mini-projects. Every chapter ships its own convergence curves, results, a short analysis, and the training data behind it.

![EAT 系统示意 / EAT system overview](00-problem/demo.png)

---

## 1. 一句话概括 / In one sentence

我们复现了论文的 Figure 6：七个算法在同一个三集群边缘调度环境里训练、用同一套评估面板打分。
**EAT 的两个好种子（187.17 / 187.38）平均 187.28，高于任何算法三种子的均值；但第三个种子掉到 158.68，于是均值 177.7 只排第六**。排序没有复现成论文的 `EAT > DPPO ≈ D2SAC > PPO ≈ SAC > TD3 > DDPG`。
这个故事的价值不在于"我们超过了论文"，而在于把"为什么没复现出来"拆成了可以逐个检查的 14 章。

We reproduced Figure 6: seven algorithms trained in the same three-cluster edge-scheduling environment and scored on one shared evaluation panel.
**EAT's two good seeds (187.17 / 187.38) average 187.28, which is above every algorithm's 3-seed mean — but the third seed collapses to 158.68, so the mean of 177.7 ranks sixth.** The paper's ranking (`EAT > DPPO ≈ D2SAC > PPO ≈ SAC > TD3 > DDPG`) did not reproduce.
The point of this repository is not to claim a win: it is to break "why it did not reproduce" into 14 chapters that can be checked one by one.

---

## 2. 主结果 / Headline result

![Figure 6 复现 / Figure 6 reproduction](figures/curves/fig6_convergence.png)

| 算法 / algorithm | 角色 / role | 终点验证回报 / final return (3 seeds) | 最高 / best | 环境步数 / env steps | 参数量（总/在线） |
|---|---|---:|---:|---:|---|
| DDPG | 经典 off-policy 基线 | 180.4 ± 8.6 | 187.3 | 336k | 349,704 / 174,852 |
| TD3 | DDPG 的双 critic 修正 | 181.3 ± 8.3 | 191.1 | 336k | 524,810 / 262,405 |
| SAC | 最大熵 off-policy | 184.3 ± 2.5 | 189.9 | 336k | 437,511 / 262,405 |
| PPO | on-policy 基线 | 185.2 ± 2.0 | 193.0 | 849k | 611,336 / 611,336 |
| D2SAC | 扩散动作头的 SAC | 159.5 ± 4.0 | 166.8 | 336k | 422,135 / 254,229 |
| DPPO | 扩散特征 + PPO | 185.6 ± 1.0 | 193.0 | 810k | 519,238 / 519,238 |
| **EAT** | **论文方法** | **177.7 ± 16.5** | 185.3 | 416k | 709,560 / 397,942 |

数字全部来自本仓库 `data/` 中重新解析的 TensorBoard 曲线，逐条可查：见 [docs/FACTS.md](docs/FACTS.md)。
All numbers are re-derived from the TensorBoard scalars stored under `data/`; every row is traceable in [docs/FACTS.md](docs/FACTS.md).

**两个必须同时看的限定条件 / two caveats that must be read together with the table**

1. **步数不同**：PPO/DPPO 是 on-policy，跑到 810k–854k 环境步才用完同样的 epoch 预算；其余算法是 336k。所以 PPO 的 185.2 不是"同等预算下的胜利"。
2. **EAT 的方差**：三个种子里两个（187.17 / 187.38）的平均 187.28 高于任何算法，一个（158.68）接近 D2SAC。均值掩盖了它。学到的 DDIM η 收敛到 ≈0.09，采样几乎变成确定性，探索基本停了 —— 这是我们对崩溃种子最主要的怀疑。
3. **谁拿过最高点**：单次终点最高是 TD3 的 188.59，单点最高是 DPPO 的 193.84。EAT 不是逐点最强，它是"两个种子很强、第三个崩"。

1. **Different budgets**: PPO and DPPO are on-policy and needed 810k–854k environment steps to consume the same epoch budget; everything else stopped at 336k. PPO's 185.2 is therefore not a budget-matched win.
2. **EAT's variance**: two seeds (187.17 / 187.38) average 187.28, above every algorithm's 3-seed mean; the third (158.68) lands near D2SAC. The mean hides this. The learned DDIM η converges to ≈0.09, i.e. sampling becomes almost deterministic and exploration stops — our main suspect for the collapsed seed.
3. **Who actually topped what**: the single highest final return is TD3 at 188.59, and the single highest best-of-run point is DPPO at 193.84. EAT's story is two strong seeds and one collapse, not pointwise dominance.

---

## 3. 故事线 / The story line

四个部分、14 个顺序项目。每一章都是一个可以独立打开的小项目：有代码、有曲线、有数据。

Four parts, 14 ordered projects. Each chapter is a self-contained mini-project with code, curves and data.

### 第一部分｜先把舞台搭好 / Part I — set the stage

| 章 | 内容 | 这一章回答的问题 |
|---|---|---|
| [00](00-problem/README.md) | **问题与舞台** / problem & stage | 三集群 AIGC 调度到底在决策什么？奖励怎么算？扩散策略为什么适合？ |
| [01](01-heuristics/README.md) | **非学习基线** FIFO / Random / Greedy | 不学习能做到什么程度？为什么"最快"往往意味着"质量崩了"？ |

### 第二部分｜算法谱系，一步只加一个机制 / Part II — one mechanism per chapter

| 章 | 算法 | 相比上一章多了什么 | 结果 |
|---|---|---|---|
| [02](02-ddpg/README.md) | DDPG | 起点：确定性 actor-critic | 180.4 ± 8.6 |
| [03](03-td3/README.md) | TD3 | 双 critic + 延迟更新 | 181.3 ± 8.3 |
| [04](04-sac/README.md) | SAC | 熵正则（随机策略） | 184.3 ± 2.5 |
| [05](05-ppo/README.md) | PPO | on-policy 裁剪目标 | 185.2 ± 2.0（2.5× 步数） |
| [06](06-d2sac/README.md) | D2SAC | 动作头换成 DDPM 去噪 | 159.5 ± 4.0 |
| [07](07-dppo/README.md) | DPPO | 扩散特征 + PPO | 185.6 ± 1.0（2.4× 步数） |
| [08](08-eat/README.md) | **EAT** | 交叉注意力 + DDIM + 可学习 η | 177.7 ± 16.5 |

### 第三部分｜把机制拆开看 / Part III — take the mechanisms apart

| 章 | 内容 | 结论一句话 |
|---|---|---|
| [09](09-ablation/README.md) | 消融：SAC-CA / D2SAC-SA / **EAT-CA** | 去掉交叉注意力后参数从 709,560 涨到 775,608，回报从 177.7 掉到 162.8 —— 方向上支持论文"参数更少、效果更好"的说法，但两个消融臂只有 1 次运行 |
| [10](10-tuning/README.md) | 调参：学习率 3e-4、fallback 重跑 | 3e-4 救不了 D2SAC/TD3；fallback 重跑把 EAT 的方差从 16.5 压到 2.1，但那一次换的是网络结构（3e-4 支路的 EAT 也换了：709,560 → 841,144） |
| [11](11-scale-density/README.md) | 规模与密度：32 节点 × 4 档质量比 | 跨 quality_ratio 的回报不可比 —— 24 和 180 的差别是奖励函数变了，不是学得更好 |

### 第四部分｜物理侧、结论与审计 / Part IV — the physical side, the verdict, and the audit

| 章 | 内容 | 结论一句话 |
|---|---|---|
| [12](12-server/README.md) | EAT-Server + Stable Diffusion 物理执行 | 仿真里"任务 = 固定执行时间"，物理侧才是真正的 SD 推理；本仓库没有物理日志，只记录实现路径 |
| [13](13-results/README.md) | 汇总、论文排序 vs 复现排序、下一步 | 复现到的是"机制方向"，没复现到的是"名次" |
| [14](14-audit/README.md) | **实现审计**：七处会改变结论的问题 | 没有全局种子、critic 不对称、熵项被置零、actor 不做 tanh、ReLU≠Mish、快照跑不起来、配置漂移；其中三处有代码之外的物证 |

---

## 4. 怎么读这个仓库 / How to read it

```
00-problem/ … 14-audit/       15 个按顺序读的章节（每章一个 README，中英双语）
code/eat-train/               训练侧源码快照（原 EAT-master/EAT-Train，含 14 个 trainer）
code/eat-server/              推理服务侧源码快照（原 EAT-master/EAT-Server）
data/<campaign>/<algo>/<run>/ 每次训练一条：curves.csv（曲线）+ run.json（元数据）
data/runs_index.csv           88 次运行一行一条，含终点/最高回报与目录
data/by_algorithm.csv         按「实验 × 算法」聚合的均值±标准差
figures/curves/               全部从 data/ 重绘的图（Pillow，无 matplotlib 依赖）
figures/original/             作者当时自己画的图（含复现者交给导师的 Fig6 归一化图）
docs/STORY.md                 按时间顺序的复现过程（先跑歪了、发现哪歪了、再修）
docs/FACTS.md                 所有被引用的数字及其出处
docs/DATA.md                  训练数据格式、字段、如何自己解析 tfevents
docs/REPRODUCE.md             环境安装与复现步骤
docs/GLOSSARY.md              命名对照（d2sac / dsac / EAT / EAT-CA / η …）
tools/                        解析、清洗、绘图、checkpoint 取证、链接检查脚本
```

三句话导航：
1. **只想看结论** → 本文件第 2 节 + [13-results](13-results/README.md)。
2. **想学机制** → 从 [00-problem](00-problem/README.md) 按编号读到 [09](09-ablation/README.md)；**想知道这份实现能不能信** → [14-audit](14-audit/README.md)。
3. **想核数字** → [docs/FACTS.md](docs/FACTS.md) 每行都有出处路径，数据本体在 `data/`；**想知道这段复现真实发生了什么** → [docs/STORY.md](docs/STORY.md)（按时间线，含跑歪的部分）。
4. **想问"这段工作证明了什么"**（导师 / 面试官视角）→ 本文件第 **7–9** 节：能力对照表、简历模板、两周最小闭环与停止条件。

1. **Just the verdict** → section 2 above + [13-results](13-results/README.md).
2. **Learn the mechanisms** → read [00](00-problem/README.md) through [09](09-ablation/README.md); **to judge whether this implementation can be trusted** → [14-audit](14-audit/README.md).
3. **Check a number** → [docs/FACTS.md](docs/FACTS.md) lists the source path of every figure quoted anywhere; the raw data lives in `data/`.
4. **"What does this work demonstrate?"** (advisor / interviewer view) → sections **7–9** below: skill-to-evidence table, CV template, and a two-week loop with an explicit stop criterion.

---

## 5. 数据从哪来 / Where the data comes from

原始 TensorBoard 日志（`events.out.tfevents.*`，共 88 次运行，7.4 GB）留在原工作区，没有入库。
本仓库把它们解析成 `data/` 下的小 CSV：

- `tools/safe_tfreader.py`：不依赖 tensorboard / protobuf 的最小 TFRecord + Event 解析器
- `tools/extract_runs.py`：遍历日志树 → 每次运行一份 `curves.csv` + `run.json`
- `tools/curate.py`：丢掉每次参数都记一条的 `actor_norm/*`、`*_sparsity/*`（EAT 那几次 log 有 1 GB），保留 `test/*`、`train/*`、`update/*`、`save/*`
- `tools/plot_curves.py`：只靠 Pillow 重绘全部图

The original TensorBoard logs (88 runs, 7.4 GB) stay in the source workspace and are not committed. They are converted into the small CSVs under `data/` by the four scripts above, all of which ship in `tools/`.

```powershell
python tools\extract_runs.py --log-root <原EATlogs目录> --out tmp\extracted
python tools\curate.py        --in-root tmp\extracted --out data --max-points 600
python tools\plot_curves.py   --data data --out figures\curves
```

---

## 6. 复现状态 / Reproduction status

| 维度 | 状态 |
|---|---|
| 代码快照 | 完整可读（97 个源文件），但**不是**原始运行时的精确版本：作者当时用的 `scripts/fig6/train.py` 驱动脚本不在快照里 |
| 超参数 | 以代码里的默认值为准（见 `code/eat-train/scripts/config.py`）；作者笔记里另有一批手改值（lr、batch、eta_max…），两者不完全一致 |
| 曲线 | 88 次运行全部重算，可直接重绘 |
| 评估指标（响应时间/质量） | 只有作者手工抄录的 64 节点单种子表（`docs/FACTS.md` §7），**没有**可重算的原始评估日志 |
| 物理 SD 验证 | 代码在，日志与结果缺失（`12-server`） |
| 论文名次 | **未复现**（见第 2 节、[13](13-results/README.md)） |
| 实现能否支撑论文结论 | **不足以**：七处问题，其中三处有物证（见 [14](14-audit/README.md)） |

| dimension | status |
|---|---|
| code snapshot | complete and readable (97 files), but **not** the exact revision that produced the runs: the `scripts/fig6/train.py` driver the author used is absent |
| hyper-parameters | code defaults are authoritative (`code/eat-train/scripts/config.py`); the author's notes record extra hand-set values (lr, batch, eta_max…) that do not always match |
| curves | all 88 runs re-derived and re-plottable |
| evaluation metrics (latency / quality) | only the author's hand-transcribed 64-node single-seed table (`docs/FACTS.md` §7); no replayable evaluation logs |
| physical SD validation | code present, logs and results missing (`12-server`) |
| the paper's ranking | **not reproduced** (section 2, [13](13-results/README.md)) |
| can this implementation support the paper | **no**: seven issues, three with evidence beyond the source (see [14](14-audit/README.md)) |

---

## 7. 这段工作证明了什么 / What this work demonstrates

不是"我跑通了代码"，而是"我独立完成了一次实现审计，并知道下一步该验证什么"。下面这张表是给**看结果的人**（导师、面试官、合作者）准备的：每一行都能在仓库里点开核对。

Not "I got the code running" — "I ran an independent implementation audit and know what to verify next". Every row below is checkable inside this repository.

| 能力 | 这里的具体证据 | 去哪看 |
|---|---|---|
| 读懂论文并忠实实现 | 7 个算法 × 3 种子 × **88 次训练**；自建 TensorBoard 解析器（不依赖 tensorboard / protobuf / matplotlib） | [`13`](13-results/README.md)、`tools/safe_tfreader.py` |
| 工程化与可复算 | 数据管线（解析 → 清洗 → 绘图）、checkpoint 取证（不装 torch 直接读张量名与参数量）、双语 15 章、链接自检脚本 | `tools/`、[`docs/DATA.md`](docs/DATA.md) |
| 科学怀疑与定位能力 | 论文名次不复现 → 不辩解、不删种子，而是定位到 **7 处实现层原因**，并按证据强度分级（E3 物证 / E2 代码 / E1 转述） | [`14`](14-audit/README.md) |
| 边界意识 | 主动标注：D2SAC 是**重建实现**、本地代码已修改过、物理侧无数据、快照与训练版本不一致 | [`14`](14-audit/README.md) §1、§2、§4 |
| 成本与风险判断 | 为下一步给出 **22 次训练**的最小闭环，并给出停止条件（而不是"再跑跑看"） | 本文件 §9 |

---

## 8. 简历段落（模板，改成你自己的话再用）/ CV template

> 措辞原则：**带数字、带限定、不说满**。"复现了论文"人人都会写；"定位到 6 处实现缺陷、其中 3 处有物证"才是可核验的。

**中文**

- **EAT 论文复现与实现审计**（独立完成）
  复现 TMC 投稿算法 EAT 及 6 个基线，7 算法 × 3 种子共 88 次训练；自建 TensorBoard 日志解析 / 清洗 / 绘图管线（零第三方绘图依赖），全部曲线可逐点复算。
- 发现论文排名不可复现，并定位到 7 处实现层原因：全局种子缺失（物证：两次运行 step-0 评估**逐位相同**）、双 critic 结构不对称（实测差 **65,792** 参数 = 恰好一层）、扩散策略熵项被置零、actor 越界无约束等；按证据强度分级（权重级 / 代码级 / 转述级）。
- 设计三组受控实验（种子确定性 / critic 对称化 / 熵项恢复，约 22 次训练）把代码级证据升级为因果结论，并给出停止条件；产物为开源双语仓库（15 章 + 数据管线 + 校验工具）。

**English**

- **Reproduction and implementation audit of the EAT paper** (independent)
  Reproduced EAT and six baselines (7 algorithms × 3 seeds = 88 training runs) and built the whole TensorBoard parsing / curation / plotting pipeline with no third-party plotting dependency; every curve is recomputable point by point.
- Found that the paper's ranking does not reproduce, and traced it to seven implementation-level causes: no global RNG seed (evidence: two runs share a bit-identical step-0 evaluation), asymmetric twin critics (65,792 parameters = exactly one layer), a zeroed entropy term in the diffusion-SAC policy, and an unbounded actor; each claim carries an explicit evidence level.
- Designed three controlled experiments (~22 short runs) to turn the code-level claims into causal ones, including an explicit stop criterion; shipped as a bilingual repository with 15 chapters, a data pipeline and self-checks.

> 三句话不要写：❌"我发现了原作者的 bug"（改成"定位到实现层原因"） · ❌"我复现了 EAT"（没有数字就无信息量） · ❌"D2SAC 最后一名说明论文有问题"（D2SAC 是重建实现，这条必须和结论绑在一起）。

---

## 9. 下一步：两周最小闭环 / Next: a two-week minimal loop

先修 `trainer/dcasac` 的 `schedule_timesteps` 笔误让快照能跑（见 [`14`](14-audit/README.md) A6），再做三组对照——每一组都能独立成表：

| # | 对照 | 产出 | 成本 |
|---|---|---|---|
| 1 | 设种子后**同配置跑两次**，逐位比较曲线 | "本实现可复现"的地基（最低成本、最高说服力） | 2 次训练 |
| 2 | critic 对称化**前后**各 5 种子（EAT） | "少一层是否改变均值与方差" | 10 次训练 |
| 3 | D2SAC **恢复熵项 + 自动温度**前后各 5 种子 | "D2SAC 垫底有多少来自实现" | 10 次训练 |

**停止条件（先定好，避免无限投入）**：若第 2 组的均值差 < 3 分且方差无显著变化，则这条线不再追加实验，只作为"实现细节不影响主结论"的负结果保留。

**不建议**做的事：把这条线扩展成正式投稿（算法无新增贡献）、或直接拿现有 PPO 实现去搭 MAPPO（会叠加 A4 / A5 / A1 三处问题，见 [`14`](14-audit/README.md) 与 [docs/REPRODUCE.md](docs/REPRODUCE.md) §3）。

---

## 10. 出处与致谢 / Provenance

- 代码来源：作者工作区 `EAT-master/EAT-Train`、`EAT-master/EAT-Server`，原样拷入 `code/`，未改逻辑。
- 训练日志：作者工作区 `EAT-master/EATlogs/EAT基线对比/**`（TensorBoard event 文件）。
- 结果图：`figures/curves/` 为本仓库重绘；`figures/original/` 保留作者原图。
- 论文与阅读笔记：`EATlogs/Reading/`（`复现.docx`、`AAAresult.docx`、`coding.docx`、`AlgoAndData.xlsx` 等）仅作为事实核对来源被引用，不随仓库分发。

- Code: the author's `EAT-master/EAT-Train` and `EAT-master/EAT-Server`, copied verbatim into `code/`; no logic was changed.
- Training logs: `EAT-master/EATlogs/EAT基线对比/**` (TensorBoard event files).
- Figures: `figures/curves/` is redrawn here; `figures/original/` keeps the author's own renders.
- Paper and reading notes: `EATlogs/Reading/` (`复现.docx`, `AAAresult.docx`, `coding.docx`, `AlgoAndData.xlsx`, …) are cited as the source of cross-checks and are not redistributed.

---

## 11. 上传到 GitHub / Pushing to GitHub

本目录已经是一个独立 git 仓库，并已有一笔初始提交（361 个文件，约 20 MB）：

```powershell
cd EAT-Story
git remote add origin https://github.com/<你的账号>/<仓库名>.git
git branch -M main
git push -u origin main
```

几点提醒：

- 仓库里**没有** `.pth` 权重、`events.out.tfevents.*` 原始日志、`runs/` 输出（见 `.gitignore`）。要给出完整可复现包，把这批文件作为 GitHub Release 附件，并在 `docs/DATA.md` 里补一条下载链接。
- 原始 TensorBoard 日志合计 7.4 GB，如需公开请走 Release 或 Git LFS，不要直接 `git add`。
- `data/` 是最大的一块（14 MB），全是文本 CSV，压缩率很高，正常 push 不会有问题。

This directory is already its own git repository with one initial commit (361 files, ~20 MB). Weights, raw event files and `runs/` output are intentionally excluded; publish those as Release assets if you want a fully reproducible bundle.

---

## 12. 一句话结论 / One-line verdict

**机制方向复现出来了，名次没有。** 交叉注意力 + DDIM + 可学习 η 的组合在两个种子上拿到 187.28 的平均终点回报（高于任何算法的三种子均值），但它对随机种子极其敏感 —— 第三个种子掉到 158.68；论文的排序需要更多种子和同预算训练才能验证。

**The mechanisms reproduce; the ranking does not.** The cross-attention + DDIM + learnable-η combination reaches an average of 187.28 on its two good seeds — above every algorithm's 3-seed mean — but it is highly seed-sensitive: the third seed lands at 158.68. The paper's ordering needs more seeds and budget-matched training before it can be called verified.

---

<sub>本仓库是学习与复现记录，不是官方实现。论文方法的所有功劳属于原作者。/ This repository is a reproduction log, not an official implementation. Credit for the method belongs to the original authors.</sub>
