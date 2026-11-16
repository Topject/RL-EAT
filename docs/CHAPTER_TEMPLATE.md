# 章节模板 / Chapter template

每个算法章节都必须严格按下面的骨架写，**中英双语**，中文在前、英文摘要紧随其后，长度控制在 120–220 行。
Every algorithm chapter follows this skeleton, bilingual with Chinese first, 120–220 lines.

```markdown
# NN 算法名, 一句话它解决什么
# NN ALGO, one line: what it solves

> 上一章 / previous：[链接](../0X-xxx/README.md) · 下一章 / next：[链接](../0Y-yyy/README.md)

## 0. 中英摘要 / Abstract

中文 4–6 句：这一章在整条故事线里的位置、它相比上一章多做了什么、结论是什么（含关键数字）。
English 3–5 sentences: same content, no new facts.

## 1. 为什么会有这一步 / Why this step

- 上一章留下的具体问题（引用上一章的数字或结论）
- 论文/代码里对应的是哪一处（文件 + 行号）
- 本章要验证的假设

## 2. 机制 / Mechanism

| 组件 | 实现 | 代码位置 |
|---|---|---|
| actor | ... | `code/eat-train/trainer/<x>/trainer.py#L<n>` |
| critic | ... | ... |
| 采样器 | ... | `code/eat-train/model/...` |

再用 3–6 行讲清楚：动作怎么产生、梯度怎么来、和上一章的差别只有哪几处。

## 3. 实验设置 / Setup

表格：环境规模、到达率、质量比、种子数、预算（epoch/步数）、学习率、扩散步数、参数量。

## 4. 收敛过程 / Convergence

- 贴图：`![...](../figures/curves/<file>.png)`
- 表：每次运行的终点回报 + 均值±标准差 + 最高回报（数字只能来自 `docs/FACTS.md`）
- 2–4 条观察：谁先收敛、谁震荡、哪一次运行异常

## 5. 结果与对比 / Results

同协议下的横向位置（引用 `docs/FACTS.md` 的数字），必要时给指标（响应时间/质量）。

## 6. 简要分析 / Analysis

3–5 条，必须落到**证据**：参数量、η 轨迹、曲线形状、日志里的具体数值。
明确写出"我们复现到了什么、没复现到什么"。

## 7. 训练数据导航 / Training data

| 内容 | 路径 |
|---|---|
| 本算法全部运行 | [`data/<campaign>/<algo>/`](../data/<campaign>/<algo>/) |
| 每一次运行的曲线 | `data/<campaign>/<algo>/<run>/curves.csv`（列：tag,step,value） |
| 运行的元数据 | `data/<campaign>/<algo>/<run>/run.json` |
| 全部运行索引 | [`data/runs_index.csv`](../data/runs_index.csv) |
| 按算法聚合 | [`data/by_algorithm.csv`](../data/by_algorithm.csv) |
| 原始 TensorBoard 日志 | `EATlogs/EAT基线对比/<...>/events.out.tfevents.*`（未入库，见 `docs/DATA.md`） |

## 8. 复现命令 / Reproduce

```powershell
# 只贴能跑通的命令；没有验证过的命令不要写
python scripts/<x>_train.py
```

---
上一章 / previous · 下一章 / next · 返回目录 / index: [README](../README.md)
```

## 硬性规则 / Hard rules

1. **数字只能来自 `docs/FACTS.md`**。没有的数字写成“未记录”，不要估算、不要编造。
2. 图片路径必须是仓库内真实存在的文件；生成脚本是 `tools/plot_curves.py`。
3. 代码引用写仓库内相对路径 `code/eat-train/...`，并带行号（用 `grep`/`read` 确认过）。
4. 事实与推断分开：推断必须写“推断 / inferred”，并给出依据（例如 checkpoint 的 tensor 名）。
5. 不写“我们的结果不如论文”这类空话，要写清楚**差在哪、差多少、可能因为什么**。
6. 不承诺未验证的结论（例如“再训练 1000 epoch 就能超过”）。
