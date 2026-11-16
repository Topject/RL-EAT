# 导师速览：EAT 复现与实现审计 / For advisors

> 一页。想看结论去第 1 节，想看证据去第 2 节。值不值得投时间，第 4 节。

## 1. 这是什么

我复现了 IEEE TMC 投稿论文 EAT（注意力引导的扩散强化学习，用于边缘 AIGC 调度），以及它的六个基线。
仓库里没有一张跑通截图。88 次训练都留了 TensorBoard 日志，曲线可以逐点重算，每个数字都标了出处。

## 2. 四组可核对的数字

| 事实 | 数字 | 怎么核 |
|---|---|---|
| EAT 复现规模 | 7 算法 × 3 种子 = 88 次训练 | [`data/runs_index.csv`](data/runs_index.csv) |
| 论文名次没复现 | 论文写 `EAT > DPPO ≈ D2SAC > PPO ≈ SAC > TD3 > DDPG`。我跑出来是 `PPO ≈ DPPO > SAC > TD3 > DDPG > EAT > D2SAC` | [13-results](13-results/README.md) |
| 原因定位到实现层 | 七处问题，其中三处有代码之外的物证 | [14-audit](14-audit/README.md) |
| 多智能体扩展 | 三集群 14 个算法、五集群 5 个算法，合计 170 次正式训练 | [docs/MARL.md](docs/MARL.md) |

三处物证，不是读代码之后的感觉。

1. **没有全局种子。** 两次 DPPO 运行的 step-0 评估都是 `148.589691`，逐位相同。第三次是 151.780380。
2. **两个 critic 不对称。** 17 个 checkpoint 里，critic1 是 221,601 个参数，critic2 是 155,809，差 65,792，恰好一层 256×256+256。
3. **代码按现状跑不起来。** `trainer/dcasac#L116` 传 `schedule_timesteps=`，而 `RLDiffusionActor` 只接受 `n_timesteps`。

## 3. 这些大概能说明什么

- 读得懂论文，也能把扩散策略、交叉注意力、SAC/PPO 的改造落到代码上。
- 结果不对的时候，我去查具体的代码行和权重张量，没有停在辩解上。
- 大致知道自己的证据强弱。每条结论标了 E3（有物证）、E2（代码可核，影响未量化）、E1（转述）。

这张表在 [README 第 7 节](README.md)。措辞偏保守，欢迎您直接否定。

## 4. 想请您判断的两件事

1. **底座要不要换。** 这个 PPO 看起来不像标准实现：`unbounded=True` 没有 tanh，隐藏层用 ReLU 而非 Mish，也没有全局种子。我倾向换成 CleanRL 的标准实现，只保留 EAT 的环境模型。这个判断可能过重，想听您的意见。
2. **副线值不值得投。** 把实现修复与可复现性分析做成技术报告或短文，可行吗，还是把时间全给主线。

## 5. 需要说明的限制

- D2SAC 是我重建的实现。原作者说环境不兼容，也没有上传，所以 D2SAC 垫底这件事不能算到论文方法头上。
- 我本地的算法代码改过。审计对象是仓库里那份原始快照，不是我当前的工作代码。
- 物理侧的 Stable Diffusion 验证只有实现路径，没有日志，也没有数字。
- 快照和训练出 checkpoint 的版本不完全一致，有报错为证。

---

细节在主 [README](README.md)，多智能体那部分在 [docs/MARL.md](docs/MARL.md)，时间线在 [docs/STORY.md](docs/STORY.md)，数字出处看 [docs/FACTS.md](docs/FACTS.md)，审计在 [14-audit](14-audit/README.md)。
