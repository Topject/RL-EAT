# 代码快照说明 / Code snapshot notes

`code/eat-train/` 与 `code/eat-server/` 是作者工作区 `EAT-master/` 的**原样拷贝**（去掉 `__pycache__` 与 `.pyc`），**逻辑未做任何修改**。
The two directories are verbatim copies of the author's `EAT-master/`, with `__pycache__` and `.pyc` removed. **No logic was changed.**

| 目录 | 内容 | 文件数 |
|---|---|---|
| `code/eat-train/` | 训练侧：`env/`（环境与节点/任务管理）、`model/`（注意力与扩散）、`policy/`（3 个自定义策略）、`trainer/`（14 个算法 trainer）、`scripts/`（配置、训练与评估入口） | 62 |
| `code/eat-server/` | 推理侧：`app.py`、`server/`、`distrifuser/`、`scripts/`（SDXL、指标、性能剖析） | 34 |

---

## 1. 和"能一键复现"的差距 / Gap to one-command reproduction

1. **驱动脚本缺失**：作者笔记里的 `python scripts/fig6/train.py --algo eat --epoch 500 --device cuda --seed 43` 是实际使用的入口，但 `scripts/fig6/` 不在这个快照里。快照里只有每个算法各自的 `*_train.py`，且都是"改常量再跑"风格，没有 argparse。
2. **配置双份**：`env/config.py` 的 `EnvConfig`（奖励尺度、可见任务数）与 `scripts/config.py` 的 `GlobalConfig`/`TrainConfig`（节点数、网络宽度、扩散步数）各管一半，改实验要两边都看。
3. **随机种子**：训练脚本里 `TrainConfig.seed` 默认 42，但本快照没有把种子写进运行目录名，所以 `runs/` 下无法从目录名看出是哪个种子。这也是本仓库 `data/` 里只能用时间戳标识运行的原因。

## 2. 读代码时值得注意的地方 / Things worth knowing while reading

| 位置 | 现象 | 影响 |
|---|---|---|
| `model/attention/cross_attention.py#L137-L231` 与 `#L234-L328` | 同一个类 `CrossAttentionQPreprocessNet_` 定义了两次（名字带下划线） | 后面的定义覆盖前面的；两者实现一致，属于复制粘贴残留 |
| `model/attention/cross_attention.py#L330` | 真正的 `CrossAttentionQPreprocessNet`（无下划线）在第三处定义，多了 `action_embed` | 训练实际用的是这一个，容易看错 |
| `env/env.py#L516-L519` | `if reward <= 0: print()` 然后 `assert reward > 0` | 奖励非正会直接中断训练；调试时表现成"莫名崩掉" |
| `trainer/dicasac/trainer.py#L137-L169` | `critic1` 用完整 `critic_hidden_sizes`，`critic2` 只用 `tail` | 两个 critic 不对称，作者笔记里也把这一点列为待修（第 10 章） |
| `trainer/dppo/trainer.py#L100` | critic 的 `preprocess_net` 直接复用扩散 backbone | 同一个扩散块被 actor 和 critic 共享，参数量因此只有 519k |
| `policy/sac_diffusion.py` | 一个文件里定义 `SACDDPMPolicy` 与 `SACDDIMPolicy` 两个策略 | D2SAC / EAT 的差别主要就落在采样器上 |
| `scripts/utils.py#L144` | `find_latest_ckpt` 用 mtime 排序选"最新一次运行" | 复制过目录后可能选错 checkpoint，评估前先确认真实目录 |

## 3. 保留但没用到的东西 / Present but unused by the Figure-6 arms

- `trainer/{appo,cappo,asac,casac,dappo,dcappo,dcasac,dicasac,dsac}` 共 9 个 trainer 里，主对比只用到 `dcasac`(→d2sac)、`dppo`、`dicasac`(→eat)、`casac`(→sac_ca)、`dsac`(→d2sac_sa)。
- `model/diffusion/discrete_compound_diffusion.py`、`policy/diffusion_compound_ppo.py` 是离散/复合动作的早期尝试，连续动作版本没有引用它们。
- `scripts/eval_real_FIFO.py`、`eval_real_random.py`、`env/backend/` 属于物理执行路径（第 12 章）。

保留它们是有意的：读这些"没被用到的分支"能看出作者试过哪些方向。

They are kept on purpose: the unused branches show which directions were tried.

## 4. 编码 / Encoding

所有源文件为 UTF-8（含中文注释与文档字符串）。原始工作区里有些文件在 Windows 控制台下会显示成乱码，那是终端编码问题，不是文件内容问题。
All sources are UTF-8 (with Chinese comments). Files may look garbled in a Windows console with a non-UTF-8 code page; the files themselves are fine.
