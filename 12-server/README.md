# 12 EAT-Server, 把"一个任务"变成一次真实的扩散推理
# 12 EAT-SERVER, turning a task into a real diffusion inference

> 上一章 / previous：[11 规模与任务密度](../11-scale-density/README.md) · 下一章 / next：[13 全部结果](../13-results/README.md)

## 0. 中英摘要 / Abstract

前十一章的所有数字都来自同一处：一个仿真环境。在这个环境里，一个任务的执行时间是算出来的，`a × steps + b`，再除以并行权重和节点算力（`code/eat-train/env/node.py#L72-L93`、`env/env.py#L269`）。也就是说，"任务有多贵"是参数，不是测量。

`code/eat-server/` 是把这句话翻过来的那一半：它用 DistriFuser 把 Stable Diffusion 的 UNet 按 patch / tensor 切开分布在多张卡上，用 FastAPI 暴露 `/init`、`/draw`、`/destroy`、`/remain_time`，让调度器可以真的下发一次推理并等它返回。训练侧的 `env/backend/real.py` 就是这套 HTTP 接口的客户端。

本章必须说清楚一件不体面的事：**本仓库没有任何物理执行日志或结果。** 原始工作区的 `EATlogs/` 里也没有。因此本章只写代码路径、接口形状和"要测什么"，所有物理数字（单任务延迟、吞吐、GPU 型号、图像质量）一律 未记录 / not recorded。

English: every number in chapters 01–11 comes from one simulator, where a task's execution time is computed as `a × steps + b` divided by a parallelism weight and node compute power, a parameter, not a measurement. `code/eat-server/` is the other half of that sentence: it shards a Stable Diffusion UNet across GPUs through DistriFuser, exposes `/init`, `/draw`, `/destroy`, `/remain_time` through FastAPI, and lets the scheduler issue real inferences. `env/backend/real.py` is the client of exactly those endpoints. This repository contains no physical execution logs or results, so this chapter describes code paths and what would have to be measured; every physical number is marked not recorded.

## 1. 为什么会有这一步 / Why this step

- 仿真里的时间成本是参数化的：`Node.receive()` 用 `time_pred_a × steps + time_pred_b` 得到基础执行时间，再按并行度权重与节点算力折算（`code/eat-train/env/node.py#L76-L91`）。这个模型能表达"步数越多越慢、并行越多越快"，但它不知道真实 GPU 上发生了什么。
- 更麻烦的是奖励：奖励里的质量项直接用 `steps` 当质量代理（`code/eat-train/env/env.py#L490-L503`）。也就是说，仿真里"质量"和"时间"都来自同一个数 `steps`。如果真实推理的时间/质量关系不是这样，那整个奖励信号都要重估。
- 于是有了这一半代码：把"任务的执行"换成真的 SD 推理，让时间与质量都变成可测量的量。对应 `RuntimeConfig.mode` 的两个取值 `"simulated"` / `"real"`（`code/eat-train/scripts/config.py#L81`）。
- 这一节回答的是另一类问题：物理侧的代码长什么样，接上它需要测哪些数。物理侧跑得怎么样，目前未记录 / not recorded。这是一份接口说明书，不是一份实验报告。

## 2. 代码里有什么 / What is in the box

| 文件 | 做什么 | 关键位置 |
|---|---|---|
| `code/eat-server/app.py` | FastAPI 服务；启动时加载 SD 1.4（`CompVis/stable-diffusion-v1-4`），推理走 DistriFuser 的 `DistriSDPipeline` | `#L8`、`#L24`、`#L75-L101` |
| 同上 | 端点：`/init`（建进程组并重配流水线）、`/draw`（出图）、`/destroy`（拆组）、`/remain_time`（剩余时间） | `#L124-L201`、`#L225-L298`、`#L103-L122`、`#L300-L302` |
| 同上 | 每步回调累计 `STEP_TIMES`，用平均步时乘剩余步数得到 `ESTIMATED_REMAIN_TIME` | `#L204-L223` |
| 同上 | 入口参数只有 `--port`（默认 8000）与 `--device-id`（默认 0） | `#L304-L316` |
| `code/eat-server/server/app.py` | 同构但换成 SDXL（`stabilityai/stable-diffusion-xl-base-1.0`）+ `DistriSDXLPipeline`；只有 `/init` 与 `/draw` | `#L53-L66`、`#L69-L141`、`#L144-L196` |
| `code/eat-server/distrifuser/pipelines.py` | `DistriSDXLPipeline` 与 `DistriSDPipeline`：`from_pretrained` 加载一次权重，`reconfigure` 按新配置重切分片，`teardown` 释放 | `#L10`、`#L21`、`#L172`、`#L194`、`#L206`、`#L263` |
| `code/eat-server/distrifuser/models/` | 三种切法：`distri_sdxl_unet_pp.py`（patch 并行）、`distri_sdxl_unet_tp.py`（tensor 并行）、`naive_patch_sdxl.py` | 目录 |
| `code/eat-server/distrifuser/modules/pp/`、`modules/tp/` | 逐层的分布式实现：GroupNorm / Conv2d / Attention / ResNet | 目录 |
| `code/eat-server/distrifuser/utils.py` | `DistriConfig`：`world_size` 必须是 2 的幂；开 CFG 时 batch 折半分配设备 | `#L20-L21`、`#L72-L79` |
| `code/eat-server/api.json` | `/init`、`/draw`、`/destroy` 的字段与返回码，可作为接口契约 | `#L1-L64` |
| `code/eat-server/utils.py` | 调试用 `inspect_pipeline_state()`：打印 device、parallelism、CUDA graph、进程组、显存占用 | `#L28-L109` |
| `code/eat-server/setup.py` | `distrifuser` 的打包定义 | 全文 |

推理之外，仓库里还有四个脚本，分别负责"造数据 / 算指标 / 看结果 / 算算力"：

| 脚本 | 做什么 | 关键位置 |
|---|---|---|
| `scripts/run_sdxl.py` | 两种模式：`--mode generation` 出单图，`--mode benchmark` 跑延迟（warmup 后测多次，去掉最慢/最快的 `ignore_ratio` 部分） | `#L113-L123`、`#L124-L155` |
| `scripts/generate_coco.py` | 用 COCO 2014 captions 验证集的 prompt 批量出图，可 `--split` 切片并行 | `#L96-L131` |
| `scripts/compute_metrics.py` | 两个目录逐图对齐后算 PSNR / LPIPS / FID | `#L62-L78` |
| `scripts/export_html.py` | 把多个目录里的同名图拼成一张对照 HTML 表 | `#L41-L83` |
| `scripts/profile_macs.py` | 只 profile SDXL 的 UNet 一次前向的 MACs | `#L20-L45` |

## 3. 一个任务怎么变成一次推理 / How a task becomes an inference

训练侧的真后端在 `code/eat-train/env/backend/real.py`，它把一次任务拆成三步（`TaskManager.execute_task()`，`#L282-L442`）：

| 步骤 | 做什么 | 代码 |
|---|---|---|
| a. destroy | 需要重载的节点组同时收到 `/destroy`，拆掉旧进程组 | `#L296-L311`、`#L133-L147` |
| b. init | 为新任务生成一个 `job_id`，从端口池取一个 master 端口，向所有分配到的节点并发发 `/init` | `#L313-L354`、`#L149-L196` |
| c. draw | 所有节点同时发 `/draw`，rank 0 回图；请求在后台线程里等，主流程立刻返回 | `#L356-L442` |

几个和"真实感"直接相关的实现细节：

- 端口池是 `29500–29599` 一百个端口（`real.py#L42`），每次 `job_id` 独占一个，失败会归还（`#L49-L80`）。这说明服务端是"一次任务一个进程组"的模型，进程组重建是常态成本。
- `/draw` 是阻塞调用，超时由 `RuntimeConfig.api_timeout` 决定（`real.py#L23`、`#L238`）；根目录 `app.py` 里 `dist.barrier()` 之后才返回（`#L272-L279`）。
- 服务端对 `guidance_scale` 与实际配置的一致性有硬检查：请求的 `guidance_scale` 与当前 `do_classifier_free_guidance` 不匹配就直接 400（`app.py#L242-L245`）。也就是**改 CFG 必须重新 `/init`**，不能在一次任务中途改。
- `world_size` 必须是 2 的幂（`app.py#L136-L138`、`distrifuser/utils.py#L20-L21`），所以节点组合法长度受限。这会影响调度器能选哪些并行度。

### 3.1 接口契约 / The endpoint contract

| 端点 | 方法 | 关键字段 | 返回 / 错误 |
|---|---|---|---|
| `/init` | POST | `job_id`、`master_addr`、`master_port`、`world_size`、`rank`、`local_rank`，加上尺寸、`mode`、`parallelism`、`split_batch`、`use_cuda_graph`、`guidance_scale` | 200 回 `{status, job_id, rank, world_size}`；`world_size` 非 2 的幂回 400；建组失败回 500 |
| `/draw` | POST | `job_id`、`prompt`、`num_inference_steps`、`guidance_scale`、`seed` | rank 0 回 `image/png`（文件名 `result.png`），其余 rank 回 JSON；进程组未初始化回 400，`job_id` 不匹配回 409 |
| `/destroy` | POST | 无 | 拆掉进程组；未初始化时回 `{status: "not initialized"}` |
| `/remain_time` | GET | 无 | `{remain_time: <秒>}`，值来自每步回调的均值估计 |

字段定义的三处来源彼此不完全一致：`api.json#L1-L64` 是文档版，`code/eat-server/app.py#L49-L73` 是 pydantic 模型版，`code/eat-train/scripts/config.py#L78-L105` 是客户端默认值版。三者的尺寸默认值不同（见第 5 节第 2 条），这是复现时最容易踩的坑。

## 4. 真后端里"真"到什么程度 / How real is "real"

这是本章最该讲清楚的一节。`mode="real"` 并不等于"所有量都来自测量"：

| 量 | 来源 | 是测量吗 |
|---|---|---|
| 节点忙碌 / 空闲 | `GET /remain_time`，值来自服务端 `step_callback` 的步时均值 × 剩余步数（`app.py#L204-L223`） | 部分是：步时是实测，剩余步数只是估计 |
| 任务开始时的剩余时间 | `task.steps × 0.1` 的硬编码估计（`real.py#L550-L552`） | 不是，是个常数 |
| 节点状态向量 | `[remaining / 30, model_id / 30]`（`real.py#L498-L501`；仿真侧同式，`node.py#L69`） | 归一化后的估计值 |
| 任务"质量" | 仍用 `steps` 当代理（`env.py#L490-L503`） | 两边都不是测量 |
| 图像实际质量（PSNR/LPIPS/FID） | `scripts/compute_metrics.py` 可以算 | **未记录 / not recorded**：仓库里没有跑过的输出 |

所以物理模式下的"时间"仍然是估计与实测的混合体：一条任务刚下发时用的是 `steps × 0.1`，之后靠轮询 `/remain_time` 修正。这一点对复现很重要，如果只把 `mode` 从 `simulated` 改成 `real`，而不去核对这两条时间来源，得到的仍然是一个半仿真环境（推断 / inferred，依据是上面两处代码路径）。

## 5. 想知道物理侧表现，必须测什么 / What would have to be measured

下面每一项在 `docs/FACTS.md` 里都没有对应数字，因此本章一律写 未记录 / not recorded：

| 指标 | 怎么测（代码里已有的入口） | 本仓库的值 |
|---|---|---|
| 单任务端到端延迟（按 steps / 并行度分层） | `scripts/run_sdxl.py --mode benchmark` 已实现 warmup + 多次取均值 | 未记录 / not recorded |
| `/remain_time` 的估计误差 | 对比服务端估计与客户端实际墙钟 | 未记录 / not recorded |
| 进程组重建开销（destroy→init 的固定成本） | `real.py` 的三步流程天然可计时，但代码里没有打点 | 未记录 / not recorded |
| 吞吐（图/秒、任务/分钟） | 需连续跑 N 个任务并统计 | 未记录 / not recorded |
| GPU 型号与卡数 | 服务端启动时固定用 `cuda:0`（`app.py#L79-L81`）并可 `--device-id` 指定 | 未记录 / not recorded（作者笔记里出现过 RTX 4080 SUPER / 笔记本，见 `docs/FACTS.md` §8；那不是本仓库的测量） |
| 图像质量（PSNR / LPIPS / FID） | `scripts/compute_metrics.py` 已经写好 | 未记录 / not recorded |
| UNet 单次前向 MACs | `scripts/profile_macs.py` 会打印 `MACs: …G` | 未记录 / not recorded |
| 物理模式下的回报 / 完成数 / 平均响应时间 | `eval_real_FIFO.py`、`eval_real_random.py` 会把结果写进 `runs/real_FIFO/`、`runs/real_random/` | 未记录 / not recorded（目录不在本仓库） |

再补两条"要复现就会撞上"的实现事实：

1. 根目录 `app.py` 在加载权重前后设置了本地代理（`HTTPS_PROXY` / `HTTP_PROXY` = `127.0.0.1:20171`，`#L91-L92`），并且以 `local_files_only=True` 加载（`#L96`）。也就是说它假定权重已经在本地缓存里；换机器复现时这一步会直接失败，而不是自动下载。
2. 接口文档与实现在默认尺寸上不一致：`api.json` 写 `height`/`width` 默认 512（`#L14-L15`），而根目录 `app.py` 的 pydantic 模型默认 1024（`#L58-L59`）；两个真实评测脚本自己也不同（`eval_real_FIFO.py#L39-L41` 用 512×512 / 20 步，`eval_real_random.py#L39-L41` 用 1024×1024 / 50 步）。复现时必须显式传参，不能依赖默认值。

## 6. 物理侧的评测入口 / Physical-mode evaluation entries

| 入口 | 策略 | 规模设置 | 输出 |
|---|---|---|---|
| [`code/eat-train/scripts/eval_real_FIFO.py`](../code/eat-train/scripts/eval_real_FIFO.py) | 固定动作 `[1, 0, -1]`（FIFO），`#L74-L76` | `NUM_NODES = 4`（`#L19`），重复 8 次（`#L15`），种子 42 起（`#L17`） | `runs/real_FIFO/<N>/eval_run.csv` 与 `eval_tasks.csv`（`#L48-L58`） |
| [`code/eat-train/scripts/eval_real_random.py`](../code/eat-train/scripts/eval_real_random.py) | 动作均匀随机采样（`#L74-L76`） | `NUM_NODES = 8`（`#L17`），重复 2 次（`#L14`），种子 42 起（`#L16`） | `runs/real_random/<N>/…`（`#L19`） |
| 后端 | `RealBackend` / `TaskManager`，HTTP 客户端 | 节点地址是 `ip:port` 列表（两脚本都列了 8 个 `172.16.0.*:8000`，`#L26-L35`） | `outputs_real/` 下的 PNG（`real.py#L411-L421`） |

注意两处不对称，都会影响复现：FIFO 脚本写 `NUM_NODES = 4` 却给了 8 个节点地址（`#L19` vs `#L26-L35`），随机脚本写 `NUM_NODES = 8`；两个脚本的推理尺寸与步数也不同（见上节第 2 条）。所以"物理侧 FIFO 与 Random 的对比"在代码层面就不是同一套参数，本仓库也没有跑过它们，无法给出任何结论。

### 6.1 三条必须先补的测量

如果下一轮要把物理侧接进故事线，顺序应该是：

1. **先把单卡基线钉住。** 用 `run_sdxl.py --mode benchmark` 在固定尺寸、固定 scheduler、固定 `num_inference_steps` 下测一组延迟，明确记录 GPU 型号与卡数。这是所有后续数字的分母，缺了它，多卡分片的收益无从判断。
2. **再测进程组重建成本。** 物理模式下一个任务要 destroy → init → draw（`real.py#L282-L442`），其中 destroy / init 是纯开销。必须在服务端两侧打点，否则调度器会把"重建进程组"的时间错算成"推理"的时间。
3. **最后才谈质量。** `compute_metrics.py` 能出 PSNR / LPIPS / FID，但需要成对的参考图目录（`--input_root0` / `--input_root1`），而物理模式的出图在 `outputs_real/`，参考图从哪来 未记录 / not recorded。在这一点定下来之前，任何"物理侧质量"的说法都不成立。

这三步与第 5 节的表格是同一张清单的两个视角：上面列的是"缺什么数"，这里列的是"先补哪个"。

## 7. 数据导航 / Data

本章**没有**数据目录。这是本章最重要的一个事实。

| 内容 | 路径 | 状态 |
|---|---|---|
| 物理执行日志 | 原工作区 `EATlogs/` 下的 real 模式记录 | 未记录 / not recorded（本仓库没有；原始工作区的 `EATlogs` 也没有） |
| 物理模式评测 CSV | `runs/real_FIFO/`、`runs/real_random/` | 未记录 / not recorded（不在本仓库） |
| 物理模式出图 | `outputs_real/` | 未记录 / not recorded（不在本仓库） |
| 指标脚本的输出 | `compute_metrics.py` 打印的 PSNR / LPIPS / FID | 未记录 / not recorded |
| 服务端与后端代码 | [`code/eat-server/`](../code/eat-server/)、[`code/eat-train/env/backend/`](../code/eat-train/env/backend/) | 已入库，可逐行核对 |
| 训练侧全部数字 | [`data/runs_index.csv`](../data/runs_index.csv) | 全部来自 `simulated` 模式 |

第三列是本仓库唯一诚实的写法。任何"物理侧延迟 X 秒 / FID Y"的说法，只要没有落在上面某个路径里，就是编的。

## 8. 复现命令 / Reproduce

下面两条命令是从代码里的 argparse 定义直接读出来的，**本机没有运行过**（需要 GPU 与本地权重缓存）：

```powershell
# 服务端（根目录 app.py，SD 1.4，默认端口 8000、GPU 0）
python code\eat-server\app.py --port 8000 --device-id 0

# 单机 SDXL 延迟基准（第 5 节表格里第一行的入口）
python code\eat-server\scripts\run_sdxl.py --mode benchmark
```

第二条在没有初始化 torch.distributed 时会退化成单卡（`distrifuser/utils.py#L43-L51` 捕获异常后把 `world_size` 当成 1），所以它测的是单卡延迟，不是分布式延迟。这一点在报告数字时必须写清楚。

---

上一章 / previous：[11 规模与任务密度](../11-scale-density/README.md) · 下一章 / next：[13 全部结果](../13-results/README.md) · 返回目录 / index: [README](../README.md)
