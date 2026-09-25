# laya-apple

[English](README.md) | [繁體中文](README.zh-TW.md) | **简体中文**

[![CI](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml/badge.svg)](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/laya-apple)](https://pypi.org/project/laya-apple/)
![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

> 本文档是 [`README.md`](README.md) 的译文。如有出入，以英文版为准。

**经过正确性验证、面向 Apple 芯片的异构 [Laya](https://github.com/NandhaKishorM/laya)
运行时：** 同时使用 MLX GPU 和 Apple 神经网络引擎（Neural Engine，ANE）。

## 在 Mac 上用本地后端替代 Jev API

`laya-apple serve` 是 Jev API 的本地替代方案。它在回环地址（loopback）上提供 Jev 兼容 API
（`POST /v1/systemone`），并用在你的 Mac 上运行的上游 Laya 给出回答。
只需通过现有 Jev 客户端的 base URL 设置把它指向这里；客户端代码无需任何改动。

```bash
pip install 'laya-apple[serve,ane]'
laya-apple serve                                   # http://127.0.0.1:8642
export TYPESAFE_BASE_URL=http://127.0.0.1:8642     # then run your Jev client as usual
```

Jev 客户端需要一个 API key 才能启动。除非你在服务器端设置了 `LAYA_API_KEY`，
否则任意占位值都可以，例如 `TYPESAFE_API_KEY=local-placeholder`。

![终端画面：laya-apple serve 在本地启动；未经修改的已发布版 Python typesafe-sdk 0.7.1 通过 TYPESAFE_BASE_URL 指向它，得到回答 fix_code；用 curl 发送同一请求，显示是 laya-apple 用 laya checkpoint 在 Apple 神经网络引擎上作答](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/serve-demo.svg)

- **已在不做修改的情况下，用 7 个 Jev 客户端的已发布版本进行测试**，包括 Python 和 JS SDK，
  以及两个 Claude Code 插件。每个客户端都只通过 base URL 设置指向这里，并完成了各自的请求
  （[`integrations/jev-plugins/README.md`](integrations/jev-plugins/README.md)）。
- **回答来自 Laya，而不是 Jev。** 所保证的是与上游 Laya 的一致性：792 个请求中有 792 个
  在 FP16 parity 门限内与未经修改的上游 `laya.serve` 0.3.20 一致，其中 365 个由神经网络引擎作答
  （[`benchmarks/serve-compat/`](benchmarks/serve-compat/README.md)）。Laya 是另一个、
  规模更小的模型，因此不声称具备 Jev 级别的准确率；如果客户端的阈值是针对 Jev 调优的，
  它可能会更频繁地进入回退（fallback）路径。
- laya-apple 与 TypeSafe 或 Jev 没有任何关联。上图是一次录制的运行
  （[`scripts/capture_serve_demo.py`](scripts/capture_serve_demo.py)）。

模型、API、安全性和限制：[`docs/serve.md`](docs/serve.md)。

## 有 Mac？试试 Switchyard。

```bash
uvx laya-apple switchyard
```

![Switchyard：同一份录制的时刻表分别以仅 GPU 和 GPU + ANE 回放；仅 GPU 时，列车在高峰时段停在红灯信号前排队，GPU + ANE 时则顺畅驶入站台；随后是结果卡片：仅 GPU 时 1,422 趟中有 1,407 趟晚点，GPU + ANE 时为 0 趟](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/switchyard-launch.gif)

**每一趟列车都是一次真实的 Laya 决策。** 同一份时刻表，仅 GPU 对比 GPU + ANE，
运行于 Apple M4 Max：

- **仅 GPU：** 1,422 趟中有 1,407–1,408 趟晚点，P99 决策延迟约 3.1 s。
- **GPU + ANE：** 1,422 趟中 0 趟晚点，P99 约 54.6 ms。

游戏与基准测试的对应关系：

- **列车** → 一个请求：“哪个站台是空的？”
- **红灯信号** → 列车正在等待模型的回答
- **道岔** → 决策：道岔扳向模型选中的站台
- **晚点** → 100 ms 内没有得到回答
- **高峰时段** → 突发负载，同时 GPU 上有长时间运行的后台请求

基准测试先以无头模式（headless）运行，随后浏览器回放录制的请求轨迹；动画本身不属于测量的一部分。

### 同一份时刻表。同一个模型。GPU + 神经网络引擎。

| | 仅 MLX GPU | MLX GPU + 神经网络引擎 |
|---|---:|---:|
| 晚点列车 | 1,407–1,408 / 1,422 | **0 / 1,422** |
| P99 决策延迟 | 3,107.7–3,170.6 ms | **54.5–54.7 ms** |
| P99 排队等待 | 3,095.9–3,158.8 ms | **39.6–42.8 ms** |

在同一台 Apple M4 Max（macOS 26.6.2，`laya-typed-decisions`）上的三次标准运行；
其他 Mac 的结果会有所不同。差距主要并不来自单个请求在神经网络引擎上的速度：而是短决策不再需要在
GPU 队列中等待，长请求则继续在 GPU 上运行。方法和原始数据：
[`benchmarks/switchyard/README.md`](benchmarks/switchyard/README.md)。

## 为什么要做这个项目

Laya 通过一次前向计算（forward pass）回答关于一段上下文的类型化问题（`choice`、`score`、`noul`）。
在 Mac 上有两个引擎可以运行它，各有所长。

1. **正确的 ANE 执行。** 速度快的 Core ML 导出不一定正确。在测试用的 Mac 上，普通的 Core ML
   导出在神经网络引擎上运行时没有报任何错误，却让多达 85 个决策与上游 PyTorch 不一致。
   laya-apple 只有在 ANE artifact 于实际使用它的机器上通过 parity 门限后才会采用
   （[`docs/correctness.md`](docs/correctness.md)）。
2. **自动路由。** 较短的、经过验证的单问题请求发往 ANE；较长的或多问题的请求发往 MLX GPU。
   路由器在请求运行之前做出决定，并记录原因。
3. **GPU + ANE 并发服务。** 两个引擎同时处理相互独立的请求，因此短请求不再排在长请求后面。

## 安装

Apple 芯片，Python 3.11–3.13：

```bash
pip install laya-apple
```

可选的 extras：

```bash
pip install "laya-apple[ane]"       # + the Neural Engine runtime (coremltools 9.0)
pip install "laya-apple[convert]"   # + building ANE artifacts on this Mac (torch 2.7.0)
pip install "laya-apple[serve]"     # + laya-apple serve, the local Jev-compatible server
laya-apple artifacts build laya-typed-decisions   # optional: build + parity-validate ANE artifacts here (~5 min)
```

如果没有安装 `ane` extra，或者没有已构建的 artifact，所有计算都在 MLX GPU 上运行。
如需从源码运行或参与开发 laya-apple，请参阅 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 快速上手（30 秒）

```python
from laya_apple import Laya

model = Laya.from_pretrained(
    "convaiinnovations/laya-typed-decisions",
    device="auto",
)

result = model.predict(
    context="The customer was charged twice for the same invoice and is frustrated.",
    questions={
        "urgency": {
            "type": "choice",
            "instructions": "How urgent is this?",
            "criteria": ["low", "medium", "high"],
        }
    },
)
print(result.answers["urgency"]["choice"], result.answers["urgency"]["probabilities"])

rt = result.runtime
print(rt.backend, rt.device, rt.routing_reason, f"{rt.latency_ms:.1f} ms")
```

在测试机器上：

```text
high {'low': 0.1713, 'medium': 0.3358, 'high': 0.4929}
coreml ane validated_short_single_question_path 11.2 ms
```

- 首次调用会下载锁定版本的 checkpoint，之后即可离线使用（`local_files_only=True`）。
- 没有 ANE artifact 时，同一请求会在 MLX 上运行，并由 `routing_reason` 说明原因。
- 更多内容：[`examples/`](examples/)（`basic.py`、`auto_routing.py`、
  `heterogeneous_serving.py`）以及[用户指南](docs/guide.md)。

## 自动路由的工作原理

![不超过 128 个 token、只有一个问题且有经过验证的 artifact 的请求发往 Apple 神经网络引擎；更长、多问题或未经验证的请求发往 MLX GPU；使用 execution="workers" 时，两个引擎并发处理相互独立的请求](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/architecture.svg)

| 请求 | 发往 | 原因（在测试用的 Mac 上测得） |
|---|---|---|
| 一个问题、≤ 128 个 token、有经过验证的 artifact | **ANE** | 更快：laya-typed-decisions 在 L128 时，ANE 耗时 9.9 ms，MLX 耗时 12.2 ms（forward P50） |
| 更长的上下文 | **MLX GPU** | 这种情况下 MLX 更快：L256 为 19.2 ms，L1024 为 71.0 ms |
| 多个问题 | **MLX GPU** | MLX 会对这些问题做批处理；ANE 则逐个运行 |
| 未经验证的 Mac、缺少 artifact，或没有 Core ML | **MLX GPU** | 记录为 `platform_not_validated`、`ane_artifact_unavailable` 或 `ane_runtime_unavailable` |

**生产环境使用的阈值比实测的交叉点更保守。**
laya-multilingual 在恰好 256 个 token 时在 ANE 上略快（8.3 ms 对 8.6 ms），但这个 bucket
仍然只在显式指定时使用，因为它在前一个 bucket（128 个 token）上并没有胜过 MLX。
每个结果都带有 `routing_reason`。

阈值如何推导：[`docs/support-matrix.md`](docs/support-matrix.md)。各组件如何协同：
[`docs/architecture.md`](docs/architecture.md)。

## Switchyard 基准测试

`laya-apple switchyard` 以无头模式测量冻结的 `switchyard-v1` 工作负载，然后写出
`result.json`、`trace.jsonl` 和一个自包含的 `replay.html`
（[`docs/switchyard.md`](docs/switchyard.md)）。

- **工作负载。** Seed 11、60 s、名义 40 req/s 的突发到达：共 2,410 个请求，其中
  1,422 个是单问题的列车决策，截止时间为 100 ms。其余是中等长度、较长和多问题的请求，用于给 GPU 加负载。
- **轮次。** `gpu_only`（`device="gpu"`），以及在神经网络引擎就绪时运行的 `hybrid`
  （`device="auto"`），两者都通过 `Laya(execution="workers")` 在完全相同的时刻表上运行。
- **测量。** 决策延迟从每趟列车的计划到达时间算起，直到得到回答为止，包含排队时间。超时率
  分别在 25、50、100、250 和 500 ms 下报告，因此结果不依赖于游戏中 100 ms 的截止时间。
- **检查。** 实际上没有用到神经网络引擎的 hybrid 轮次会被标记为不可比较。在 M4 Max 的
  测试中，两个轮次对每一趟列车都给出了相同的回答，且没有任何列车被错误路由。
- **耗时。** 首次运行会下载锁定版本的 checkpoint（约 800 MB）；之后一次标准运行约需
  2–3 分钟。如果没有已构建的神经网络引擎 artifact，它会运行 `gpu_only` 并打印设置命令
  `uvx --from "laya-apple[convert]" laya-apple switchyard --setup-ane`。

**局限。** 只有一台机器（Apple M4 Max）。轮次顺序由 seed 决定，因此三次运行中都是 `hybrid`
先运行（`design.counterbalance` 为 `"none"`）。这些数字不能与下方 v1.0 的开环（open-loop）表格直接比较：
两者的速率、测量边界和工作负载都不同（[`docs/switchyard.md`](docs/switchyard.md)）。

每次运行的结果、跨运行的离散程度以及复现命令：
[`benchmarks/switchyard/README.md`](benchmarks/switchyard/README.md)。原始结果：
[`benchmarks/switchyard/v1-m4-max/raw/`](benchmarks/switchyard/v1-m4-max/raw/)。

## GPU + ANE 异构服务

![同一个模型分别在 MLX GPU 和 Apple 神经网络引擎上玩 Lane Runner，得到相同的分数；随后同一批突发请求分别以仅 GPU 和 GPU + ANE 处理：仅 GPU 时，短请求在 GPU 队列中最多等待 1.5 s，而在 GPU + ANE 下，路由器把它们发往 ANE，请求一到即运行](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/heterogeneous-serving.gif)

本节所依据的 v1.0 基准测试的动画版。laya-apple 把有经过验证的 artifact 的短单问题请求发往
Apple 神经网络引擎，更长的请求则继续在 MLX GPU 上运行。动画首先回放一局在 M4 Max 上录制的
[Lane Runner](https://github.com/tc3oliver/laya-playground-apple) 游戏：同一个模型分别在各设备上
运行、分别录制，每次一个请求。随后根据基准测试到达序列的逐请求轨迹，回放 laya-typed-decisions
突发工作负载中的一次突发。其中的 P99 值就是下表中已发布的 v1.0 数字；每个数字的来源都在
[`docs/media/README.md`](docs/media/README.md)。

![相对于仅 GPU 服务的混合工作负载吞吐量：laya 从 41.9 提升到 122.5 req/s（2.92×），laya-multilingual 从 55.7 提升到 241.8 req/s（4.34×），laya-typed-decisions 从 24.0 提升到 109.6 req/s（4.57×）](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/hero-throughput.svg)

收益来自同时使用两个引擎，而不是 ANE 本身的延迟。这是在一台 Apple M4 Max、macOS 26.6.2 上的
v1.0 基准测试：一条短请求流和一条长请求流，经由同一个 `Laya(execution="workers")` 实例。
其他 Mac 不在此基准测试范围内；它们的结果收录在[社区矩阵](#community-benchmarks)中。方法和原始数据见
[`benchmarks/v1.0.md`](benchmarks/v1.0.md)。

```python
with Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers") as model:
    futures = [model.submit(context=c, questions=q) for c, q in requests]   # thread-safe
```

- GPU 在一个 worker 进程中运行，ANE 则在自己的 dispatcher 上运行。
- 每个请求只在一个设备上运行，由路由器选择。
- 在有负载时，路由器还会比较各队列的积压情况。

要查看某个请求为什么慢，可以传入一个回调函数。每完成一个请求，它都会收到一个 `RequestTrace`：
路由器做决定时所依据的队列快照、它选择的设备和原因，以及从提交、排队、处理到响应的单调时间戳。

```python
def on_trace(trace):
    print(trace.request_id, trace.target, trace.routing_reason, trace.queue_ms, trace.e2e_ms)

model = Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers", trace=on_trace)
```

使用默认的 `trace=None` 时不会记录任何内容。回调函数在对应设备的 dispatcher 线程上运行，
因此请保持其开销足够小。

**开环突发到达下的短请求 P99**，从到达时刻开始测量，包含排队时间
（v1.0，两者使用相同的到达序列）：

| 模型 | 仅 GPU | GPU + ANE |
|---|---:|---:|
| laya（施加负载 46.2 req/s） | 1538.0 ms | 108.5 ms |
| laya-multilingual（83.8 req/s） | 2052.3 ms | 29.6 ms |
| laya-typed-decisions（35.8 req/s） | 1592.9 ms | 79.5 ms |

单个短请求在 ANE 上并没有快很多（例如 9.9 对 12.2 ms）。收益来自同时使用两个引擎。

<a id="community-benchmarks"></a>

## 社区基准测试

上面的所有基准测试都只覆盖 M4 Max。[社区矩阵](docs/community-benchmarks.md)
以独立运行的形式收录其他 Mac 的结果，不与上面的数字混在一起。目前已有 M4 Pro、M4 和
M2 Pro（仅 MLX）的外部结果。如果你有其他型号的 Mac，一条命令就能把它加进来。
**无需修改任何代码。**

- M1 / M2 → [#1](https://github.com/tc3oliver/laya-apple/issues/1)
- M3 Max → [#2](https://github.com/tc3oliver/laya-apple/issues/2)
- M5 或其他任何 Mac → [添加你的 Mac](docs/community-benchmarks.md#add-your-mac)

```bash
uv run python scripts/hardware_report.py --quick
```

上述 issue 和指南给出了从 `git clone` 到 pull request 的完整步骤，大约需要 10 分钟。

## 正确性

以 PyTorch CPU FP32 上的上游 Laya 为基准、针对随附 golden rows 的 parity 结果（v1.0）。
每个单元格依次给出硬性不一致数和最大概率误差。

| 测试用 Mac 上的实现 | laya | laya-multilingual | laya-typed-decisions |
|---|---|---|---|
| 普通 Core ML 导出 · `CPU_AND_NE` | ❌ 12, 0.56 | ❌ 85, 1.0 | ❌ 19, 0.42 |
| 普通 Core ML 导出 · `CPU_AND_GPU` | ✅ 0, 0.0066 | ✅ 0, 0.0059 | ✅ 0, 0.0028 |
| **laya-apple MLX FP16** | ✅ 0, 0.0037 | ✅ 0, 0.0045 | ✅ 0, 0.0017 |
| **laya-apple ANE FP16** | ✅ 0 (1 near-tie), 0.012 | ✅ 0, 0.013 | ✅ 0, 0.0077 |

- **FP16 门限：** 概率误差 ≤ 0.02，且硬性不一致为 0。
- **Near-tie 翻转**（上游前两名的差值 < 0.04）会被列出，而不是被隐藏。
- **显式指定 ANE 的请求绝不回退。** 它们要么运行经过验证的 artifact，要么抛出异常；
  而且每个加载的 artifact 还会与 `CPU_ONLY` 对比计时，以发现被悄悄放到 CPU 上运行的情况。
- 定义、所有测试过的配置以及回退审计，见
  [`docs/correctness.md`](docs/correctness.md) 和
  [`docs/no-silent-fallback.md`](docs/no-silent-fallback.md)。

## 支持的模型和平台

| 模型 | max_len | MLX GPU | ANE bucket（显式指定） | `auto` 使用的 ANE bucket |
|---|---:|---|---|---|
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | 512 | FP16 / FP32，任意长度 | 64, 96, 128 | 64, 96, 128 |
| [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | 1024 | FP16 / FP32，任意长度 | 64, 96, 128, 256 | 64, 96, 128 |
| [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 1024 | FP16 / FP32，任意长度 | 64, 96, 128 | 64, 96, 128 |

**已测试：**
- Apple M4 Max、macOS 26.6.2、MLX 0.32.2、coremltools 9.0；
- Python 3.11–3.13。

**其他 Apple 芯片：**
- 预计 MLX 可以正常工作。
- 在该机器上构建并校准 artifact 之前（`laya-apple calibrate`），`auto` 会一直使用 MLX。

参见 [`docs/compatibility.md`](docs/compatibility.md)，以及
[`docs/community-benchmarks.md`](docs/community-benchmarks.md) 中的社区矩阵。

## 复现

上面每个主要数字都可以追溯到 [`docs/reproducibility.md`](docs/reproducibility.md) 中的
报告、原始数据、命令和环境。完整的 v1.0 测试套件对比了 PyTorch CPU/MPS、普通 Core ML 导出、
MLX 和 laya-apple，见 [`benchmarks/v1.0.md`](benchmarks/v1.0.md)。在你自己的 Mac 上快速检查：

```bash
uv run python scripts/hardware_report.py --quick
```

## 参与贡献

最有价值的首次贡献，是提交一台非 M4 Max 的 Mac 的基准测试结果：运行上面的命令，
然后带上 `hardware-results/` 提交一个 PR（[具体做法](docs/community-benchmarks.md)）。

- [`CONTRIBUTING.md`](CONTRIBUTING.md) 介绍了环境搭建、测试层级（某个改动实际需要跑哪些测试）、
  parity 检查以及后端改动。
- 待处理的工作标有 `good first issue`、`help wanted` 和 `research` 标签。
- 编程智能体（coding agent）：[`AGENTS.md`](AGENTS.md) 列出了本仓库的规则。

## 局限

- **只有一台测试机器。** 所有基准测试都来自一台运行 macOS 26.6.2 的 Apple M4 Max。
  不假定路由阈值在其他 Apple SoC 上同样成立。
- **长上下文留在 MLX 上，** 因为 MLX 在这种情况下更快。ANE 路径仅支持 batch 1。
- **隔离只是部分的。** 在并发情况下，每条请求流的 P99 都高于其单独运行时的值。
- 在全新的 artifact 位置上**冷启动**时，每个模型需要 3–5 分钟的 Core ML 编译。
  `ane_startup="background"` 会在此期间用 MLX 提供服务。
- **`choice` 决策可能受选项顺序影响。** 这来自上游 Laya，laya-apple 完全复现了这一行为
  （[`research/option-order/`](research/option-order/)）。
- **尚未测量：** 能耗、量化 artifact 以及跨 SoC 验证。
- **`laya-apple serve` 用 Laya 作答，而不是 Jev。** 只测量了与上游 Laya 的一致性，
  没有测量 Jev 级别的准确率。阈值针对 Jev 调优的客户端可能会更频繁地进入回退路径；
  在测试的七个客户端中有两个出现了这种情况
  （[`integrations/jev-plugins/README.md`](integrations/jev-plugins/README.md)）。
- **`laya-apple serve` 不做任何性能声明。** 它的延迟、吞吐量，以及它对共享 GPU 的本地 LLM
  的影响都没有测量。
- **客户端兼容性只测试过一次，** 时间为 2026-09-25，使用列出的客户端版本，在测试用的 M4 Max 上进行。
  客户端之后的版本可能会改变它发送或接受的内容。
- **`serve --model auto` 只在英文模型和多语言模型之间路由。** 上游可选启用的 typed-decisions
  工作流检测（`LAYA_AUTO_TASK`）以及调用方的语言提示尚未实现
  （[`docs/serve.md`](docs/serve.md#limits)）。

## 更多

- 用户指南：[`docs/guide.md`](docs/guide.md)。
- 本地 Jev 兼容服务器：[`docs/serve.md`](docs/serve.md)。
- 稳定 API：[`docs/api.md`](docs/api.md)。
- 架构：[`docs/architecture.md`](docs/architecture.md)。
- 变更记录：[`CHANGELOG.md`](CHANGELOG.md)。
- 安全：[`SECURITY.md`](SECURITY.md)。

Apache-2.0；见 [`LICENSE`](LICENSE) 和 [`NOTICE`](NOTICE)。模型权重从其锁定的 Hugging Face
revision 下载，不会再分发。这是一个独立项目，并非 Convai Innovations、Apple 或 MLX 的官方发布。
