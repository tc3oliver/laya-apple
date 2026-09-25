# laya-apple

[English](README.md) | [繁體中文](README.zh-TW.md) | **简体中文**

[![CI](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml/badge.svg)](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/laya-apple)](https://pypi.org/project/laya-apple/)
![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

> 本文档是 [`README.md`](README.md) 的中文版。如有出入，以英文版为准。

**经过正确性验证、可同时使用 MLX GPU 和 Apple Neural Engine（ANE）的
[Laya](https://github.com/NandhaKishorM/laya) 运行时，面向 Apple 芯片。**

## 本地 LLM 忙碌时，决策依然很快

`laya-apple serve` 在你的 Mac 上用上游 Laya 回答 Jev 客户端发来的决策请求。默认的
`--device auto` 会把简短的单问题决策交给 Apple Neural Engine 运行，不用在 GPU 上排在本地 LLM
后面等待。

```bash
pip install 'laya-apple[serve,ane]'
laya-apple serve                                   # http://127.0.0.1:8642
export TYPESAFE_BASE_URL=http://127.0.0.1:8642     # then run your Jev client as usual
```

![laya-apple serve 与忙碌的本地 LLM 同时运行，在 Apple M4 Max 上用 Qwen3.8-27B-oQ4e-mtp 跑一次的结果：LLM 生成时，简短决策的 P99 在 serve auto 下为 47.2 ms，--device gpu 下为 122.3 ms（LLM 空闲时分别为 43.2 和 55.9 ms）；LLM 吞吐量单独运行时为 42.2 tok/s，搭配 serve --device gpu 时为 40.0，搭配 serve auto 时为 40.5](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/serve-llm-load.svg)

以下是在一台 Apple M4 Max 上跑一次的结果：27B 本地 LLM（oMLX 上的 `Qwen3.8-27B-oQ4e-mtp`）
满载生成，同时每秒发送 8 个决策请求，其中 80% 是简短的单问题决策
（[`benchmarks/serve/`](benchmarks/serve/README.md)，第 2 次运行）：

- **LLM 忙碌时，简短决策的 P99：`auto` 为 47.2 ms，`--device gpu` 为 122.3 ms。** LLM 空闲时，
  `auto` 为 43.2 ms，`--device gpu` 为 55.9 ms。
- **LLM 的吞吐量以单独运行时的 42.2 tok/s 为基准，搭配 serve `auto` 时下降 4.0%，搭配
  `--device gpu` 时下降 5.3%。** 两者相差 1.31 个百分点，只比同一次运行中 LLM 自身各 window
  之间的波动（1.28 个百分点）略高。
- **负载下结果依然正确：** 7,728 个决策中 0 个 hard mismatch、0 个错误；每个决策都与同一个服务器
  在 LLM 空闲时的回答做了比对。
- `auto` 的简短决策约有 4% 按设计改走 GPU：当 Neural Engine 的 backlog 等待时间更长时就会这样。
  上面的 P99 已包含这些请求。
- 这个 benchmark 的第 1 次运行按其自身的有效性检查判定为无效，不从中得出任何结论。

API 与 Jev 兼容；laya-apple 与 TypeSafe 或 Jev 没有任何关联，回答来自 Laya，而不是 Jev。

## 在 Mac 上用本地后端替代 Jev API

`laya-apple serve` 可以在本地替代 Jev API。它在 loopback 地址上提供 Jev 兼容 API
（`POST /v1/systemone`），并在你的 Mac 上用上游 Laya 给出回答。
现有的 Jev 客户端只需把 base URL 配置指向这里即可，代码无需任何改动。

```bash
pip install 'laya-apple[serve,ane]'
laya-apple serve                                   # http://127.0.0.1:8642
export TYPESAFE_BASE_URL=http://127.0.0.1:8642     # then run your Jev client as usual
```

Jev 客户端启动时需要 API key。只要服务端没有设置 `LAYA_API_KEY`，填任意占位值都可以，
例如 `TYPESAFE_API_KEY=local-placeholder`。

![终端画面：laya-apple serve 在本地启动；未经修改的 Python typesafe-sdk 0.7.1 正式版通过 TYPESAFE_BASE_URL 指向它，得到回答 fix_code；用 curl 发送同一请求，可以看到是 laya-apple 用 laya checkpoint 在 Apple Neural Engine 上作答的](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/serve-demo.svg)

- **已用 7 个 Jev 客户端的正式版本实测，客户端未做任何修改**，包括 Python 和 JS SDK，
  以及两个 Claude Code 插件。每个客户端都只改了 base URL 配置，所有请求均顺利完成
  （[`integrations/jev-plugins/README.md`](integrations/jev-plugins/README.md)）。
- **回答来自 Laya，而不是 Jev。** 这里只保证结果与上游 Laya 一致：792 个请求全部在 FP16
  parity 检查范围内与未经修改的上游 `laya.serve` 0.3.20 一致，其中 365 个由 ANE 作答
  （[`benchmarks/serve-compat/`](benchmarks/serve-compat/README.md)）。Laya 是另一个更小的模型，
  因此不声称具备 Jev 级别的准确率；如果客户端的阈值是针对 Jev 调优的，可能会更频繁地触发 fallback。
- laya-apple 与 TypeSafe 或 Jev 没有任何关联。上图是一次实际运行的录制结果
  （[`scripts/capture_serve_demo.py`](scripts/capture_serve_demo.py)）。

模型、API、安全性和限制：[`docs/serve.md`](docs/serve.md)。

## 手上有 Mac？试试 Switchyard

```bash
uvx laya-apple switchyard
```

![Switchyard：同一份录制好的时刻表分别以仅 GPU 和 GPU + ANE 回放；仅 GPU 时，列车在高峰时段堵在红灯前排队，GPU + ANE 时则顺畅驶入站台；最后是结果卡片：仅 GPU 时 1,422 趟中有 1,407 趟晚点，GPU + ANE 时为 0 趟](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/switchyard-launch.gif)

**每一趟列车都是一次真实的 Laya 决策。** 下面是在 Apple M4 Max 上运行、用同一份时刻表对比仅 GPU
与 GPU + ANE 的结果：

- **仅 GPU：** 1,422 趟中有 1,407–1,408 趟晚点，P99 决策延迟约 3.1 s。
- **GPU + ANE：** 1,422 趟中 0 趟晚点，P99 约 54.6 ms。

游戏画面与 benchmark 的对应关系：

- **列车** → 一个请求：“哪个站台是空的？”
- **红灯** → 列车正在等模型回答
- **道岔** → 决策：道岔扳向模型选中的站台
- **晚点** → 100 ms 内没有拿到回答
- **高峰时段** → 突发负载，同时 GPU 上还有长时间运行的后台请求

Benchmark 会先以 headless 模式跑完，浏览器再回放录制的请求 trace；动画本身不计入测量。

### 同一份时刻表，同一个模型，GPU + Neural Engine

| | 仅 MLX GPU | MLX GPU + Neural Engine |
|---|---:|---:|
| 晚点列车 | 1,407–1,408 / 1,422 | **0 / 1,422** |
| P99 决策延迟 | 3,107.7–3,170.6 ms | **54.5–54.7 ms** |
| P99 排队等待 | 3,095.9–3,158.8 ms | **39.6–42.8 ms** |

以上是同一台 Apple M4 Max（macOS 26.6.2，`laya-typed-decisions`）上的三次标准运行；
换成其他 Mac 结果会有所不同。差距主要不是因为单个请求在 ANE 上跑得更快，而是短请求不必再在
GPU 队列里等待，长请求则继续留在 GPU 上运行。方法和原始数据：
[`benchmarks/switchyard/README.md`](benchmarks/switchyard/README.md)。

## 为什么做这个项目

Laya 只需一次 forward pass，就能回答一段 context 中的 typed questions（`choice`、`score`、`noul`）。
Mac 上有两个引擎可以运行它，各有所长。

1. **ANE 执行的正确性。** Core ML 导出跑得快，不代表结果一定正确。在我们测试的 Mac 上，普通的
   Core ML 导出在 ANE 上运行时没有报任何错误，却有多达 85 个决策与上游 PyTorch 不一致。
   ANE artifact 只有在实际运行它的 Mac 上通过 parity 检查后才会被使用
   （[`docs/correctness.md`](docs/correctness.md)）。
2. **自动路由。** 较短且已验证的单问题请求发往 ANE；较长或包含多个问题的请求发往 MLX GPU。
   路由器在请求运行之前就做出决定，并记录原因。
3. **GPU + ANE 并发服务。** 两个引擎同时处理相互独立的请求，短请求不再排在长请求后面。

## 安装

需要 Apple 芯片和 Python 3.11–3.13：

```bash
pip install laya-apple
```

可选 extras：

```bash
pip install "laya-apple[ane]"       # + the Neural Engine runtime (coremltools 9.0)
pip install "laya-apple[convert]"   # + building ANE artifacts on this Mac (torch 2.7.0)
pip install "laya-apple[serve]"     # + laya-apple serve, the local Jev-compatible server
laya-apple artifacts build laya-typed-decisions   # optional: build + parity-validate ANE artifacts here (~5 min)
```

没有安装 `ane` extra，或者还没有构建 artifact 时，所有计算都会在 MLX GPU 上运行。
如需从源码运行或参与开发，请参阅 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

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

在测试机器上的输出：

```text
high {'low': 0.1713, 'medium': 0.3358, 'high': 0.4929}
coreml ane validated_short_single_question_path 11.2 ms
```

- 首次调用会下载锁定版本的 checkpoint，之后即可离线使用（`local_files_only=True`）。
- 没有 ANE artifact 时，同一请求会改在 MLX 上运行，`routing_reason` 会说明原因。
- 更多示例：[`examples/`](examples/)（`basic.py`、`auto_routing.py`、
  `heterogeneous_serving.py`）以及[用户指南](docs/guide.md)。

## 自动路由的工作原理

![不超过 128 个 token、只有一个问题且有已验证 artifact 的请求发往 Apple Neural Engine；更长、包含多个问题或未经验证的请求发往 MLX GPU；使用 execution="workers" 时，两个引擎并发处理相互独立的请求](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/architecture.svg)

| 请求 | 发往 | 原因（在测试用的 Mac 上测得） |
|---|---|---|
| 一个问题、≤ 128 个 token、有已验证的 artifact | **ANE** | 更快：laya-typed-decisions 在 L128 时，ANE 耗时 9.9 ms，MLX 耗时 12.2 ms（forward P50） |
| Context 较长 | **MLX GPU** | 这种情况下 MLX 更快：L256 为 19.2 ms，L1024 为 71.0 ms |
| 多个问题 | **MLX GPU** | MLX 会把问题批量处理；ANE 只能逐个运行 |
| 未经验证的 Mac、缺少 artifact，或没有 Core ML | **MLX GPU** | 记录为 `platform_not_validated`、`ane_artifact_unavailable` 或 `ane_runtime_unavailable` |

**生产环境使用的阈值比实测的交叉点更保守。**
laya-multilingual 在恰好 256 个 token 时 ANE 略快一些（8.3 ms 对 8.6 ms），但这个 bucket
仍然只在显式指定时使用，因为在前一个 bucket（128 个 token）上它并没有胜过 MLX。
每个结果都带有 `routing_reason`。

阈值如何推导：[`docs/support-matrix.md`](docs/support-matrix.md)。整体架构：
[`docs/architecture.md`](docs/architecture.md)。

## Switchyard benchmark

`laya-apple switchyard` 以 headless 模式测量固定的 `switchyard-v1` workload，然后输出
`result.json`、`trace.jsonl` 和一个自包含的 `replay.html`
（[`docs/switchyard.md`](docs/switchyard.md)）。

- **Workload。** Seed 11、60 s、突发到达，名义速率 40 req/s：共 2,410 个请求，其中
  1,422 个是单问题的列车决策，deadline 为 100 ms。其余是中等长度、较长和多问题的请求，用来给 GPU 加负载。
- **轮次。** `gpu_only`（`device="gpu"`），以及 Neural Engine 就绪时运行的 `hybrid`
  （`device="auto"`），两者都通过 `Laya(execution="workers")` 跑完全相同的时刻表。
- **测量。** 决策延迟从每趟列车的计划到达时间算起，直到拿到回答为止，包含排队时间。Deadline miss rate
  分别以 25、50、100、250 和 500 ms 为阈值统计，因此结论不依赖于游戏设定的 100 ms deadline。
- **检查。** 如果 hybrid 轮次实际上没有用到 Neural Engine，会被标记为不可比较。在 M4 Max 的测试中，
  两个轮次对每一趟列车给出的回答都相同，也没有任何列车被路由错。
- **耗时。** 首次运行会下载锁定版本的 checkpoint（约 800 MB）；之后一次标准运行约需
  2–3 分钟。如果还没有构建 Neural Engine artifact，它只会运行 `gpu_only`，并打印设置命令
  `uvx --from "laya-apple[convert]" laya-apple switchyard --setup-ane`。

**局限。** 只在一台机器（Apple M4 Max）上测过。轮次顺序由 seed 决定，所以三次运行都是 `hybrid`
先跑（`design.counterbalance` 为 `"none"`）。这些数字不能与下方 v1.0 的 open-loop 表格直接比较：
两者的速率、测量边界和 workload 都不同（[`docs/switchyard.md`](docs/switchyard.md)）。

每次运行的结果、多次运行之间的差异以及复现命令：
[`benchmarks/switchyard/README.md`](benchmarks/switchyard/README.md)。原始结果：
[`benchmarks/switchyard/v1-m4-max/raw/`](benchmarks/switchyard/v1-m4-max/raw/)。

## GPU + ANE 异构服务

![同一个模型分别在 MLX GPU 和 Apple Neural Engine 上玩 Lane Runner，得到相同的分数；随后同一批突发请求分别以仅 GPU 和 GPU + ANE 处理：仅 GPU 时，短请求在 GPU 队列中最多等待 1.5 s，而在 GPU + ANE 下，路由器把它们发往 ANE，请求一到即运行](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/heterogeneous-serving.gif)

这段动画展示的是本节所用的 v1.0 benchmark。laya-apple 会把有已验证 artifact 的短单问题请求发往
Apple Neural Engine，较长的请求则留在 MLX GPU 上运行。动画先回放一局在 M4 Max 上录制的
[Lane Runner](https://github.com/tc3oliver/laya-playground-apple) 游戏：同一个模型分别在两个设备上
各录一次，每次只处理一个请求。随后根据 benchmark 到达序列的逐请求 trace，回放
laya-typed-decisions 突发 workload 中的一次突发。其中的 P99 就是下表中已发布的 v1.0 数字；
每个数字的来源都列在 [`docs/media/README.md`](docs/media/README.md)。

![与仅 GPU 相比的混合 workload 吞吐量：laya 从 41.9 提升到 122.5 req/s（2.92×），laya-multilingual 从 55.7 提升到 241.8 req/s（4.34×），laya-typed-decisions 从 24.0 提升到 109.6 req/s（4.57×）](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/hero-throughput.svg)

性能提升来自同时使用两个引擎，而不是 ANE 本身的延迟更低。这是在一台 Apple M4 Max、macOS 26.6.2
上运行的 v1.0 benchmark：一条短请求流和一条长请求流，送入同一个 `Laya(execution="workers")` 实例。
其他 Mac 不在这个 benchmark 的范围内，相关结果见[社区矩阵](#community-benchmarks)。方法和原始数据见
[`benchmarks/v1.0.md`](benchmarks/v1.0.md)。

```python
with Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers") as model:
    futures = [model.submit(context=c, questions=q) for c, q in requests]   # thread-safe
```

- GPU 在独立的 worker 进程中运行，ANE 则有自己的 dispatcher。
- 每个请求只在一个设备上运行，由路由器选择。
- 有负载时，路由器还会比较各队列的积压情况。

想知道某个请求为什么慢，可以传入一个回调函数。每完成一个请求，它都会收到一个 `RequestTrace`：
路由器做决定时看到的队列快照、它选择的设备和原因，以及从提交、排队、执行到响应的 monotonic 时间戳。

```python
def on_trace(trace):
    print(trace.request_id, trace.target, trace.routing_reason, trace.queue_ms, trace.e2e_ms)

model = Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers", trace=on_trace)
```

默认的 `trace=None` 不会记录任何内容。回调函数在对应设备的 dispatcher 线程上运行，
所以请尽量保持轻量。

**Open-loop 突发到达下的短请求 P99**，从请求到达时刻开始计时，包含排队时间
（v1.0，两种配置使用相同的到达序列）：

| 模型 | 仅 GPU | GPU + ANE |
|---|---:|---:|
| laya（offered load 46.2 req/s） | 1538.0 ms | 108.5 ms |
| laya-multilingual（83.8 req/s） | 2052.3 ms | 29.6 ms |
| laya-typed-decisions（35.8 req/s） | 1592.9 ms | 79.5 ms |

单个短请求在 ANE 上并没有快很多（例如 9.9 ms 对 12.2 ms）。性能提升来自同时使用两个引擎。

<a id="community-benchmarks"></a>

## 社区 benchmark

上面所有 benchmark 都只在 M4 Max 上跑过。[社区矩阵](docs/community-benchmarks.md)
收录其他 Mac 的结果，每台单独记录，不与上面的数字混在一起。目前已有 M4 Pro、M4 和
M2 Pro（仅 MLX）的外部结果。如果你有其他型号的 Mac，一条命令就能把它加进来。
**无需修改任何代码。**

- M1 / M2 → [#1](https://github.com/tc3oliver/laya-apple/issues/1)
- M3 Max → [#2](https://github.com/tc3oliver/laya-apple/issues/2)
- M5 或其他任何 Mac → [添加你的 Mac](docs/community-benchmarks.md#add-your-mac)

```bash
uv run python scripts/hardware_report.py --quick
```

上面的 issue 和指南给出了从 `git clone` 到提交 pull request 的完整步骤，大约需要 10 分钟。

## 正确性

以 PyTorch CPU FP32 上的上游 Laya 为基准，用随附的 golden rows 对比 parity（v1.0）。
每个单元格依次是 hard mismatch 数和最大概率误差。

| 测试用 Mac 上的实现 | laya | laya-multilingual | laya-typed-decisions |
|---|---|---|---|
| 普通 Core ML 导出 · `CPU_AND_NE` | ❌ 12, 0.56 | ❌ 85, 1.0 | ❌ 19, 0.42 |
| 普通 Core ML 导出 · `CPU_AND_GPU` | ✅ 0, 0.0066 | ✅ 0, 0.0059 | ✅ 0, 0.0028 |
| **laya-apple MLX FP16** | ✅ 0, 0.0037 | ✅ 0, 0.0045 | ✅ 0, 0.0017 |
| **laya-apple ANE FP16** | ✅ 0 (1 near-tie), 0.012 | ✅ 0, 0.013 | ✅ 0, 0.0077 |

- **FP16 检查标准：** 概率误差 ≤ 0.02，且 hard mismatch 为 0。
- **Near-tie 翻转**（上游前两名的差值 < 0.04）会明确列出，不会隐藏。
- **显式指定 ANE 的请求绝不 fallback。** 要么运行已验证的 artifact，要么直接抛出异常；
  每个加载的 artifact 还会与 `CPU_ONLY` 对比运行时间，以发现被悄悄放到 CPU 上运行的情况。
- 定义、所有测试过的配置以及 fallback 审计，见
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
- MLX 预计可以正常工作。
- 在该机器上构建并校准 artifact（`laya-apple calibrate`）之前，`auto` 会一直使用 MLX。

参见 [`docs/compatibility.md`](docs/compatibility.md)，以及
[`docs/community-benchmarks.md`](docs/community-benchmarks.md) 中的社区矩阵。

## 复现

上面每个主要数字，都能在 [`docs/reproducibility.md`](docs/reproducibility.md) 中找到对应的
报告、原始数据、命令和环境。完整的 v1.0 测试对比了 PyTorch CPU/MPS、普通 Core ML 导出、
MLX 和 laya-apple，见 [`benchmarks/v1.0.md`](benchmarks/v1.0.md)。想在自己的 Mac 上快速测一下：

```bash
uv run python scripts/hardware_report.py --quick
```

## 参与贡献

如果想开始参与贡献，最有价值的是提交非 M4 Max 的 Mac 的 benchmark 结果：运行上面的命令，
然后带上 `hardware-results/` 提交一个 PR（[具体做法](docs/community-benchmarks.md)）。

- [`CONTRIBUTING.md`](CONTRIBUTING.md) 介绍了环境搭建、测试分级（每种改动实际需要跑哪些测试）、
  parity 检查以及后端改动。
- 待处理的工作标有 `good first issue`、`help wanted` 和 `research` 标签。
- Coding agent：仓库规则见 [`AGENTS.md`](AGENTS.md)。

## 局限

- **只有一台测试机器。** 所有 benchmark 都来自同一台运行 macOS 26.6.2 的 Apple M4 Max。
  我们不假定路由阈值在其他 Apple SoC 上同样成立。
- **长 context 留在 MLX 上运行，** 因为 MLX 在这种情况下更快。ANE 路径仅支持 batch 1。
- **并发的请求流之间无法完全隔离。** 并发运行时，每条请求流的 P99 都高于单独运行时。
- **Cold start 需要编译：** 如果 artifact 目录是全新的，cold start 时每个模型需要 3–5 分钟编译 Core ML。
  设置 `ane_startup="background"` 时，这段时间会先用 MLX 提供服务。
- **`choice` 决策可能受选项顺序影响。** 这是上游 Laya 本身的行为，laya-apple 完全复现了这一点
  （[`research/option-order/`](research/option-order/)）。
- **尚未测量：** 能耗、量化 artifact 以及跨 SoC 验证。
- **`laya-apple serve` 用 Laya 作答，而不是 Jev。** 只测量了结果与上游 Laya 是否一致，
  没有测量 Jev 级别的准确率。阈值针对 Jev 调优的客户端可能会更频繁地触发 fallback；
  测试的七个客户端中有两个出现了这种情况
  （[`integrations/jev-plugins/README.md`](integrations/jev-plugins/README.md)）。
- **`laya-apple serve` 的性能只在一种配置下跑过一次：** 一台 M4 Max、一个 LLM（oMLX 上的
  `Qwen3.8-27B-oQ4e-mtp`，以 decode 为主、prompt 很短）、`--model laya`、每秒 8 个决策请求
  （[`benchmarks/serve/`](benchmarks/serve/README.md)）。其他 LLM 服务器和模型、以 prefill
  为主的 LLM 负载、`--model auto` 以及其他请求速率都还没有测量。
- **serve `auto` 仍会拉低 LLM 的吞吐量：** 那次运行中下降 4.0%，只比 `--device gpu` 少 1.31
  个百分点，而 LLM 单独运行时各 window 之间的波动就有 1.28 个百分点。
- **多问题决策始终在 GPU 上运行，** LLM 忙碌时它们的 P99 会升高：`auto` 下为 117.1 ms，
  空闲时为 84.1 ms。
- **客户端兼容性只测试过一次：** 2026-09-25，使用列出的客户端版本，在测试用的 M4 Max 上进行。
  客户端之后的版本可能会改变它发送或接受的内容。
- **`serve --model auto` 只在英文模型和多语言模型之间路由。** 上游需手动开启的 typed-decisions
  工作流检测（`LAYA_AUTO_TASK`）以及调用方的语言提示都尚未实现
  （[`docs/serve.md`](docs/serve.md#limits)）。

## 更多

- 用户指南：[`docs/guide.md`](docs/guide.md)。
- 本地 Jev 兼容服务器：[`docs/serve.md`](docs/serve.md)。
- 稳定 API：[`docs/api.md`](docs/api.md)。
- 架构：[`docs/architecture.md`](docs/architecture.md)。
- 变更记录：[`CHANGELOG.md`](CHANGELOG.md)。
- 安全：[`SECURITY.md`](SECURITY.md)。

采用 Apache-2.0 许可证；见 [`LICENSE`](LICENSE) 和 [`NOTICE`](NOTICE)。模型权重从指定的
Hugging Face revision 下载，本项目不会再分发。这是一个独立项目，并非 Convai Innovations、Apple 或 MLX
的官方发布。
