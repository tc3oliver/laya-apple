# laya-apple

[English](README.md) | [繁體中文](README.zh-TW.md) | **简体中文**

[![CI](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml/badge.svg)](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/laya-apple)](https://pypi.org/project/laya-apple/)
![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

> 本文档是 [`README.md`](README.md) 的中文版。如有出入，以英文版为准。

**在 Apple 芯片上运行、经过正确性验证的自适应 [Laya](https://github.com/NandhaKishorM/laya)
推理：MLX GPU 和 Apple Neural Engine 同时提供服务。**

## 一次调用直达 Neural Engine，无需本地构建（1.6）

`from_pretrained("auto")` → 下载并验证 artifact → `predict` / `predict_shortlist`。

```bash
pip install -U 'laya-apple[ane]'
laya-apple artifacts fetch laya                  # prebuilt ANE artifacts, validated here
laya-apple artifacts fetch laya-multilingual
```

```python
from laya_apple import Laya

with Laya.from_pretrained("auto") as model:       # laya or laya-multilingual, per request
    result = model.predict(context=context, questions=questions)
    rt = result.runtime
    print(rt.model, rt.model_routing, rt.device, rt.routing_reason)
```

- **按语言路由。** `Laya.from_pretrained("auto")` 返回一个 `LayaRouter`，它同时加载 `laya` 和
  `laya-multilingual`，按每个请求 context 的语言选择其中一个，判断方式与 `laya-apple serve --model auto`
  使用的是同一个函数。选择的理由记录在 `RuntimeInfo.model_routing`（[`docs/api.md`](docs/api.md)）。
- **预构建的 ANE artifact，在你的 Mac 上验证。** `laya-apple artifacts fetch MODEL` 从
  [`tc3oliver/laya-apple-artifacts`](https://huggingface.co/tc3oliver/laya-apple-artifacts)
  下载预构建的 artifact，再交给会做验证的 `artifacts import` 注册：核对索引中的 SHA-256、manifest
  和平台 profile、compute plan、完整的 FP16 parity 关卡以及 placement probe，全部在接收端机器上执行。
  下载来的 artifact 不会比本地构建的更受信任。它省去了本地构建及其 PyTorch 依赖，不需要 `convert` extra。
- **`predict` 不变；`predict_shortlist` 需手动启用。** 标签很多的 `choice` 问题可以用
  `predict_shortlist(..., embed_fn, k=20)`：先保留与请求最相近的 `k` 个标签，再运行一次 `predict`。
  `predict` 本身没有变化（[`examples/auto_fetch_shortlist.py`](examples/auto_fetch_shortlist.py)）。

局限：
- **预构建的 artifact 目前只有一个平台 profile：Apple M4 Max、macOS 26、coremltools 9.0。**
  其他 Mac 都会得到 `ArtifactMissingError`，错误信息会给出构建命令，照旧在本地构建
  （`laya-apple artifacts build MODEL`，需要 `convert` extra）。
- **已发布的 artifact 经过一次空缓存下载检查：** 在该 profile 上，全部 10 组 model/bucket 的
  fetch、verify、parity 和 placement 都通过
  （[`benchmarks/prebuilt-artifacts-1.6.0.md`](benchmarks/prebuilt-artifacts-1.6.0.md)）。
  这项检查是在构建 artifact 的同一台机器上、用空缓存执行的，不是在第二台机器上。
- **第一次加载 Core ML 仍然要在设备上编译：** 约 4.5 分钟。在一次实测中，laya-typed-decisions
  （bucket 64/96/128）在新位置冷启动用了 273.5 s，热启动只要 2.7 s
  （[`research/coreml-compile-cache/screen.md`](research/coreml-compile-cache/screen.md)）。
  下载 artifact 无法省掉这段时间。设置 `ane_startup="background"` 时，这段时间会先用 MLX 提供服务。

1.6 另外还有这些，但不是重点：MLX 快速路径（token-id 缓存默认开启；`mx.compile` 和按长度分组的
batching 需手动启用；测量范围见[局限](#局限)）；
以及没有任何成果发布的 W8 ANE 研究（见[局限](#局限)）。Release notes：
[`docs/releases/v1.6.0.md`](docs/releases/v1.6.0.md)。

## 自适应 GPU + Neural Engine 服务（1.5）

laya-apple 把简短的单问题决策交给 Apple Neural Engine（ANE），较长或包含多个问题的任务留在
MLX GPU 上，两个引擎同时服务。从 v1.0 起，这种分工就让短请求不必排在耗时的 GPU 任务后面。

**GPU 早就算完了，Python 还在等。** ANE 跑在同一进程的线程上时，同步的 Core ML 调用在每次预测中
有很大一部分时间都持有 GIL。一个 GPU 已经算完的请求，要等这次调用返回才能把结果交回去
（[`research/coreml-gil-completion-path/`](research/coreml-gil-completion-path/README.md)）。
1.5 让 laya 和 laya-typed-decisions 中符合条件的 ANE 请求改走异步 Core ML 执行，避开同步路径
长时间持有 GIL 的问题。运行时会监控这条更快的路径，一旦持续变慢，就退回已知安全的 1.4 同步
路径。在 `execution="workers"`、`device="auto"` 下默认开启，无需再改代码。

![GPU 已经算完，但结果卡在同步 Core ML predict 持有的 GIL 前面；改用异步 Core ML 后结果直接通过，GPU 结果返回从 8.60 降到 0.043 ms；接着是一段异步路径变慢的实测记录，以及受控恢复实验中 1.5 检测到变慢、回退到 1.4 路径并恢复：12 个 episode 全部在 164–414 ms 内恢复](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/release15-social.gif)

前 10 秒是示意动画。画面上每个数字的来源：
[`docs/media/release15-social.md`](docs/media/release15-social.md)。

| 相比 1.4 路径（一台 Apple M4 Max） | laya | laya-typed-decisions |
|---|---:|---:|
| GPU 返回（P50） | 4.28–4.29 → **0.035–0.037 ms** | 8.60 → **0.043 ms** |
| 吞吐量 | **1.042×** | **1.038×** |

*GPU 返回*指从 GPU worker 算完一个请求，到结果送达调用方的时间，不是 GPU 的计算时间。
在 1.4 路径上，已经算完的结果要等 4.3–8.6 ms，几乎全部是在等 GIL。

- **验证。** 154 个生产环境验证 episode（76 个 laya、78 个 typed-decisions，包含突发负载和
  soak 测试）。每个 episode 在 handoff 之后都留在异步路径上，没有 mismatch、路由失败、请求丢失或崩溃。
  这几轮验证都没有出现变慢状态，因此 fallback 从未触发
  （[`val_tables.md`](research/coreml-adaptive-breaker/val_tables.md)）。
- **恢复（另一组独立的受控实验）。** 异步路径已经变慢的 12 个 episode，运行时全部在 40 ms 内
  检测到，并在 164–414 ms 内让延迟回到 1.4 路径的水平
  （[`phase1_tables.md`](research/coreml-adaptive-breaker/phase1_tables.md)）。

[试试 Switchyard](#亲自看看switchyard) ·
[安装](#安装) · [本地 Jev 兼容服务器](#本地-jev-兼容服务器)

## 安装

需要 Apple 芯片和 Python 3.11–3.13：

```bash
pip install 'laya-apple[ane]'   # MLX GPU + the Neural Engine runtime
```

| Extra | 增加的内容 |
|---|---|
| （无） | 仅 MLX GPU 运行时 |
| `ane` | Neural Engine 运行时（coremltools 9.0、pyobjc-framework-CoreML） |
| `convert` | 在这台 Mac 上构建 ANE artifact（torch 2.7.0）；`artifacts fetch` 不需要 |
| `serve` | `laya-apple serve`，本地 Jev 兼容服务器 |

没有安装 `ane` extra，或还没构建 ANE artifact 时，所有计算都在 MLX GPU 上运行。
如需从源码运行或参与开发，请参阅 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 快速上手（30 秒）

```python
from laya_apple import Laya

model = Laya.from_pretrained("convaiinnovations/laya-typed-decisions", device="auto")

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
rt = result.runtime
print(result.answers["urgency"]["choice"])
print(rt.device, rt.routing_reason, f"{rt.latency_ms:.1f} ms")
```

在测试机器上的输出：

```text
high
ane validated_short_single_question_path 11.2 ms
```

- 第一次调用会下载固定版本的 checkpoint，之后即可离线使用（`local_files_only=True`）。
- 没有 ANE artifact 时，同一个请求会改在 MLX 上运行，`routing_reason` 会说明原因。在 Apple M4 Max、
  macOS 26、coremltools 9.0 上，可以下载预构建、并在这台 Mac 上验证的 artifact：
  `laya-apple artifacts fetch laya-typed-decisions`。其他 Mac 则在本地构建并通过 parity 验证
  （需要 `convert` extra）：`laya-apple artifacts build laya-typed-decisions`。
- 概率、其他问题类型和完整 API：[`examples/`](examples/)、[用户指南](docs/guide.md) 和
  [`docs/api.md`](docs/api.md)。

要让 GPU + ANE 并发服务，请使用 `execution="workers"`，请求可以从任意线程提交：

```python
with Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers") as model:
    futures = [model.submit(context=c, questions=q) for c, q in requests]   # thread-safe
```

## 工作原理

Laya 只需一次 forward pass，就能回答一段 context 中的 typed questions（`choice`、`score`、`noul`）。
Mac 上有两个引擎可以运行它，分别擅长不同的请求。

![不超过 128 个 token、只有一个问题且有已验证 artifact 的请求发往 Apple Neural Engine；更长、包含多个问题或未经验证的请求发往 MLX GPU；使用 execution="workers" 时，两个引擎并发处理相互独立的请求](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/architecture.svg)

1. **ANE 执行的正确性。** Core ML 导出跑得快，不代表结果正确。在测试用的 Mac 上，普通的 Core ML
   导出在 ANE 上运行时没有报任何错误，却有多达 85 个决策与上游不一致。ANE artifact 必须在实际
   运行它的 Mac 上通过 parity 检查才会被使用（[正确性](#正确性)）。
2. **自动路由。** Router 在请求运行前就做出决定，并在 `routing_reason` 中记录原因。有负载时，
   它还会比较两个队列的积压情况。
3. **GPU + ANE 并发服务。** 使用 `execution="workers"` 时，GPU 在 worker 进程中运行，ANE 有自己的
   dispatcher，短请求不再排在长请求后面。
4. **自适应 ANE 执行（1.5）。** GPU 与 ANE 同时繁忙的每一段时间（overlap）开始时，先保守地走
   同步路径，之后符合条件的 ANE 请求改用异步 Core ML，已经算完的 GPU 结果就不会被同步路径长时间持有的 GIL 卡住。
   运行时根据每个请求的时间数据判断这条路径是否变慢；持续变慢时，这段时间内剩余的请求会退回
   已知安全的同步路径。设置 `ane_handoff=False` 即可关闭
   （[指南](docs/guide.md#adaptive-ane-execution-in-process-ane-the-default-since-15)）。

| 请求 | 发往 | 原因（在测试用的 Mac 上测量） |
|---|---|---|
| 一个问题、≤ 128 个 token、有已验证的 artifact | **ANE** | 更快：laya-typed-decisions 在 L128 时，ANE 需要 9.9 ms，MLX 需要 12.2 ms（forward P50） |
| Context 较长 | **MLX GPU** | 这时 MLX 更快：L256 为 19.2 ms，L1024 为 71.0 ms |
| 多个问题 | **MLX GPU** | MLX 会批量处理这些问题；ANE 只能逐个运行 |
| 未经验证的 Mac、缺少 artifact，或没有 Core ML | **MLX GPU** | 记录为 `platform_not_validated`、`ane_artifact_unavailable` 或 `ane_runtime_unavailable` |

完整的路由阈值和校准依据：[`docs/support-matrix.md`](docs/support-matrix.md)。

已完成的请求可以通过 `trace=` 输出 `RequestTrace`（路由决策，以及排队、执行和响应的时间）
（[`docs/api.md`](docs/api.md)）。自适应执行依据的也是这些逐请求的计时数据，无需另外传入 `trace=`。
架构：[`docs/architecture.md`](docs/architecture.md)。

## 亲自看看：Switchyard

```bash
uvx laya-apple switchyard
```

![Switchyard：同一份录制好的时刻表分别以仅 GPU 和 GPU + ANE 回放；仅 GPU 时，列车在高峰时段堵在红灯前排队，GPU + ANE 时则顺畅驶入站台；最后是结果卡片：仅 GPU 时 1,422 趟中有 1,407 趟晚点，GPU + ANE 时为 0 趟](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/switchyard-launch.gif)

**每一趟列车都是一次真实的 Laya 决策**（“哪个站台是空的？”）。红灯表示列车正在等模型回答；
100 ms 内没有拿到回答即为晚点。高峰时段会加入突发负载，同时 GPU 上还有耗时较长的后台请求。

| 同一份时刻表，同一个模型 | 仅 MLX GPU | MLX GPU + Neural Engine |
|---|---:|---:|
| 晚点列车 | 1,407–1,408 / 1,422 | **0 / 1,422** |
| P99 决策延迟 | 3,107.7–3,170.6 ms | **54.5–54.7 ms** |
| P99 排队等待 | 3,095.9–3,158.8 ms | **39.6–42.8 ms** |

- 同一台 Apple M4 Max（macOS 26.6.2，`laya-typed-decisions`）上的三次标准运行。每次运行中，
  两个轮次对每一趟列车给出的回答都相同。
- 差距主要来自短决策不必再在 GPU 队列里等待，长请求则继续在 GPU 上运行。
- Benchmark 在 headless 模式下测量；动画只是回放，不属于测量本身。这些数字不能与下方 v1.0
  的结果直接比较（速率、测量边界和 workload 都不同）。

首次下载、设置、方法和原始数据：[`docs/switchyard.md`](docs/switchyard.md) 和
[`benchmarks/switchyard/README.md`](benchmarks/switchyard/README.md)。

## 本地 Jev 兼容服务器

`laya-apple serve` 可以在本地替代 Jev API。它在 loopback 上提供相同的 API
（`POST /v1/systemone`），并在你的 Mac 上用上游 Laya 作答。现有的 Jev 客户端只需把 base URL
设置指向这里，代码无需任何改动。

```bash
pip install 'laya-apple[serve,ane]'
laya-apple serve                                   # http://127.0.0.1:8642
export TYPESAFE_BASE_URL=http://127.0.0.1:8642     # then run your Jev client as usual
```

![终端画面：laya-apple serve 在本地启动；未经修改的 Python typesafe-sdk 0.7.1 正式版通过 TYPESAFE_BASE_URL 指向它，得到回答 fix_code；用 curl 发送同一个请求，可以看到是 laya-apple 用 laya checkpoint 在 Apple Neural Engine 上作答的](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/serve-demo.svg)

- **已用 7 个 Jev 客户端的正式版本实测，客户端未做任何修改**，包括 Python 和 JS SDK，以及两个
  Claude Code 插件（[`integrations/jev-plugins/README.md`](integrations/jev-plugins/README.md)）。
- **回答来自 Laya，而不是 Jev。** 792 个请求全部在 FP16 parity 检查范围内与未经修改的上游
  `laya.serve` 0.3.20 一致（[`benchmarks/serve-compat/`](benchmarks/serve-compat/README.md)）。
  本项目不对与 Jev 相比的准确率做任何声明。

API key、模型、安全性和客户端注意事项：[`docs/serve.md`](docs/serve.md)。

### 较早的 serve benchmark

在 1.3 路径上测量，同一台 Mac 上有一个满载生成的 27B 本地 LLM 时，简短决策的 P99 在异构 `auto`
服务下为 **47.2 ms**，仅用 GPU 时为 **122.3 ms**（一台 M4 Max、一次运行，`--model laya`）。
第 3 次运行使用 laya-apple 1.5.0 的默认设置，`auto` 为 **41.7 ms**，`--device gpu` 为
**79.5 ms**，所有预先登记的标准都通过。自适应执行已开启，但在 3,222 次 Neural Engine forward 中
异步路径一次都没有触发，因此第 3 次运行测到的是 1.5 serve 发布时的行为，而不是异步路径
（[`benchmarks/serve/m4-max-r3/tables.md`](benchmarks/serve/m4-max-r3/tables.md)）。各次运行是不同的测量，只做描述性比较。对 LLM 吞吐量的影响、方法和局限：
[`docs/serve.md`](docs/serve.md#beside-a-local-llm)。

## 正确性

以 PyTorch CPU FP32 上的上游 Laya 为基准，用随附的 golden rows 对比 parity（v1.0）。
每个单元格依次是 hard mismatch 数和最大概率误差。

| 测试用 Mac 上的实现 | laya | laya-multilingual | laya-typed-decisions |
|---|---|---|---|
| 普通 Core ML 导出 · `CPU_AND_NE` | ❌ 12, 0.56 | ❌ 85, 1.0 | ❌ 19, 0.42 |
| 普通 Core ML 导出 · `CPU_AND_GPU` | ✅ 0, 0.0066 | ✅ 0, 0.0059 | ✅ 0, 0.0028 |
| **laya-apple MLX FP16** | ✅ 0, 0.0037 | ✅ 0, 0.0045 | ✅ 0, 0.0017 |
| **laya-apple ANE FP16** | ✅ 0 (1 near-tie), 0.012 | ✅ 0, 0.013 | ✅ 0, 0.0077 |

- **FP16 检查标准：** 概率误差 ≤ 0.02，且 hard mismatch 为 0。Near-tie 翻转会明确列出，不会隐藏。
- **不会悄悄 fallback：** 显式指定 ANE 的请求，要么运行已验证的 artifact，要么直接抛出异常。
- **1.5 的异步路径**在 golden rows 上的输出必须与 coremltools 完全一致。

定义、所有测试过的配置以及 fallback 审计：[`docs/correctness.md`](docs/correctness.md) 和
[`docs/no-silent-fallback.md`](docs/no-silent-fallback.md)。

## 最初的异构服务 benchmark（v1.0）

这些结果证明了 GPU + ANE 服务的价值，是在 1.5 引入自适应执行之前测得的。

| 模型 | 相比仅 GPU 的吞吐量 | 短请求 P99，仅 GPU | 短请求 P99，GPU + ANE |
|---|---:|---:|---:|
| laya | **2.92×** | 1538.0 ms | **108.5 ms** |
| laya-multilingual | **4.34×** | 2052.3 ms | **29.6 ms** |
| laya-typed-decisions | **4.57×** | 1592.9 ms | **79.5 ms** |

一台 Apple M4 Max，一条短请求流和一条长请求流送入同一个 `Laya(execution="workers")`。短请求
P99 在 open-loop 突发负载下从请求到达时开始计时，因此包含排队时间。性能提升来自两个引擎同时
工作，而不是 ANE 本身更快。方法和原始数据：[`benchmarks/v1.0.md`](benchmarks/v1.0.md)。

## 支持的模型和平台

| 模型 | max_len | MLX GPU | ANE bucket（显式指定） | `auto` 使用的 ANE bucket | 自适应 ANE 执行 |
|---|---:|---|---|---|---|
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | 512 | FP16 / FP32，任意长度 | 64, 96, 128 | 64, 96, 128 | 是 |
| [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | 1024 | FP16 / FP32，任意长度 | 64, 96, 128, 256 | 64, 96, 128 | 否（ANE 在 worker 进程中运行） |
| [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 1024 | FP16 / FP32，任意长度 | 64, 96, 128 | 64, 96, 128 | 是 |

自适应 ANE 执行仅在 `execution="workers"`、`device="auto"` 下启用。仅在一台运行 macOS 26.6.2 的 Apple M4 Max 上验证过。在其他 Mac 上，需要先在该机器上构建并校准
ANE artifact（`laya-apple calibrate`），`auto` 才会使用 ANE，在此之前一直走 MLX。预构建的 artifact
（`laya-apple artifacts fetch`）只提供 Apple M4 Max、macOS 26、coremltools 9.0。参见
[`docs/compatibility.md`](docs/compatibility.md)。

<a id="community-benchmarks"></a>

## 社区 benchmark

上面所有 benchmark 都是在 M4 Max 上跑的。[社区矩阵](docs/community-benchmarks.md)单独记录
其他 Mac 的结果，目前已有 M4 Pro、M4 和 M2 Pro（仅 MLX）。一条命令就能把你的 Mac 加进来：

```bash
uv run python scripts/hardware_report.py --quick
```

从 `git clone` 到提交 pull request 的完整步骤，见[指南](docs/community-benchmarks.md#add-your-mac)。

## 复现

上面每个主要数字，都能在 [`docs/reproducibility.md`](docs/reproducibility.md) 中找到对应的
报告、原始数据、命令和环境。1.6 预构建 artifact 的检查记录在
[`benchmarks/prebuilt-artifacts-1.6.0.md`](benchmarks/prebuilt-artifacts-1.6.0.md)。1.5 的相关证据见
[`research/coreml-adaptive-breaker/`](research/coreml-adaptive-breaker/README.md)。
走到 1.5 的整条研究脉络，包括失败的路线，整理在 [`research/README.md`](research/README.md)。

## 局限

- **只有一台测试机器。** 所有 benchmark，包括 1.5 的验证和恢复实验，都来自同一台运行
  macOS 26.6.2 的 Apple M4 Max。我们不假定路由阈值在其他 Apple SoC 上同样成立。
- **自适应执行的 handoff 很保守：** GPU 与 ANE 每次同时繁忙时，最先的一批 ANE 请求会先走
  1.4 路径。
  laya-multilingual 的 ANE 在 worker 进程中运行，不使用这项功能。
- **`laya-apple serve` 中还没有测到 1.5 自适应执行的异步路径。** serve 默认使用自适应执行；
  第 3 次在本地 LLM 旁的 benchmark 开启了它，但 3,222 次 Neural Engine forward 全都走 1.4 的
  Core ML 路径（[`benchmarks/serve/m4-max-r3/tables.md`](benchmarks/serve/m4-max-r3/tables.md)）。
- **长请求和多问题请求留在 GPU 上，** 因为 GPU 处理它们更快。ANE 路径仅支持 batch 1。
- **无法完全隔离。** 并发运行时，每条请求流的 P99 都高于单独运行时。
- **冷启动：** 如果 artifact 目录是全新的，每个模型需要 3–5 分钟编译 Core ML。
  设置 `ane_startup="background"` 时，这段时间会先用 MLX 提供服务。
- **预构建的 artifact 省不掉冷启动。** 下载来的 artifact 第一次加载时仍要在设备上编译：一次实测中
  laya-typed-decisions（bucket 64/96/128）用了 273.5 s，热启动为 2.7 s
  （[`research/coreml-compile-cache/screen.md`](research/coreml-compile-cache/screen.md)）。
- **Core ML 的编译缓存只增不减：** 每个新的 artifact 位置（一次 fetch、一次 import、移动过的缓存）
  每个 bucket 会多出 0.7–1.4 GB。laya-apple 从不清理它（[用户指南](docs/guide.md#artifact-lifecycle)）。
- **预构建的 artifact 只有一个平台 profile**（Apple M4 Max、macOS 26、coremltools 9.0），而且
  只在构建它们的那台机器上用空缓存下载验证过
  （[`benchmarks/prebuilt-artifacts-1.6.0.md`](benchmarks/prebuilt-artifacts-1.6.0.md)），
  没有在第二台机器上验证。
- **没有任何量化 artifact 发布。** W8 研究中只有 laya-typed-decisions L64 `w8-pt` 的结果得到复现
  （延迟为 FP16 的 0.650）；laya 和 laya-multilingual 没有通过 parity 关卡
  （[`research/ane-w8/README.md`](research/ane-w8/README.md)）。
- **MLX 快速路径的提升（约 1–4%）来自筛选测试，** 不是完整的 benchmark
  （[`benchmarks/mlx-fast-path-screen/README.md`](benchmarks/mlx-fast-path-screen/README.md)）。
- **Switchyard 没有平衡轮次顺序：** 使用标准 seed 时，三次运行都是 GPU + ANE 先跑。

各个 benchmark 自身的局限记录在对应的实验文档中，例如选项顺序见
[`docs/correctness.md`](docs/correctness.md#option-order)，serve 和客户端的注意事项见
[`docs/serve.md`](docs/serve.md#limits)，尚未测量的内容见
[`docs/compatibility.md`](docs/compatibility.md#not-measured)。

## 参与贡献

最有价值的第一份贡献，是提交 M4 Max 以外机型的 benchmark 结果（[具体做法](docs/community-benchmarks.md)）。

[`CONTRIBUTING.md`](CONTRIBUTING.md) 介绍了环境搭建、测试分级和 parity 检查；coding agent 请遵循
[`AGENTS.md`](AGENTS.md)。待处理的工作标有 `good first issue`、`help wanted` 和 `research`。

更多：[用户指南](docs/guide.md) · [稳定 API](docs/api.md) · [架构](docs/architecture.md) ·
[变更记录](CHANGELOG.md) · [安全](SECURITY.md)。

采用 Apache-2.0 许可证；见 [`LICENSE`](LICENSE) 和 [`NOTICE`](NOTICE)。模型权重从指定的
Hugging Face revision 下载，本项目不会再分发。这是一个独立项目，并非 Convai Innovations、Apple 或 MLX
的官方版本。
