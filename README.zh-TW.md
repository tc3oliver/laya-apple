# laya-apple

[English](README.md) | **繁體中文** | [简体中文](README.zh-CN.md)

[![CI](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml/badge.svg)](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/laya-apple)](https://pypi.org/project/laya-apple/)
![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

> 本文件為 [`README.md`](README.md) 的翻譯。若內容有出入，以英文版為準。

**經正確性驗證、適用於 Apple 晶片的異質 [Laya](https://github.com/NandhaKishorM/laya)
執行環境：** 同時使用 MLX GPU 與 Apple 神經網路引擎（Neural Engine，ANE）。

## 在 Mac 上以本機後端取代 Jev API

`laya-apple serve` 是 Jev API 的本機替代方案。它在 loopback 上提供 Jev 相容 API
（`POST /v1/systemone`），並以在你的 Mac 上執行的上游 Laya 作答。
只要透過既有 Jev 用戶端的 base URL 設定將它指向這裡即可；用戶端程式碼不需修改。

```bash
pip install 'laya-apple[serve,ane]'
laya-apple serve                                   # http://127.0.0.1:8642
export TYPESAFE_BASE_URL=http://127.0.0.1:8642     # then run your Jev client as usual
```

Jev 用戶端需要某個 API key 才能啟動。除非你在伺服器端設定了 `LAYA_API_KEY`，
否則任何佔位值都可以，例如 `TYPESAFE_API_KEY=local-placeholder`。

![終端機畫面：laya-apple serve 在本機啟動；未經修改的已發布版 Python typesafe-sdk 0.7.1 透過 TYPESAFE_BASE_URL 指向它，得到答案 fix_code；以 curl 送出同一請求，顯示是 laya-apple 以 laya checkpoint 在 Apple 神經網路引擎上作答](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/serve-demo.svg)

- **已在不修改的情況下以 7 個 Jev 用戶端的已發布版本測試**，包括 Python 與 JS SDK，
  以及兩個 Claude Code 外掛。每個用戶端都只透過 base URL 設定指向這裡，並完成了它的請求
  （[`integrations/jev-plugins/README.md`](integrations/jev-plugins/README.md)）。
- **答案來自 Laya，而不是 Jev。** 所保證的是對上游 Laya 的忠實度：792 個請求中有 792 個
  在 FP16 parity 閘門內與未經修改的上游 `laya.serve` 0.3.20 一致，其中 365 個由神經網路引擎作答
  （[`benchmarks/serve-compat/`](benchmarks/serve-compat/README.md)）。Laya 是另一個、
  規模更小的模型，因此不宣稱具有 Jev 等級的準確度；若用戶端的閾值是針對 Jev 調校的，
  它可能會更常走到 fallback 路徑。
- laya-apple 與 TypeSafe 或 Jev 並無任何關係。上圖是一次錄製的執行
  （[`scripts/capture_serve_demo.py`](scripts/capture_serve_demo.py)）。

模型、API、安全性與限制：[`docs/serve.md`](docs/serve.md)。

## 有 Mac 嗎？試試 Switchyard。

```bash
uvx laya-apple switchyard
```

![Switchyard：同一份錄製的時刻表分別以僅 GPU 與 GPU + ANE 重播；僅 GPU 時，列車在尖峰時段停在紅燈號誌前排隊，GPU + ANE 時則順暢駛入月台；接著是結果卡：僅 GPU 有 1,422 班中的 1,407 班誤點，GPU + ANE 為 0 班](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/switchyard-launch.gif)

**每一班列車都是一次真實的 Laya 決策。** 同一份時刻表，僅 GPU 對比 GPU + ANE，
於 Apple M4 Max 上執行：

- **僅 GPU：** 1,422 班中有 1,407–1,408 班誤點，P99 決策延遲約 3.1 s。
- **GPU + ANE：** 1,422 班中 0 班誤點，P99 約 54.6 ms。

遊戲與基準測試的對應關係：

- **列車** → 一個請求：「哪個月台是空的？」
- **紅燈號誌** → 列車正在等待模型的答案
- **道岔** → 決策：轉轍器扳向模型選擇的月台
- **誤點** → 100 ms 內沒有得到答案
- **尖峰時段** → 突發負載，且 GPU 上有長時間執行的背景請求

基準測試先以 headless 方式執行，之後瀏覽器再重播錄製的請求軌跡；動畫本身不屬於量測的一部分。

### 同一份時刻表。同一個模型。GPU + 神經網路引擎。

| | 僅 MLX GPU | MLX GPU + 神經網路引擎 |
|---|---:|---:|
| 誤點列車 | 1,407–1,408 / 1,422 | **0 / 1,422** |
| P99 決策延遲 | 3,107.7–3,170.6 ms | **54.5–54.7 ms** |
| P99 佇列等待 | 3,095.9–3,158.8 ms | **39.6–42.8 ms** |

在同一台 Apple M4 Max（macOS 26.6.2，`laya-typed-decisions`）上的三次標準執行；
其他 Mac 的結果會不同。差距主要並非來自單一請求在神經網路引擎上的速度：而是短決策不必再於
GPU 佇列中等待，長請求則持續在 GPU 上執行。方法與原始資料：
[`benchmarks/switchyard/README.md`](benchmarks/switchyard/README.md)。

## 為什麼有這個專案

Laya 以一次前向傳遞（forward pass）回答關於某段上下文的型別化問題（`choice`、`score`、`noul`）。
在 Mac 上有兩個引擎可以執行它，各有所長。

1. **正確的 ANE 執行。** 快速的 Core ML 匯出不一定正確。在受測的 Mac 上，一般的 Core ML
   匯出在神經網路引擎上執行時沒有任何錯誤，卻讓多達 85 個決策與上游 PyTorch 不同。
   laya-apple 只有在 ANE artifact 於實際使用它的機器上通過 parity 閘門後才會採用
   （[`docs/correctness.md`](docs/correctness.md)）。
2. **自動路由。** 短的、經驗證的單一問題請求送往 ANE；長的或多問題的請求送往 MLX GPU。
   路由器在請求執行前做出決定，並記錄原因。
3. **GPU + ANE 並行服務。** 兩個引擎同時服務彼此獨立的請求，因此短請求不必再排在長請求後面。

## 安裝

Apple 晶片，Python 3.11–3.13：

```bash
pip install laya-apple
```

選用的 extras：

```bash
pip install "laya-apple[ane]"       # + the Neural Engine runtime (coremltools 9.0)
pip install "laya-apple[convert]"   # + building ANE artifacts on this Mac (torch 2.7.0)
pip install "laya-apple[serve]"     # + laya-apple serve, the local Jev-compatible server
laya-apple artifacts build laya-typed-decisions   # optional: build + parity-validate ANE artifacts here (~5 min)
```

若沒有安裝 `ane` extra，或沒有已建置的 artifact，所有運算都在 MLX GPU 上執行。
若要從原始碼執行或開發 laya-apple，請參閱 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 快速開始（30 秒）

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

在受測機器上：

```text
high {'low': 0.1713, 'medium': 0.3358, 'high': 0.4929}
coreml ane validated_short_single_question_path 11.2 ms
```

- 第一次呼叫會下載固定版本的 checkpoint，之後即可離線使用（`local_files_only=True`）。
- 沒有 ANE artifact 時，同一個請求會在 MLX 上執行，並由 `routing_reason` 說明原因。
- 更多內容：[`examples/`](examples/)（`basic.py`、`auto_routing.py`、
  `heterogeneous_serving.py`）與[使用指南](docs/guide.md)。

## 自動路由如何運作

![不超過 128 個 token、只有一個問題且有經驗證 artifact 的請求送往 Apple 神經網路引擎；較長、多問題或未經驗證的請求送往 MLX GPU；使用 execution="workers" 時，兩個引擎並行服務彼此獨立的請求](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/architecture.svg)

| 請求 | 送往 | 原因（於受測 Mac 上量測） |
|---|---|---|
| 一個問題、≤ 128 個 token、有經驗證的 artifact | **ANE** | 較快：laya-typed-decisions 在 L128 時，ANE 需 9.9 ms，MLX 需 12.2 ms（forward P50） |
| 較長的上下文 | **MLX GPU** | 此時 MLX 較快：L256 為 19.2 ms，L1024 為 71.0 ms |
| 多個問題 | **MLX GPU** | MLX 會批次處理這些問題；ANE 則逐一執行 |
| 未經驗證的 Mac、缺少 artifact，或沒有 Core ML | **MLX GPU** | 記錄為 `platform_not_validated`、`ane_artifact_unavailable` 或 `ane_runtime_unavailable` |

**正式環境的閾值比量測到的交叉點更保守。**
laya-multilingual 在剛好 256 個 token 時於 ANE 上略快（8.3 ms 對 8.6 ms），但這個 bucket
仍只在明確指定時使用，因為它在前一個 bucket（128 個 token）並未勝過 MLX。
每個結果都帶有 `routing_reason`。

閾值如何推導：[`docs/support-matrix.md`](docs/support-matrix.md)。各元件如何組合：
[`docs/architecture.md`](docs/architecture.md)。

## Switchyard 基準測試

`laya-apple switchyard` 以 headless 方式量測凍結的 `switchyard-v1` 工作負載，然後寫出
`result.json`、`trace.jsonl` 與一個自成一體的 `replay.html`
（[`docs/switchyard.md`](docs/switchyard.md)）。

- **工作負載。** Seed 11、60 s、名目 40 req/s 的突發到達：共 2,410 個請求，其中
  1,422 個是單一問題的列車決策，期限為 100 ms。其餘是中等長度、長的與多問題的請求，用來加重 GPU 負載。
- **回合。** `gpu_only`（`device="gpu"`）以及在神經網路引擎就緒時的 `hybrid`
  （`device="auto"`），兩者都透過 `Laya(execution="workers")` 在完全相同的時刻表上執行。
- **量測。** 決策延遲從每班列車的預定到達時間算到得到答案為止，包含排隊時間。未達期限比例
  分別以 25、50、100、250 與 500 ms 報告，因此結果不依賴遊戲的 100 ms 期限。
- **檢查。** 實際上沒有使用神經網路引擎的 hybrid 回合會被標記為不可比較。在 M4 Max 的
  測試中，兩個回合對每一班列車都給出相同答案，且沒有任何列車被誤路由。
- **時間。** 第一次執行會下載固定版本的 checkpoint（約 800 MB）；之後一次標準執行約需
  2–3 分鐘。若沒有已建置的神經網路引擎 artifact，它會執行 `gpu_only` 並印出設定指令
  `uvx --from "laya-apple[convert]" laya-apple switchyard --setup-ane`。

**限制。** 僅一台機器（Apple M4 Max）。回合順序由 seed 決定，因此三次執行都是 `hybrid`
先跑（`design.counterbalance` 為 `"none"`）。這些數字無法與下方 v1.0 開放迴路（open-loop）表格比較：
兩者的速率、量測邊界與工作負載都不同（[`docs/switchyard.md`](docs/switchyard.md)）。

各次執行的結果、跨執行的離散程度與重現指令：
[`benchmarks/switchyard/README.md`](benchmarks/switchyard/README.md)。原始結果：
[`benchmarks/switchyard/v1-m4-max/raw/`](benchmarks/switchyard/v1-m4-max/raw/)。

## GPU + ANE 異質服務

![同一個模型分別在 MLX GPU 與 Apple 神經網路引擎上玩 Lane Runner，得到相同分數；接著同一批突發請求分別以僅 GPU 與 GPU + ANE 服務：僅 GPU 時，短請求在 GPU 佇列中等待長達 1.5 s，而 GPU + ANE 時，路由器將它們送往 ANE，請求一到就執行](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/heterogeneous-serving.gif)

本節所依據的 v1.0 基準測試的動畫版。laya-apple 將有經驗證 artifact 的短、單一問題請求送往
Apple 神經網路引擎，較長的請求則繼續在 MLX GPU 上執行。動畫首先重播一場在 M4 Max 上錄製的
[Lane Runner](https://github.com/tc3oliver/laya-playground-apple) 遊戲：同一個模型分別在各裝置上
執行、分開錄製，一次一個請求。接著依據基準測試到達序列的逐請求軌跡，重播 laya-typed-decisions
突發工作負載中的一段突發。其 P99 值即下表中已發布的 v1.0 數字；每個數字的來源都在
[`docs/media/README.md`](docs/media/README.md)。

![相對於僅 GPU 服務的混合工作負載吞吐量：laya 從 41.9 提升到 122.5 req/s（2.92×），laya-multilingual 從 55.7 提升到 241.8 req/s（4.34×），laya-typed-decisions 從 24.0 提升到 109.6 req/s（4.57×）](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/hero-throughput.svg)

增益來自同時使用兩個引擎，而不是 ANE 本身的延遲。這是在一台 Apple M4 Max、macOS 26.6.2 上的
v1.0 基準測試：一條短請求串流與一條長請求串流，經由同一個 `Laya(execution="workers")` 實例。
其他 Mac 不在此基準測試範圍內；它們的結果收錄在[社群矩陣](#community-benchmarks)。方法與原始資料在
[`benchmarks/v1.0.md`](benchmarks/v1.0.md)。

```python
with Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers") as model:
    futures = [model.submit(context=c, questions=q) for c, q in requests]   # thread-safe
```

- GPU 在一個 worker 行程中執行，ANE 則在自己的 dispatcher 上執行。
- 每個請求只在一個裝置上執行，由路由器選擇。
- 在負載下，路由器也會比較各佇列的積壓量。

若要了解某個請求為什麼慢，可傳入一個回呼函式。每完成一個請求，它就會收到一個 `RequestTrace`：
路由器做決定時依據的佇列快照、它選擇的裝置與原因，以及從提交、排隊、服務到回應的單調時間戳記。

```python
def on_trace(trace):
    print(trace.request_id, trace.target, trace.routing_reason, trace.queue_ms, trace.e2e_ms)

model = Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers", trace=on_trace)
```

使用預設的 `trace=None` 時不會記錄任何內容。回呼函式在該裝置的 dispatcher 執行緒上執行，
因此請保持輕量。

**開放迴路突發到達下的短請求 P99**，從到達開始量測並包含排隊時間
（v1.0，兩者使用相同的到達序列）：

| 模型 | 僅 GPU | GPU + ANE |
|---|---:|---:|
| laya（提供負載 46.2 req/s） | 1538.0 ms | 108.5 ms |
| laya-multilingual（83.8 req/s） | 2052.3 ms | 29.6 ms |
| laya-typed-decisions（35.8 req/s） | 1592.9 ms | 79.5 ms |

單一短請求在 ANE 上並沒有快很多（例如 9.9 對 12.2 ms）。增益來自同時使用兩個引擎。

<a id="community-benchmarks"></a>

## 社群基準測試

上面的每一項基準測試都只涵蓋 M4 Max。[社群矩陣](docs/community-benchmarks.md)
以獨立執行的方式收錄其他 Mac 的結果，不與上面的數字混在一起。目前已有 M4 Pro、M4 與
M2 Pro（僅 MLX）的外部結果。如果你有其他 Mac，一個指令就能加入。
**不需要修改任何程式碼。**

- M1 / M2 → [#1](https://github.com/tc3oliver/laya-apple/issues/1)
- M3 Max → [#2](https://github.com/tc3oliver/laya-apple/issues/2)
- M5 或任何其他 Mac → [加入你的 Mac](docs/community-benchmarks.md#add-your-mac)

```bash
uv run python scripts/hardware_report.py --quick
```

上述 issue 與指南列有完整步驟，從 `git clone` 到 pull request，約需 10 分鐘。

## 正確性

以 PyTorch CPU FP32 上的上游 Laya 為基準、針對隨附 golden rows 的 parity 結果（v1.0）。
每一格依序列出硬性不一致數，以及最大機率誤差。

| 受測 Mac 上的實作 | laya | laya-multilingual | laya-typed-decisions |
|---|---|---|---|
| 一般 Core ML 匯出 · `CPU_AND_NE` | ❌ 12, 0.56 | ❌ 85, 1.0 | ❌ 19, 0.42 |
| 一般 Core ML 匯出 · `CPU_AND_GPU` | ✅ 0, 0.0066 | ✅ 0, 0.0059 | ✅ 0, 0.0028 |
| **laya-apple MLX FP16** | ✅ 0, 0.0037 | ✅ 0, 0.0045 | ✅ 0, 0.0017 |
| **laya-apple ANE FP16** | ✅ 0 (1 near-tie), 0.012 | ✅ 0, 0.013 | ✅ 0, 0.0077 |

- **FP16 閘門：** 機率誤差 ≤ 0.02，且硬性不一致為 0。
- **Near-tie 翻轉**（上游前兩名的差距 < 0.04）會列出，而不是隱藏。
- **明確指定 ANE 的請求絕不 fallback。** 它們要嘛執行經驗證的 artifact，要嘛拋出例外；
  而且每個載入的 artifact 也會與 `CPU_ONLY` 比較計時，以抓出悄悄落到 CPU 的情形。
- 定義、所有測試過的組態以及 fallback 稽核，請見
  [`docs/correctness.md`](docs/correctness.md) 與
  [`docs/no-silent-fallback.md`](docs/no-silent-fallback.md)。

## 支援的模型與平台

| 模型 | max_len | MLX GPU | ANE bucket（明確指定） | `auto` 使用的 ANE bucket |
|---|---:|---|---|---|
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | 512 | FP16 / FP32，任意長度 | 64, 96, 128 | 64, 96, 128 |
| [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | 1024 | FP16 / FP32，任意長度 | 64, 96, 128, 256 | 64, 96, 128 |
| [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 1024 | FP16 / FP32，任意長度 | 64, 96, 128 | 64, 96, 128 |

**已測試：**
- Apple M4 Max、macOS 26.6.2、MLX 0.32.2、coremltools 9.0；
- Python 3.11–3.13。

**其他 Apple 晶片：**
- 預期 MLX 可以運作。
- 在該機器上建置並校準 artifact 之前（`laya-apple calibrate`），`auto` 會一直使用 MLX。

請參閱 [`docs/compatibility.md`](docs/compatibility.md) 與
[`docs/community-benchmarks.md`](docs/community-benchmarks.md) 中的社群矩陣。

## 重現

上面每個主要數字都可追溯到 [`docs/reproducibility.md`](docs/reproducibility.md) 中的
報告、原始資料、指令與環境。完整的 v1.0 測試套件比較了 PyTorch CPU/MPS、一般 Core ML 匯出、
MLX 與 laya-apple，收錄在 [`benchmarks/v1.0.md`](benchmarks/v1.0.md)。在你自己的 Mac 上快速檢查：

```bash
uv run python scripts/hardware_report.py --quick
```

## 貢獻

最有幫助的第一個貢獻，是提供一台非 M4 Max 的 Mac 的基準測試結果：執行上面的指令，
並以 `hardware-results/` 開一個 PR（[做法](docs/community-benchmarks.md)）。

- [`CONTRIBUTING.md`](CONTRIBUTING.md) 說明環境設定、測試層級（某項變更實際需要哪些測試）、
  parity 檢查與後端變更。
- 待處理的工作標有 `good first issue`、`help wanted` 與 `research` 標籤。
- 程式碼代理（coding agent）：[`AGENTS.md`](AGENTS.md) 列有本儲存庫的規則。

## 限制

- **只有一台測試機器。** 所有基準測試都來自一台 macOS 26.6.2 上的 Apple M4 Max。
  不假設路由閾值在其他 Apple SoC 上同樣成立。
- **長上下文留在 MLX，** 因為 MLX 在那裡較快。ANE 路徑僅支援 batch 1。
- **隔離只是部分的。** 在並行情況下，每條串流的 P99 都高於其單獨執行時的值。
- 在全新的 artifact 位置上**冷啟動**，每個模型需要 3–5 分鐘的 Core ML 編譯。
  `ane_startup="background"` 會在此期間以 MLX 提供服務。
- **`choice` 決策可能受選項順序影響。** 這源自上游 Laya，laya-apple 完全重現了此行為
  （[`research/option-order/`](research/option-order/)）。
- **尚未量測：** 能耗、量化 artifact 與跨 SoC 驗證。
- **`laya-apple serve` 以 Laya 作答，而不是 Jev。** 只量測了對上游 Laya 的忠實度，
  沒有量測 Jev 等級的準確度。閾值針對 Jev 調校的用戶端可能更常走到 fallback 路徑；
  受測的七個用戶端中有兩個出現這種情況
  （[`integrations/jev-plugins/README.md`](integrations/jev-plugins/README.md)）。
- **`laya-apple serve` 不做任何效能宣稱。** 它的延遲、吞吐量，以及它對共用 GPU 的本機 LLM
  的影響都尚未量測。
- **用戶端相容性只測試過一次，** 於 2026-09-25，使用列出的用戶端版本，在受測的 M4 Max 上進行。
  之後的用戶端版本可能會改變它送出或接受的內容。
- **`serve --model auto` 只在英文與多語模型之間路由。** 上游選擇性啟用的 typed-decisions
  工作流程偵測（`LAYA_AUTO_TASK`）與呼叫端的語言提示並未實作
  （[`docs/serve.md`](docs/serve.md#limits)）。

## 更多

- 使用指南：[`docs/guide.md`](docs/guide.md)。
- 本機 Jev 相容伺服器：[`docs/serve.md`](docs/serve.md)。
- 穩定 API：[`docs/api.md`](docs/api.md)。
- 架構：[`docs/architecture.md`](docs/architecture.md)。
- 變更紀錄：[`CHANGELOG.md`](CHANGELOG.md)。
- 安全性：[`SECURITY.md`](SECURITY.md)。

Apache-2.0；請見 [`LICENSE`](LICENSE) 與 [`NOTICE`](NOTICE)。模型權重從其固定的 Hugging Face
revision 下載，不會重新散布。這是一個獨立專案，並非 Convai Innovations、Apple 或 MLX 的官方發布。
