# laya-apple

[English](README.md) | **繁體中文** | [简体中文](README.zh-CN.md)

[![CI](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml/badge.svg)](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/laya-apple)](https://pypi.org/project/laya-apple/)
![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

> 本文件是 [`README.md`](README.md) 的中文版。內容如有出入，以英文版為準。

**經過正確性驗證，可同時使用 MLX GPU 與 Apple Neural Engine（ANE）的
[Laya](https://github.com/NandhaKishorM/laya) runtime，適用於 Apple silicon。**

## 本機 LLM 忙碌時，簡短決策依然快速

`laya-apple serve` 在你的 Mac 上用上游 Laya 回答 Jev 用戶端送來的決策請求。預設的
`--device auto` 會把簡短的單一問題決策交給 Apple Neural Engine 執行，不必在 GPU 上排在本機 LLM
後面等待。

```bash
pip install 'laya-apple[serve,ane]'
laya-apple serve                                   # http://127.0.0.1:8642
export TYPESAFE_BASE_URL=http://127.0.0.1:8642     # then run your Jev client as usual
```

![laya-apple serve 與忙碌的本機 LLM 同時執行，在 Apple M4 Max 上以 Qwen3.8-27B-oQ4e-mtp 跑一次的結果：LLM 生成時，簡短決策的 P99 在 serve auto 下是 47.2 ms，--device gpu 則是 122.3 ms（LLM 閒置時分別是 43.2 與 55.9 ms）；LLM 吞吐量單獨執行時 42.2 tok/s，搭配 serve --device gpu 時 40.0，搭配 serve auto 時 40.5](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/serve-llm-load.svg)

以下是 `laya-apple serve --model laya` 在一台 Apple M4 Max 上跑一次的結果：27B 本機 LLM
（oMLX 上的 `Qwen3.8-27B-oQ4e-mtp`）滿載生成，同時以每秒 8 個的速率送出決策請求，其中 80%
是簡短的單一問題決策（[`benchmarks/serve/`](benchmarks/serve/README.md)，第 2 次執行）。
這些數字是用 `--model laya` 量測的；預設的 `--model auto` 沒有量測。

- **LLM 忙碌時，簡短決策的 P99：`auto` 為 47.2 ms，`--device gpu` 為 122.3 ms。** LLM 閒置時，
  `auto` 為 43.2 ms，`--device gpu` 為 55.9 ms。
- **LLM 的吞吐量從單獨執行時的 42.2 tok/s，搭配 serve `auto` 時下降 4.0%，搭配 `--device gpu`
  時下降 5.3%。** 兩者相差 1.31 個百分點，只比同一次執行中 LLM 本身各 window 之間的波動
  （1.28 個百分點）略高。
- **負載下結果依然正確：** 7,728 個決策中 0 個 hard mismatch、0 個錯誤；每個決策都與同一個伺服器
  在 LLM 閒置時的回答比對。
- `auto` 的簡短決策約有 4% 依設計改送 GPU：當 Neural Engine 的 backlog 等待時間較長時就會這樣做。
  上面的 P99 已包含這些請求。
- 這個 benchmark 的第 1 次執行依它自己的有效性檢查判定無效，不從中得出任何結論。

API 與 Jev 相容；laya-apple 與 TypeSafe 或 Jev 沒有任何關係，回答來自 Laya，不是 Jev。

## 在 Mac 上用本機 backend 取代 Jev API

`laya-apple serve` 可以在本機取代 Jev API。它在 loopback 上提供 Jev 相容的 API
（`POST /v1/systemone`），並在你的 Mac 上用上游 Laya 產生回答。
既有的 Jev 用戶端只要改 base URL 設定指到這裡就能用，程式碼完全不用改。

```bash
pip install 'laya-apple[serve,ane]'
laya-apple serve                                   # http://127.0.0.1:8642
export TYPESAFE_BASE_URL=http://127.0.0.1:8642     # then run your Jev client as usual
```

Jev 用戶端啟動時需要 API key。只要伺服器端沒有設定 `LAYA_API_KEY`，填任何佔位值都可以，
例如 `TYPESAFE_API_KEY=local-placeholder`。

![終端機畫面：laya-apple serve 在本機啟動；未經修改的 Python typesafe-sdk 0.7.1 正式版透過 TYPESAFE_BASE_URL 指向它，得到回答 fix_code；用 curl 送出同一個請求，可以看到是 laya-apple 以 laya checkpoint 在 Apple Neural Engine 上回答的](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/serve-demo.svg)

- **已用 7 個 Jev 用戶端的正式版本實測，完全沒有修改用戶端**，包括 Python 與 JS SDK，
  以及兩個 Claude Code plugin。每個用戶端都只改了 base URL 設定，所有請求都順利完成
  （[`integrations/jev-plugins/README.md`](integrations/jev-plugins/README.md)）。
- **回答來自 Laya，不是 Jev。** 這裡只保證結果與上游 Laya 一致：792 個請求全部在 FP16
  parity 檢查範圍內與未經修改的上游 `laya.serve` 0.3.20 一致，其中 365 個由 ANE 回答
  （[`benchmarks/serve-compat/`](benchmarks/serve-compat/README.md)）。Laya 是另一個較小的模型，
  因此不宣稱有 Jev 等級的準確度；如果用戶端的門檻值是針對 Jev 調校的，可能會更常觸發 fallback。
- laya-apple 與 TypeSafe 或 Jev 沒有任何關係。上圖是一次實際執行的錄製結果
  （[`scripts/capture_serve_demo.py`](scripts/capture_serve_demo.py)）。

模型、API、安全性與限制：[`docs/serve.md`](docs/serve.md)。

## 手邊有 Mac？試試 Switchyard

```bash
uvx laya-apple switchyard
```

![Switchyard：同一份錄製好的時刻表分別以僅 GPU 與 GPU + ANE 重播；僅 GPU 時，列車在尖峰時段卡在紅燈前排隊，GPU + ANE 時則順利駛入月台；最後是結果卡：僅 GPU 有 1,422 班中的 1,407 班誤點，GPU + ANE 則是 0 班](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/switchyard-launch.gif)

**每一班列車都是一次真實的 Laya 決策。** 以下是在 Apple M4 Max 上，以同一份時刻表比較僅 GPU
與 GPU + ANE 的結果：

- **僅 GPU：** 1,422 班中有 1,407–1,408 班誤點，P99 決策延遲約 3.1 s。
- **GPU + ANE：** 1,422 班中 0 班誤點，P99 約 54.6 ms。

遊戲畫面與 benchmark 的對應：

- **列車** → 一個請求：「哪個月台是空的？」
- **紅燈** → 列車正在等模型回答
- **道岔** → 決策：轉轍器扳向模型選的月台
- **誤點** → 100 ms 內沒有拿到回答
- **尖峰時段** → 突發負載，同時 GPU 上還有長時間執行的背景請求

Benchmark 會先以 headless 模式跑完，瀏覽器再重播錄下來的 request trace；動畫本身不算在量測裡。

### 同一份時刻表，同一個模型，GPU + Neural Engine

| | 僅 MLX GPU | MLX GPU + Neural Engine |
|---|---:|---:|
| 誤點列車 | 1,407–1,408 / 1,422 | **0 / 1,422** |
| P99 決策延遲 | 3,107.7–3,170.6 ms | **54.5–54.7 ms** |
| P99 佇列等待 | 3,095.9–3,158.8 ms | **39.6–42.8 ms** |

以上是同一台 Apple M4 Max（macOS 26.6.2，`laya-typed-decisions`）上的三次標準執行；
換成其他 Mac 結果會不同。差異主要不是因為單一請求在 ANE 上跑得更快，而是短請求不必再等待
GPU 佇列，長請求則繼續留在 GPU 上執行。方法與原始資料：
[`benchmarks/switchyard/README.md`](benchmarks/switchyard/README.md)。

## 為什麼做這個專案

Laya 只要一次 forward pass，就能回答一段 context 中的 typed questions（`choice`、`score`、`noul`）。
Mac 上有兩個引擎可以跑它，各有擅長的地方。

1. **ANE 執行正確性。** Core ML 匯出跑得快，不代表結果一定正確。在我們測試的 Mac 上，一般的
   Core ML 匯出在 ANE 上執行時完全沒有報錯，卻有多達 85 個決策與上游 PyTorch 不同。
   ANE artifact 只有在實際執行的 Mac 上通過 parity 檢查後才會使用
   （[`docs/correctness.md`](docs/correctness.md)）。
2. **自動路由。** 較短且已驗證的單一問題請求送往 ANE；較長或包含多個問題的請求送往 MLX GPU。
   Router 在請求執行前就決定去向，並記錄原因。
3. **GPU + ANE 同時服務。** 兩個引擎同時處理彼此獨立的請求，短請求不必再排在長請求後面。

## 安裝

需要 Apple silicon 與 Python 3.11–3.13：

```bash
pip install laya-apple
```

可選 extras：

```bash
pip install "laya-apple[ane]"       # + the Neural Engine runtime (coremltools 9.0)
pip install "laya-apple[convert]"   # + building ANE artifacts on this Mac (torch 2.7.0)
pip install "laya-apple[serve]"     # + laya-apple serve, the local Jev-compatible server
laya-apple artifacts build laya-typed-decisions   # optional: build + parity-validate ANE artifacts here (~5 min)
```

沒有安裝 `ane` extra，或還沒有建置 artifact 時，所有運算都會在 MLX GPU 上執行。
如果要從原始碼執行或參與開發，請參考 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

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

在測試機上的輸出：

```text
high {'low': 0.1713, 'medium': 0.3358, 'high': 0.4929}
coreml ane validated_short_single_question_path 11.2 ms
```

- 第一次呼叫會下載指定版本的 checkpoint，之後就能離線使用（`local_files_only=True`）。
- 沒有 ANE artifact 時，同一個請求會改在 MLX 上執行，`routing_reason` 會說明原因。
- 更多範例：[`examples/`](examples/)（`basic.py`、`auto_routing.py`、
  `heterogeneous_serving.py`）與[使用指南](docs/guide.md)。

## 自動路由怎麼運作

![不超過 128 個 token、只有一個問題且有已驗證 artifact 的請求送往 Apple Neural Engine；較長、包含多個問題或未經驗證的請求送往 MLX GPU；使用 execution="workers" 時，兩個引擎同時處理彼此獨立的請求](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/architecture.svg)

| 請求 | 送往 | 原因（在測試用的 Mac 上量測） |
|---|---|---|
| 一個問題、≤ 128 個 token、有已驗證的 artifact | **ANE** | 比較快：laya-typed-decisions 在 L128 時，ANE 要 9.9 ms，MLX 要 12.2 ms（forward P50） |
| Context 較長 | **MLX GPU** | 這時 MLX 比較快：L256 為 19.2 ms，L1024 為 71.0 ms |
| 多個問題 | **MLX GPU** | MLX 會把問題批次處理；ANE 只能一個一個跑 |
| 未經驗證的 Mac、缺少 artifact，或沒有 Core ML | **MLX GPU** | 記錄為 `platform_not_validated`、`ane_artifact_unavailable` 或 `ane_runtime_unavailable` |

**正式使用的門檻比實測的交叉點更保守。**
laya-multilingual 在剛好 256 個 token 時，ANE 略快一些（8.3 ms 對 8.6 ms），但這個 bucket
仍然只在明確指定時才使用，因為在前一個 bucket（128 個 token）它並沒有贏過 MLX。
每個結果都會附上 `routing_reason`。

門檻怎麼推導：[`docs/support-matrix.md`](docs/support-matrix.md)。整體架構：
[`docs/architecture.md`](docs/architecture.md)。

## Switchyard benchmark

`laya-apple switchyard` 會以 headless 模式量測固定的 `switchyard-v1` workload，接著輸出
`result.json`、`trace.jsonl` 與一個獨立的 `replay.html`
（[`docs/switchyard.md`](docs/switchyard.md)）。

- **Workload。** Seed 11、60 s、突發到達，nominal rate 為 40 req/s：共 2,410 個請求，其中
  1,422 個是單一問題的列車決策，deadline 為 100 ms。其餘是中等長度、長的與多問題請求，用來讓 GPU 保持忙碌。
- **回合。** `gpu_only`（`device="gpu"`），以及 Neural Engine 就緒時的 `hybrid`
  （`device="auto"`），兩者都透過 `Laya(execution="workers")` 跑完全相同的時刻表。
- **量測。** 決策延遲從每班列車的預定到達時間算到拿到回答為止，包含排隊時間。Deadline miss rate
  分別以 25、50、100、250 與 500 ms 為門檻統計，所以結論不受遊戲設定的 100 ms deadline 影響。
- **檢查。** 如果 hybrid 回合實際上沒有用到 Neural Engine，會被標為不可比較。在 M4 Max 的測試中，
  兩個回合對每一班列車給出的回答都相同，也沒有任何列車被送錯路線。
- **時間。** 第一次執行會下載指定版本的 checkpoint（約 800 MB）；之後一次標準執行約需
  2–3 分鐘。如果還沒建置 Neural Engine artifact，它只會跑 `gpu_only`，並印出設定指令
  `uvx --from "laya-apple[convert]" laya-apple switchyard --setup-ane`。

**限制。** 只在一台機器（Apple M4 Max）上測過。回合順序由 seed 決定，所以三次執行都是 `hybrid`
先跑（`design.counterbalance` 為 `"none"`）。這些數字不能和下方 v1.0 open-loop 表格直接比較：
兩者的速率、量測範圍與 workload 都不一樣（[`docs/switchyard.md`](docs/switchyard.md)）。

每次執行的結果、各次執行之間的差異與重現指令：
[`benchmarks/switchyard/README.md`](benchmarks/switchyard/README.md)。原始結果：
[`benchmarks/switchyard/v1-m4-max/raw/`](benchmarks/switchyard/v1-m4-max/raw/)。

## GPU + ANE 異質服務

![同一個模型分別在 MLX GPU 與 Apple Neural Engine 上玩 Lane Runner，拿到相同分數；接著同一批突發請求分別以僅 GPU 與 GPU + ANE 處理：僅 GPU 時，短請求在 GPU 佇列中最多要等 1.5 s，GPU + ANE 時，router 把它們送往 ANE，請求一到就執行](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/heterogeneous-serving.gif)

這段動畫呈現的是本節使用的 v1.0 benchmark。laya-apple 會把有已驗證 artifact 的短單一問題請求送往
Apple Neural Engine，較長的請求則留在 MLX GPU 上執行。動畫先重播一場在 M4 Max 上錄製的
[Lane Runner](https://github.com/tc3oliver/laya-playground-apple) 遊戲：同一個模型分別在兩個裝置上
各錄一次，每次只處理一個請求。接著根據 benchmark 到達序列的逐請求 trace，重播
laya-typed-decisions 突發 workload 中的一段突發。動畫中的 P99 就是下表中已發布的 v1.0 數字；
每個數字的來源都列在 [`docs/media/README.md`](docs/media/README.md)。

![與僅 GPU 相比的混合 workload 吞吐量：laya 從 41.9 提升到 122.5 req/s（2.92×），laya-multilingual 從 55.7 提升到 241.8 req/s（4.34×），laya-typed-decisions 從 24.0 提升到 109.6 req/s（4.57×）](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/hero-throughput.svg)

效能提升來自同時使用兩個引擎，而不是 ANE 本身的延遲比較低。這是在一台 Apple M4 Max、macOS 26.6.2
上跑的 v1.0 benchmark：一條短請求串流與一條長請求串流，送進同一個 `Laya(execution="workers")` instance。
其他 Mac 不在這個 benchmark 的範圍內，相關結果請看[社群矩陣](#社群-benchmark)。方法與原始資料在
[`benchmarks/v1.0.md`](benchmarks/v1.0.md)。

```python
with Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers") as model:
    futures = [model.submit(context=c, questions=q) for c, q in requests]   # thread-safe
```

- GPU 在獨立的 worker process 中執行，ANE 則有自己的 dispatcher。
- 每個請求只會在一個裝置上執行，由 router 決定。
- 有負載時，router 也會比較各佇列的積壓狀況。

**GPU + Neural Engine 服務會隨系統負載調整（1.5）。**
- **Mac 空閒時：** laya 與 laya-typed-decisions 會讓 Neural Engine 走較快的非同步路徑，不再拖慢
  GPU 回傳結果。
- **Mac 變得繁忙時：** 自動退回 1.4 的執行路徑。
- **在一台 M4 Max 上、與 1.4 路徑相比：** 154 個 production episode 全程留在快速路徑：
  - GPU 結果回傳從 4.3 ms（laya）與 8.6 ms（typed-decisions）降到 0.04 ms；
  - throughput 為 1.4 路徑的 1.04 倍；
  - P99 低 0.2–0.5 ms。
- **恢復：** 快速路徑確實變慢的情況下（記錄到的 12 個 episode），延遲在 0.42 s 內回到 1.4 的水準。

預設開啟，不需改程式。來源：
[`research/coreml-adaptive-breaker/`](research/coreml-adaptive-breaker/README.md)；運作方式：
[`docs/guide.md`](docs/guide.md#adaptive-ane-execution-in-process-ane-the-default-since-15)。

想知道某個請求為什麼慢，可以傳入 callback。每完成一個請求，它就會收到一個 `RequestTrace`：
router 做決定時看到的佇列快照、選擇的裝置與原因，以及從提交、排隊、執行到回應的 monotonic timestamps。

```python
def on_trace(trace):
    print(trace.request_id, trace.target, trace.routing_reason, trace.queue_ms, trace.e2e_ms)

model = Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers", trace=on_trace)
```

預設的 `trace=None` 不會記錄任何東西。Callback 會在該裝置的 dispatcher thread 上執行，
所以請盡量保持輕量。

**Open-loop 突發到達下的短請求 P99**，從請求到達開始計時，包含排隊時間
（v1.0，兩種設定使用相同的到達序列）：

| 模型 | 僅 GPU | GPU + ANE |
|---|---:|---:|
| laya（offered load 46.2 req/s） | 1538.0 ms | 108.5 ms |
| laya-multilingual（83.8 req/s） | 2052.3 ms | 29.6 ms |
| laya-typed-decisions（35.8 req/s） | 1592.9 ms | 79.5 ms |

單一短請求在 ANE 上並沒有快很多（例如 9.9 ms 對 12.2 ms）。效能提升來自同時使用兩個引擎。

<a id="community-benchmarks"></a>

## 社群 benchmark

上面所有 benchmark 都只在 M4 Max 上跑過。[社群矩陣](docs/community-benchmarks.md)
收錄其他 Mac 的結果，每台分開記錄，不會和上面的數字混在一起。目前已有 M4 Pro、M4 與
M2 Pro（僅 MLX）的外部結果。如果你有其他型號的 Mac，一個指令就能加入。
**完全不需要改程式碼。**

- M1 / M2 → [#1](https://github.com/tc3oliver/laya-apple/issues/1)
- M3 Max → [#2](https://github.com/tc3oliver/laya-apple/issues/2)
- M5 或其他任何 Mac → [加入你的 Mac](docs/community-benchmarks.md#add-your-mac)

```bash
uv run python scripts/hardware_report.py --quick
```

上面的 issue 與指南有完整步驟，從 `git clone` 到開 pull request，大約 10 分鐘。

## 正確性

以 PyTorch CPU FP32 上的上游 Laya 為基準，用隨附的 golden rows 比對 parity（v1.0）。
每一格依序是 hard mismatch 數與最大機率誤差。

| 在測試用 Mac 上的實作 | laya | laya-multilingual | laya-typed-decisions |
|---|---|---|---|
| 一般 Core ML 匯出 · `CPU_AND_NE` | ❌ 12, 0.56 | ❌ 85, 1.0 | ❌ 19, 0.42 |
| 一般 Core ML 匯出 · `CPU_AND_GPU` | ✅ 0, 0.0066 | ✅ 0, 0.0059 | ✅ 0, 0.0028 |
| **laya-apple MLX FP16** | ✅ 0, 0.0037 | ✅ 0, 0.0045 | ✅ 0, 0.0017 |
| **laya-apple ANE FP16** | ✅ 0 (1 near-tie), 0.012 | ✅ 0, 0.013 | ✅ 0, 0.0077 |

- **FP16 檢查標準：** 機率誤差 ≤ 0.02，且 hard mismatch 為 0。
- **Near-tie 翻轉**（上游前兩名的差距 < 0.04）會明確列出，不會隱藏。
- **明確指定 ANE 的請求絕不 fallback。** 要嘛執行已驗證的 artifact，要嘛直接拋出例外；
  每個載入的 artifact 也會和 `CPU_ONLY` 比較執行時間，以抓出悄悄跑到 CPU 上的情況。
- 定義、所有測過的設定與 fallback 稽核，請見
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

**其他 Apple silicon：**
- MLX 預期可以正常運作。
- 在該機器上建置並校準 artifact（`laya-apple calibrate`）之前，`auto` 會一律使用 MLX。

詳見 [`docs/compatibility.md`](docs/compatibility.md) 與
[`docs/community-benchmarks.md`](docs/community-benchmarks.md) 中的社群矩陣。

## 重現結果

上面每個主要數字，都能在 [`docs/reproducibility.md`](docs/reproducibility.md) 找到對應的
報告、原始資料、指令與環境。完整的 v1.0 測試比較了 PyTorch CPU/MPS、一般 Core ML 匯出、
MLX 與 laya-apple，收錄在 [`benchmarks/v1.0.md`](benchmarks/v1.0.md)。想在自己的 Mac 上快速測一下：

```bash
uv run python scripts/hardware_report.py --quick
```

## 參與貢獻

如果想開始參與貢獻，最有幫助的是提供非 M4 Max 的 Mac 的 benchmark 結果：執行上面的指令，
再把 `hardware-results/` 開成一個 PR（[做法](docs/community-benchmarks.md)）。

- [`CONTRIBUTING.md`](CONTRIBUTING.md) 說明環境設定、測試分級（每種變更實際需要跑哪些測試）、
  parity 檢查與 backend 修改。
- 待處理的工作會標上 `good first issue`、`help wanted` 與 `research` 標籤。
- Coding agent：repo 規則請見 [`AGENTS.md`](AGENTS.md)。

## 限制

- **只有一台測試機。** 所有 benchmark 都來自同一台 macOS 26.6.2 的 Apple M4 Max。
  我們不假設路由門檻在其他 Apple SoC 上也成立。
- **長 context 留在 MLX 上執行，** 因為 MLX 在這種情況下比較快。ANE 路徑只支援 batch 1。
- **並行的請求串流之間無法完全隔離。** 同時執行時，每條串流的 P99 都比單獨執行時高。
- **自適應 ANE 執行只在一台 M4 Max 上量測過。**
  - 154 個驗證 episode 都沒有出現變慢狀態，所以退回後的恢復數據來自同一台機器上另外 12 個
    episode 的實驗。
  - 每段 GPU + ANE 重疊的前 64 個 Neural Engine 請求走 1.4 路徑。
  - laya-multilingual 的 Neural Engine 在 worker process 中執行，不使用此功能。
- **Cold start 需要編譯：** 如果 artifact 目錄是全新的，cold start 時每個模型需要 3–5 分鐘編譯 Core ML。
  設定 `ane_startup="background"` 時，這段期間會先用 MLX 提供服務。
- **`choice` 決策可能受選項順序影響。** 這是上游 Laya 本身的行為，laya-apple 完全照樣重現
  （[`research/option-order/`](research/option-order/)）。
- **尚未量測：** 能耗、量化 artifact 與跨 SoC 驗證。
- **`laya-apple serve` 用 Laya 回答，不是 Jev。** 只量測了結果與上游 Laya 是否一致，
  沒有量測 Jev 等級的準確度。門檻值針對 Jev 調校的用戶端可能會更常觸發 fallback；
  測試的七個用戶端中有兩個出現這種情況
  （[`integrations/jev-plugins/README.md`](integrations/jev-plugins/README.md)）。
- **`laya-apple serve` 的效能只在一種設定下跑過一次：** 一台 M4 Max、一個 LLM（oMLX 上的
  `Qwen3.8-27B-oQ4e-mtp`，以 decode 為主、prompt 很短）、`--model laya`、每秒送出 8 個決策請求
  （[`benchmarks/serve/`](benchmarks/serve/README.md)）。還沒量測的有：其他 LLM 伺服器與模型、
  以 prefill 為主的 LLM 負載、其他請求速率、serve 的最大決策吞吐量（第 2 次執行固定以每秒 8 個
  請求送出），以及其他 checkpoint（`laya-typed-decisions`、`--model laya-multilingual`）與
  `--model auto`。
- **serve `auto` 仍會拖慢 LLM 的吞吐量：** 那次執行中下降 4.0%，只比 `--device gpu` 少 1.31
  個百分點，而 LLM 單獨執行時各 window 之間的波動就有 1.28 個百分點。
- **多問題決策一律在 GPU 上執行，** LLM 忙碌時它們的 P99 會變高：`auto` 下為 117.1 ms，
  閒置時為 84.1 ms。
- **用戶端相容性只測過一次：** 2026-09-25，使用列出的用戶端版本，在測試用的 M4 Max 上進行。
  之後的用戶端版本可能會改變它送出或接受的內容。
- **`serve --model auto` 只會在英文與多語模型之間路由。** 上游需手動啟用的 typed-decisions
  workflow 偵測（`LAYA_AUTO_TASK`）以及呼叫端的語言提示都還沒實作
  （[`docs/serve.md`](docs/serve.md#limits)）。

## 更多資訊

- 使用指南：[`docs/guide.md`](docs/guide.md)。
- 本機 Jev 相容伺服器：[`docs/serve.md`](docs/serve.md)。
- 穩定 API：[`docs/api.md`](docs/api.md)。
- 架構：[`docs/architecture.md`](docs/architecture.md)。
- 變更紀錄：[`CHANGELOG.md`](CHANGELOG.md)。
- 安全性：[`SECURITY.md`](SECURITY.md)。

採用 Apache-2.0 授權；請見 [`LICENSE`](LICENSE) 與 [`NOTICE`](NOTICE)。模型權重從指定的
Hugging Face revision 下載，本專案不會重新散布。這是獨立專案，並非 Convai Innovations、Apple 或 MLX
的官方發布。
