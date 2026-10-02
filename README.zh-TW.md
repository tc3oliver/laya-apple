# laya-apple

[English](README.md) | **繁體中文** | [简体中文](README.zh-CN.md)

[![CI](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml/badge.svg)](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/laya-apple)](https://pypi.org/project/laya-apple/)
![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

> 本文件是 [`README.md`](README.md) 的中文版。內容如有出入，以英文版為準。

**讓 Mac 的 GPU 和 Neural Engine 同時跑 [Laya](https://github.com/NandhaKishorM/laya)，而且
Neural Engine 只在結果跟上游一致時才上場。**

Laya 讀一段 context，一次 forward pass 就能回答多個有型別的問題（`choice`、`score`、`noul`）。
laya-apple 在 Mac 的兩個引擎上跑上游 Laya，每個請求自動挑一個引擎。

1.6.1 更新：`artifacts fetch` 更可靠了；M4 Max 在 macOS 27 上建好 artifact 後，`auto` 也會用 ANE
（[release notes](docs/releases/v1.6.1.md)）。

## 為什麼要用

- **ANE 的結果要對，不只是快。** 在測試機上，直接用 Core ML 匯出的模型在 Neural Engine（ANE）上跑，
  一個錯誤都沒報，卻有多達 85 個決策跟上游不一樣。所以 ANE artifact 一定要先在你的 Mac 上通過 parity 檢查才會
  啟用（[正確性](#正確性)）。
- **短決策不用再排在長任務後面。** 有驗證過的 artifact 時，單一問題的請求交給 ANE；較長或多個問題的
  請求留在 GPU。開啟 `execution="workers"` 後，兩個引擎同時接請求。數據都來自同一台 Apple M4 Max；
  跑一下 `uvx laya-apple switchyard` 就能親眼看到
  （[Switchyard](#親眼看看switchyard)、[GPU + ANE benchmark](#gpu--ane-benchmark)）。
- **在本機取代 Jev API。** `laya-apple serve` 在你的 Mac 上用上游 Laya 回答既有的 Jev 用戶端，
  用戶端完全不用改（[本機 Jev 相容伺服器](#本機-jev-相容伺服器)）。
- **每個路由決策都查得到原因。** 結果裡會記錄裝置和 `routing_reason`。明確指定 ANE 時，不是跑驗證過的
  artifact，就是直接丟出例外，絕不偷偷 fallback（[運作方式](#運作方式)）。

## 安裝

需要 Apple silicon 與 Python 3.11–3.13：

```bash
pip install 'laya-apple[ane]'   # MLX GPU + the Neural Engine runtime
```

| Extra | 加入的功能 |
|---|---|
| （無） | 只有 MLX GPU runtime |
| `ane` | Neural Engine runtime（coremltools 9.0、pyobjc-framework-CoreML） |
| `convert` | 在這台 Mac 上建置 ANE artifact（torch 2.7.0）；`artifacts fetch` 不需要 |
| `serve` | `laya-apple serve`，本機 Jev 相容伺服器 |

沒有裝 `ane` extra，或還沒建置 ANE artifact 時，所有運算都在 MLX GPU 上執行。
如果要從原始碼執行或參與開發，請參考 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 快速開始

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

在測試機上的輸出：

```text
high
ane validated_short_single_question_path 11.2 ms
```

- 第一次呼叫會下載固定版本的 checkpoint，之後就能離線使用（`local_files_only=True`）。
- 沒有 ANE artifact 時，同一個請求會改在 MLX 上跑，`routing_reason` 會說明原因。
- Apple M4 Max + macOS 26 + coremltools 9.0：直接下載預先建置的 artifact，會在你的 Mac 上驗證：
  `laya-apple artifacts fetch laya-typed-decisions`。
- 其他環境就自己建置一個（需要 `convert` extra）：`laya-apple artifacts build laya-typed-decisions`。
- 機率、其他問題類型與完整 API：[`examples/`](examples/)、[使用指南](docs/guide.md) 與
  [`docs/api.md`](docs/api.md)。

要讓 GPU + ANE 同時服務，請使用 `execution="workers"`，請求可以從任何執行緒送出：

```python
with Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers") as model:
    futures = [model.submit(context=c, questions=q) for c, q in requests]   # thread-safe
```

### 先取得經過驗證的 ANE artifact，再依語言路由（1.6）

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

- **依語言路由。** `Laya.from_pretrained("auto")` 會回傳一個 `LayaRouter`，它同時載入 `laya` 與
  `laya-multilingual`，依每個請求 context 的語言挑選其中一個，判斷方式與 `laya-apple serve --model auto`
  用的是同一個函式。挑選的理由記錄在 `RuntimeInfo.model_routing`（[`docs/api.md`](docs/api.md)）。
- **預先建置的 ANE artifact，在你的 Mac 上驗證。** `laya-apple artifacts fetch MODEL` 從
  [`tc3oliver/laya-apple-artifacts`](https://huggingface.co/tc3oliver/laya-apple-artifacts)
  下載預先建置的 artifact，再交給會做驗證的 `artifacts import` 註冊：先比對索引裡的 SHA-256，檢查
  manifest、平台 profile 與 compute plan，再跑完整的 FP16 parity 關卡和 placement probe，全部在你的
  機器上完成。下載的 artifact 跟本機建置的一視同仁，該過的檢查一個都不少。fetch 固定讀取這個
  repository 中驗證過的 commit，不會讀可變動的 `main`。有了它就不用在本機建置，也不必裝 PyTorch 和
  `convert` extra。
- **`predict` 照舊；`predict_shortlist` 需自行開啟。** 標籤很多的 `choice` 問題可以用
  `predict_shortlist(..., embed_fn, k=20)`：先留下與請求最相近的 `k` 個標籤，再跑一次 `predict`。
  `predict` 本身沒有改變（[`examples/auto_fetch_shortlist.py`](examples/auto_fetch_shortlist.py)）。

限制：
- **預先建置的 artifact 目前只有一個平台 profile：Apple M4 Max、macOS 26、coremltools 9.0。**
  其他 Mac 都會得到 `ArtifactMissingError`，錯誤訊息會附上建置指令，照舊在本機建置
  （`laya-apple artifacts build MODEL`，需要 `convert` extra）。
- **已發布的 artifact 經過一次空快取下載檢查：** 在該 profile 上，全部 10 組 model/bucket 的
  fetch、verify、parity 與 placement 都通過
  （[`benchmarks/prebuilt-artifacts-1.6.0.md`](benchmarks/prebuilt-artifacts-1.6.0.md)）。
  這項檢查是在建置 artifact 的同一台機器上、以空的快取執行。每一台接收端機器在註冊 artifact 前，
  都會再做一次完整性、平台、parity 與 placement 檢查。
- **Core ML 第一次載入模型時仍要在裝置上編譯：** 約 4.5 分鐘。在一次實測中，laya-typed-decisions
  （bucket 64/96/128）在新位置冷啟動花了 273.5 s，暖啟動只要 2.7 s
  （[`research/coreml-compile-cache/screen.md`](research/coreml-compile-cache/screen.md)）。
  下載 artifact 無法省掉這段時間。設定 `ane_startup="background"` 時，這段期間會先用 MLX 服務。

1.6 其他較小的變動：MLX 快速路徑（token-id 快取預設開啟；`mx.compile` 與依長度分組的
batching 需自行開啟；量測範圍見[限制](#限制)），以及沒有納入正式版的 W8 ANE 研究（見[限制](#限制)）。Release notes：
[`docs/releases/v1.6.0.md`](docs/releases/v1.6.0.md)。

## GPU + ANE benchmark

除了社群結果，這一節的數字都來自同一台 Apple M4 Max。其他 Mac 的結果另外整理在
[社群 benchmark](#community-benchmarks)。

### 親眼看看：Switchyard

```bash
uvx laya-apple switchyard
```

![Switchyard：同一份錄製好的時刻表分別以僅 GPU 與 GPU + ANE 重播；僅 GPU 時，列車在尖峰時段卡在紅燈前排隊，GPU + ANE 時則順利駛入月台；最後是結果卡：僅 GPU 有 1,422 班中的 1,407 班誤點，GPU + ANE 則是 0 班](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/switchyard-launch.gif)

**每一班列車都是一次真實的 Laya 決策**（「哪個月台是空的？」）。紅燈代表列車正在等模型回答；
100 ms 內沒拿到回答就算誤點。尖峰時段會加入突發負載，同時 GPU 上還有長時間的背景請求。

| 同一份時刻表，同一個模型 | 僅 MLX GPU | MLX GPU + Neural Engine |
|---|---:|---:|
| 誤點列車 | 1,407–1,408 / 1,422 | **0 / 1,422** |
| P99 決策延遲 | 3,107.7–3,170.6 ms | **54.5–54.7 ms** |
| P99 佇列等待 | 3,095.9–3,158.8 ms | **39.6–42.8 ms** |

- 同一台 Apple M4 Max（macOS 26.6.2，`laya-typed-decisions`）上的三次標準執行。每次執行中，
  兩個回合對每一班列車給的回答都一樣。
- 差距大多來自短決策不必再等 GPU 佇列，長請求則繼續在 GPU 上跑。
- Benchmark 在 headless 模式下量測；動畫只是重播，不是量測本身。這些數字不能和下面的 v1.0
  結果直接比較（速率、量測範圍與 workload 都不同）。

第一次下載、設定、方法與原始資料：[`docs/switchyard.md`](docs/switchyard.md) 與
[`benchmarks/switchyard/README.md`](benchmarks/switchyard/README.md)。

### 最初的異質服務 benchmark（v1.0）

這些結果證明了 GPU + ANE 服務的價值，是在 1.5 推出自適應執行之前量測的。

| 模型 | 與僅 GPU 相比的吞吐量 | 短請求 P99，僅 GPU | 短請求 P99，GPU + ANE |
|---|---:|---:|---:|
| laya | **2.92×** | 1538.0 ms | **108.5 ms** |
| laya-multilingual | **4.34×** | 2052.3 ms | **29.6 ms** |
| laya-typed-decisions | **4.57×** | 1592.9 ms | **79.5 ms** |

一台 Apple M4 Max，一條短請求與一條長請求串流送進同一個 `Laya(execution="workers")`。短請求
P99 在 open-loop 突發負載下從請求到達開始計時，所以包含排隊時間。效能提升來自同時使用兩個引擎，
而不是 ANE 本身比較快。方法與原始資料：[`benchmarks/v1.0.md`](benchmarks/v1.0.md)。

### 自適應 ANE 執行（1.5）

ANE 跑在同一個 process 的執行緒上時，同步的 Core ML 呼叫在每次預測中有很大一部分時間都握著 GIL，
所以一個 GPU 已經算完的請求，要等那次呼叫結束才能把結果交回來
（[`research/coreml-gil-completion-path/`](research/coreml-gil-completion-path/README.md)）。
1.5 讓 laya 與 laya-typed-decisions 中符合條件的 ANE 請求改走非同步的 Core ML 執行；一旦持續變慢，
runtime 就退回已知安全的 1.4 同步路徑。在 `execution="workers"`、`device="auto"` 下預設開啟，
不需要再改程式。

![GPU 已經算完，但結果卡在同步 Core ML predict 持有的 GIL 前面；改用非同步 Core ML 後結果直接通過，GPU 結果回傳從 8.60 降到 0.043 ms；接著是一段非同步路徑變慢的實測紀錄，以及受控恢復實驗中 1.5 偵測到變慢、退回 1.4 路徑並恢復：12 個 episode 全部在 164–414 ms 內恢復](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/media/release15-social.gif)

前 10 秒是示意動畫。畫面上每個數字的來源：
[`docs/media/release15-social.md`](docs/media/release15-social.md)。

| 相較於 1.4 路徑（一台 Apple M4 Max） | laya | laya-typed-decisions |
|---|---:|---:|
| GPU 回傳（P50） | 4.28–4.29 → **0.035–0.037 ms** | 8.60 → **0.043 ms** |
| 吞吐量 | **1.042×** | **1.038×** |

*GPU 回傳*指的是從 GPU worker 算完一個請求，到結果交到呼叫端手上的時間，不是 GPU 的運算時間。
在 1.4 路徑上，已經算完的結果要等 4.3–8.6 ms，幾乎全都是在等 GIL。

- **驗證。** 154 個正式環境驗證 episode（76 個 laya、78 個 typed-decisions，包含突發負載與
  soak 測試）。每個 episode 在 handoff 之後都留在非同步路徑，沒有 mismatch、路由失敗、遺失請求或當機。
  這幾輪驗證都沒有出現變慢狀態，因此 fallback 從未觸發
  （[`val_tables.md`](research/coreml-adaptive-breaker/val_tables.md)）。
- **恢復（另一組獨立的受控實驗）。** 非同步路徑已經變慢的 12 個 episode，runtime 全都在 40 ms
  內偵測到，並在 164–414 ms 內讓延遲回到 1.4 路徑的水準
  （[`phase1_tables.md`](research/coreml-adaptive-breaker/phase1_tables.md)）。

完整脈絡，從 GIL 的診斷到 adaptive breaker：
[`research/coreml-gil-completion-path/`](research/coreml-gil-completion-path/README.md)、
[`research/coreml-adaptive-breaker/`](research/coreml-adaptive-breaker/README.md) 與
[1.5 release notes](docs/releases/v1.5.0.md)。

<a id="community-benchmarks"></a>

### 社群 benchmark

本 README 中所有 benchmark 都是在 M4 Max 上跑的。[社群矩陣](docs/community-benchmarks.md)分開記錄
其他 Mac 的結果，目前已有 M4 Pro、M4 與 M2 Pro（僅 MLX）。一個指令就能加入你的 Mac：

```bash
uv run python scripts/hardware_report.py --quick
```

從 `git clone` 到開 pull request 的完整步驟，請見[指南](docs/community-benchmarks.md#add-your-mac)。

## 本機 Jev 相容伺服器

`laya-apple serve` 可以在本機取代 Jev API。它在 loopback 上提供相同的 API
（`POST /v1/systemone`），並在你的 Mac 上用上游 Laya 回答。既有的 Jev 用戶端只要把 base URL
設定指到這裡，程式碼完全不用改。

```bash
pip install 'laya-apple[serve,ane]'
laya-apple serve                                   # http://127.0.0.1:8642
export TYPESAFE_BASE_URL=http://127.0.0.1:8642     # then run your Jev client as usual
```

![終端機畫面：laya-apple serve 在本機啟動；未經修改的 Python typesafe-sdk 0.7.1 正式版透過 TYPESAFE_BASE_URL 指向它，得到回答 fix_code；用 curl 送出同一個請求，可以看到是 laya-apple 以 laya checkpoint 在 Apple Neural Engine 上回答的](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/serve-demo.svg)

- **已用 7 個 Jev 用戶端的正式版本實測，用戶端完全沒改**，包括 Python 與 JS SDK，以及兩個
  Claude Code plugin（[`integrations/jev-plugins/README.md`](integrations/jev-plugins/README.md)）。
- **回答來自 Laya，不是 Jev。** 792 個請求全部在 FP16 parity 檢查範圍內與未經修改的上游
  `laya.serve` 0.3.20 一致（[`benchmarks/serve-compat/`](benchmarks/serve-compat/README.md)）。
  至於準確度比 Jev 高還是低，這裡不做任何比較。

API key、模型、安全性與用戶端注意事項：[`docs/serve.md`](docs/serve.md)。

### 較早的 serve benchmark

在 1.3 路徑上量測，同一台 Mac 上有一個滿載生成的 27B 本機 LLM 時，簡短決策的 P99 在異質 `auto`
服務下為 **47.2 ms**，只用 GPU 時為 **122.3 ms**（一台 M4 Max、一次執行，`--model laya`）。
第 3 次執行使用 laya-apple 1.5.0 的預設值，`auto` 為 **41.7 ms**，`--device gpu` 為
**79.5 ms**，所有預先登記的標準都通過。自適應執行有開啟，但 3,222 次 Neural Engine forward 中
非同步路徑一次都沒有觸發，所以第 3 次執行量到的是 1.5 正式版 serve 的實際行為，而不是非同步路徑
（[`benchmarks/serve/m4-max-r3/tables.md`](benchmarks/serve/m4-max-r3/tables.md)）。
這幾次是各自獨立的量測，彼此只做描述性對照。LLM 吞吐量的影響、方法與限制：
[`docs/serve.md`](docs/serve.md#beside-a-local-llm)。

## 運作方式

Mac 上有兩個引擎可以跑 Laya，各自擅長不同的請求。

![不超過 128 個 token、只有一個問題且有已驗證 artifact 的請求送往 Apple Neural Engine；較長、包含多個問題或未經驗證的請求送往 MLX GPU；使用 execution="workers" 時，兩個引擎同時處理彼此獨立的請求](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/architecture.svg)

1. **ANE 的結果要對。** Core ML 匯出跑得快，不代表結果正確。ANE artifact 必須在實際執行的
   那台 Mac 上通過 parity 檢查才會使用（[正確性](#正確性)）。
2. **自動路由。** Router 在請求執行前就決定去向，並把原因記在 `routing_reason`。有負載時，
   它也會比較兩個佇列的 backlog。
3. **GPU + ANE 同時服務。** 使用 `execution="workers"` 時，GPU 在 worker process 裡執行，ANE 則有
   自己的 dispatcher，短請求不必再排在長請求後面。
4. **自適應 ANE 執行（1.5）。** GPU 與 ANE 同時忙碌的每一段期間（overlap）開始時，會先保守地
   走同步路徑，之後符合條件的 ANE 請求改用非同步 Core ML，已經算完的 GPU 結果就不會被同步路徑長時間持有的 GIL 卡住。
   Runtime 靠每個請求的時間資料判斷這條路徑有沒有變慢；持續變慢時，這段期間剩下的請求會退回
   已知安全的同步路徑。設定 `ane_handoff=False` 即可關閉
   （[指南](docs/guide.md#adaptive-ane-execution-in-process-ane-the-default-since-15)）。

| 請求 | 送往 | 原因（在測試用的 Mac 上量測） |
|---|---|---|
| 一個問題、≤ 128 個 token、有已驗證的 artifact | **ANE** | 比較快：laya-typed-decisions 在 L128 時，ANE 要 9.9 ms，MLX 要 12.2 ms（forward P50） |
| Context 較長 | **MLX GPU** | 這時 MLX 比較快：L256 為 19.2 ms，L1024 為 71.0 ms |
| 多個問題 | **MLX GPU** | MLX 會把問題批次處理；ANE 只能一個一個跑 |
| 未經驗證的 Mac、缺少 artifact，或沒有 Core ML | **MLX GPU** | 記錄為 `platform_not_validated`、`ane_artifact_unavailable` 或 `ane_runtime_unavailable` |

完整的路由門檻與校準依據：[`docs/support-matrix.md`](docs/support-matrix.md)。

完成的請求可以透過 `trace=` 回報 `RequestTrace`（路由決策，以及排隊、執行與回應的時間）
（[`docs/api.md`](docs/api.md)）。自適應執行看的也是這些每個請求的計時資料，不必另外傳入 `trace=`。
架構：[`docs/architecture.md`](docs/architecture.md)。

### 支援的模型與平台

| 模型 | max_len | MLX GPU | ANE bucket（明確指定） | `auto` 使用的 ANE bucket | 自適應 ANE 執行 |
|---|---:|---|---|---|---|
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | 512 | FP16 / FP32，任意長度 | 64, 96, 128 | 64, 96, 128 | 是 |
| [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | 1024 | FP16 / FP32，任意長度 | 64, 96, 128, 256 | 64, 96, 128 | 否（ANE 在 worker process 中執行） |
| [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 1024 | FP16 / FP32，任意長度 | 64, 96, 128 | 64, 96, 128 | 是 |

自適應 ANE 執行只在 `execution="workers"`、`device="auto"` 下啟用，目前只在一台 macOS 26.6.2 的
Apple M4 Max 上驗證過。內建路由 profile 支援 Apple M4 Max + coremltools 9.0，macOS 26.6.2 和 27.0
都有：只要建置好 artifact，`auto` 就會用 ANE。macOS 27 的 profile 適用所有 27.x，但實際只測過 27.0。
macOS 27 上只量測了建置、parity 與路由；自適應執行、`serve`
與發行 benchmark 都沒有量測（[`benchmarks/routing-macos27/`](benchmarks/routing-macos27/README.md)）。
在其他 Mac 上，要先在那台機器建置並校準 ANE artifact（`laya-apple calibrate`），`auto` 才會使用 ANE，
在那之前都走 MLX。預先建置的 artifact
（`laya-apple artifacts fetch`）只提供 Apple M4 Max、macOS 26、coremltools 9.0。詳見
[`docs/compatibility.md`](docs/compatibility.md)。

## 正確性

以 PyTorch CPU FP32 上的上游 Laya 為基準，用隨附的 golden rows 比對 parity（v1.0）。
每一格依序是 hard mismatch 數與最大機率誤差。

| 在測試用 Mac 上的實作 | laya | laya-multilingual | laya-typed-decisions |
|---|---|---|---|
| 一般 Core ML 匯出 · `CPU_AND_NE` | ❌ 12, 0.56 | ❌ 85, 1.0 | ❌ 19, 0.42 |
| 一般 Core ML 匯出 · `CPU_AND_GPU` | ✅ 0, 0.0066 | ✅ 0, 0.0059 | ✅ 0, 0.0028 |
| **laya-apple MLX FP16** | ✅ 0, 0.0037 | ✅ 0, 0.0045 | ✅ 0, 0.0017 |
| **laya-apple ANE FP16** | ✅ 0 (1 near-tie), 0.012 | ✅ 0, 0.013 | ✅ 0, 0.0077 |

- **FP16 檢查標準：** 機率誤差 ≤ 0.02，且 hard mismatch 為 0。Near-tie 翻轉會明確列出，不會隱藏。
- **不會悄悄 fallback：** 明確指定 ANE 的請求，要嘛跑已驗證的 artifact，要嘛直接拋出例外。
- **1.5 的非同步路徑**在 golden rows 上的輸出必須和 coremltools 完全相同。

定義、所有測過的設定與 fallback 稽核：[`docs/correctness.md`](docs/correctness.md) 與
[`docs/no-silent-fallback.md`](docs/no-silent-fallback.md)。

## 限制

- **只有一台測試機。** 所有 benchmark，包括 1.5 的驗證與恢復實驗，都來自同一台 Apple M4 Max：
  macOS 26.6.2，以及 macOS 27.0 上的建置、parity 與路由。我們不假設路由門檻在其他 Apple SoC 上也成立。
- **自適應執行的 handoff 很保守：** GPU 與 ANE 每次同時忙碌時，最先的一批 ANE 請求會先走
  1.4 路徑。
  laya-multilingual 的 ANE 在 worker process 中執行，不使用這項功能。
- **`laya-apple serve` 裡還沒有量測到 1.5 自適應執行的非同步路徑。** serve 預設使用自適應執行；
  第 3 次在本機 LLM 旁的 benchmark 有開啟它，但 3,222 次 Neural Engine forward 全都走 1.4 的
  Core ML 路徑（[`benchmarks/serve/m4-max-r3/tables.md`](benchmarks/serve/m4-max-r3/tables.md)）。
- **長請求與多問題請求留在 GPU，** 因為 GPU 處理它們比較快。ANE 路徑只支援 batch 1。
- **無法完全隔離。** 同時執行時，每條串流的 P99 都比單獨執行時高。
- **冷啟動：** 如果 artifact 目錄是全新的，每個模型需要 3–5 分鐘編譯 Core ML。
  設定 `ane_startup="background"` 時，這段期間會先用 MLX 服務。
- **預先建置的 artifact 省不掉冷啟動。** 下載來的 artifact 第一次載入時仍要在裝置上編譯：一次實測中
  laya-typed-decisions（bucket 64/96/128）花了 273.5 s，暖啟動為 2.7 s
  （[`research/coreml-compile-cache/screen.md`](research/coreml-compile-cache/screen.md)）。
- **Core ML 的編譯快取只增不減：** 每個新的 artifact 位置（一次 fetch、一次 import、搬移過的快取）
  每個 bucket 會多出 0.7–1.4 GB。laya-apple 從不清除它（[使用指南](docs/guide.md#artifact-lifecycle)）。
- **預先建置的 artifact 只有一個平台 profile**（Apple M4 Max、macOS 26、coremltools 9.0），而且
  只在建置它們的那台機器上以空快取下載驗證過
  （[`benchmarks/prebuilt-artifacts-1.6.0.md`](benchmarks/prebuilt-artifacts-1.6.0.md)），
  目前還沒有在第二台同 profile 的機器上做獨立驗證。發布前建議做，但不是必要條件
  （[`docs/publishing.md`](docs/publishing.md)）；每一台接收端機器仍會在註冊前驗證每個 artifact。
- **目前沒有釋出任何量化 artifact。** W8 研究中只有 laya-typed-decisions L64 `w8-pt` 的結果獲得重現
  （延遲為 FP16 的 0.650）；laya 與 laya-multilingual 沒有通過 parity 關卡
  （[`research/ane-w8/README.md`](research/ane-w8/README.md)）。
- **MLX 快速路徑的提升（約 1–4%）來自篩選測試，** 不是完整的 benchmark
  （[`benchmarks/mlx-fast-path-screen/README.md`](benchmarks/mlx-fast-path-screen/README.md)）。
- **Switchyard 沒有平衡回合順序：** 使用標準 seed 時，三次執行都是 GPU + ANE 先跑。

各個 benchmark 自己的限制，記錄在對應的實驗文件中，例如選項順序見
[`docs/correctness.md`](docs/correctness.md#option-order)，serve 與用戶端的注意事項見
[`docs/serve.md`](docs/serve.md#limits)，尚未量測的項目見
[`docs/compatibility.md`](docs/compatibility.md#not-measured)。

## 研究、版本與重現

- **重現結果。** 上面每個主要數字，都能在 [`docs/reproducibility.md`](docs/reproducibility.md)
  找到對應的報告、原始資料、指令與環境。1.6 預先建置 artifact 的檢查記錄在
  [`benchmarks/prebuilt-artifacts-1.6.0.md`](benchmarks/prebuilt-artifacts-1.6.0.md)。1.5 的相關證據收錄在
  [`research/coreml-adaptive-breaker/`](research/coreml-adaptive-breaker/README.md)。
- **研究。** 1.5 與 1.6 版本背後的研究脈絡，包括失敗的路線，整理在 [`research/README.md`](research/README.md)。
- **版本。** 所有變更都記錄在 [`CHANGELOG.md`](CHANGELOG.md)；release notes 放在
  [`docs/releases/`](docs/releases/)，最新的是 [1.6.1](docs/releases/v1.6.1.md)、
  [1.6.0](docs/releases/v1.6.0.md) 與 [1.5.0](docs/releases/v1.5.0.md)。

## 參與貢獻

最有幫助的第一個貢獻，是提供 M4 Max 以外機型的 benchmark 結果（[做法](docs/community-benchmarks.md)）。

[`CONTRIBUTING.md`](CONTRIBUTING.md) 說明環境設定、測試分級與 parity 檢查；coding agent 請遵守
[`AGENTS.md`](AGENTS.md)。待處理的工作會標上 `good first issue`、`help wanted` 與 `research`。

更多：[使用指南](docs/guide.md) · [穩定 API](docs/api.md) · [架構](docs/architecture.md) ·
[變更紀錄](CHANGELOG.md) · [安全性](SECURITY.md)。

採用 Apache-2.0 授權；請見 [`LICENSE`](LICENSE) 與 [`NOTICE`](NOTICE)。模型權重從指定的
Hugging Face revision 下載，本專案不會重新散布。這是獨立專案，並非 Convai Innovations、Apple 或 MLX
的官方版本。
