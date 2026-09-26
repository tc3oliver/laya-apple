# laya-apple

[English](README.md) | **繁體中文** | [简体中文](README.zh-CN.md)

[![CI](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml/badge.svg)](https://github.com/tc3oliver/laya-apple/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/laya-apple)](https://pypi.org/project/laya-apple/)
![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

> 本文件是 [`README.md`](README.md) 的中文版。內容如有出入，以英文版為準。

**在 Apple silicon 上執行、經過正確性驗證的自適應 [Laya](https://github.com/NandhaKishorM/laya)
推論：MLX GPU 與 Apple Neural Engine 同時提供服務。**

## 自適應 GPU + Neural Engine 服務（1.5）

laya-apple 把簡短的單一問題決策交給 Apple Neural Engine（ANE），較長或有多個問題的工作留在
MLX GPU，兩個引擎同時服務。從 v1.0 開始，這樣的分工就讓短請求不必排在長時間的 GPU 工作後面。

**GPU 早就算完了，Python 還在等。** ANE 跑在同一個 process 的執行緒上時，同步的 Core ML 呼叫在
每次預測中有很大一部分時間都握著 GIL。一個 GPU 已經算完的請求，要等那次呼叫結束才能把結果交回來
（[`research/coreml-gil-completion-path/`](research/coreml-gil-completion-path/README.md)）。
1.5 讓 laya 與 laya-typed-decisions 中符合條件的 ANE 請求改走非同步的 Core ML 執行，避開同步路徑
長時間持有 GIL 的問題。Runtime 會監看這條較快的路徑，一旦持續變慢，就退回已知安全的 1.4 同步
路徑。在 `execution="workers"`、`device="auto"` 下預設開啟，不需要再改程式。

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

[試試 Switchyard](#親眼看看switchyard) ·
[安裝](#安裝) · [本機 Jev 相容伺服器](#本機-jev-相容伺服器)

## 安裝

需要 Apple silicon 與 Python 3.11–3.13：

```bash
pip install 'laya-apple[ane]'   # MLX GPU + the Neural Engine runtime
```

| Extra | 加入的功能 |
|---|---|
| （無） | 只有 MLX GPU runtime |
| `ane` | Neural Engine runtime（coremltools 9.0、pyobjc-framework-CoreML） |
| `convert` | 在這台 Mac 上建置 ANE artifact（torch 2.7.0） |
| `serve` | `laya-apple serve`，本機 Jev 相容伺服器 |

沒有裝 `ane` extra，或還沒建置 ANE artifact 時，所有運算都在 MLX GPU 上執行。
如果要從原始碼執行或參與開發，請參考 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 快速開始（30 秒）

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
- 沒有 ANE artifact 時，同一個請求會改在 MLX 上跑，`routing_reason` 會說明原因。要在這台 Mac 上
  建置並通過 parity 驗證（需要 `convert` extra）：`laya-apple artifacts build laya-typed-decisions`。
- 機率、其他問題類型與完整 API：[`examples/`](examples/)、[使用指南](docs/guide.md) 與
  [`docs/api.md`](docs/api.md)。

要讓 GPU + ANE 同時服務，請使用 `execution="workers"`，請求可以從任何執行緒送出：

```python
with Laya.from_pretrained("convaiinnovations/laya-typed-decisions", execution="workers") as model:
    futures = [model.submit(context=c, questions=q) for c, q in requests]   # thread-safe
```

## 運作方式

Laya 只要一次 forward pass，就能回答一段 context 中的 typed questions（`choice`、`score`、`noul`）。
Mac 上有兩個引擎可以跑它，各自擅長不同的請求。

![不超過 128 個 token、只有一個問題且有已驗證 artifact 的請求送往 Apple Neural Engine；較長、包含多個問題或未經驗證的請求送往 MLX GPU；使用 execution="workers" 時，兩個引擎同時處理彼此獨立的請求](https://raw.githubusercontent.com/tc3oliver/laya-apple/main/docs/readme/architecture.svg)

1. **ANE 執行正確性。** Core ML 匯出跑得快，不代表結果正確。在測試用的 Mac 上，一般的 Core ML
   匯出在 ANE 上執行時完全沒有報錯，卻有多達 85 個決策和上游不同。ANE artifact 必須在實際執行的
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

## 親眼看看：Switchyard

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
  本專案不對與 Jev 相比的準確度做任何宣稱。

API key、模型、安全性與用戶端注意事項：[`docs/serve.md`](docs/serve.md)。

### 較早的 serve benchmark

在 1.3 路徑上量測，同一台 Mac 上有一個滿載生成的 27B 本機 LLM 時，簡短決策的 P99 在異質 `auto`
服務下為 **47.2 ms**，只用 GPU 時為 **122.3 ms**（一台 M4 Max、一次執行，`--model laya`）。
這項結果尚未在 1.5 自適應執行下重新量測。LLM 吞吐量的影響、方法與限制：
[`docs/serve.md`](docs/serve.md#beside-a-local-llm)。

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

## 最初的異質服務 benchmark（v1.0）

這些結果證明了 GPU + ANE 服務的價值，是在 1.5 推出自適應執行之前量測的。

| 模型 | 與僅 GPU 相比的吞吐量 | 短請求 P99，僅 GPU | 短請求 P99，GPU + ANE |
|---|---:|---:|---:|
| laya | **2.92×** | 1538.0 ms | **108.5 ms** |
| laya-multilingual | **4.34×** | 2052.3 ms | **29.6 ms** |
| laya-typed-decisions | **4.57×** | 1592.9 ms | **79.5 ms** |

一台 Apple M4 Max，一條短請求與一條長請求串流送進同一個 `Laya(execution="workers")`。短請求
P99 在 open-loop 突發負載下從請求到達開始計時，所以包含排隊時間。效能提升來自同時使用兩個引擎，
而不是 ANE 本身比較快。方法與原始資料：[`benchmarks/v1.0.md`](benchmarks/v1.0.md)。

## 支援的模型與平台

| 模型 | max_len | MLX GPU | ANE bucket（明確指定） | `auto` 使用的 ANE bucket | 自適應 ANE 執行 |
|---|---:|---|---|---|---|
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | 512 | FP16 / FP32，任意長度 | 64, 96, 128 | 64, 96, 128 | 是 |
| [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | 1024 | FP16 / FP32，任意長度 | 64, 96, 128, 256 | 64, 96, 128 | 否（ANE 在 worker process 中執行） |
| [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 1024 | FP16 / FP32，任意長度 | 64, 96, 128 | 64, 96, 128 | 是 |

自適應 ANE 執行只在 `execution="workers"`、`device="auto"` 下啟用。只在一台 macOS 26.6.2 的 Apple M4 Max 上驗證過。在其他 Mac 上，要先在那台機器建置並校準 ANE
artifact（`laya-apple calibrate`），`auto` 才會使用 ANE，在那之前都走 MLX。詳見
[`docs/compatibility.md`](docs/compatibility.md)。

<a id="community-benchmarks"></a>

## 社群 benchmark

上面所有 benchmark 都是在 M4 Max 上跑的。[社群矩陣](docs/community-benchmarks.md)分開記錄
其他 Mac 的結果，目前已有 M4 Pro、M4 與 M2 Pro（僅 MLX）。一個指令就能加入你的 Mac：

```bash
uv run python scripts/hardware_report.py --quick
```

從 `git clone` 到開 pull request 的完整步驟，請見[指南](docs/community-benchmarks.md#add-your-mac)。

## 重現結果

上面每個主要數字，都能在 [`docs/reproducibility.md`](docs/reproducibility.md) 找到對應的
報告、原始資料、指令與環境。1.5 的相關證據收錄在
[`research/coreml-adaptive-breaker/`](research/coreml-adaptive-breaker/README.md)。

## 限制

- **只有一台測試機。** 所有 benchmark，包括 1.5 的驗證與恢復實驗，都來自同一台 macOS 26.6.2
  的 Apple M4 Max。我們不假設路由門檻在其他 Apple SoC 上也成立。
- **自適應執行的 handoff 很保守：** GPU 與 ANE 每次同時忙碌時，最先的一批 ANE 請求會先走
  1.4 路徑。
  laya-multilingual 的 ANE 在 worker process 中執行，不使用這項功能。
- **`laya-apple serve` 預設會使用 1.5 的自適應執行，但還沒有在這條路徑上做過 benchmark；**
  上面的 serve benchmark 是在 1.3 路徑上量測的。
- **長請求與多問題請求留在 GPU，** 因為 GPU 處理它們比較快。ANE 路徑只支援 batch 1。
- **無法完全隔離。** 同時執行時，每條串流的 P99 都比單獨執行時高。
- **冷啟動：** 如果 artifact 目錄是全新的，每個模型需要 3–5 分鐘編譯 Core ML。
  設定 `ane_startup="background"` 時，這段期間會先用 MLX 服務。
- **Switchyard 沒有平衡回合順序：** 使用標準 seed 時，三次執行都是 GPU + ANE 先跑。

各個 benchmark 自己的限制，記錄在對應的實驗文件中，例如選項順序見
[`docs/correctness.md`](docs/correctness.md#option-order)，serve 與用戶端的注意事項見
[`docs/serve.md`](docs/serve.md#limits)，尚未量測的項目見
[`docs/compatibility.md`](docs/compatibility.md#not-measured)。

## 參與貢獻

最有幫助的第一個貢獻，是提供 M4 Max 以外機型的 benchmark 結果（[做法](docs/community-benchmarks.md)）。

[`CONTRIBUTING.md`](CONTRIBUTING.md) 說明環境設定、測試分級與 parity 檢查；coding agent 請遵守
[`AGENTS.md`](AGENTS.md)。待處理的工作會標上 `good first issue`、`help wanted` 與 `research`。

更多：[使用指南](docs/guide.md) · [穩定 API](docs/api.md) · [架構](docs/architecture.md) ·
[變更紀錄](CHANGELOG.md) · [安全性](SECURITY.md)。

採用 Apache-2.0 授權；請見 [`LICENSE`](LICENSE) 與 [`NOTICE`](NOTICE)。模型權重從指定的
Hugging Face revision 下載，本專案不會重新散布。這是獨立專案，並非 Convai Innovations、Apple 或 MLX
的官方版本。
