#!/bin/zsh
# v1.0 comprehensive benchmark, the parts not produced by release_bench.py / bench_concurrency.py --part a.
# Run on a quiet machine, nothing else using the GPU or ANE. Outputs go to benchmarks/v1.0/;
# recorded Phase -1 evidence under research/ is never overwritten.
#
#   LAYA_APPLE_CACHE=/path/to/cache LAYA_APPLE_ARTIFACTS=/path/to/research-artifacts \
#     HF_HUB_OFFLINE=1 scripts/bench_v1.sh
#
# 1. Comparators, with the Phase -1 harness (research/phase-0-feasibility), forward P50 at
#    every length each model supports, one question: upstream PyTorch CPU / MPS FP32 and the
#    ordinary Core ML graph (laya-coreml) fixed-shape on CPU_AND_NE and CPU_AND_GPU, plus the
#    enumerated-shape export at L128 / L512.
# 2. Comparator correctness against the upstream PyTorch FP32 goldens.
# 3. laya-apple open-loop mixed workload (Poisson and bursty arrivals, P50/P95/P99 from arrival).
set -eu
ROOT=${0:A:h:h}
OUT=$ROOT/benchmarks/v1.0
RESEARCH=$ROOT/research/phase-0-feasibility
MODELS=(${=MODELS:-laya-typed-decisions laya-multilingual laya})
mkdir -p $OUT
# Stop before the Core ML matrix starts if free disk is below the minimum (a large E5 cache only warns).
$ROOT/.venv/bin/python $ROOT/scripts/bench_preflight.py --workspace $OUT

cd $RESEARCH
for spec in '{"backend":"torch","device":"cpu"}' '{"backend":"torch","device":"mps"}' \
            '{"backend":"coreml","units":"cpu_ne"}' '{"backend":"coreml","units":"cpu_gpu"}'; do
  for m in $MODELS; do
    uv run python scripts/bench.py --model $m --spec "$spec" --questions 1 --modes forward \
      --budget ${BUDGET:-4} --tag v1.0 --output $OUT/comparators-latency.jsonl 2>&1 | grep -E "fwd p50|Error|Traceback" || true
  done
done
for m in $MODELS; do
  uv run python scripts/bench.py --model $m --spec '{"backend":"coreml","units":"cpu_gpu","enumerated":true}' \
    --lengths 128 512 --questions 1 --modes forward --min-samples 20 --budget 3 --tag v1.0 \
    --output $OUT/comparators-latency.jsonl 2>&1 | grep -E "fwd p50|Error" || true
done

for m in $MODELS; do
  for spec in '{"backend":"torch","device":"mps"}' '{"backend":"coreml","units":"cpu_ne"}' \
              '{"backend":"coreml","units":"cpu_gpu"}' '{"backend":"coreml","units":"all"}'; do
    PARITY_OUT=$OUT/parity uv run python scripts/parity.py --model $m --spec "$spec" 2>&1 | grep -E "passed=|Error" | tail -1 || true
  done
done

cd $ROOT
# Same lengths and offered rates as the v0.2 report (benchmarks/v0.2/concurrency-*.json).
typeset -A SHORT LONG MEDIUM RATES
SHORT=(laya 128 laya-multilingual 96 laya-typed-decisions 128)
LONG=(laya 512 laya-multilingual 1024 laya-typed-decisions 1024)
MEDIUM=(laya 256 laya-multilingual 512 laya-typed-decisions 512)
RATES=(laya "25 45" laya-multilingual "40 80" laya-typed-decisions "20 35")
for m in $MODELS; do
  .venv/bin/python scripts/bench_concurrency.py --model $m --short $SHORT[$m] --long $LONG[$m] \
    --medium $MEDIUM[$m] --rates ${=RATES[$m]} --part b --output $OUT/concurrency-$m.json 2>&1 | grep -E "wrote|Error" || true
done
