# CortexNet Benchmarking and Evaluation

Version: 3.2.1

---

## 1. Purpose

This document defines benchmark scope, metric conventions, and execution practices for reproducible CortexNet evaluation.

---

## 2. Evaluation dimensions

1. latency: first-token and end-to-end generation latency
2. throughput: tokens per second
3. resources: peak RSS memory and device pressure
4. stability: run-to-run variance and failure rate
5. quality proxies: rule checks and side-by-side output comparisons

---

## 3. Script matrix

### 3.1 Runtime comparison

- `scripts/benchmarks/benchmark_cortexnet_vs_transformer.py`

Reports:

1. first-token latency
2. decode throughput
3. peak memory
4. fixed prompt answer comparison

### 3.2 Long context

- `scripts/benchmarks/benchmark_long_context.py`

Measures scaling behavior under increasing context lengths.

### 3.3 Coldstart

- `scripts/benchmarks/benchmark_coldstart.py`

Measures startup and first-request behavior with cache/lazy options.

### 3.4 Quantization

- `scripts/benchmarks/benchmark_quantization.py`

Compares quantization strategies for speed/resource tradeoffs.

### 3.5 Lite baseline

- `scripts/benchmarks/benchmark_lite.py`

Quick local comparison across Lite/Full/Transformer paths.

### 3.6 Capability evaluation

- `scripts/eval/eval_ability_compare.py`

Provides rule-based pass rates and output similarity snapshots.

---

## 4. Standard execution protocol

1. fix software/hardware environment
2. fix random seeds
3. separate coldstart vs warm-start reporting
4. save JSON outputs with commit metadata

---

## 5. Metric definitions

1. `first_token_latency_s`: interactive responsiveness signal
2. `decode_tokens_per_s`: sustained generation efficiency
3. `peak_rss_gb`: host memory footprint
4. `checks_pass_rate`: rule-based quality proxy
5. `similarity`: output-text similarity to baseline reference

---

## 6. Example commands

```bash
python scripts/benchmarks/benchmark_cortexnet_vs_transformer.py \
  --model-path "/path/to/model" \
  --device auto \
  --dtype auto \
  --save-json benchmark_compare.json

python scripts/benchmarks/benchmark_long_context.py \
  --model-path "/path/to/model" \
  --context-lengths 512,1024,2048,4096 \
  --save-json benchmark_long_context.json

python scripts/benchmarks/benchmark_coldstart.py \
  --model-path "/path/to/model" \
  --save-json benchmark_coldstart.json

python scripts/eval/eval_ability_compare.py \
  --model-path "/path/to/model" \
  --engine cortex
```

---

## 7. Interpretation guidance

1. report both averages and tail values (for example P95)
2. do not mix coldstart and warm-start metrics
3. keep prompt sets and check rules versioned
4. always include runtime config snapshot in reports

---

## 8. Common distortion sources

1. background load and thermal throttling
2. tokenizer/preprocessing overhead mixed into core metrics
3. inconsistent cache state between runs
4. inconsistent sampling parameters

Controls:

1. isolated process runs
2. explicit warmup strategy
3. standardized report schema

---

## 9. CI integration recommendation

Use lightweight performance gates:

1. run small benchmark subsets per PR
2. alert on threshold regressions
3. store artifacts with commit IDs

---

## 10. Release report template

1. environment snapshot
2. coldstart summary
3. warm runtime summary
4. quality summary
5. baseline delta and release decision
