# CortexNet Adaptation and Compatibility Design

Version: 3.2.1

---

## 1. Goals

This document explains how CortexNet preserves a unified runtime interface while supporting heterogeneous source model families.

Primary goals:

1. one canonical public entry class: `CortexNet`
2. backward-compatible aliases for legacy code
3. adapter layer to isolate family-specific conversion logic

---

## 2. Adaptation pipeline

`from_pretrained` pipeline:

1. detect source model family (`ModelRegistry`)
2. convert source config to `CortexNetConfig`
3. map weights (`WeightAdapter`)
4. apply architecture-level behavior adaptation (`ArchitectureAdapter`)
5. set generation behavior defaults (`InferenceAdapter`)
6. run optional lightweight calibration

---

## 3. Supported families

Current registry includes:

1. LLaMA
2. Qwen2 / Qwen3
3. Mistral / Mixtral
4. Baichuan
5. ChatGLM
6. Phi
7. Yi
8. DeepSeek
9. InternLM
10. Gemma
11. Falcon
12. CodeLlama

If a model is not recognized, extend registry + mapping rules.

---

## 4. Config conversion strategy

Typical field mapping includes:

1. `num_hidden_layers -> num_layers`
2. `num_attention_heads -> num_heads`
3. `num_key_value_heads -> num_kv_heads`
4. `intermediate_size -> expert_ff_dim/intermediate_size`
5. `rms_norm_eps/layer_norm_epsilon -> norm_eps`
6. `rope_theta/rope_scaling`

Default conversion favors conservative migration behavior:

1. `lite=True` by default for lower overhead
2. `compatibility_mode=True` for closer source-path behavior when needed

---

## 5. Weight mapping strategy

### 5.1 Principles

1. exact mapping first for core tensors
2. safe shape transforms where deterministic
3. explicit accounting for unmapped and shape mismatch tensors

### 5.2 Typical edge cases

1. merged QKV layouts that require splitting
2. GQA head expansion or preservation logic
3. tied embedding/output head consistency

---

## 6. Runtime compatibility

### 6.1 Canonical class policy

Use `CortexNet` in new code.

Legacy names remain as compatibility aliases only.

### 6.2 Inference behavior adaptation

`InferenceAdapter` sets family-aware generation defaults to reduce behavior drift.

### 6.3 Device and dtype normalization

Use runtime helpers:

1. `resolve_device_string`
2. `resolve_dtype_for_device`

Supported families include `cpu/cuda/mps/npu/mlu`.

---

## 7. Coldstart compatibility strategy

For large migrated models, combine:

1. mapped cache
2. lazy runtime warmup
3. optional calibration controls

Tune these based on first-token latency vs total startup tradeoff.

---

## 8. Extending to a new model family

Minimum steps:

1. add `ModelInfo` in `adapter/model_registry.py`
2. add rules in `adapter/weight_adapter.py`
3. update `arch_adapter.py` and `inference_adapter.py` if needed
4. add tests for conversion + mapping + end-to-end load path

---

## 9. Compatibility regression checklist

Before release:

1. `detect_model_type` correctness
2. `from_pretrained(..., load_weights=True)` stability
3. no obvious generation regression on fixed prompts
4. acceptable long-context/coldstart benchmark deltas

---

## 10. Boundaries

1. highly customized source naming may require custom mapping rules
2. default generation settings vary by model family
3. migration success must be validated by task-level evaluation, not only loading success
