# CortexNet API Reference

Version: 3.2.1

---

## 1. Canonical exports

Primary public API from `cortexnet/__init__.py`:

1. `CortexNet`
2. `CortexNetConfig`
3. `TrainingConfig`

---

## 2. `CortexNetConfig`

`CortexNetConfig` controls architecture and runtime behavior.

Key groups:

1. base shape: `vocab_size`, `hidden_size`, `num_layers`, `num_heads`, `max_seq_len`
2. attention: `top_k_ratio`, `attention_k_mode`, `num_kv_heads`
3. SSM: `num_scales`, `ssm_state_size`, `ssm_expand_factor`
4. memory: `memory_dim`, `memory_decay_init`
5. MoE: `num_experts`, `num_active_experts`, `expert_ff_dim`
6. compatibility/runtime: `compatibility_mode`, `lite`, `lazy_device_load`

---

## 3. `CortexNet`

### 3.1 Initialization

```python
from cortexnet import CortexNet, CortexNetConfig
model = CortexNet(CortexNetConfig())
```

### 3.2 Forward

```python
out = model(input_ids, labels=None)
# Returns logits; training mode can also return loss/aux_loss
```

### 3.3 Generation

```python
gen = model.generate(
    input_ids,
    max_new_tokens=128,
    temperature=0.7,
    top_k=20,
    top_p=0.9,
    repetition_penalty=1.05,
)
```

### 3.4 Migration loading

```python
model = CortexNet.from_pretrained(
    model_path="/path/to/hf_model",
    device="auto",
    dtype="auto",
    compatibility_mode=True,
)
```

### 3.5 Smart generation

```python
gen = model.smart_generate(input_ids, max_new_tokens=128)
```

---

## 4. Device/runtime helpers

From `cortexnet.ops`:

1. `resolve_device_string(requested, auto_priority, allow_fallback)`
2. `resolve_dtype_for_device(dtype, device)`
3. `get_best_device_info()`
4. `DeviceManager`

---

## 5. Adapter APIs

From `cortexnet.adapter`:

1. `detect_model_type(model_path)`
2. `get_cortexnet_config(model_path, model_type)`
3. `WeightAdapter`
4. `ArchitectureAdapter`
5. `InferenceAdapter`
6. `LightweightCalibrator`

---

## 6. Training utility APIs

From `cortexnet.training_utils`:

1. `set_seed(seed)`
2. `create_optimizer_and_scheduler(...)`
3. `safe_clip_grad_norm_(model, max_norm)`
4. `check_gradients_finite(model)`
5. `GradientMonitor`

---

## 7. Distributed APIs

From `cortexnet.distributed`:

1. `setup_distributed(backend="nccl")`
2. `wrap_fsdp(model, mixed_precision="bf16")`
3. `wrap_ddp(model)`
4. `cleanup_distributed()`
5. `get_rank()` / `get_world_size()` / `is_main_process()`

---

## 8. Legacy compatibility aliases

Available for backward compatibility:

1. `CortexNetV2`
2. `CortexNetV3`
3. `CortexBlockV2`
4. `CortexBlockV3`

New code should use `CortexNet` and `CortexBlock`.

---

## 9. CLI

```bash
python -m cortexnet --version
python -m cortexnet --smoke-test
```

---

## 10. Stability notes

Current stability priority:

1. high: canonical model/config API and CLI basics
2. medium: adapter/runtime utility surface
3. evolving: selected advanced capability modules
