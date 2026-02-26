# CortexNet API 参考（中文）

版本：3.2.1

---

## 1. 核心导出

主入口来自 `cortexnet/__init__.py`，推荐优先使用：

1. `CortexNet`
2. `CortexNetConfig`
3. `TrainingConfig`

---

## 2. `CortexNetConfig`

`CortexNetConfig` 是模型结构与运行行为配置中心，关键字段：

1. 基础结构：`vocab_size`、`hidden_size`、`num_layers`、`num_heads`、`max_seq_len`
2. 注意力：`top_k_ratio`、`attention_k_mode`、`num_kv_heads`
3. SSM：`num_scales`、`ssm_state_size`、`ssm_expand_factor`
4. 记忆：`memory_dim`、`memory_decay_init`
5. MoE：`num_experts`、`num_active_experts`、`expert_ff_dim`
6. 兼容与部署：`compatibility_mode`、`lite`、`lazy_device_load`

---

## 3. `CortexNet`

### 3.1 初始化

```python
from cortexnet import CortexNet, CortexNetConfig
model = CortexNet(CortexNetConfig())
```

### 3.2 前向

```python
out = model(input_ids, labels=None)
# out 包含 logits；训练时可包含 loss 与 aux_loss
```

### 3.3 生成

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

### 3.4 迁移加载

```python
model = CortexNet.from_pretrained(
    model_path="/path/to/hf_model",
    device="auto",
    dtype="auto",
    compatibility_mode=True,
)
```

### 3.5 智能生成

```python
gen = model.smart_generate(input_ids, max_new_tokens=128)
```

---

## 4. 设备工具 API

来自 `cortexnet.ops`：

1. `resolve_device_string(requested, auto_priority, allow_fallback)`
2. `resolve_dtype_for_device(dtype, device)`
3. `get_best_device_info()`
4. `DeviceManager`

---

## 5. 适配器 API

来自 `cortexnet.adapter`：

1. `detect_model_type(model_path)`
2. `get_cortexnet_config(model_path, model_type)`
3. `WeightAdapter`
4. `ArchitectureAdapter`
5. `InferenceAdapter`
6. `LightweightCalibrator`

---

## 6. 训练辅助 API

来自 `cortexnet.training_utils`：

1. `set_seed(seed)`
2. `create_optimizer_and_scheduler(...)`
3. `safe_clip_grad_norm_(model, max_norm)`
4. `check_gradients_finite(model)`
5. `GradientMonitor`

---

## 7. 分布式 API

来自 `cortexnet.distributed`：

1. `setup_distributed(backend="nccl")`
2. `wrap_fsdp(model, mixed_precision="bf16")`
3. `wrap_ddp(model)`
4. `cleanup_distributed()`
5. `get_rank()` / `get_world_size()` / `is_main_process()`

---

## 8. 兼容别名说明

库中仍提供以下兼容别名：

1. `CortexNetV2`
2. `CortexNetV3`
3. `CortexBlockV2`
4. `CortexBlockV3`

建议：新代码统一使用 `CortexNet` 与 `CortexBlock`。

---

## 9. CLI 入口

```bash
python -m cortexnet --version
python -m cortexnet --smoke-test
```

---

## 10. 稳定性说明

当前 API 稳定优先级：

1. 高稳定：`CortexNet`、`CortexNetConfig`、CLI 基础命令
2. 中稳定：适配器与设备工具（可能持续演进）
3. 实验性：部分进阶能力模块的细节行为
