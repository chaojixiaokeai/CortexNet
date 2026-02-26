# CortexNet 适配与兼容设计

版本：3.2.1

---

## 1. 目标

本文件说明 CortexNet 如何兼容主流开源模型并保持统一推理入口，核心目标：

1. 用 `CortexNet` 统一外部调用。
2. 保留历史别名兼容旧代码。
3. 通过适配层降低不同模型家族的接入复杂度。

---

## 2. 适配架构总览

`from_pretrained` 关键流程：

1. `ModelRegistry` 识别模型类型。
2. 将源 `config.json` 转换为 `CortexNetConfig`。
3. `WeightAdapter` 进行权重命名和形状映射。
4. `ArchitectureAdapter` 做行为级适配。
5. `InferenceAdapter` 应用模型族推理参数策略。
6. `LightweightCalibrator` 可选轻量校准。

---

## 3. 支持模型族

当前注册表支持（以代码为准）：

1. LLaMA 系列
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

如果遇到未识别模型，可通过扩展注册表规则支持。

---

## 4. 配置兼容策略

映射过程中重点处理：

1. `num_hidden_layers -> num_layers`
2. `num_attention_heads -> num_heads`
3. `num_key_value_heads -> num_kv_heads`
4. `intermediate_size -> expert_ff_dim/intermediate_size`
5. `rms_norm_eps/layer_norm_epsilon -> norm_eps`
6. `rope_theta/rope_scaling`

默认策略偏保守：

- 默认 `lite=True`，优先降低额外参数与计算开销。
- 可通过 `compatibility_mode=True` 强化兼容路径。

---

## 5. 权重映射策略

### 5.1 映射原则

1. 优先精确匹配核心权重。
2. 对形状可推断场景进行安全变换（如 GQA 扩展）。
3. 记录 unmapped 与 shape mismatch，用于后续修复。

### 5.2 常见特殊情况

1. 合并 QKV（如部分 Baichuan 权重格式）需拆分。
2. GQA 头数与全头数不同，需要扩展或保持原生 KV 维度。
3. tie word embeddings 需与源配置保持一致。

---

## 6. 运行时兼容策略

### 6.1 统一主类

- 新代码只建议使用 `CortexNet`。
- `CortexNetV2/CortexNetV3` 保留为兼容别名，不作为主入口。

### 6.2 推理参数适配

`InferenceAdapter` 会根据模型族提供默认参数策略，减少迁移后的行为偏差。

### 6.3 设备与 dtype

通过 `resolve_device_string` 和 `resolve_dtype_for_device` 统一解析，兼容：

- `cpu`
- `cuda`
- `mps`
- `npu`
- `mlu`

---

## 7. 冷启动优化兼容

在大模型迁移场景下，推荐结合：

1. 映射缓存（mapped cache）
2. lazy 设备预热
3. 可选校准开关

这三者的组合可根据“首 token 时延 vs 总加载时长”目标进行权衡。

---

## 8. 扩展新模型族指南

最小步骤：

1. 在 `adapter/model_registry.py` 新增 `ModelInfo`。
2. 在 `adapter/weight_adapter.py` 增加映射规则。
3. 视需要更新 `arch_adapter.py` / `inference_adapter.py`。
4. 增加至少一组测试（配置转换 + 权重映射 + 端到端加载）。

---

## 9. 兼容性回归清单

发布前建议检查：

1. `detect_model_type` 是否正确识别。
2. `from_pretrained(..., load_weights=True)` 是否无崩溃。
3. 典型 prompt 生成是否存在明显行为退化。
4. 长上下文/冷启动基准是否在可接受阈值内。

---

## 10. 风险与边界

1. 极端定制权重命名不在默认映射范围。
2. 不同模型家族默认采样策略差异可能影响主观质量。
3. 迁移后质量应使用任务集复核，不能只看能跑通。
