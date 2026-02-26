# CortexNet 训练与部署手册

版本：3.2.1

---

## 1. 总览

本手册给出 CortexNet 从训练到部署的工程化路径，覆盖：

1. 配置设计
2. 单机训练
3. 分布式训练
4. 推理部署
5. 观测与排障

---

## 2. 训练配置建议

核心配置来自 `CortexNetConfig` 与 `TrainingConfig`。

建议起步参数：

1. `hidden_size=512~2048`
2. `num_layers=4~24`
3. `num_heads` 与 `hidden_size` 整除
4. `num_experts=1` 先跑通，后续按资源扩展 MoE
5. `dropout=0.0~0.1` 视任务规模调整

---

## 3. 单机训练最小样例

```python
import torch
from cortexnet import CortexNet, CortexNetConfig
from cortexnet.training_utils import create_optimizer_and_scheduler, safe_clip_grad_norm_

cfg = CortexNetConfig(
    vocab_size=32000,
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    max_seq_len=2048,
)

model = CortexNet(cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer, scheduler = create_optimizer_and_scheduler(
    model,
    lr=3e-4,
    weight_decay=0.01,
    warmup_steps=200,
    total_steps=20000,
)

model.train()
for step in range(1000):
    input_ids = torch.randint(0, cfg.vocab_size, (2, 256), device=device)
    labels = input_ids.clone()

    out = model(input_ids, labels=labels)
    loss = out["loss"]

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    safe_clip_grad_norm_(model, max_norm=1.0)
    optimizer.step()
    scheduler.step()
```

---

## 4. 混合精度与稳定性

建议：

1. CUDA/NPU/MLU 优先使用 `bfloat16`（硬件支持时）。
2. 长序列训练中开启梯度裁剪，减少梯度爆炸风险。
3. 对异常 batch 进行 NaN/Inf 梯度检查。

可用工具：

- `check_gradients_finite(model)`
- `GradientMonitor(model)`

---

## 5. 分布式训练

`cortexnet.distributed` 提供：

- `setup_distributed`
- `wrap_fsdp`
- `wrap_ddp`
- `cleanup_distributed`

最小流程：

```python
from cortexnet.distributed import setup_distributed, wrap_fsdp, cleanup_distributed

is_dist = setup_distributed()
if is_dist:
    model = wrap_fsdp(model, mixed_precision="bf16")

# ... train ...
cleanup_distributed()
```

说明：

1. 若未检测到分布式环境变量，会自动回到单机模式。
2. 非 CUDA 环境会自动将 `nccl` 回退到 `gloo`。

---

## 6. 推理部署路径

### 6.1 原生模型加载

适用于从头训练或已保存 CortexNet 权重的场景。

### 6.2 `from_pretrained` 迁移加载

适用于将 HuggingFace 模型目录快速接入 CortexNet 推理。

建议生产参数：

1. `compatibility_mode=True`
2. 固定 `device` 和 `dtype`
3. 根据冷启动要求评估 `lazy_device_load`

---

## 7. 冷启动与加载优化

可结合以下策略：

1. 映射缓存（mapped cache）
2. lazy 设备预热
3. 首轮预热请求

建议使用仓库脚本定量评估：

```bash
python scripts/benchmarks/benchmark_coldstart.py --model-path "/path/to/model"
```

---

## 8. 线上运行建议

1. 推理服务固定版本和配置快照。
2. 对核心指标设门限：P50/P95 延迟、tokens/s、内存峰值。
3. 发布前做回归基准，不接受“体感变慢”型变更。
4. 对异常输出加入安全与格式后处理。

---

## 9. 发布流水线（GitHub + PyPI）

仓库已提供：

1. `ci.yml`：静态检查 + 测试 + 构建检查
2. `publish.yml`：Release 发布触发 PyPI 上传

需要配置 GitHub Secret：`PYPI_API_TOKEN`。

---

## 10. 排障手册

1. 加载失败：确认 `config.json`、权重文件、tokenizer 文件完整。
2. 设备不匹配：通过 `resolve_device_string` 统一解析。
3. OOM：降低 `max_seq_len`、batch size、dtype 或使用更稀疏配置。
4. 输出重复：适当提高 `repetition_penalty`，调整 `top_k/top_p`。
