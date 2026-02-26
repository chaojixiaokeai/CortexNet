# CortexNet 快速开始与使用指南

版本：3.2.1

---

## 1. 环境要求

- Python `>=3.9`
- PyTorch `>=2.0`
- 推荐：安装 `build`、`pytest`、`ruff` 用于开发校验

---

## 2. 安装方式

### 2.1 从源码安装（开发推荐）

```bash
git clone https://github.com/chaojixiaokeai/CortexNet.git
cd CortexNet
pip install -e .
```

### 2.2 安装开发依赖

```bash
pip install -e .[dev]
```

### 2.3 通过 PyPI 安装（发布后）

```bash
pip install cortexnet
```

---

## 3. 安装后快速验证

### 3.1 查看版本

```bash
python -m cortexnet --version
```

### 3.2 运行最小冒烟测试

```bash
python -m cortexnet --smoke-test
```

期望输出包含 `smoke_ok logits_shape=...`。

---

## 4. 最小推理示例

```python
import torch
from cortexnet import CortexNet, CortexNetConfig

config = CortexNetConfig(
    vocab_size=32000,
    hidden_size=512,
    num_layers=4,
    num_heads=8,
    max_seq_len=2048,
)

model = CortexNet(config).eval()
input_ids = torch.randint(0, config.vocab_size, (1, 16))

with torch.no_grad():
    out = model(input_ids)

print(out["logits"].shape)
```

---

## 5. 文本生成

```python
import torch
from cortexnet import CortexNet, CortexNetConfig

cfg = CortexNetConfig(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=4)
model = CortexNet(cfg).eval()

prompt = torch.randint(0, cfg.vocab_size, (1, 8))
with torch.no_grad():
    gen = model.generate(
        prompt,
        max_new_tokens=32,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.05,
    )

print(gen.shape)
```

---

## 6. 从 HuggingFace 模型目录迁移加载

```python
import torch
from cortexnet import CortexNet

model = CortexNet.from_pretrained(
    "/path/to/hf_model_dir",
    device="auto",
    dtype=torch.bfloat16,
    compatibility_mode=True,
    auto_calibrate=False,
)
```

说明：

1. 对外主类统一使用 `CortexNet`。
2. 历史命名（如 `CortexNetV3`）仅保留兼容别名，不建议新代码继续使用。

---

## 7. 常用脚本

### 7.1 运行完整测试

```bash
python -m pytest -q
python scripts/dev/run_tests.py
```

### 7.2 一键流程（回归 + 聊天）

```bash
python scripts/dev/one_click_cortexnet.py --model-path "/path/to/Qwen3-8B"
# 或：
export CORTEXNET_MODEL_PATH="/path/to/Qwen3-8B"
python scripts/dev/one_click_cortexnet.py
```

### 7.3 基准对比

```bash
python scripts/benchmarks/benchmark_cortexnet_vs_transformer.py --model-path "/path/to/model"
python scripts/benchmarks/benchmark_long_context.py --model-path "/path/to/model"
python scripts/benchmarks/benchmark_coldstart.py --model-path "/path/to/model"
python scripts/benchmarks/benchmark_quantization.py
```

### 7.4 能力评测

```bash
python scripts/eval/eval_ability_compare.py --model-path "/path/to/model" --engine cortex
```

---

## 8. 常见问题

### 8.1 `torch_npu` / `torch_mlu` 不可用

- 属于可选后端，未安装时会自动回退。
- 可先用 `device=cpu` 或 `device=cuda` 验证流程。

### 8.2 第一次加载慢

- 大模型首次映射权重耗时正常。
- 可开启映射缓存相关配置降低重复加载成本。

### 8.3 想优先稳定上线

建议优先：

1. `compatibility_mode=True`
2. 关闭不必要的进阶模块
3. 固定 `dtype/device` 与采样参数

---

## 9. 最佳实践

1. 新项目统一使用 `CortexNet` 主类。
2. 所有训练或推理脚本固定随机种子，便于回归比较。
3. 提交前执行：`make lint && make test-all && make check`。
4. 公开发布时同步更新 `CHANGELOG.md` 与文档。
