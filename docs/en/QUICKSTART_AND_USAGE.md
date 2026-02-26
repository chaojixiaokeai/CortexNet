# CortexNet Quickstart and Usage

Version: 3.2.1

---

## 1. Requirements

- Python `>=3.9`
- PyTorch `>=2.0`
- Optional dev tools: `pytest`, `ruff`, `build`, `twine`

---

## 2. Installation

### 2.1 Install from source

```bash
git clone https://github.com/chaojixiaokeai/CortexNet.git
cd CortexNet
pip install -e .
```

### 2.2 Install dev extras

```bash
pip install -e .[dev]
```

### 2.3 Install from PyPI

```bash
pip install cortexnet
```

---

## 3. Post-install smoke checks

```bash
python -m cortexnet --version
python -m cortexnet --smoke-test
```

A successful smoke test prints `smoke_ok logits_shape=...`.

---

## 4. Minimal forward example

```python
import torch
from cortexnet import CortexNet, CortexNetConfig

cfg = CortexNetConfig(
    vocab_size=32000,
    hidden_size=512,
    num_layers=4,
    num_heads=8,
    max_seq_len=2048,
)

model = CortexNet(cfg).eval()
input_ids = torch.randint(0, cfg.vocab_size, (1, 16))

with torch.no_grad():
    out = model(input_ids)

print(out["logits"].shape)
```

---

## 5. Generation example

```python
import torch
from cortexnet import CortexNet, CortexNetConfig

cfg = CortexNetConfig(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=4)
model = CortexNet(cfg).eval()
prompt = torch.randint(0, cfg.vocab_size, (1, 8))

with torch.no_grad():
    generated = model.generate(
        prompt,
        max_new_tokens=32,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.05,
    )

print(generated.shape)
```

---

## 6. Load from a HuggingFace model directory

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

Notes:

1. Use `CortexNet` as the canonical class in all new code.
2. Legacy names are compatibility aliases only.

---

## 7. Common scripts

### 7.1 Tests

```bash
python -m pytest -q
python scripts/dev/run_tests.py
```

### 7.2 One-click flow

```bash
python scripts/dev/one_click_cortexnet.py --model-path "/path/to/Qwen3-8B"
# or:
export CORTEXNET_MODEL_PATH="/path/to/Qwen3-8B"
python scripts/dev/one_click_cortexnet.py
```

### 7.3 Benchmarks

```bash
python scripts/benchmarks/benchmark_cortexnet_vs_transformer.py --model-path "/path/to/model"
python scripts/benchmarks/benchmark_long_context.py --model-path "/path/to/model"
python scripts/benchmarks/benchmark_coldstart.py --model-path "/path/to/model"
python scripts/benchmarks/benchmark_quantization.py
```

### 7.4 Capability evaluation

```bash
python scripts/eval/eval_ability_compare.py --model-path "/path/to/model" --engine cortex
```

---

## 8. Troubleshooting

1. Device backend unavailable (`npu/mlu`): run with `device=cpu` or `device=cuda` first.
2. First load is slow: expected for large model mapping; evaluate mapped cache options.
3. Repetitive outputs: tune `repetition_penalty`, `top_k`, and `top_p`.

---

## 9. Recommended local quality gate

```bash
make lint
make test-all
make check
```
