# CortexNet Training and Deployment Guide

Version: 3.2.1

---

## 1. Overview

This guide covers practical training and deployment workflow for CortexNet:

1. configuration setup
2. single-node training
3. distributed training
4. inference deployment
5. observability and troubleshooting

---

## 2. Training configuration

Core knobs are in `CortexNetConfig` and `TrainingConfig`.

Suggested starting range:

1. `hidden_size=512~2048`
2. `num_layers=4~24`
3. `num_heads` divisible with `hidden_size`
4. start with `num_experts=1`, then scale MoE
5. tune `dropout` based on data scale

---

## 3. Minimal single-node training loop

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

## 4. Mixed precision and stability

1. Prefer `bfloat16` on supported accelerators.
2. Keep gradient clipping enabled for long-context training.
3. Use finite-gradient checks on unstable runs.

Helpers:

1. `check_gradients_finite(model)`
2. `GradientMonitor(model)`

---

## 5. Distributed training

`cortexnet.distributed` provides:

1. `setup_distributed`
2. `wrap_fsdp`
3. `wrap_ddp`
4. `cleanup_distributed`

Minimal pattern:

```python
from cortexnet.distributed import setup_distributed, wrap_fsdp, cleanup_distributed

is_dist = setup_distributed()
if is_dist:
    model = wrap_fsdp(model, mixed_precision="bf16")

# ... train ...
cleanup_distributed()
```

---

## 6. Deployment paths

### 6.1 Native CortexNet weights

Use when model is trained/saved directly in CortexNet format.

### 6.2 Migration deployment (`from_pretrained`)

Use when rehosting a HuggingFace model directory.

Production defaults often start with:

1. `compatibility_mode=True`
2. explicit `device` and `dtype`
3. measured choice of `lazy_device_load`

---

## 7. Coldstart optimization

Recommended strategies:

1. mapped weight cache
2. lazy runtime warmup
3. warmup request before serving traffic

Evaluate with:

```bash
python scripts/benchmarks/benchmark_coldstart.py --model-path "/path/to/model"
```

---

## 8. Runtime operations

1. lock model/config version for each deployment
2. monitor latency percentiles and tokens/s
3. monitor memory peaks
4. run regression benchmarks before release

---

## 9. GitHub and PyPI release flow

Workflows already exist:

1. `ci.yml`: checks + tests + packaging validation
2. `publish.yml`: release-triggered PyPI publish

Required secret: `PYPI_API_TOKEN`.

---

## 10. Troubleshooting

1. load failure: verify `config.json`, tokenizer, and weights
2. device mismatch: use `resolve_device_string`
3. OOM: reduce context length, batch size, precision, or path complexity
4. repetitive outputs: tune sampling and penalty parameters
