---
language:
- en
license: apache-2.0
library_name: cortexnet
tags:
- cortexnet
- llm
- state-space-model
- sparse-attention
---

# CortexNet

CortexNet is a unified neural architecture implementation for language modeling and reasoning.

This Hugging Face model repository is used as the public project card and release artifact index.

## Install

```bash
pip install cortexnet
```

## Quickstart

```python
import torch
from cortexnet import CortexNet, CortexNetConfig

cfg = CortexNetConfig(vocab_size=32000, hidden_size=256, num_layers=4, num_heads=8, lite=True)
model = CortexNet(cfg).eval()
input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
with torch.no_grad():
    out = model(input_ids)
print(out["logits"].shape)
```

## Links

- GitHub: https://github.com/chaojixiaokeai/CortexNet
- PyPI: https://pypi.org/project/cortexnet/
- GitHub Release: https://github.com/chaojixiaokeai/CortexNet/releases
- Docs Hub: https://github.com/chaojixiaokeai/CortexNet/tree/main/docs

## Included Files

- `README.md`: this model card
- `PROJECT_README.md`: project root README
- `DOCS_INDEX.md`: bilingual docs index
- `reports/`: release smoke benchmark reports and JSON artifacts
