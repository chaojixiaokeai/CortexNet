# CortexNet Examples

These examples are intentionally minimal and runnable.

## 1. Minimal forward inference

```bash
python examples/minimal_infer.py
```

## 2. Migration loading from a HuggingFace-style model directory

```bash
python examples/from_pretrained_migration.py --model-path /path/to/model --device auto
```

## 3. Tiny training loop

```bash
python examples/train_tiny.py --steps 20 --device auto
```
