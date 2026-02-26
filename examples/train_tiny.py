from __future__ import annotations

import argparse

import torch

from cortexnet import CortexNet, CortexNetConfig, set_seed
from cortexnet.ops.device_manager import resolve_device_string


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tiny CortexNet training example")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/mps/npu/mlu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = resolve_device_string(args.device)

    cfg = CortexNetConfig(
        vocab_size=4096,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=max(128, args.seq_len),
        lite=True,
    )

    model = CortexNet(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for step in range(1, args.steps + 1):
        input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)
        labels = input_ids.clone()

        out = model(input_ids, labels=labels)
        loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 5 == 0 or step == args.steps:
            print(f"step={step} loss={loss.detach().item():.4f}")


if __name__ == "__main__":
    main()
