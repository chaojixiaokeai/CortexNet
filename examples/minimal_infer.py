from __future__ import annotations

import torch

from cortexnet import CortexNet, CortexNetConfig


def main() -> None:
    config = CortexNetConfig(
        vocab_size=32000,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        max_seq_len=512,
    )

    model = CortexNet(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 12))

    with torch.no_grad():
        out = model(input_ids)

    print("logits shape:", tuple(out["logits"].shape))


if __name__ == "__main__":
    main()
