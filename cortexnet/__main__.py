"""CortexNet CLI entrypoint."""

from __future__ import annotations

import argparse

import torch

from . import __version__, CortexNet, CortexNetConfig


def _run_smoke_test() -> None:
    cfg = CortexNetConfig(vocab_size=128, hidden_size=64, num_layers=1, num_heads=2, lite=False)
    model = CortexNet(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(input_ids)
    print(f"smoke_ok logits_shape={tuple(out['logits'].shape)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CortexNet CLI")
    parser.add_argument("--version", action="store_true", help="Print installed CortexNet version")
    parser.add_argument("--smoke-test", action="store_true", help="Run a minimal forward-pass smoke test")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        print(__version__)
        return 0

    if args.smoke_test:
        _run_smoke_test()
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
