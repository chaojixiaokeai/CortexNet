from __future__ import annotations

import argparse

from cortexnet import CortexNet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CortexNet from_pretrained migration example")
    parser.add_argument("--model-path", required=True, help="Path to a HuggingFace-style model directory")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/mps/npu/mlu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--compatibility-mode", action="store_true")
    parser.add_argument("--disable-calibrate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    model = CortexNet.from_pretrained(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        compatibility_mode=args.compatibility_mode,
        auto_calibrate=not args.disable_calibrate,
    )

    print("loaded model:", model.__class__.__name__)
    if hasattr(model, "config"):
        print("hidden_size:", model.config.hidden_size)
        print("num_layers:", model.config.num_layers)


if __name__ == "__main__":
    main()
