#!/usr/bin/env python3
"""Release smoke benchmark for CortexNet.

Purpose:
1) Provide a stable, fast benchmark artifact per release.
2) Keep runtime short for CI and release validation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from cortexnet import CortexNet, CortexNetConfig, __version__
from cortexnet.ops.device_manager import resolve_device_string


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Release smoke benchmark for CortexNet")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/mps/npu/mlu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--output", type=str, default="")
    return parser


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def sync_if_needed(device: str) -> None:
    kind = device.split(":", 1)[0]
    if kind == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    args = build_parser().parse_args()

    device = resolve_device_string(args.device, auto_priority=("cuda", "mps", "cpu"), allow_fallback=True)
    dtype = parse_dtype(args.dtype)

    cfg = CortexNetConfig(
        vocab_size=4096,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=max(args.seq_len, 128),
        lite=True,
    )

    model = CortexNet(cfg).to(device)
    model = model.to(dtype=dtype)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(input_ids)

        times_ms = []
        for _ in range(args.iters):
            sync_if_needed(device)
            t0 = time.perf_counter()
            out = model(input_ids)
            sync_if_needed(device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            times_ms.append(dt_ms)

    avg_ms = statistics.mean(times_ms)
    if times_ms:
        sorted_times = sorted(times_ms)
        p95_idx = max(0, min(len(sorted_times) - 1, int(len(sorted_times) * 0.95) - 1))
        p95_ms = sorted_times[p95_idx]
    else:
        p95_ms = avg_ms
    tokens = args.batch_size * args.seq_len
    throughput = tokens / (avg_ms / 1000.0)

    result = {
        "version": __version__,
        "device": device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "warmup": args.warmup,
        "iters": args.iters,
        "avg_forward_ms": round(avg_ms, 4),
        "p95_forward_ms": round(p95_ms, 4),
        "throughput_tokens_per_sec": round(throughput, 4),
        "logits_shape": list(out["logits"].shape),
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
