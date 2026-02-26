#!/usr/bin/env python3
"""
量化策略对比基准（面向 CortexNet 推理）。

对比项：
1) 前向延迟 (ms)
2) 前向吞吐 (tokens/s)
3) state_dict 大小 (MB)
4) 峰值 RSS (GB)
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List

import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from cortexnet.ops.device_manager import (
    resolve_device_string,
    resolve_dtype_for_device,
    get_device_type,
)


def get_peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 ** 3)
    return rss / (1024 ** 2)


def resolve_device(device: str) -> str:
    return resolve_device_string(
        device,
        auto_priority=("npu", "cuda", "mps", "cpu"),
        allow_fallback=True,
    )


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    return resolve_dtype_for_device(dtype, device)


def parse_strategies(raw: str) -> List[str]:
    vals = [s.strip() for s in raw.split(",") if s.strip()]
    if not vals:
        raise ValueError("strategies is empty")
    return vals


def _effective_tensor_bytes(obj: object, visited: set[int] | None = None) -> int:
    if visited is None:
        visited = set()
    oid = id(obj)
    if oid in visited:
        return 0
    visited.add(oid)

    if torch.is_tensor(obj):
        module_name = type(obj).__module__
        if module_name.startswith("torchao"):
            inner = 0
            for attr_name in ("tensor_impl", "original_weight_tensor"):
                if hasattr(obj, attr_name):
                    inner += _effective_tensor_bytes(getattr(obj, attr_name), visited)
            for v in getattr(obj, "__dict__", {}).values():
                inner += _effective_tensor_bytes(v, visited)
            if inner > 0:
                return inner
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, dict):
        return sum(_effective_tensor_bytes(v, visited) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(_effective_tensor_bytes(v, visited) for v in obj)
    if hasattr(obj, "__dict__"):
        return sum(_effective_tensor_bytes(v, visited) for v in obj.__dict__.values())
    return 0


def _serialized_state_dict_bytes(model: torch.nn.Module) -> int:
    fd, path = tempfile.mkstemp(prefix="cortex_quant_", suffix=".pt")
    os.close(fd)
    try:
        torch.save(model.state_dict(), path)
        return int(os.path.getsize(path))
    except Exception:
        return 0
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def state_dict_stats(model: torch.nn.Module) -> Dict[str, Any]:
    logical_total = 0
    effective_total = 0
    int8_total = 0
    dtype_bytes: Dict[str, int] = {}
    for _, tensor in model.state_dict().items():
        if torch.is_tensor(tensor):
            nbytes = tensor.numel() * tensor.element_size()
            logical_total += nbytes
            effective_total += _effective_tensor_bytes(tensor)
            dtype_name = str(tensor.dtype).replace("torch.", "")
            dtype_bytes[dtype_name] = dtype_bytes.get(dtype_name, 0) + nbytes
            if tensor.dtype in {torch.int8, torch.uint8, torch.qint8, torch.quint8}:
                int8_total += nbytes
    return {
        "logical_total_bytes": logical_total,
        "effective_total_bytes": effective_total,
        "int8_bytes": int8_total,
        "dtype_bytes": dtype_bytes,
        "serialized_total_bytes": _serialized_state_dict_bytes(model),
    }


def run_worker(args: argparse.Namespace) -> Dict[str, Any]:
    from cortexnet.config import CortexNetConfig
    from cortexnet import CortexNet
    from cortexnet.quantization import QuantizationWrapper

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    seed_device = get_device_type(args.device)
    if seed_device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    elif seed_device == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "manual_seed_all"):
        torch.npu.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    config = CortexNetConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=max(1, min(args.num_kv_heads, args.num_heads)),
        intermediate_size=args.intermediate_size,
        max_seq_len=max(args.max_seq_len, args.seq_len),
        dropout=0.0,
        lite=True,
    )
    model = CortexNet(config).eval().to(device=device, dtype=dtype)

    quant_start = time.perf_counter()
    qmodel = QuantizationWrapper(model, strategy=args.strategy)
    quant_sec = time.perf_counter() - quant_start
    backend = getattr(qmodel, "backend", args.strategy)

    ids = torch.randint(
        0,
        config.vocab_size,
        (args.batch_size, args.seq_len),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        for _ in range(max(args.warmup, 0)):
            _ = qmodel(ids)

        start = time.perf_counter()
        for _ in range(args.iters):
            out = qmodel(ids)
            _ = out["logits"]
        total_sec = time.perf_counter() - start

    avg_ms = (total_sec / max(args.iters, 1)) * 1000.0
    tok_per_sec = (args.batch_size * args.seq_len * args.iters) / max(total_sec, 1e-6)
    sd_stats = state_dict_stats(qmodel)

    return {
        "strategy": args.strategy,
        "backend": backend,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "quant_sec": round(quant_sec, 4),
        "avg_forward_ms": round(avg_ms, 3),
        "tokens_per_sec": round(tok_per_sec, 2),
        "state_dict_mb": round(sd_stats["logical_total_bytes"] / (1024 ** 2), 3),
        "effective_state_mb": round(sd_stats["effective_total_bytes"] / (1024 ** 2), 3),
        "serialized_state_mb": round(sd_stats["serialized_total_bytes"] / (1024 ** 2), 3),
        "int8_state_mb": round(sd_stats["int8_bytes"] / (1024 ** 2), 3),
        "dtype_mb": {
            k: round(v / (1024 ** 2), 3)
            for k, v in sorted(sd_stats["dtype_bytes"].items(), key=lambda x: x[0])
        },
        "peak_rss_gb": round(get_peak_rss_gb(), 3),
    }


def run_parent(args: argparse.Namespace) -> int:
    script_path = os.path.abspath(__file__)
    strategies = parse_strategies(args.strategies)

    results = []
    for strategy in strategies:
        cmd = [
            sys.executable,
            script_path,
            "--worker",
            "--strategy",
            strategy,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--hidden-size",
            str(args.hidden_size),
            "--num-layers",
            str(args.num_layers),
            "--num-heads",
            str(args.num_heads),
            "--num-kv-heads",
            str(args.num_kv_heads),
            "--intermediate-size",
            str(args.intermediate_size),
            "--vocab-size",
            str(args.vocab_size),
            "--max-seq-len",
            str(args.max_seq_len),
            "--batch-size",
            str(args.batch_size),
            "--seq-len",
            str(args.seq_len),
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--seed",
            str(args.seed),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[ERROR] strategy={strategy} failed", file=sys.stderr)
            print(proc.stdout, file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"worker failed: {strategy}")

        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("JSON_RESULT:"):
                payload = line[len("JSON_RESULT:") :].strip()
        if payload is None:
            raise RuntimeError(f"No JSON_RESULT for strategy={strategy}. stdout={proc.stdout}")
        results.append(json.loads(payload))

    baseline = None
    for r in results:
        if r["strategy"] == "none":
            baseline = r
            break

    print("=== Quantization Benchmark (CortexNet Lite) ===")
    print(
        f"{'strategy':<18}"
        f"{'backend':<24}"
        f"{'avg_ms':>10}"
        f"{'tok/s':>12}"
        f"{'state_mb':>11}"
        f"{'eff_mb':>10}"
        f"{'ser_mb':>10}"
        f"{'int8_mb':>11}"
        f"{'peak_rss_gb':>13}"
        f"{'lat_delta%':>12}"
        f"{'mem_delta%':>12}"
    )
    for r in results:
        lat_delta = 0.0
        mem_delta = 0.0
        if baseline is not None and r["strategy"] != "none":
            lat_delta = (
                (float(r["avg_forward_ms"]) - float(baseline["avg_forward_ms"]))
                / max(float(baseline["avg_forward_ms"]), 1e-6)
            ) * 100.0
            mem_delta = (
                (float(r["effective_state_mb"]) - float(baseline["effective_state_mb"]))
                / max(float(baseline["effective_state_mb"]), 1e-6)
            ) * 100.0
        print(
            f"{r['strategy']:<18}"
            f"{r['backend']:<24}"
            f"{float(r['avg_forward_ms']):>10.3f}"
            f"{float(r['tokens_per_sec']):>12.2f}"
            f"{float(r['state_dict_mb']):>11.3f}"
            f"{float(r['effective_state_mb']):>10.3f}"
            f"{float(r['serialized_state_mb']):>10.3f}"
            f"{float(r['int8_state_mb']):>11.3f}"
            f"{float(r['peak_rss_gb']):>13.3f}"
            f"{lat_delta:>12.2f}"
            f"{mem_delta:>12.2f}"
        )

    if args.save_json:
        out_path = os.path.abspath(args.save_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON report: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantization benchmark for CortexNet")
    parser.add_argument("--strategies", type=str, default="none,dynamic_int8,weight_only_int8")
    parser.add_argument("--strategy", type=str, default="none", help=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/mps/cuda/npu/mlu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", type=str, default="")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.worker:
            result = run_worker(args)
            print("JSON_RESULT:" + json.dumps(result, ensure_ascii=False))
            return 0
        return run_parent(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
