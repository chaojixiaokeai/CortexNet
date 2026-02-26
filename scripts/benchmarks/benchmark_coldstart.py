#!/usr/bin/env python3
"""
冷启动基准（CortexNet）。

默认对比三组：
1) baseline_no_lazy: 关闭 lazy
2) lazy_cache_build: 开启 lazy（首次构建 mapped cache）
3) lazy_cache_hit: 开启 lazy（复用已构建 mapped cache）

输出：
- load_sec
- first_token_sec
- tokens_per_sec
- peak_rss_gb
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List


def _run_cortex_worker(args: argparse.Namespace, *, lazy_device_load: bool) -> Dict[str, Any]:
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_long_context.py")
    cmd = [
        sys.executable,
        script_path,
        "--worker",
        "--engine",
        "cortex",
        "--model-path",
        args.model_path,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--context-lengths",
        str(args.context_len),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--repetition-penalty",
        str(args.repetition_penalty),
        "--seed",
        str(args.seed),
        "--warmup-runs",
        str(args.warmup_runs),
        "--compatibility-mode" if args.compatibility_mode else "--no-compatibility-mode",
        "--lazy-device-load" if lazy_device_load else "--no-lazy-device-load",
        "--lazy-background-warmup" if args.lazy_background_warmup else "--no-lazy-background-warmup",
        "--lazy-cpu-fallback" if args.lazy_cpu_fallback else "--no-lazy-cpu-fallback",
    ]
    if args.disable_calibrate:
        cmd.append("--disable-calibrate")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"coldstart worker failed (lazy={lazy_device_load})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("JSON_RESULT:"):
            payload = line[len("JSON_RESULT:") :].strip()
    if payload is None:
        raise RuntimeError(f"No JSON_RESULT found. stdout:\n{proc.stdout}")
    return json.loads(payload)


def _cache_count(cache_dir: str) -> int:
    if not os.path.isdir(cache_dir):
        return 0
    return sum(1 for n in os.listdir(cache_dir) if n.endswith(".safetensors"))


def _extract_row(case_name: str, worker_result: Dict[str, Any], cache_dir: str) -> Dict[str, Any]:
    ctx = (worker_result.get("context_results") or [{}])[0]
    lazy_status = worker_result.get("lazy_status") or {}
    return {
        "case": case_name,
        "load_sec": float(worker_result.get("load_sec", 0.0)),
        "first_token_sec": float(ctx.get("first_token_sec", 0.0)),
        "tokens_per_sec": float(ctx.get("tokens_per_sec", 0.0)),
        "peak_rss_gb": float(worker_result.get("peak_rss_gb", 0.0)),
        "device": worker_result.get("device", "unknown"),
        "dtype": worker_result.get("dtype", "unknown"),
        "cache_files": _cache_count(cache_dir),
        "lazy_enabled": bool(lazy_status.get("enabled", False)),
        "lazy_ready": bool(lazy_status.get("ready", False)),
        "lazy_target_device": lazy_status.get("target_device"),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    cache_dir = os.path.abspath(os.path.expanduser(args.cache_dir))
    if args.reset_cache and os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)

    rows: List[Dict[str, Any]] = []
    rows.append(
        _extract_row(
            "baseline_no_lazy",
            _run_cortex_worker(args, lazy_device_load=False),
            cache_dir,
        )
    )
    rows.append(
        _extract_row(
            "lazy_cache_build",
            _run_cortex_worker(args, lazy_device_load=True),
            cache_dir,
        )
    )
    rows.append(
        _extract_row(
            "lazy_cache_hit",
            _run_cortex_worker(args, lazy_device_load=True),
            cache_dir,
        )
    )

    baseline = rows[0]
    for row in rows[1:]:
        row["load_delta_pct_vs_baseline"] = round(
            (row["load_sec"] - baseline["load_sec"]) / max(baseline["load_sec"], 1e-6) * 100.0,
            2,
        )
        row["first_token_delta_pct_vs_baseline"] = round(
            (row["first_token_sec"] - baseline["first_token_sec"])
            / max(baseline["first_token_sec"], 1e-6)
            * 100.0,
            2,
        )

    return {
        "model_path": os.path.abspath(args.model_path),
        "cache_dir": cache_dir,
        "args": {
            "device": args.device,
            "dtype": args.dtype,
            "context_len": args.context_len,
            "max_new_tokens": args.max_new_tokens,
            "compatibility_mode": args.compatibility_mode,
            "disable_calibrate": args.disable_calibrate,
            "lazy_background_warmup": args.lazy_background_warmup,
            "lazy_cpu_fallback": args.lazy_cpu_fallback,
        },
        "results": rows,
    }


def print_report(payload: Dict[str, Any]) -> None:
    rows = payload["results"]
    print("=== Coldstart Benchmark (CortexNet) ===")
    print(
        f"{'case':<20}"
        f"{'load_sec':>10}"
        f"{'first_tok_s':>13}"
        f"{'tok/s':>10}"
        f"{'peak_rss':>10}"
        f"{'lazy':>7}"
        f"{'cache':>8}"
    )
    for r in rows:
        print(
            f"{r['case']:<20}"
            f"{r['load_sec']:>10.3f}"
            f"{r['first_token_sec']:>13.4f}"
            f"{r['tokens_per_sec']:>10.3f}"
            f"{r['peak_rss_gb']:>10.3f}"
            f"{('on' if r.get('lazy_enabled') else 'off'):>7}"
            f"{int(r['cache_files']):>8d}"
        )
    if len(rows) >= 3:
        lazy_hit = rows[2]
        print(
            "\nvs baseline: "
            f"lazy_cache_hit load_delta={lazy_hit.get('load_delta_pct_vs_baseline', 0.0):.2f}%, "
            f"first_token_delta={lazy_hit.get('first_token_delta_pct_vs_baseline', 0.0):.2f}%"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coldstart benchmark for CortexNet")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/Users/pengjiajun/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/mps/cuda/npu/mlu")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument(
        "--compatibility-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--disable-calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lazy-background-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lazy-cpu-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="~/.cache/cortexnet/mapped_weights",
    )
    parser.add_argument(
        "--reset-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-json", type=str, default="benchmark_coldstart_report.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run(args)
    print_report(payload)

    if args.save_json:
        out = os.path.abspath(args.save_json)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
