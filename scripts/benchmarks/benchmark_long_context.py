#!/usr/bin/env python3
"""
长上下文性能基准（CortexNet vs Transformer）。

目标：
1) 在不同上下文长度下对比首 token 延迟
2) 对比解码吞吐 (tokens/s)
3) 对比峰值进程内存 (RSS)

说明：
- 使用合成 token 序列（固定 seed）避免 tokenizer 预处理噪声。
- 每个引擎在独立子进程中运行，避免内存统计互相污染。
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from cortexnet.ops.device_manager import resolve_device_string, resolve_dtype_for_device


def get_peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 ** 3)  # macOS: bytes
    return rss / (1024 ** 2)  # Linux: KB


def resolve_device(device: str) -> str:
    return resolve_device_string(
        device,
        auto_priority=("npu", "cuda", "mps", "cpu"),
        allow_fallback=True,
    )


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    return resolve_dtype_for_device(dtype, device)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if temperature <= 0:
        safe_logits = logits.clamp(-100, 100)
        bad = torch.isnan(safe_logits) | torch.isinf(safe_logits)
        safe_logits = torch.where(bad, torch.zeros_like(safe_logits), safe_logits)
        return torch.argmax(safe_logits.to(torch.float32), dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-8)

    if top_k > 0 and top_k < logits.size(-1):
        values, _ = torch.topk(logits, top_k, dim=-1)
        threshold = values[:, -1].unsqueeze(-1)
        logits = torch.where(
            logits < threshold,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        remove_mask = torch.zeros_like(logits, dtype=torch.bool)
        remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(remove_mask, float("-inf"))

    # 数值安全：长上下文 + 半精度时避免 NaN/Inf 破坏采样。
    logits = logits.clamp(-100, 100)
    bad = torch.isnan(logits) | torch.isinf(logits)
    logits = torch.where(bad, torch.zeros_like(logits), logits)

    probs = torch.softmax(logits, dim=-1)
    probs = probs.clamp(min=1e-8)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.multinomial(probs, num_samples=1)


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty <= 0 or abs(repetition_penalty - 1.0) < 1e-6:
        return logits
    vocab = logits.size(-1)
    for b in range(generated.size(0)):
        seen_ids = torch.unique(
            generated[b].clamp(min=0, max=vocab - 1).to(device=logits.device)
        )
        seen_logits = logits[b, seen_ids]
        positive = seen_logits > 0
        seen_logits[positive] = seen_logits[positive] / repetition_penalty
        seen_logits[~positive] = seen_logits[~positive] * repetition_penalty
        logits[b, seen_ids] = seen_logits
    return logits


@torch.no_grad()
def generate_cortex(
    model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> Tuple[torch.Tensor, float, float]:
    generated = input_ids
    past_cache = None
    t_start = time.perf_counter()
    first_token_latency: Optional[float] = None

    for _ in range(max_new_tokens):
        if past_cache is None:
            max_seq = getattr(model.config, "max_seq_len", generated.shape[1])
            idx_cond = generated if generated.shape[1] <= max_seq else generated[:, -max_seq:]
        else:
            idx_cond = generated[:, -1:]

        outputs = model.forward(
            idx_cond,
            past_cache=past_cache,
            use_cache=True,
            start_warmup_after_infer=False,
        )
        logits = outputs["logits"][:, -1, :]
        past_cache = outputs.get("past_cache")

        logits = apply_repetition_penalty(logits, generated, repetition_penalty)
        next_token = sample_next_token(logits, temperature, top_k, top_p)
        next_token = next_token.clamp(min=0, max=logits.size(-1) - 1)
        if generated.device != next_token.device:
            generated = generated.to(next_token.device)
        generated = torch.cat([generated, next_token], dim=1)

        if first_token_latency is None:
            first_token_latency = time.perf_counter() - t_start

    total = time.perf_counter() - t_start
    return generated, first_token_latency or total, total


@torch.no_grad()
def generate_transformer(
    model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> Tuple[torch.Tensor, float, float]:
    generated = input_ids
    past_key_values = None
    t_start = time.perf_counter()
    first_token_latency: Optional[float] = None

    for _ in range(max_new_tokens):
        idx_cond = generated if past_key_values is None else generated[:, -1:]
        outputs = model(input_ids=idx_cond, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        logits = apply_repetition_penalty(logits, generated, repetition_penalty)
        next_token = sample_next_token(logits, temperature, top_k, top_p)
        next_token = next_token.clamp(min=0, max=logits.size(-1) - 1)
        if generated.device != next_token.device:
            generated = generated.to(next_token.device)
        generated = torch.cat([generated, next_token], dim=1)

        if first_token_latency is None:
            first_token_latency = time.perf_counter() - t_start

    total = time.perf_counter() - t_start
    return generated, first_token_latency or total, total


def make_synth_input(
    vocab_size: int,
    seq_len: int,
    seed: int,
    device: str,
) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + seq_len * 9973)
    ids = torch.randint(0, vocab_size, (1, seq_len), generator=g, dtype=torch.long)
    return ids.to(device)


def parse_context_lengths(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    vals = [int(p) for p in parts]
    vals = [v for v in vals if v > 0]
    if not vals:
        raise ValueError("context lengths must contain at least one positive integer")
    return vals


def save_trend_plot(
    cortex: Dict[str, Any],
    transformer: Dict[str, Any],
    plot_path: str,
) -> Optional[str]:
    """保存趋势图（tokens/s 与首 token 延迟）。"""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    c_sorted = sorted(cortex["context_results"], key=lambda x: int(x["seq_len"]))
    t_map = {int(r["seq_len"]): r for r in transformer["context_results"]}
    lengths = [int(r["seq_len"]) for r in c_sorted]
    c_tps = [float(r["tokens_per_sec"]) for r in c_sorted]
    t_tps = [float(t_map[int(r["seq_len"])]["tokens_per_sec"]) for r in c_sorted]
    c_first = [float(r["first_token_sec"]) for r in c_sorted]
    t_first = [float(t_map[int(r["seq_len"])]["first_token_sec"]) for r in c_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=140)
    ax0, ax1 = axes

    ax0.plot(lengths, c_tps, marker="o", linewidth=2, label="CortexNet")
    ax0.plot(lengths, t_tps, marker="o", linewidth=2, label="Transformer")
    ax0.set_title("Decode Throughput Trend")
    ax0.set_xlabel("Context Length")
    ax0.set_ylabel("Tokens / sec")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1.plot(lengths, c_first, marker="o", linewidth=2, label="CortexNet")
    ax1.plot(lengths, t_first, marker="o", linewidth=2, label="Transformer")
    ax1.set_title("First Token Latency Trend")
    ax1.set_xlabel("Context Length")
    ax1.set_ylabel("Seconds")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    fig.suptitle("Long Context Benchmark: CortexNet vs Transformer", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.abspath(plot_path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_worker(args: argparse.Namespace) -> Dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    lengths = parse_context_lengths(args.context_lengths)

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if device == "mps":
        torch.mps.manual_seed(args.seed)

    load_start = time.perf_counter()
    if args.engine == "cortex":
        from cortexnet import CortexNet

        model = CortexNet.from_pretrained(
            args.model_path,
            device=device,
            dtype=dtype,
            compatibility_mode=args.compatibility_mode,
            auto_calibrate=(not args.disable_calibrate),
            load_weights=True,
            lazy_device_load=args.lazy_device_load,
            lazy_background_warmup=args.lazy_background_warmup,
            lazy_cpu_fallback=args.lazy_cpu_fallback,
        )
        generate_fn = generate_cortex
        engine_name = "CortexNet(compat)"
        vocab_size = int(getattr(model.config, "vocab_size", 32000))
    elif args.engine == "transformer":
        from transformers import AutoModelForCausalLM
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=dtype,
            trust_remote_code=True,
        )
        model = model.to(device)
        model.eval()
        generate_fn = generate_transformer
        engine_name = "Transformer(Qwen3 native)"
        vocab_size = int(getattr(model.config, "vocab_size", 32000))
    else:
        raise ValueError(f"Unsupported engine: {args.engine}")

    load_sec = time.perf_counter() - load_start

    runtime_device = device
    lazy_status = None
    if args.engine == "cortex" and hasattr(model, "get_lazy_status"):
        lazy_status = model.get_lazy_status()
        if lazy_status.get("enabled") and not lazy_status.get("ready"):
            runtime_device = "cpu"
        elif lazy_status.get("enabled") and lazy_status.get("ready") and lazy_status.get("target_device"):
            runtime_device = str(lazy_status.get("target_device"))

    per_length = []
    for seq_len in lengths:
        input_ids = make_synth_input(vocab_size, seq_len, args.seed, runtime_device)

        for _ in range(max(args.warmup_runs, 0)):
            _ = generate_fn(
                model=model,
                input_ids=input_ids,
                max_new_tokens=max(1, args.max_new_tokens // 2),
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )

        _, first_token_sec, gen_sec = generate_fn(
            model=model,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        tokens_per_sec = args.max_new_tokens / max(gen_sec, 1e-6)
        per_length.append(
            {
                "seq_len": int(seq_len),
                "first_token_sec": round(first_token_sec, 4),
                "gen_sec": round(gen_sec, 4),
                "tokens_per_sec": round(tokens_per_sec, 3),
            }
        )

    return {
        "engine": args.engine,
        "engine_name": engine_name,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "load_sec": round(load_sec, 3),
        "peak_rss_gb": round(get_peak_rss_gb(), 3),
        "context_results": per_length,
        "lazy_status": lazy_status,
    }


def run_parent(args: argparse.Namespace) -> int:
    script_path = os.path.abspath(__file__)

    def _run_engine(engine: str) -> Dict[str, Any]:
        cmd = [
            sys.executable,
            script_path,
            "--worker",
            "--engine",
            engine,
            "--model-path",
            args.model_path,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--context-lengths",
            args.context_lengths,
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
        ]
        if args.compatibility_mode:
            cmd.append("--compatibility-mode")
        else:
            cmd.append("--no-compatibility-mode")
        if args.disable_calibrate:
            cmd.append("--disable-calibrate")
        if args.lazy_device_load:
            cmd.append("--lazy-device-load")
        else:
            cmd.append("--no-lazy-device-load")
        if args.lazy_background_warmup:
            cmd.append("--lazy-background-warmup")
        else:
            cmd.append("--no-lazy-background-warmup")
        if args.lazy_cpu_fallback:
            cmd.append("--lazy-cpu-fallback")
        else:
            cmd.append("--no-lazy-cpu-fallback")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[ERROR] Engine {engine} failed.\n", file=sys.stderr)
            print(proc.stdout, file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"Benchmark worker failed: {engine}")

        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("JSON_RESULT:"):
                payload = line[len("JSON_RESULT:") :].strip()
        if payload is None:
            raise RuntimeError(f"No JSON_RESULT found for engine={engine}. stdout={proc.stdout}")
        return json.loads(payload)

    print("=== Long Context Benchmark: CortexNet vs Transformer ===")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}, DType: {args.dtype}")
    print(f"Context lengths: {args.context_lengths}, New tokens: {args.max_new_tokens}")
    print("")

    cortex = _run_engine("cortex")
    transformer = _run_engine("transformer")

    t_map = {int(r["seq_len"]): r for r in transformer["context_results"]}
    print("=== Per-length Speed ===")
    print(
        f"{'seq_len':<10}"
        f"{'cortex_first(s)':>16}"
        f"{'trans_first(s)':>16}"
        f"{'cortex_tok/s':>14}"
        f"{'trans_tok/s':>14}"
        f"{'delta_tok/s':>14}"
    )
    for c in cortex["context_results"]:
        L = int(c["seq_len"])
        t = t_map[L]
        delta = float(c["tokens_per_sec"]) - float(t["tokens_per_sec"])
        print(
            f"{L:<10}"
            f"{float(c['first_token_sec']):>16.4f}"
            f"{float(t['first_token_sec']):>16.4f}"
            f"{float(c['tokens_per_sec']):>14.3f}"
            f"{float(t['tokens_per_sec']):>14.3f}"
            f"{delta:>14.3f}"
        )

    print("")
    print("=== Load & Memory ===")
    print(
        f"{'engine':<16}{'load_sec':>12}{'peak_rss_gb':>14}"
    )
    print(f"{cortex['engine_name']:<16}{float(cortex['load_sec']):>12.3f}{float(cortex['peak_rss_gb']):>14.3f}")
    print(
        f"{transformer['engine_name']:<16}"
        f"{float(transformer['load_sec']):>12.3f}"
        f"{float(transformer['peak_rss_gb']):>14.3f}"
    )

    merged = {"cortex": cortex, "transformer": transformer}
    if args.save_json:
        out_path = os.path.abspath(args.save_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"\nSaved JSON report: {out_path}")

    if args.plot_path:
        plotted = save_trend_plot(cortex, transformer, args.plot_path)
        if plotted is not None:
            print(f"Saved trend plot: {plotted}")
        else:
            print("Trend plot skipped: matplotlib is not available.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Long-context benchmark for CortexNet vs Transformer")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/Users/pengjiajun/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
        help="本地模型目录（含 config.json / safetensors）",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/mps/cuda/npu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--context-lengths",
        type=str,
        default="512,1024,2048",
        help="逗号分隔的上下文长度，如 512,1024,2048",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16, help="每个长度的解码 token 数")
    parser.add_argument("--warmup-runs", type=int, default=1, help="每个长度的预热轮数")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", type=str, default="", help="保存对比结果 JSON 路径")
    parser.add_argument(
        "--plot-path",
        type=str,
        default="",
        help="趋势图输出路径（png），如 reports/long_context_trend.png",
    )

    parser.add_argument(
        "--compatibility-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CortexNet 兼容模式（建议开启）",
    )
    parser.add_argument(
        "--lazy-device-load",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否启用惰性上设备（仅 CortexNet 有效）",
    )
    parser.add_argument(
        "--lazy-background-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="惰性模式下首次推理后是否后台预热（仅 CortexNet 有效）",
    )
    parser.add_argument(
        "--lazy-cpu-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="惰性模式下未完成预热时 CPU 兜底（仅 CortexNet 有效）",
    )
    parser.add_argument(
        "--disable-calibrate",
        action="store_true",
        help="关闭 CortexNet 无感校准",
    )

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--engine", type=str, default="cortex", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model path not found: {args.model_path}", file=sys.stderr)
        return 2

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
