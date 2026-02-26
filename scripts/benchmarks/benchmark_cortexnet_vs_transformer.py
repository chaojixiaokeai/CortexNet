#!/usr/bin/env python3
"""
一键对比 CortexNet 与传统 Transformer（Qwen3 原生）的推理表现。

输出内容：
1) 首 token 延迟
2) 解码速度 (tokens/s)
3) 峰值进程内存 (RSS)
4) 3 个固定问题的回答对照

用法示例：
  python benchmark_cortexnet_vs_transformer.py \
    --model-path "/path/to/Qwen3-8B"
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
from tokenizers import Tokenizer

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from cortexnet.ops.device_manager import resolve_device_string, resolve_dtype_for_device


PERF_PROMPT = "请用一句话介绍 Transformer 和 CortexNet 在推理路径上的核心差异。"
QUALITY_PROMPTS = [
    "请用三点总结大模型落地时最容易踩的坑。",
    "写一个 Python 函数，返回前 10 项斐波那契数列。",
    "请给出 3 条可执行的 AI 安全上线检查项。",
]


def get_peak_rss_gb() -> float:
    """返回当前进程峰值 RSS（GB）。"""
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
        return torch.argmax(logits, dim=-1, keepdim=True)

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

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty <= 0 or abs(repetition_penalty - 1.0) < 1e-6:
        return logits

    for b in range(generated.size(0)):
        seen_ids = torch.unique(generated[b].to(device=logits.device))
        seen_logits = logits[b, seen_ids]
        positive = seen_logits > 0
        seen_logits[positive] = seen_logits[positive] / repetition_penalty
        seen_logits[~positive] = seen_logits[~positive] * repetition_penalty
        logits[b, seen_ids] = seen_logits
    return logits


def encode_prompt(tokenizer: Tokenizer, prompt: str, device: str) -> torch.Tensor:
    encoded = tokenizer.encode(prompt)
    return torch.tensor([encoded.ids], dtype=torch.long, device=device)


def decode_new_tokens(
    tokenizer: Tokenizer,
    generated: torch.Tensor,
    prompt_len: int,
) -> str:
    ids = generated[0, prompt_len:].tolist()
    return tokenizer.decode(ids)


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
        if generated.device != next_token.device:
            generated = generated.to(next_token.device)
        generated = torch.cat([generated, next_token], dim=1)

        if first_token_latency is None:
            first_token_latency = time.perf_counter() - t_start

    total = time.perf_counter() - t_start
    return generated, first_token_latency or total, total


def run_worker(args: argparse.Namespace) -> Dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if device == "mps":
        torch.mps.manual_seed(args.seed)

    tokenizer_path = os.path.join(args.model_path, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

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
    else:
        raise ValueError(f"Unsupported engine: {args.engine}")

    load_sec = time.perf_counter() - load_start

    runtime_device = device
    if args.engine == "cortex" and hasattr(model, "get_lazy_status"):
        lazy_status = model.get_lazy_status()
        if lazy_status.get("enabled") and not lazy_status.get("ready"):
            runtime_device = "cpu"
        elif lazy_status.get("enabled") and lazy_status.get("ready") and lazy_status.get("target_device"):
            runtime_device = str(lazy_status.get("target_device"))

    perf_input = encode_prompt(tokenizer, PERF_PROMPT, runtime_device)
    perf_prompt_tokens = perf_input.shape[1]
    perf_output, first_token_sec, perf_gen_sec = generate_fn(
        model=model,
        input_ids=perf_input,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    perf_answer = decode_new_tokens(tokenizer, perf_output, perf_prompt_tokens)

    quality_results: List[Dict[str, str]] = []
    for prompt in QUALITY_PROMPTS:
        runtime_device = device
        if args.engine == "cortex" and hasattr(model, "get_lazy_status"):
            lazy_status = model.get_lazy_status()
            if lazy_status.get("enabled") and not lazy_status.get("ready"):
                runtime_device = "cpu"
            elif lazy_status.get("enabled") and lazy_status.get("ready") and lazy_status.get("target_device"):
                runtime_device = str(lazy_status.get("target_device"))
        q_input = encode_prompt(tokenizer, prompt, runtime_device)
        q_out, _, _ = generate_fn(
            model=model,
            input_ids=q_input,
            max_new_tokens=args.quality_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        quality_results.append(
            {
                "prompt": prompt,
                "answer": decode_new_tokens(tokenizer, q_out, q_input.shape[1]).strip(),
            }
        )

    new_tokens = perf_output.shape[1] - perf_input.shape[1]
    tokens_per_sec = new_tokens / max(perf_gen_sec, 1e-6)

    result = {
        "engine": args.engine,
        "engine_name": engine_name,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "load_sec": round(load_sec, 3),
        "first_token_sec": round(first_token_sec, 3),
        "gen_sec": round(perf_gen_sec, 3),
        "new_tokens": int(new_tokens),
        "tokens_per_sec": round(tokens_per_sec, 3),
        "peak_rss_gb": round(get_peak_rss_gb(), 3),
        "perf_prompt": PERF_PROMPT,
        "perf_answer": perf_answer.strip(),
        "quality": quality_results,
    }
    if args.engine == "cortex" and hasattr(model, "get_lazy_status"):
        result["lazy_status"] = model.get_lazy_status()
    return result


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
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--quality-new-tokens",
            str(args.quality_new_tokens),
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

    print("=== CortexNet vs Transformer Benchmark ===")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}, DType: {args.dtype}")
    print(f"Perf tokens: {args.max_new_tokens}, Quality tokens: {args.quality_new_tokens}")
    print("")

    cortex = _run_engine("cortex")
    transformer = _run_engine("transformer")

    print("=== Speed & Memory ===")
    print(
        f"{'Metric':<20}"
        f"{'CortexNet':>18}"
        f"{'Transformer':>18}"
        f"{'Delta(Cortex-Trans)':>22}"
    )
    rows = [
        ("load_sec", cortex["load_sec"], transformer["load_sec"]),
        ("first_token_sec", cortex["first_token_sec"], transformer["first_token_sec"]),
        ("gen_sec", cortex["gen_sec"], transformer["gen_sec"]),
        ("tokens_per_sec", cortex["tokens_per_sec"], transformer["tokens_per_sec"]),
        ("peak_rss_gb", cortex["peak_rss_gb"], transformer["peak_rss_gb"]),
    ]
    for name, c_val, t_val in rows:
        delta = float(c_val) - float(t_val)
        print(f"{name:<20}{c_val:>18.3f}{t_val:>18.3f}{delta:>22.3f}")

    print("")
    print("=== Perf Prompt Answer ===")
    print(f"Prompt: {cortex['perf_prompt']}")
    print(f"[CortexNet] {cortex['perf_answer']}")
    print(f"[Transformer] {transformer['perf_answer']}")

    print("")
    print("=== Quality Prompt Comparison (3 fixed prompts) ===")
    for idx, item in enumerate(cortex["quality"], start=1):
        prompt = item["prompt"]
        c_answer = item["answer"]
        t_answer = transformer["quality"][idx - 1]["answer"]
        print(f"{idx}. Prompt: {prompt}")
        print(f"   CortexNet: {c_answer}")
        print(f"   Transformer: {t_answer}")
        print("")

    merged = {"cortex": cortex, "transformer": transformer}
    if args.save_json:
        out_path = os.path.abspath(args.save_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON report: {out_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark CortexNet vs Transformer")
    default_model_path = os.getenv("CORTEXNET_MODEL_PATH", "")
    parser.add_argument(
        "--model-path",
        type=str,
        default=default_model_path,
        help="本地模型目录（默认读取环境变量 CORTEXNET_MODEL_PATH）",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/mps/cuda/npu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=64, help="性能测试生成 token 数")
    parser.add_argument("--quality-new-tokens", type=int, default=96, help="质量对比每题生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", type=str, default="", help="保存对比结果 JSON 路径")

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

    if not args.model_path:
        print("[ERROR] Missing --model-path. Set --model-path or CORTEXNET_MODEL_PATH.", file=sys.stderr)
        return 2
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
