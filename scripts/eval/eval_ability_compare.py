#!/usr/bin/env python3
"""
能力对比评测（CortexNet vs Transformer）。

输出内容：
1) 固定题集答案对照
2) Cortex vs Transformer 的文本相似度
3) 规则化能力检查（sanity checks）通过率
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from difflib import SequenceMatcher
from typing import Any, Dict, List

import torch
from tokenizers import Tokenizer

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from cortexnet.ops.device_manager import resolve_device_string, resolve_dtype_for_device


ABILITY_CASES: List[Dict[str, Any]] = [
    {
        "id": "arch_diff",
        "prompt": "请用一句话介绍 Transformer 和 CortexNet 在推理路径上的核心差异。",
        "checks": {"contains_any": ["Transformer", "CortexNet"]},
    },
    {
        "id": "llm_risks",
        "prompt": "请用三点总结大模型落地时最容易踩的坑。",
        "checks": {"min_chars": 20},
    },
    {
        "id": "fib_code",
        "prompt": "写一个 Python 函数，返回前 10 项斐波那契数列。",
        "checks": {"contains_all": ["def", "return"]},
    },
    {
        "id": "safety_checks",
        "prompt": "请给出 3 条可执行的 AI 安全上线检查项。",
        "checks": {"min_chars": 20},
    },
    {
        "id": "email_regex",
        "prompt": "写一个 Python 函数，用正则判断邮箱是否合法，并给出一个测试示例。",
        "checks": {"contains_all": ["def", "re"]},
    },
]


def resolve_device(device: str) -> str:
    return resolve_device_string(
        device,
        auto_priority=("npu", "cuda", "mps", "cpu"),
        allow_fallback=True,
    )


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    return resolve_dtype_for_device(dtype, device)


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-8)
    if top_k > 0 and top_k < logits.size(-1):
        values, _ = torch.topk(logits, top_k, dim=-1)
        threshold = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove_mask = torch.zeros_like(logits, dtype=torch.bool)
        remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
        logits = logits.masked_fill(remove_mask, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, repetition_penalty: float) -> torch.Tensor:
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


@torch.no_grad()
def generate_cortex(
    model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> torch.Tensor:
    generated = input_ids
    past_cache = None
    for _ in range(max_new_tokens):
        idx_cond = generated if past_cache is None else generated[:, -1:]
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
    return generated


@torch.no_grad()
def generate_transformer(
    model: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> torch.Tensor:
    generated = input_ids
    past_key_values = None
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
    return generated


def run_checks(answer: str, checks: Dict[str, Any]) -> Dict[str, bool]:
    text = answer.strip()
    results: Dict[str, bool] = {}

    min_chars = int(checks.get("min_chars", 0))
    if min_chars > 0:
        results["min_chars"] = len(text) >= min_chars

    contains_all = checks.get("contains_all", [])
    for kw in contains_all:
        results[f"contains_all:{kw}"] = kw in text

    contains_any = checks.get("contains_any", [])
    if contains_any:
        results["contains_any"] = any(kw in text for kw in contains_any)

    if not results:
        results["non_empty"] = len(text) > 0
    return results


def check_score(results: Dict[str, bool]) -> float:
    if not results:
        return 0.0
    ok = sum(1 for v in results.values() if v)
    return ok / len(results)


def run_worker(args: argparse.Namespace) -> Dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    tokenizer_path = os.path.join(args.model_path, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

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

    runtime_device = device
    if args.engine == "cortex" and hasattr(model, "get_lazy_status"):
        lazy_status = model.get_lazy_status()
        if lazy_status.get("enabled") and not lazy_status.get("ready"):
            runtime_device = "cpu"
        elif lazy_status.get("enabled") and lazy_status.get("ready") and lazy_status.get("target_device"):
            runtime_device = str(lazy_status.get("target_device"))

    outputs = []
    for case in ABILITY_CASES:
        ids = torch.tensor([tokenizer.encode(case["prompt"]).ids], dtype=torch.long, device=runtime_device)
        generated = generate_fn(
            model=model,
            input_ids=ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        answer_ids = generated[0, ids.shape[1] :].to("cpu").tolist()
        answer = tokenizer.decode(answer_ids).strip()
        checks = run_checks(answer, case.get("checks", {}))
        outputs.append(
            {
                "id": case["id"],
                "prompt": case["prompt"],
                "answer": answer,
                "checks": checks,
                "check_score": round(check_score(checks), 4),
            }
        )

    avg_score = sum(o["check_score"] for o in outputs) / max(len(outputs), 1)
    return {
        "engine": args.engine,
        "engine_name": engine_name,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "avg_check_score": round(avg_score, 4),
        "outputs": outputs,
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
            raise RuntimeError(f"Eval worker failed: {engine}")

        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("JSON_RESULT:"):
                payload = line[len("JSON_RESULT:") :].strip()
        if payload is None:
            raise RuntimeError(f"No JSON_RESULT found for engine={engine}. stdout={proc.stdout}")
        return json.loads(payload)

    print("=== Ability Eval: CortexNet vs Transformer ===")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}, DType: {args.dtype}, max_new_tokens={args.max_new_tokens}")
    print("")

    cortex = _run_engine("cortex")
    transformer = _run_engine("transformer")
    t_map = {o["id"]: o for o in transformer["outputs"]}

    exact_match = 0
    sim_scores: List[float] = []

    for c in cortex["outputs"]:
        t = t_map[c["id"]]
        sim = SequenceMatcher(None, c["answer"], t["answer"]).ratio()
        sim_scores.append(sim)
        if c["answer"] == t["answer"]:
            exact_match += 1

    avg_sim = sum(sim_scores) / max(len(sim_scores), 1)
    total = len(cortex["outputs"])

    print("=== Summary ===")
    print(f"Cortex avg check score:      {cortex['avg_check_score']:.4f}")
    print(f"Transformer avg check score: {transformer['avg_check_score']:.4f}")
    print(f"Exact match count:           {exact_match}/{total}")
    print(f"Avg similarity ratio:        {avg_sim:.4f}")
    print("")

    print("=== Per-case Comparison ===")
    for idx, c in enumerate(cortex["outputs"], start=1):
        t = t_map[c["id"]]
        sim = SequenceMatcher(None, c["answer"], t["answer"]).ratio()
        print(f"{idx}. [{c['id']}] {c['prompt']}")
        print(f"   Cortex (score={c['check_score']:.2f}): {c['answer']}")
        print(f"   Trans  (score={t['check_score']:.2f}): {t['answer']}")
        print(f"   Similarity: {sim:.4f}")
        print("")

    merged = {
        "summary": {
            "cortex_avg_check_score": cortex["avg_check_score"],
            "transformer_avg_check_score": transformer["avg_check_score"],
            "exact_match_count": exact_match,
            "total_cases": total,
            "avg_similarity": round(avg_sim, 4),
        },
        "cortex": cortex,
        "transformer": transformer,
    }
    if args.save_json:
        out_path = os.path.abspath(args.save_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON report: {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ability eval for CortexNet vs Transformer")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/Users/pengjiajun/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
        help="本地模型目录（含 config.json / tokenizer.json / safetensors）",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/mps/cuda/npu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
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
