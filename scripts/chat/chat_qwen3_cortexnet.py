#!/usr/bin/env python3
"""
Qwen3-8B + CortexNet 聊天脚本

用法示例：
  python chat_qwen3_cortexnet.py
  python chat_qwen3_cortexnet.py --prompt "你好，请介绍一下你自己"
  python chat_qwen3_cortexnet.py --model-path "/path/to/Qwen3-8B" --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import torch
from tokenizers import Tokenizer

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from cortexnet import CortexNet
from cortexnet.ops.device_manager import resolve_device_string, resolve_dtype_for_device


class LocalTokenizer:
    """基于 tokenizer.json 的本地 tokenizer（不依赖 transformers）。"""

    def __init__(self, model_path: str):
        tok_file = os.path.join(model_path, "tokenizer.json")
        if not os.path.exists(tok_file):
            raise FileNotFoundError(f"缺少 tokenizer.json: {tok_file}")
        self._tok = Tokenizer.from_file(tok_file)

        cfg_file = os.path.join(model_path, "config.json")
        if os.path.exists(cfg_file):
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.eos_token_id = int(cfg.get("eos_token_id", -1))
            self.bos_token_id = int(cfg.get("bos_token_id", -1))
        else:
            self.eos_token_id = -1
            self.bos_token_id = -1

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> torch.Tensor:
        ids = self._tok.encode(text).ids
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids_list = ids.tolist()
        else:
            ids_list = list(ids)
        return self._tok.decode(ids_list, skip_special_tokens=skip_special_tokens)


def choose_device(device_arg: str) -> str:
    return resolve_device_string(
        device_arg,
        auto_priority=("npu", "cuda", "mps", "cpu"),
        allow_fallback=True,
    )


def choose_dtype(dtype_arg: str, device: str) -> torch.dtype:
    return resolve_dtype_for_device(dtype_arg, device)


def build_prompt(
    system_prompt: str,
    history: List[Tuple[str, str]],
    user_text: str,
    thinking_enabled: bool = False,
) -> str:
    """构建 Qwen3 ChatML 格式 prompt。"""
    current_user_text = user_text
    if (not thinking_enabled) and "/no_think" not in user_text:
        current_user_text = f"/no_think {user_text}"

    parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>\n"]
    for user_turn, assistant_turn in history:
        parts.append(f"<|im_start|>user\n{user_turn}<|im_end|>\n")
        parts.append(f"<|im_start|>assistant\n{assistant_turn}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{current_user_text}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def apply_thinking_mode(system_prompt: str, enabled: bool) -> str:
    if not enabled:
        return system_prompt + "\n请直接给出最终回答，不要输出思考过程或 <think> 标签。"
    return (
        system_prompt
    )


def strip_think_content(text: str) -> str:
    """移除 <think>...</think>，若未闭合则截断到 <think> 前。"""
    out = text
    while True:
        s = out.find("<think>")
        if s < 0:
            return out
        e = out.find("</think>", s + len("<think>"))
        if e < 0:
            return out[:s]
        out = out[:s] + out[e + len("</think>"):]


def trim_stop_markers(text: str, thinking_enabled: bool = False) -> Tuple[str, bool]:
    out = text

    # 清理前缀控制标记（部分 tokenizer 会把特殊 token 解码成字面量文本）。
    leading_tokens = [
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<|im_start|>",
        "<|im_end|>",
        "</think>",
        "<think>",
    ]
    changed = True
    while changed:
        changed = False
        for token in leading_tokens:
            if out.startswith(token):
                out = out[len(token):].lstrip()
                changed = True

    if not thinking_enabled:
        think_start = out.find("<think>")
        if think_start >= 0:
            think_end = out.find("</think>", think_start + len("<think>"))
            if think_end >= 0:
                # 思考块在前缀时直接丢弃，保留其后的最终回答部分。
                if think_start == 0:
                    out = out[think_end + len("</think>"):].lstrip()
                else:
                    return out[:think_end], True

    stop_tokens = ["<|im_end|>", "<|im_start|>", "\n用户：", "\n系统："]
    if not thinking_enabled:
        stop_tokens.insert(0, "</think>")
    for stop_token in stop_tokens:
        idx = out.find(stop_token)
        if idx > 0:
            return out[:idx], True
        if idx == 0:
            out = out[len(stop_token):].lstrip()
            return out, True
    return out, False


def lcp_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def resolve_runtime_device(model: CortexNet, preferred_device: str) -> str:
    if hasattr(model, "get_lazy_status"):
        status = model.get_lazy_status()
        if status.get("enabled") and not status.get("ready"):
            return "cpu"
        if status.get("enabled") and status.get("ready") and status.get("target_device"):
            return str(status.get("target_device"))
    return preferred_device


def generate_once(
    model: CortexNet,
    tokenizer: LocalTokenizer,
    prompt: str,
    device: str,
    max_context_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    thinking_enabled: bool = False,
) -> str:
    input_ids = tokenizer.encode(prompt)

    if input_ids.shape[1] > max_context_tokens:
        input_ids = input_ids[:, -max_context_tokens:]

    run_device = resolve_runtime_device(model, device)
    input_ids = input_ids.to(run_device)

    def _decode_generated(out_tensor: torch.Tensor) -> str:
        new_ids_local = out_tensor[0, input_ids.shape[1]:].to("cpu")
        text_local = tokenizer.decode(new_ids_local, skip_special_tokens=True)
        text_local, _ = trim_stop_markers(text_local, thinking_enabled=thinking_enabled)
        if not thinking_enabled:
            cleaned = strip_think_content(text_local).strip()
            if cleaned:
                return cleaned
            # 兜底：避免空回复，至少返回去标签后的内容。
            return text_local.replace("<think>", "").replace("</think>", "").strip()
        return text_local.strip()

    with torch.no_grad():
        out = model.smart_generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    text = _decode_generated(out)
    if text or thinking_enabled:
        return text

    # 非思考模式兜底：若首轮被思考块耗尽导致空输出，自动放宽生成长度重试一次。
    retry_tokens = max(max_new_tokens * 2, 256)
    with torch.no_grad():
        out_retry = model.smart_generate(
            input_ids,
            max_new_tokens=retry_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
    return _decode_generated(out_retry)


def stream_generate_once(
    model: CortexNet,
    tokenizer: LocalTokenizer,
    prompt: str,
    device: str,
    max_context_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    thinking_enabled: bool = False,
) -> str:
    input_ids = tokenizer.encode(prompt)
    if input_ids.shape[1] > max_context_tokens:
        input_ids = input_ids[:, -max_context_tokens:]
    run_device = resolve_runtime_device(model, device)
    input_ids = input_ids.to(run_device)

    token_iter = None
    if hasattr(model, "_inference_adapter"):
        token_iter = model._inference_adapter.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stream=True,
        )

    if token_iter is None:
        return generate_once(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_context_tokens=max_context_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    generated_ids: List[int] = []
    prev_text = ""
    for next_token in token_iter:
        token_id = int(next_token[0, 0].item())
        if tokenizer.eos_token_id >= 0 and token_id == tokenizer.eos_token_id:
            break
        generated_ids.append(token_id)
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        text, should_stop = trim_stop_markers(text, thinking_enabled=thinking_enabled)
        if not thinking_enabled:
            text = strip_think_content(text)
        prefix = lcp_len(prev_text, text)
        if prefix < len(prev_text):
            # 出现回写时暂不输出，等待后续稳定增量，避免重复打印
            delta = ""
        else:
            delta = text[prefix:]
        if delta:
            print(delta, end="", flush=True)
        prev_text = text
        if should_stop:
            break
    print()
    return prev_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Qwen3-8B + CortexNet 聊天脚本")
    parser.add_argument(
        "--model-path",
        default="/Users/pengjiajun/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
        help="本地模型目录路径",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto/cpu/mps/cuda/npu/mlu（支持索引，如 npu:0）",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="推理精度",
    )
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否流式输出（默认开启）",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否显示思考摘要（不输出详细推理过程）",
    )
    parser.add_argument(
        "--system-prompt",
        default="你是一个严谨、简洁、可靠的中文 AI 助手。",
        help="系统提示词",
    )
    parser.add_argument("--prompt", default=None, help="单轮提问（设置后执行一次并退出）")
    parser.add_argument(
        "--lazy-device-load",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否启用惰性上设备（首次推理 CPU 兜底，后台预热到目标设备）",
    )
    parser.add_argument(
        "--lazy-background-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="惰性模式下是否在首次推理后自动后台预热",
    )
    parser.add_argument(
        "--lazy-cpu-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="惰性模式下未完成预热时是否允许 CPU 兜底推理",
    )
    parser.add_argument(
        "--compatibility-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="兼容模式（推荐开启，显著降低额外参数与显存开销）",
    )
    parser.add_argument(
        "--disable-calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否关闭加载后校准（推荐关闭，降低冷启动开销）",
    )
    parser.add_argument(
        "--mapped-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用映射后权重缓存（第二次加载明显更快）",
    )
    parser.add_argument(
        "--mapped-cache-dir",
        default="",
        help="映射缓存目录（可选）",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"模型目录不存在: {args.model_path}")

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)

    print(f"[加载] tokenizer.json: {args.model_path}")
    tokenizer = LocalTokenizer(args.model_path)

    print(f"[加载] CortexNet: device={device}, dtype={dtype}")
    model = CortexNet.from_pretrained(
        args.model_path,
        model_type="qwen3",
        load_weights=True,
        device=device,
        dtype=dtype,
        compatibility_mode=args.compatibility_mode,
        auto_calibrate=(not args.disable_calibrate),
        mapped_cache_enabled=args.mapped_cache,
        mapped_cache_dir=(args.mapped_cache_dir or None),
        lazy_device_load=args.lazy_device_load,
        lazy_background_warmup=args.lazy_background_warmup,
        lazy_cpu_fallback=args.lazy_cpu_fallback,
    )
    lite_mode = getattr(model, 'lite_mode', False)
    compat_mode = getattr(model, 'compatibility_mode', False)
    params = sum(p.numel() for p in model.parameters())
    print(f"[完成] 参数量: {params/1e9:.2f}B")

    if lite_mode:
        block0 = model.blocks[0]
        ssm_rank = getattr(block0, 'ssm_rank', 'N/A')
        print(f"[架构] CortexNet Lite: SSM(rank={ssm_rank}) + Attention + Memory + FFN")
    elif compat_mode:
        print("[架构] Core Mode: SSM + Sparse Attention + Lite Fusion")
    else:
        print("[架构] Core Mode: Full CortexNet")
    if args.stream:
        print("[模式] 流式输出: ON")
    if args.thinking:
        print("[模式] 思考摘要: ON")
    if hasattr(model, "get_lazy_status"):
        lazy_status = model.get_lazy_status()
        if lazy_status.get("enabled"):
            print(
                "[模式] 惰性上设备: ON "
                f"(ready={lazy_status.get('ready')}, target={lazy_status.get('target_device')})"
            )
    if args.compatibility_mode:
        print("[模式] compatibility_mode: ON")
    if args.disable_calibrate:
        print("[模式] calibrate: OFF")
    if args.mapped_cache:
        cache_dir = args.mapped_cache_dir or os.path.expanduser("~/.cache/cortexnet/mapped_weights")
        print(f"[模式] mapped_cache: ON ({cache_dir})")

    effective_system_prompt = apply_thinking_mode(args.system_prompt, args.thinking)

    if args.prompt is not None:
        prompt = build_prompt(
            effective_system_prompt,
            [],
            args.prompt,
            thinking_enabled=args.thinking,
        )
        print(f"\n用户：{args.prompt}")
        print("助手：", end="", flush=True)
        if args.stream:
            answer = stream_generate_once(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_context_tokens=args.max_context_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                thinking_enabled=args.thinking,
            )
        else:
            answer = generate_once(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_context_tokens=args.max_context_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                thinking_enabled=args.thinking,
            )
            print(answer)
        return

    print("\n进入交互模式，输入 /exit 退出。\n")
    history: List[Tuple[str, str]] = []
    while True:
        try:
            user_text = input("用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not user_text:
            continue
        if user_text in ("/exit", "exit", "quit", "/quit"):
            print("退出。")
            break

        prompt = build_prompt(
            effective_system_prompt,
            history,
            user_text,
            thinking_enabled=args.thinking,
        )
        print("助手：", end="", flush=True)
        if args.stream:
            answer = stream_generate_once(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_context_tokens=args.max_context_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                thinking_enabled=args.thinking,
            )
        else:
            answer = generate_once(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_context_tokens=args.max_context_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                thinking_enabled=args.thinking,
            )
            print(answer)
        print()
        history.append((user_text, answer))


if __name__ == "__main__":
    main()
