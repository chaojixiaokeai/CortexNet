"""
推理适配层 (Inference Adapter)

核心功能：
  提供统一的生成接口，自动适配不同模型的默认参数、
  生成策略和加速配置。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# 各模型族的默认生成参数
DEFAULT_GENERATION_PARAMS: Dict[str, Dict[str, Any]] = {
    "llama": {
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "max_new_tokens": 512,
    },
    "qwen2": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
        "max_new_tokens": 512,
    },
    "qwen3": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
        "max_new_tokens": 512,
    },
    "mistral": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512,
    },
    "chatglm": {
        "temperature": 0.8,
        "top_p": 0.8,
        "top_k": 50,
        "repetition_penalty": 1.02,
        "max_new_tokens": 512,
    },
    "baichuan": {
        "temperature": 0.3,
        "top_p": 0.85,
        "top_k": 5,
        "repetition_penalty": 1.05,
        "max_new_tokens": 512,
    },
    "phi": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "max_new_tokens": 256,
    },
    "default": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512,
    },
}


class InferenceAdapter:
    """推理适配器：统一生成接口。

    自动匹配源模型的默认生成参数，并启用 CortexNet 加速策略。

    Args:
        cortex_model: CortexNet 模型实例
        model_type: 源模型类型
    """

    def __init__(self, cortex_model: Any, model_type: str = "default"):
        self.model = cortex_model
        self.model_type = model_type
        self.default_params = self._get_default_params()

    def _get_default_params(self) -> Dict[str, Any]:
        """获取模型族的默认生成参数。"""
        params = DEFAULT_GENERATION_PARAMS.get(
            self.model_type,
            DEFAULT_GENERATION_PARAMS["default"],
        )
        return dict(params)

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        stream: bool = False,
        use_speculative: bool = False,
        **kwargs,
    ):
        """统一生成接口。

        Args:
            input_ids: (batch, seq_len) 输入 token 索引
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-k 采样
            top_p: 核采样阈值
            repetition_penalty: 重复惩罚
            stream: 是否流式输出
            use_speculative: 是否使用推测式解码
            **kwargs: 其他生成参数

        Returns:
            stream=False: (batch, seq_len + max_new_tokens) token 索引
            stream=True: token 迭代器
        """
        # 合并参数（用户参数优先 > 模型默认参数）
        params = dict(self.default_params)
        if max_new_tokens is not None:
            params["max_new_tokens"] = max_new_tokens
        if temperature is not None:
            params["temperature"] = temperature
        if top_k is not None:
            params["top_k"] = top_k
        if top_p is not None:
            params["top_p"] = top_p
        if repetition_penalty is not None:
            params["repetition_penalty"] = repetition_penalty
        params.update(kwargs)

        if stream:
            return self._stream_generate(input_ids, params)

        # 推测式解码（如果模型支持且已开启）
        if use_speculative and hasattr(self.model, 'speculative_generate'):
            return self.model.speculative_generate(
                input_ids,
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_k=params["top_k"],
            )

        # 标准生成
        return self.model.generate(
            input_ids,
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            repetition_penalty=params.get("repetition_penalty", 1.0),
        )

    def _stream_generate(
        self,
        input_ids: torch.Tensor,
        params: Dict[str, Any],
    ) -> Iterator[torch.Tensor]:
        """流式生成：逐 token 产出。

        Yields:
            每步新生成的 token (batch, 1)
        """
        self.model.eval()
        generated = input_ids
        max_new_tokens = params["max_new_tokens"]
        temperature = params["temperature"]
        top_k = params["top_k"]
        top_p = params["top_p"]
        repetition_penalty = params.get("repetition_penalty", 1.0)
        defer_lazy_warmup = bool(
            getattr(self.model, "_lazy_enabled", False)
            and not getattr(self.model, "_lazy_ready", True)
        )
        supports_cache_forward = (
            hasattr(self.model, "forward")
            and hasattr(self.model.forward, "__code__")
            and "past_cache" in self.model.forward.__code__.co_varnames
        )
        use_repetition_penalty = repetition_penalty != 1.0 and repetition_penalty > 0

        past_cache = None
        seen_mask: Optional[torch.Tensor] = None

        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    if past_cache is None:
                        idx_cond = generated
                    else:
                        idx_cond = generated[:, -1:]

                    # 前向
                    if supports_cache_forward:
                        output = self.model.forward(
                            idx_cond,
                            past_cache=past_cache,
                            use_cache=True,
                            start_warmup_after_infer=not defer_lazy_warmup,
                        )
                        past_cache = output.get("past_cache")
                    else:
                        output = self.model.forward(idx_cond)

                    logits = output["logits"][:, -1, :] / max(temperature, 1e-8)

                    # 向量化重复惩罚：避免 Python token 循环
                    if use_repetition_penalty:
                        if seen_mask is None or seen_mask.shape[1] != logits.shape[1]:
                            seen_mask = torch.zeros(
                                logits.shape[0],
                                logits.shape[1],
                                device=logits.device,
                                dtype=torch.bool,
                            )
                            seen_ids = generated.clamp_max(logits.shape[1] - 1)
                            seen_mask.scatter_(1, seen_ids, True)
                        positive = logits > 0
                        logits = torch.where(
                            seen_mask & positive, logits / repetition_penalty, logits
                        )
                        logits = torch.where(
                            seen_mask & ~positive, logits * repetition_penalty, logits
                        )

                    # Top-k
                    if top_k > 0:
                        top_k_val = min(top_k, logits.size(-1))
                        _, top_indices = torch.topk(logits, top_k_val)
                        mask = torch.ones_like(logits, dtype=torch.bool)
                        mask.scatter_(1, top_indices, False)
                        logits.masked_fill_(mask, float("-inf"))

                    # Top-p
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_remove = cumprobs > top_p
                        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                        sorted_remove[..., 0] = 0
                        remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
                        logits[remove] = float("-inf")

                    # NaN/Inf 安全处理（float16 可能溢出）
                    logits = logits.clamp(-100, 100)
                    logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
                    probs = F.softmax(logits, dim=-1)
                    probs = probs.clamp(min=1e-8)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=1)
                    if seen_mask is not None:
                        seen_mask.scatter_(1, next_token, True)
                    yield next_token
        finally:
            if defer_lazy_warmup and hasattr(self.model, "start_background_warmup"):
                self.model.start_background_warmup()
