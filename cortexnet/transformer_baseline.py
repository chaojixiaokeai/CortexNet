"""
Transformer 基线模型 (Transformer Baseline)

标准 Transformer 语言模型，用于与 CortexNet 进行公平对比。
使用 Pre-LN (Pre-LayerNorm) 结构 + RoPE 位置编码。
"""

from __future__ import annotations

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import RMSNorm
from .attention import precompute_rope_freqs, apply_rope


class TransformerBlock(nn.Module):
    """标准 Transformer 解码器块（Pre-LN）。"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_len: int = 8192, dropout: float = 0.0,
                 rope_theta: float = 10000.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # 自注意力
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # FFN (SwiGLU)
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # RoPE
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.head_dim, max_seq_len, rope_theta),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x_norm = self.norm1(x)

        # Multi-head attention with RoPE
        q = self.q_proj(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)

        # Causal attention
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        x = residual + self.dropout(self.o_proj(attn_out))

        # SwiGLU FFN
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.dropout(
            self.down_proj(F.silu(self.gate_proj(x_norm)) * self.up_proj(x_norm))
        )

        return x


class TransformerLM(nn.Module):
    """标准 Transformer 语言模型（用于对比基线）。

    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度
        num_layers: 层数
        num_heads: 注意力头数
        d_ff: FFN 中间维度
        max_seq_len: 最大序列长度
        dropout: Dropout 比率
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, dropout, rope_theta)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, L = input_ids.shape
        x = self.embed_dropout(self.embed(input_ids))

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result
