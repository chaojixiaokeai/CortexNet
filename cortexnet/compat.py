"""
CortexNet 兼容模式组件 (Compatibility Mode Components)

为开源 LLM (LLaMA/Qwen/Mistral...) 的权重无损迁移提供轻量兼容组件：
  - _CompatAttention: GQA + KV cache 注意力
  - _CompatLiteSSM: 低秩 SSM 旁路（默认零影响）
  - _CompatFusionGate: 轻量两路融合（强偏向 Attention）
  - _CompatExpert/_CompatMoE: 单专家 FFN 兼容壳
  - _CompatCortexBlockV3: 完整的 V3 兼容块
  - _NoOpEvolutionEngine: 占位进化引擎

这些组件在 compatibility_mode=True 时替代完整 V3 模块使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from .config import CortexNetConfig
    from .blocks import RMSNorm
    from .attention import precompute_rope_freqs, apply_rope, apply_rope_with_positions
except ImportError:
    from cortexnet.config import CortexNetConfig
    from cortexnet.blocks import RMSNorm
    from cortexnet.attention import precompute_rope_freqs, apply_rope, apply_rope_with_positions


class _NoOpEvolutionEngine(nn.Module):
    """兼容模式下的轻量占位引擎。"""

    def __init__(self):
        super().__init__()

    def get_compute_budget(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.size(0), device=x.device, dtype=x.dtype)

    def get_efficiency_loss(self) -> float:
        return 0.0


class _CompatAttention(nn.Module):
    """与主流 HF 架构对齐的轻量注意力（支持 GQA + KV cache）。"""

    def __init__(self, config: CortexNetConfig):
        super().__init__()
        self.d_model = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = self.d_model // self.num_heads

        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        # Qwen2/3 常见 q_norm/k_norm
        self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.norm_eps)

        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.head_dim, config.max_seq_len, config.rope_theta),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        rope = self.rope_freqs.to(x.device)
        if past_key_value is None:
            q = apply_rope(q, rope)
            k = apply_rope(k, rope)
        else:
            past_k, past_v = past_key_value
            pos_offset = past_k.shape[2]
            pos = torch.arange(
                pos_offset, pos_offset + L, device=x.device
            ).unsqueeze(0).expand(B, -1)
            q = apply_rope_with_positions(q, rope, pos)
            k = apply_rope_with_positions(k, rope, pos)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # GQA: 将 KV 头扩展到 Query 头
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k_attn = k.repeat_interleave(repeat, dim=1)
            v_attn = v.repeat_interleave(repeat, dim=1)
        else:
            k_attn = k
            v_attn = v

        attn_out = F.scaled_dot_product_attention(
            q, k_attn, v_attn,
            attn_mask=None, dropout_p=0.0,
            is_causal=(past_key_value is None),
        )
        out = self.o_proj(attn_out.transpose(1, 2).contiguous().view(B, L, D))

        if use_cache:
            return out, (k, v)
        return out


class _CompatLiteSSM(nn.Module):
    """轻量 SSM 路径（参数开销小，默认零影响，支持后续无感校准启用）。"""

    def __init__(self, d_model: int, rank: int = 256, skip_threshold: float = 1e-6):
        super().__init__()
        self.rank = max(16, min(rank, d_model))
        self.skip_threshold = float(skip_threshold)
        self.fast_skip = True
        self.in_proj = nn.Linear(d_model, self.rank, bias=False)
        self.out_proj = nn.Linear(self.rank, d_model, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def is_effectively_disabled(self) -> bool:
        return (not self.training) and bool(self.fast_skip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_effectively_disabled():
            return torch.zeros_like(x)
        return self.alpha * self.out_proj(torch.tanh(self.in_proj(x)))


class _CompatFusionGate(nn.Module):
    """轻量融合门：仅融合 SSM/Attention 两路，初始化强偏向 Attention。"""

    def __init__(
        self, d_model: int,
        long_context_threshold: int = 2048,
        long_context_ssm_ratio: float = 0.35,
    ):
        super().__init__()
        bottleneck = max(d_model // 32, 32)
        self.long_context_threshold = int(long_context_threshold)
        self.long_context_ssm_ratio = float(min(max(long_context_ssm_ratio, 0.0), 0.95))
        self.context_proj = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, 1),
        )
        self.attn_bias = nn.Parameter(torch.tensor(10.0))
        with torch.no_grad():
            for module in self.context_proj:
                if isinstance(module, nn.Linear):
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()

    def forward(self, x_norm, ssm_out, attn_out, *, ssm_enabled=True):
        if not ssm_enabled:
            return attn_out
        context = x_norm.mean(dim=1)
        gate = torch.sigmoid(self.attn_bias + self.context_proj(context)).unsqueeze(1)
        if self.long_context_threshold > 0 and x_norm.size(1) >= self.long_context_threshold:
            max_gate = 1.0 - self.long_context_ssm_ratio
            gate = torch.clamp(gate, min=0.0, max=max_gate)
        return gate * attn_out + (1.0 - gate) * ssm_out


class _CompatExpert(nn.Module):
    """与源模型 FFN 权重命名对齐的单专家。"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _CompatMoE(nn.Module):
    """轻量 MoE 兼容壳：保留 experts.0 参数路径，前向等价单 FFN。"""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.experts = nn.ModuleList([_CompatExpert(d_model, d_ff)])
        self.router = nn.Linear(d_model, 1, bias=False)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.aux_loss = x.new_zeros(())
        return self.experts[0](x)


class _CompatCortexBlockV3(nn.Module):
    """V3 兼容块：保留 SSM + 稀疏注意力 + 轻量融合门，去除非核心大模块。"""

    def __init__(self, config: CortexNetConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, config.norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.norm_eps)
        self.ssm = _CompatLiteSSM(
            config.hidden_size,
            rank=getattr(config, "compat_ssm_rank", 256),
        )
        self.attention = _CompatAttention(config)
        self.fusion = _CompatFusionGate(
            config.hidden_size,
            long_context_threshold=getattr(config, "fusion_long_context_threshold", 2048),
            long_context_ssm_ratio=getattr(config, "fusion_long_context_ssm_ratio", 0.35),
        )
        self.moe = _CompatMoE(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, past_cache=None, use_cache=False):
        residual = x
        x_norm = self.norm1(x)
        ssm_enabled = not self.ssm.is_effectively_disabled()
        ssm_out = self.ssm(x_norm) if ssm_enabled else None

        if use_cache:
            attn_out, new_cache = self.attention(x_norm, past_key_value=past_cache, use_cache=True)
        else:
            attn_out = self.attention(x_norm, past_key_value=past_cache, use_cache=False)
            new_cache = None

        fused = self.fusion(x_norm, ssm_out, attn_out, ssm_enabled=(ssm_enabled and ssm_out is not None)) if ssm_enabled and ssm_out is not None else attn_out
        x = residual + self.dropout(fused)
        residual = x
        x = residual + self.dropout(self.moe(self.norm2(x)))

        if use_cache:
            return x, new_cache
        return x

    def get_aux_loss(self):
        return self.moe.aux_loss
