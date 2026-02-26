"""
CortexBlockLite — 精简版 CortexNet 块

仅保留三条核心路径 + SwiGLU FFN，参数量与 Transformer 相当：
  1. Multi-Scale SSM  — O(1) 解码，长序列优势
  2. Selective Attention — 兼容预训练权重
  3. Synaptic Memory   — 快速权重系统

相比 CortexBlockV3 移除了：
  - MultiAgentSystem (23.7%)
  - CollaborativeMoE → 直接 SwiGLU FFN
  - CausalReasoning (8.0%)
  - TaskAdaptiveController (6.3%)
  - GraphReasoning (5.2%)
  - MetaLearningAdapter (4.2%)
  - AdversarialShield (2.1%)

典型参数节省：60-70%
"""

from __future__ import annotations

from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:
    from .blocks import RMSNorm
    from .config import CortexNetConfig
    from .ssm import MultiScaleSSM
    from .attention import SelectiveSparseAttention
    from .memory import SynapticMemory
except ImportError:
    from cortexnet.blocks import RMSNorm
    from cortexnet.config import CortexNetConfig
    from cortexnet.ssm import MultiScaleSSM
    from cortexnet.attention import SelectiveSparseAttention
    from cortexnet.memory import SynapticMemory


class SwiGLU_FFN(nn.Module):
    """SwiGLU 前馈网络（与 LLaMA/Qwen 兼容）。

    参数量 = 3 × d_model × intermediate_size（无 bias）。
    """

    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CortexBlockLite(nn.Module):
    """精简 CortexNet 块：SSM + Attention + Memory + SwiGLU FFN。

    核心设计：
      - 三条并行路径（SSM, Attention, Memory）通过自适应融合门控合并
      - SwiGLU FFN 替代重量级 MoE（参数减少 80%+）
      - 支持 KV cache 推理和梯度检查点
      - SSM 优先模式：长序列生成时可跳过 Attention

    Args:
        config: CortexNetConfig 配置对象
        layer_idx: 当前层索引（用于 RoPE 偏移计算）
    """

    def __init__(self, config: CortexNetConfig, layer_idx: int = 0):
        super().__init__()
        D = config.hidden_size
        self.layer_idx = layer_idx
        self.use_checkpoint = config.use_gradient_checkpointing

        # ═══ 归一化 ═══
        self.norm1 = RMSNorm(D, eps=config.norm_eps)
        self.norm2 = RMSNorm(D, eps=config.norm_eps)

        # ═══ 路径 1: 低秩 SSM（O(1) 解码，O(d×rank) 参数） ═══
        # Lite 模式：SSM 在降维空间运行，大幅减少参数量
        ssm_rank = getattr(config, 'compat_ssm_rank', max(64, D // 4))
        ssm_rank = min(ssm_rank, D)  # 不超过 D
        self.ssm_rank = ssm_rank
        self.ssm_down = nn.Linear(D, ssm_rank, bias=False) if ssm_rank < D else nn.Identity()
        self.ssm = MultiScaleSSM(
            d_model=ssm_rank,
            num_scales=1,
            state_size=config.ssm_state_size,
            expand_factor=1,
        )
        self.ssm_up = nn.Linear(ssm_rank, D, bias=False) if ssm_rank < D else nn.Identity()

        # ═══ 路径 2: Selective Sparse Attention ═══
        self.attention = SelectiveSparseAttention(
            d_model=D,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            top_k_ratio=config.top_k_ratio,
            k_mode=config.attention_k_mode,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
            sliding_window_size=config.sliding_window_size,
            use_qk_norm=getattr(config, "use_qk_norm", False),
            norm_eps=config.norm_eps,
        )

        # ═══ 路径 3: Synaptic Memory ═══
        # 大模型用较小 memory_dim 避免参数膨胀
        mem_dim = min(config.memory_dim, max(32, D // 16))
        self.memory = SynapticMemory(
            d_model=D,
            memory_dim=mem_dim,
            decay_init=config.memory_decay_init,
        )

        # ═══ 轻量级三路径融合（3 个标量权重） ═══
        # 替代 AdaptiveFusionGate 的全连接网络，参数从 O(d²) 降至 3
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

        # ═══ SwiGLU FFN（替代 MoE） ═══
        ff_dim = config.intermediate_size if config.intermediate_size > 0 else config.expert_ff_dim
        self.ffn = SwiGLU_FFN(D, ff_dim)

        # ═══ Dropout ═══
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        past_cache: Optional[Tuple[Any, Any, Any]] = None,
        use_cache: bool = False,
        ssm_only: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            past_cache: (past_attn_kv, past_ssm_state, past_mem_state)
            use_cache: 是否返回缓存
            ssm_only: True → 纯 SSM 模式（跳过 Attention，极速解码）

        Returns:
            use_cache=False: output (B, L, D)
            use_cache=True: (output, (attn_cache, ssm_cache, mem_cache))
        """
        residual = x
        x_norm = self.norm1(x)

        # 解包缓存
        past_attn = past_ssm = past_mem = None
        if past_cache is not None:
            past_attn, past_ssm, past_mem = past_cache

        # ═══ SSM 路径（低秩空间） ═══
        ssm_input = self.ssm_down(x_norm)
        if use_cache:
            ssm_out_lr, new_ssm_cache = self.ssm(
                ssm_input, past_state=past_ssm, use_cache=True,
            )
        else:
            ssm_out_lr = self._maybe_checkpoint(self.ssm, ssm_input)
            new_ssm_cache = None
        ssm_out = self.ssm_up(ssm_out_lr)

        if ssm_only:
            # 纯 SSM 模式：跳过 Attention 和 Memory
            x = residual + self.dropout(ssm_out)
        else:
            # ═══ Attention 路径 ═══
            if use_cache:
                attn_out, new_attn_cache = self.attention(
                    x_norm, past_key_value=past_attn, use_cache=True,
                )
            else:
                attn_out = self._maybe_checkpoint(self.attention, x_norm)
                new_attn_cache = None

            # ═══ Memory 路径 ═══
            if use_cache:
                past_memory = past_mem[0] if isinstance(past_mem, tuple) else past_mem
                past_z = past_mem[1] if isinstance(past_mem, tuple) else None
                mem_out, new_mem, new_z = self.memory(
                    x_norm, past_memory=past_memory, past_z=past_z, use_cache=True,
                )
                new_mem_cache = (new_mem, new_z)
            else:
                mem_out = self._maybe_checkpoint(self.memory, x_norm)
                new_mem_cache = None

            # ═══ 轻量融合 ═══
            w = torch.softmax(self.fusion_weights, dim=0)
            fused = w[0] * ssm_out + w[1] * attn_out + w[2] * mem_out
            x = residual + self.dropout(fused)

        # ═══ FFN ═══
        residual = x
        x = residual + self.dropout(self.ffn(self.norm2(x)))

        if use_cache:
            cache = (
                new_attn_cache if not ssm_only else past_attn,
                new_ssm_cache,
                new_mem_cache if not ssm_only else past_mem,
            )
            return x, cache
        return x

    def _maybe_checkpoint(self, module, *args):
        """条件梯度检查点。"""
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def get_aux_loss(self) -> torch.Tensor:
        """Lite 模式无 MoE，返回零损失。"""
        return self.fusion_weights.new_zeros(())
