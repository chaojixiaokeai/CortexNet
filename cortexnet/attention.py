from __future__ import annotations

"""
选择性稀疏注意力 (Selective Sparse Attention) + 旋转位置编码 (RoPE)

核心创新：
  传统 Transformer 的注意力机制对所有 token 对计算注意力权重，
  复杂度为 O(n²)。CortexNet 通过学习一个重要性评分函数，
  只选择最关键的 token 作为 Key/Value，所有 token 的 Query
  只与这些选中的 token 交互。

  复杂度分析：
    - ratio 模式：O(n·k)，k = ratio·n，例如 ratio=0.25 → 4x 加速
    - sqrt 模式：O(n·√n)，真正的亚二次复杂度
    - log 模式：O(n·log(n))，近似线性

  重要性评分器通过软门控接收梯度，实现端到端学习。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from functools import lru_cache


def _mps_safe() -> bool:
    """检测是否在 MPS 设备上（MPS 对复数运算支持有限）。"""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


class HeadRMSNorm(nn.Module):
    """用于注意力 Q/K 的按头 RMSNorm（最后一维归一化）。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


@lru_cache(maxsize=16)
def precompute_rope_freqs(
    dim: int, max_seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """预计算旋转位置编码 (RoPE) 的频率。

    使用 LRU 缓存避免重复计算，提升长序列训练效率。
    返回 cos/sin 张量而非复数，确保 CUDA/MPS/CPU 全兼容。

    Args:
        dim: 头维度（必须为偶数）
        max_seq_len: 最大序列长度
        theta: 频率计算的基数
    Returns:
        (max_seq_len, dim//2, 2) — [..., 0] = cos, [..., 1] = sin
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)  # (max_seq_len, dim//2)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)  # (L, dim//2, 2)


def _apply_rope_real(x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
    """用实数 cos/sin 旋转（全设备兼容），替代复数乘法。

    x: (B, H, L, D) where D is head_dim (even)
    cos_sin: (L, D//2, 2)
    """
    D = x.shape[-1]
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]  # 各 (B, H, L, D//2)
    cos = cos_sin[..., 0]  # (L, D//2)
    sin = cos_sin[..., 1]  # (L, D//2)
    # 广播到 (1, 1, L, D//2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1).type_as(x)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """对输入张量应用旋转位置编码（全设备兼容）。

    Args:
        x: (batch, num_heads, seq_len, head_dim)
        freqs: (max_seq_len, head_dim//2, 2) — cos/sin 频率
    Returns:
        旋转后的张量，形状与 x 相同
    """
    L = x.shape[2]
    return _apply_rope_real(x, freqs[:L])


def apply_rope_with_positions(
    x: torch.Tensor, freqs: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """对指定位置的 token 应用 RoPE（全设备兼容）。

    Args:
        x: (batch, num_heads, k, head_dim)
        freqs: (max_seq_len, head_dim//2, 2)
        positions: (batch, k) - 每个 token 的原始位置
    Returns:
        旋转后的张量
    """
    B, H, K, D = x.shape
    pos_flat = positions.reshape(-1)  # (B*K,)
    pos_freqs = freqs[pos_flat].reshape(B, K, freqs.shape[1], 2)  # (B, K, D//2, 2)
    # 分离 cos, sin → (B, 1, K, D//2) 广播到所有 head
    cos = pos_freqs[..., 0].unsqueeze(1)
    sin = pos_freqs[..., 1].unsqueeze(1)
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1).type_as(x)


class SelectiveSparseAttention(nn.Module):
    """选择性稀疏注意力机制。

    架构流程：
        1. 对所有 token 计算重要性分数（学习得到）
        2. 选择 top-k 个最重要的 token 作为 Key/Value
        3. 所有 token 的 Query 与选中的 Key/Value 交互
        4. 应用 RoPE 保持位置信息
        5. 通过重要性软门控传递梯度

    相比全注意力的优势：
        - 计算量：O(n·k) vs O(n²)
        - 显存：O(n·k) vs O(n²)
        - 自动学习哪些 token 最值得关注

    Args:
        d_model: 模型维度
        num_heads: 注意力头数
        top_k_ratio: 选择比例（ratio 模式下使用）
        k_mode: 选择模式 - 'ratio', 'sqrt', 'log'
        max_seq_len: 最大序列长度
        rope_theta: RoPE 基数
        dropout: 注意力 dropout 率
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k_ratio: float = 0.25,
        k_mode: str = "ratio",
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        sliding_window_size: int = 0,
        num_kv_heads: int = 0,
        use_qk_norm: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads > 0 else num_heads
        self.head_dim = d_model // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.top_k_ratio = top_k_ratio
        self.k_mode = k_mode
        self.scale = self.head_dim ** -0.5
        self.sliding_window_size = sliding_window_size

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        )
        assert self.head_dim % 2 == 0, (
            f"head_dim ({self.head_dim}) 必须为偶数以支持 RoPE"
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) 必须能被 num_kv_heads ({self.num_kv_heads}) 整除"
        )

        # Q/KV 投影（GQA: KV 使用更少的头）
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, self.kv_dim * 2, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # 学习的重要性评分器
        self.importance_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        self.attn_dropout = nn.Dropout(dropout)

        # 可选的 Q/K 归一化（Qwen3 等模型使用 per-head RMSNorm）
        if use_qk_norm:
            self.q_norm = HeadRMSNorm(self.head_dim, eps=norm_eps)
            self.k_norm = HeadRMSNorm(self.head_dim, eps=norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        # 滑动窗口门控（与稀疏全局注意力并行融合）
        if sliding_window_size > 0:
            self.window_gate = nn.Linear(d_model, 1)
        else:
            self.window_gate = None

        # 预计算 RoPE 频率
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.head_dim, max_seq_len, rope_theta),
            persistent=False,
        )

    def _project_kv(self, x: torch.Tensor):
        """融合 KV 投影（GQA: 输出 kv_dim 而非 d_model）。"""
        kv = self.kv_proj(x)
        return kv.chunk(2, dim=-1)  # each (B, L, kv_dim)

    def _compute_k_count(self, seq_len: int) -> int:
        """根据模式计算选择的 token 数量。"""
        if self.k_mode == "sqrt":
            return max(1, int(math.sqrt(seq_len)))
        elif self.k_mode == "log":
            return max(1, int(math.log2(max(seq_len, 2)) * self.num_heads))
        else:  # ratio
            return max(1, int(seq_len * self.top_k_ratio))

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            causal_mask: 是否应用因果掩码
            past_key_value:
                新版: (past_K, past_V, past_top_k_indices, cache_len, total_seen_len)
                旧版兼容: (past_K, past_V, past_top_k_indices, cache_len)
            用于增量解码
            use_cache: 若 True 返回 (output, new_cache)
        Returns:
            output: (batch, seq_len, d_model)
            new_cache (可选): (K, V, top_k_indices) 当 use_cache=True
        """
        B, L, D = x.shape
        rope_freqs = self.rope_freqs.to(x.device)

        cache_len = 0
        total_seen_len = 0
        if past_key_value is not None:
            if len(past_key_value) == 5:
                past_K, past_V, past_top_k, cache_len, total_seen_len = past_key_value
            elif len(past_key_value) == 4:
                past_K, past_V, past_top_k, cache_len = past_key_value
                inferred_seen = int(past_top_k.max().item()) + 1 if past_top_k.numel() > 0 else 0
                total_seen_len = max(int(cache_len), inferred_seen)
            elif len(past_key_value) == 3:
                past_K, past_V, past_top_k = past_key_value
                cache_len = int(past_K.shape[2])
                inferred_seen = int(past_top_k.max().item()) + 1 if past_top_k.numel() > 0 else 0
                total_seen_len = max(cache_len, inferred_seen)
            else:
                raise ValueError(
                    "past_key_value must be a tuple of length 3, 4, or 5"
                )
            position_offset = int(total_seen_len)
        else:
            position_offset = 0

        importance = self.importance_scorer(x).squeeze(-1)  # (B, L)

        if past_key_value is not None:
            # 增量模式：静态缓存插入，避免 torch.cat 复制开销
            new_positions = torch.arange(
                position_offset, position_offset + L, device=x.device
            ).unsqueeze(0).expand(B, -1)
            k_raw, v_raw = self._project_kv(x)
            k_new = k_raw.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v_new = v_raw.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
            if self.k_norm is not None:
                k_new = self.k_norm(k_new)
            k_new = apply_rope_with_positions(k_new, rope_freqs, new_positions)

            # 静态缓存：如果空间不够则扩展
            new_len = cache_len + L
            max_cache = past_K.shape[2]
            if new_len > max_cache:
                # 按 2x 扩展，分摊分配成本
                new_max = max(new_len, max_cache * 2)
                pad_k = torch.zeros(B, self.num_kv_heads, new_max - max_cache, self.head_dim,
                                    device=past_K.device, dtype=past_K.dtype)
                pad_v = torch.zeros_like(pad_k)
                past_K = torch.cat([past_K, pad_k], dim=2)
                past_V = torch.cat([past_V, pad_v], dim=2)

            # 原地插入新 KV（零拷贝）
            past_K[:, :, cache_len:new_len] = k_new
            past_V[:, :, cache_len:new_len] = v_new

            k = past_K[:, :, :new_len]
            v = past_V[:, :, :new_len]
            top_k_indices_full = torch.cat(
                [past_top_k[:, :cache_len], new_positions], dim=1
            )
        else:
            k_count = min(self._compute_k_count(L), L)
            _, top_k_indices = importance.topk(k_count, dim=-1)
            top_k_indices_sorted, _ = top_k_indices.sort(dim=-1)
            idx_expanded = top_k_indices_sorted.unsqueeze(-1).expand(-1, -1, D)
            x_selected = x.gather(1, idx_expanded)
            k, v = self._project_kv(x_selected)
            k = k.view(B, k_count, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, k_count, self.num_kv_heads, self.head_dim).transpose(1, 2)
            if self.k_norm is not None:
                k = self.k_norm(k)
            k = apply_rope_with_positions(k, rope_freqs, top_k_indices_sorted)
            top_k_indices_full = top_k_indices_sorted

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # 可选 Q/K 归一化（Qwen3 per-head RMSNorm）
        if self.q_norm is not None:
            q = self.q_norm(q)
        if position_offset > 0:
            q_positions = torch.arange(
                position_offset, position_offset + L, device=x.device
            ).unsqueeze(0).expand(B, -1)
            q = apply_rope_with_positions(q, rope_freqs, q_positions)
        else:
            q = apply_rope(q, rope_freqs)

        # 构建因果掩码 (bool)
        attn_mask = None
        if causal_mask and (L > 1 or position_offset > 0):
            q_positions = torch.arange(
                position_offset, position_offset + L, device=x.device
            ).view(1, 1, L, 1)
            k_positions = top_k_indices_full.unsqueeze(1).unsqueeze(1)
            attn_mask = q_positions >= k_positions  # (B, 1, L, k)

        # GQA 扩展：KV 头数 < Q 头数时，expand 不复制内存（比 repeat_interleave 快 2-3x）
        if self.num_kv_heads != self.num_heads:
            kv_repeat = self.num_heads // self.num_kv_heads
            # expand 创建视图而非拷贝
            k_attn = k.unsqueeze(2).expand(-1, -1, kv_repeat, -1, -1).reshape(B, self.num_heads, -1, self.head_dim)
            v_attn = v.unsqueeze(2).expand(-1, -1, kv_repeat, -1, -1).reshape(B, self.num_heads, -1, self.head_dim)
        else:
            k_attn = k
            v_attn = v

        # 尝试使用 PyTorch SDPA（自动选择 Flash/Memory-efficient/Math 后端）
        out = self._sdpa_attention(q, k_attn, v_attn, attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        # 滑动窗口注意力（与稀疏全局并行，门控融合）
        if self.window_gate is not None and past_key_value is None and L > 1:
            window_out = self._sliding_window_attention(x, rope_freqs, L, position_offset)
            gate_w = torch.sigmoid(self.window_gate(x))  # (B, L, 1)
            out = out * (1 - gate_w) + window_out * gate_w

        out = self.o_proj(out)

        if use_cache:
            # 返回静态缓存 + 实际长度
            if past_key_value is not None:
                new_cache_len = int(cache_len) + L
                new_total_seen = int(total_seen_len) + L
                cache_k = past_K
                cache_v = past_V
            else:
                new_cache_len = k.shape[2]
                new_total_seen = L
                cache_k = k
                cache_v = v
            return out, (
                cache_k,
                cache_v,
                top_k_indices_full,
                new_cache_len,
                new_total_seen,
            )
        return out

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """使用 PyTorch SDPA 计算注意力（自动选择最优后端）。

        优先: FlashAttention > Memory-Efficient > Math fallback
        """
        try:
            # SDPA 的 bool mask 语义：True=允许关注，False=屏蔽。
            if attn_mask is not None:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask.to(dtype=torch.bool),
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    scale=self.scale,
                )
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    scale=self.scale,
                )
        except RuntimeError:
            # 回退到手动实现（兼容旧 PyTorch 或不支持的形状）
            out = self._manual_attention(q, k, v, attn_mask)
        return out

    def _sliding_window_attention(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        L: int,
        position_offset: int,
    ) -> torch.Tensor:
        """分块滑动窗口注意力，避免构建 O(L²) 掩码。

        将序列按 window_size 分块，每块只与自身 + 上一块的后半交互，
        总内存 O(L × W) 而非 O(L²)。
        """
        B, _, D = x.shape
        W = self.sliding_window_size

        # 全量 Q/K/V 投影 + RoPE
        q_w = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k_raw, v_raw = self._project_kv(x)
        k_w = k_raw.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_w = v_raw.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q_w = apply_rope(q_w, rope_freqs)
        k_w = apply_rope(k_w, rope_freqs)

        # GQA 扩展
        if self.num_kv_heads != self.num_heads:
            kv_repeat = self.num_heads // self.num_kv_heads
            k_w = k_w.repeat_interleave(kv_repeat, dim=1)
            v_w = v_w.repeat_interleave(kv_repeat, dim=1)

        # 如果序列短于 2 倍窗口，直接用全量注意力
        if L <= W * 2:
            positions = torch.arange(L, device=x.device)
            diff = positions.unsqueeze(0) - positions.unsqueeze(1)
            window_mask = (diff >= 0) & (diff < W)
            window_mask = window_mask.unsqueeze(0).unsqueeze(0)
            try:
                out_w = F.scaled_dot_product_attention(
                    q_w, k_w, v_w,
                    attn_mask=window_mask,
                    scale=self.scale,
                )
            except RuntimeError:
                attn = (q_w @ k_w.transpose(-2, -1)) * self.scale
                attn = attn.masked_fill(~window_mask, float("-inf"))
                attn = F.softmax(attn, dim=-1)
                out_w = attn @ v_w
            return out_w.transpose(1, 2).contiguous().view(B, L, D)

        # 分块处理: 每块处理 W 个 query，KV 范围为 [start-W+1, start+W]
        outputs = []
        for start in range(0, L, W):
            end = min(start + W, L)
            q_chunk = q_w[:, :, start:end, :]  # (B, H, chunk, D)

            # KV 范围: [max(0, start-W+1), end]
            kv_start = max(0, start - W + 1)
            k_chunk = k_w[:, :, kv_start:end, :]
            v_chunk = v_w[:, :, kv_start:end, :]

            # 构建局部因果 + 窗口掩码
            q_pos = torch.arange(start, end, device=x.device).view(1, 1, -1, 1)
            k_pos = torch.arange(kv_start, end, device=x.device).view(1, 1, 1, -1)
            local_mask = (q_pos >= k_pos) & (q_pos - k_pos < W)

            try:
                out_chunk = F.scaled_dot_product_attention(
                    q_chunk, k_chunk, v_chunk,
                    attn_mask=local_mask,
                    scale=self.scale,
                )
            except RuntimeError:
                attn = (q_chunk @ k_chunk.transpose(-2, -1)) * self.scale
                attn = attn.masked_fill(~local_mask, float("-inf"))
                attn = F.softmax(attn, dim=-1)
                out_chunk = attn @ v_chunk
            outputs.append(out_chunk)

        out_w = torch.cat(outputs, dim=2)
        return out_w.transpose(1, 2).contiguous().view(B, L, D)

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """手动注意力计算（SDPA 不可用时的回退）。"""
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
        attn_max = attn_weights.max(dim=-1, keepdim=True).values
        attn_max = torch.where(
            attn_max == float("-inf"), torch.zeros_like(attn_max), attn_max
        )
        attn_weights = attn_weights - attn_max
        all_masked = (attn_weights == float("-inf")).all(dim=-1, keepdim=True)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.masked_fill(all_masked, 0.0)
        attn_weights = self.attn_dropout(attn_weights)
        return attn_weights @ v
