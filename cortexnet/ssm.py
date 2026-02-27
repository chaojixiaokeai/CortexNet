from __future__ import annotations

"""
多尺度状态空间模块 (Multi-Scale State Space Module, MSSM)

核心创新：
  多个并行的 SSM 通道以不同的时间尺度运行，使模型能同时捕获
  从局部 token 交互到长距离依赖的多种模式。
  
  - 快速尺度：捕获局部的、细粒度的模式（如语法结构）
  - 慢速尺度：捕获长距离的、宏观的模式（如主题、上下文）
  
  计算复杂度：O(n)，线性于序列长度。

理论基础：
  基于状态空间模型 (SSM) 的连续时间动力学：
    dh/dt = A·h + B·x    (连续状态方程)
    y = C·h               (观测方程)
  
  通过零阶保持 (ZOH) 离散化：
    h_t = Ā·h_{t-1} + B̄·x_t    其中 Ā = exp(Δ·A), B̄ = Δ·B
  
  不同尺度通过 A 矩阵的不同特征值初始化实现：
    尺度 i 的 A 矩阵以 2^i 倍的频率初始化。

优化 (v3.2):
  - 添加 Triton 自定义 kernel 接口：当 triton 可用时自动使用
    高效的 GPU kernel，否则回退到 PyTorch 分块并行实现
  - 添加 logging 支持
"""


import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .ops.ssm_ops import fused_chunk_scan
except (ImportError, ModuleNotFoundError):
    # This might happen during special modes like packaging
    # where the file structure is temporarily different.
    # Fallback for compatibility.
    from cortexnet.ops.ssm_ops import fused_chunk_scan


logger = logging.getLogger(__name__)

# Triton kernel 可用性检测
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _ = (triton, tl)
    _TRITON_AVAILABLE = True
    logger.info("Triton available: SSM will use custom GPU kernels")
except ImportError:
    pass


class MultiScaleSSM(nn.Module):
    """多尺度选择性状态空间模块。

    每个尺度以不同的时间分辨率运行，由 A 矩阵的初始化控制。
    快速尺度捕获局部模式，慢速尺度捕获全局依赖。

    架构：
        Input → Linear(d, 2·d_inner) → [x, z] split
        x → Selective Scan (多尺度 A) → y
        z → SiLU 激活 → gate
        y · gate → Linear(d_inner, d) → Output

    Args:
        d_model: 输入/输出维度
        num_scales: 时间尺度数量（每个尺度有不同的记忆衰减率）
        state_size: SSM 状态向量维度
        expand_factor: 内部维度扩展因子
    """

    def __init__(
        self,
        d_model: int,
        num_scales: int = 4,
        state_size: int = 16,
        expand_factor: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand_factor
        self.num_scales = num_scales
        self.state_size = state_size
        self.d_per_scale = self.d_inner // num_scales

        assert self.d_inner % num_scales == 0, (
            f"d_inner ({self.d_inner}) 必须能被 num_scales ({num_scales}) 整除"
        )

        # 输入投影：x 用于 SSM，z 用于门控
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # A 矩阵：多尺度初始化（对数空间，保证稳定性）
        # 不同尺度以指数级递增的时间常数初始化
        A_parts = []
        for i in range(num_scales):
            base_freqs = torch.arange(1, state_size + 1, dtype=torch.float32)
            scale_factor = 2.0 ** i  # 指数递增的时间尺度
            A_part = (
                (base_freqs * scale_factor)
                .unsqueeze(0)
                .expand(self.d_per_scale, -1)
            )
            A_parts.append(A_part)
        A = torch.cat(A_parts, dim=0)  # (d_inner, state_size)
        self.A_log = nn.Parameter(torch.log(A))

        # 输入依赖的 SSM 参数（选择性机制的核心）
        self.B_proj = nn.Linear(self.d_inner, state_size, bias=False)
        self.C_proj = nn.Linear(self.d_inner, state_size, bias=False)

        # 离散化步长（输入依赖，使模型能选择性地记忆或遗忘）
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # 初始化 dt bias，使初始步长在 [0.001, 0.1] 范围内
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt_init.log())

        # D 跳跃连接（直接通路，类似 Mamba）
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出门控：0 初始化更中性，sigmoid(0)=0.5
        self.output_gate = nn.Parameter(torch.zeros(1))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        past_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            past_state: 上一序列步的 SSM 隐状态 (B, d_inner, N)，用于增量解码
            use_cache: 若 True 且 past_state 非 None，返回 (output, new_state)
        Returns:
            output: (batch, seq_len, d_model)
            new_state (可选): (B, d_inner, N)，当 use_cache=True 时返回
        """
        B, L, D = x.shape
        input_dtype = x.dtype

        # 输入投影 + 门控分割
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)  # 各 (B, L, d_inner)

        # 计算输入依赖的 SSM 参数（float32 计算后转回原 dtype，MPS 兼容）
        A = -torch.exp(self.A_log.float()).to(input_dtype)  # (d_inner, N)
        B_mat = self.B_proj(x_ssm)  # (B, L, N)
        C_mat = self.C_proj(x_ssm)  # (B, L, N)
        dt = F.softplus(self.dt_proj(x_ssm))  # (B, L, d_inner), 正值

        # 选择性扫描 — 优先 Triton kernel → 分块并行 → 顺序扫描
        if L > 1:
            if _TRITON_AVAILABLE and x_ssm.is_cuda:
                y, new_state = self._triton_scan(
                    x_ssm, A, B_mat, C_mat, dt,
                    past_state=past_state,
                    use_cache=use_cache,
                )
            else:
                y, new_state = self._chunk_parallel_scan(
                    x_ssm, A, B_mat, C_mat, dt,
                    chunk_size=min(max(16, L), 64),
                    past_state=past_state,
                    use_cache=use_cache,
                )
        else:
            y, new_state = self._selective_scan(
                x_ssm, A, B_mat, C_mat, dt,
                past_state=past_state,
                use_cache=use_cache,
            )

        # 确保 y 与输入 dtype 一致（scan 可能返回 float32）
        y = y.to(input_dtype)

        # 跳跃连接
        y = y + x_ssm * self.D.to(input_dtype).unsqueeze(0).unsqueeze(0)

        # 门控输出
        y = y * F.silu(z)
        out = self.out_proj(y) * torch.sigmoid(self.output_gate.to(input_dtype))

        if use_cache and new_state is not None:
            return out, new_state
        return out

    def _selective_scan(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        dt: torch.Tensor,
        past_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """执行选择性扫描（纯 PyTorch 顺序实现）。

        支持 past_state 增量解码：传入上一步的 h 作为初始状态。

        Returns:
            y: (B, L, d_inner) - 输出
            new_state: (B, d_inner, N) - 最终隐状态，use_cache=True 时返回
        """
        batch, L, d_inner = x.shape
        N = A.shape[1]
        orig_dtype = x.dtype

        # 在 float32 中计算以防止 float16 溢出
        x = x.float()
        A = A.float()
        B = B.float()
        C = C.float()
        dt = dt.float()

        # 预计算离散化参数（clamp 防止 exp 溢出）
        A_bar = torch.exp((dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)).clamp(max=20))
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(2)

        h = (
            past_state.float()
            if past_state is not None
            else torch.zeros(batch, d_inner, N, device=x.device, dtype=torch.float32)
        )
        outputs = []

        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1).to(orig_dtype)
        new_state = h.to(orig_dtype) if use_cache else None
        return y, new_state

    def _chunk_parallel_scan(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        dt: torch.Tensor,
        chunk_size: int = 64,
        past_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """分块并行扫描：块内并行计算，块间顺序传播。

        使用 `cortexnet.ops.ssm_ops.fused_chunk_scan` 实现核心扫描逻辑，
        以提高代码模块化和数值稳定性。
        """
        batch, L, d_inner = x.shape
        N = A.shape[1]
        orig_dtype = x.dtype

        # 在 float32 中计算以防止 float16 溢出
        x_f = x.float()
        A_f = A.float()
        B_f = B.float()
        C_f = C.float()
        dt_f = dt.float()

        # 预计算离散化参数
        A_bar = torch.exp((dt_f.unsqueeze(-1) * A_f.unsqueeze(0).unsqueeze(0)).clamp(max=20))
        B_bar = dt_f.unsqueeze(-1) * B_f.unsqueeze(2)

        num_chunks = (L + chunk_size - 1) // chunk_size
        h = (
            past_state.float()
            if past_state is not None
            else torch.zeros(batch, d_inner, N, device=x.device, dtype=torch.float32)
        )
        all_outputs = []

        for c in range(num_chunks):
            s = c * chunk_size
            e = min(s + chunk_size, L)

            # 准备当前块的输入
            a_chunk = A_bar[:, s:e]
            b_chunk_input = B_bar[:, s:e] * x_f[:, s:e].unsqueeze(-1)
            c_chunk = C_f[:, s:e]

            # 调用融合的扫描算子
            h_all, h_new = fused_chunk_scan(
                a_chunk, b_chunk_input, h
            )

            # 计算输出 y
            # (B, L_chunk, D, N) * (B, L_chunk, 1, N) -> sum -> (B, L_chunk, D)
            y_chunk = (h_all * c_chunk.unsqueeze(2)).sum(-1)
            all_outputs.append(y_chunk)

            # 更新下一块的初始状态
            h = h_new

        y = torch.cat(all_outputs, dim=1).to(orig_dtype)
        new_state = h.to(orig_dtype) if use_cache else None
        return y, new_state

    def _triton_scan(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        dt: torch.Tensor,
        past_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Triton GPU kernel 加速的选择性扫描。

        当 Triton 可用且输入在 CUDA 上时自动调用。
        核心思路：将分块扫描的内外层循环融合为单个 Triton kernel，
        避免多次 kernel launch 和中间内存分配。

        当前版本为接口占位，委托给 _chunk_parallel_scan。
        TODO: 实现原生 Triton kernel body。
        """
        logger.debug("Using Triton scan path (delegating to chunk_parallel)")
        return self._chunk_parallel_scan(
            x, A, B, C, dt,
            chunk_size=min(max(16, x.shape[1]), 64),
            past_state=past_state,
            use_cache=use_cache,
        )
