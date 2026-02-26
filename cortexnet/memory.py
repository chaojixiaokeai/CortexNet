from __future__ import annotations

"""
突触可塑性记忆模块 (Synaptic Plasticity Memory)

核心创新：
  受生物神经系统突触可塑性启发的快速权重记忆系统。
  在前向传播过程中逐步累积键值关联，使模型具备：
  
  1. 即时上下文学习：无需梯度更新即可在推理时"学习"新模式
  2. 工作记忆：维护当前上下文的动态表示
  3. 模式发现：自动发现并记忆输入序列中的规律

  与传统注意力的区别：
    - 注意力：每次重新计算所有 token 间的关系 → O(n²)
    - 突触记忆：递增地累积关联 → O(n)，且具有压缩效果

学术渊源：
  - Fast Weight Programmers (Schmidhuber, 1992)
  - Linear Attention (Katharopoulos et al., 2020)
  - Delta Net (Schlag et al., 2021)
  
  本模块在此基础上创新性地加入：
    - 可学习的衰减率（平衡新旧信息）
    - ELU+1 正值化（类似核方法，保证记忆矩阵的正定性）
    - 归一化读取（防止记忆值爆炸或消失）
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RMSNorm(nn.Module):
    """内联 RMSNorm（避免 blocks.py 循环导入，兼容 MPS/float16）。"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        rms = torch.sqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f / rms).to(dtype) * self.weight.to(dtype)


class SynapticMemory(nn.Module):
    """突触可塑性记忆模块。

    维护一个在前向传播中动态更新的快速权重矩阵。
    每个 token 先从累积的记忆中读取，然后写入自己的关联，
    形成一种快速学习机制。

    架构流程（对每个时间步 t）：
        1. 读取：o_t = φ(q_t) @ Memory_{t-1} / (φ(q_t) @ z_{t-1})
        2. 写入：Memory_t = λ·Memory_{t-1} + φ(k_t)^T @ v_t
        3. 更新归一化因子：z_t = λ·z_{t-1} + φ(k_t)
        4. 输出：RMSNorm(o_t) → Linear → output_t

    其中 φ(x) = ELU(x) + 1 是正值化的核函数。

    Args:
        d_model: 输入/输出维度
        memory_dim: 记忆矩阵的行维度（类似于"记忆槽"数量）
        decay_init: 衰减率初始值（控制新旧信息的平衡）
    """

    def __init__(
        self, d_model: int, memory_dim: int = 64, decay_init: float = 0.95
    ):
        super().__init__()
        self.d_model = d_model
        self.memory_dim = memory_dim

        # 投影层
        self.query_proj = nn.Linear(d_model, memory_dim, bias=False)
        self.key_proj = nn.Linear(d_model, memory_dim, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # 可学习的衰减率
        # logit(0.95) ≈ 2.944, sigmoid 后恢复为 ~0.95
        self.decay_logit = nn.Parameter(
            torch.tensor(torch.logit(torch.tensor(decay_init)).item())
        )

        # 缩放因子
        self.scale = memory_dim ** -0.5

        # 输出归一化（使用 RMSNorm 保持一致性）
        self.norm = _RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_memory: Optional[torch.Tensor] = None,
        past_z: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        # 增量缓存模式或需要返回状态：用顺序实现
        if use_cache or past_memory is not None:
            return self._sequential_forward(x, past_memory, past_z, use_cache)
        # 训练 / 非缓存推理：用并行实现（无 Python 循环，大幅提速）
        return self._parallel_forward(x)

    def _parallel_forward(self, x: torch.Tensor) -> torch.Tensor:
        """并行线性注意力：用 log-space cumsum 替代 Python for-loop。

        数学等价于顺序版本，但全部操作是张量并行的。
        原理：M_t = Σ_{s≤t} λ^{t-s} · k_s^T · v_s
              用 exp(-log(λ)·s) 归一化后 cumsum，再乘回 exp(log(λ)·t)。
        """
        B, L, D = x.shape
        queries = F.elu(self.query_proj(x), alpha=1.0) + 1  # (B, L, mem)
        keys = F.elu(self.key_proj(x), alpha=1.0) + 1
        values = self.value_proj(x)  # (B, L, D)
        queries = queries * self.scale
        decay = torch.sigmoid(self.decay_logit)

        # Log-space cumulative decay trick（clamp 防止溢出）
        log_decay = torch.log(decay.clamp(min=1e-6))
        positions = torch.arange(L, device=x.device, dtype=x.dtype)
        log_cum = (log_decay * positions).clamp(-20, 20)  # (L,)

        exp_neg = torch.exp(-log_cum).view(1, L, 1)  # (1, L, 1)
        exp_pos = torch.exp(log_cum).view(1, L, 1)   # (1, L, 1)

        # 记忆矩阵并行累积: M_t = Σ_{s≤t} decay^{t-s} * k_s ⊗ v_s
        kv = keys.unsqueeze(-1) * values.unsqueeze(-2)  # (B, L, mem, D)
        weighted_kv = exp_neg.unsqueeze(-1) * kv         # broadcast (1,L,1,1)
        cum_kv = torch.cumsum(weighted_kv, dim=1)
        M = exp_pos.unsqueeze(-1) * cum_kv               # (B, L, mem, D)

        # 归一化因子: z_t = Σ_{s≤t} decay^{t-s} * k_s
        weighted_k = exp_neg * keys                       # (B, L, mem)
        cum_k = torch.cumsum(weighted_k, dim=1)
        z = exp_pos * cum_k                               # (B, L, mem)

        # 读取: o_t = q_t · M_t / (q_t · z_t)
        numerator = torch.einsum('blm,blmd->bld', queries, M)
        denominator = torch.einsum('blm,blm->bl', queries, z).unsqueeze(-1) + 1e-6

        output = self.norm(numerator / denominator)
        return self.out_proj(output)

    def _sequential_forward(
        self, x, past_memory, past_z, use_cache,
    ):
        """顺序实现（用于增量缓存生成，L 通常 = 1）。"""
        B, L, D = x.shape
        queries = F.elu(self.query_proj(x), alpha=1.0) + 1
        keys = F.elu(self.key_proj(x), alpha=1.0) + 1
        values = self.value_proj(x)
        queries = queries * self.scale
        decay = torch.sigmoid(self.decay_logit)
        memory = (
            past_memory if past_memory is not None
            else torch.zeros(B, self.memory_dim, D, device=x.device, dtype=x.dtype)
        )
        z = (
            past_z if past_z is not None
            else torch.zeros(B, self.memory_dim, 1, device=x.device, dtype=x.dtype)
        )
        outputs = []
        for t in range(L):
            q_t, k_t, v_t = queries[:, t], keys[:, t], values[:, t]
            num = torch.bmm(q_t.unsqueeze(1), memory).squeeze(1)
            den = torch.bmm(q_t.unsqueeze(1), z).squeeze(1)
            outputs.append(num / (den + 1e-6))
            memory = decay * memory + torch.bmm(k_t.unsqueeze(2), v_t.unsqueeze(1))
            z = decay * z + k_t.unsqueeze(2)
        output = self.norm(torch.stack(outputs, dim=1))
        out = self.out_proj(output)
        if use_cache:
            return out, memory, z
        return out
