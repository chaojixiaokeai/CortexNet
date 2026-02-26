from __future__ import annotations

"""
分层记忆系统 (Hierarchical Memory System)

受人类认知架构启发的三层记忆系统：

  ┌─────────────────────────────────────────────────────────────┐
  │                    分层记忆架构                              │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  ┌─── 工作记忆 (Working Memory) ──────────────────────┐    │
  │  │  • 快速权重，逐 token 更新                          │    │
  │  │  • 高分辨率，小容量                                  │    │
  │  │  • 衰减快，捕获即时上下文                            │    │
  │  │  • 类似人脑前额叶的工作记忆                          │    │
  │  └────────────────────────────────────────────────────┘    │
  │                                                             │
  │  ┌─── 情景记忆 (Episodic Memory) ─────────────────────┐    │
  │  │  • 可学习的记忆槽位，跨序列积累                      │    │
  │  │  • 通过交叉注意力读写                                │    │
  │  │  • 中等容量，存储压缩的上下文快照                    │    │
  │  │  • 类似海马体的情景记忆                              │    │
  │  └────────────────────────────────────────────────────┘    │
  │                                                             │
  │  ┌─── 语义记忆 (Semantic Memory) ─────────────────────┐    │
  │  │  • 全局可学习知识库，所有层共享                      │    │
  │  │  • 通过训练缓慢更新                                  │    │
  │  │  • 大容量，存储抽象的通用知识                        │    │
  │  │  • 类似大脑皮层的长期语义记忆                        │    │
  │  └────────────────────────────────────────────────────┘    │
  │                                                             │
  │  ┌─── 记忆控制器 (Memory Controller) ─────────────────┐    │
  │  │  • 动态决定从哪层记忆读取/写入                      │    │
  │  │  • 学习不同情境下的最优记忆策略                      │    │
  │  │  • 门控融合三层记忆输出                              │    │
  │  └────────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────────┘

创新点:
  1. 三层记忆各有不同的时间尺度和容量
  2. 记忆控制器动态平衡读写策略
  3. 选择性遗忘机制防止记忆过载
  4. 工作记忆提供即时上下文，语义记忆提供长期知识
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class WorkingMemory(nn.Module):
    """工作记忆：快速权重系统，逐 token 更新。

    通过快速权重矩阵累积键值关联，每个 token 先读后写。
    加入选择性遗忘门控，防止记忆过载。

    数学原理：
        遗忘门: f_t = σ(W_f · x_t)
        读取:   o_t = φ(q_t) · M_{t-1} / norm
        写入:   M_t = f_t · M_{t-1} + φ(k_t)^T · v_t
    """

    def __init__(self, d_model: int, memory_dim: int = 64):
        super().__init__()
        self.memory_dim = memory_dim
        self.scale = memory_dim ** -0.5

        self.q_proj = nn.Linear(d_model, memory_dim, bias=False)
        self.k_proj = nn.Linear(d_model, memory_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # 选择性遗忘门控（每个记忆维度独立衰减）
        self.forget_gate = nn.Linear(d_model, memory_dim)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_memory: Optional[torch.Tensor] = None,
        past_z: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        if use_cache or past_memory is not None:
            return self._sequential_forward(x, past_memory, past_z, use_cache)
        return self._parallel_forward(x)

    def _parallel_forward(self, x: torch.Tensor) -> torch.Tensor:
        """并行线性注意力（带 per-position 遗忘门），无 Python 循环。

        log-space cumsum trick 处理 position-dependent decay：
          log_cum[t] = Σ_{i=0}^{t} log(forget[i])
          decay(t→s) = exp(log_cum[t] - log_cum[s])
        """
        B, L, D = x.shape
        q = F.elu(self.q_proj(x), alpha=1.0) + 1  # (B, L, mem)
        k = F.elu(self.k_proj(x), alpha=1.0) + 1
        v = self.v_proj(x)                          # (B, L, D)
        q = q * self.scale
        forget = torch.sigmoid(self.forget_gate(x))  # (B, L, mem)

        log_f = torch.log(forget.clamp(min=1e-6))   # (B, L, mem)
        log_f_cum = torch.cumsum(log_f, dim=1).clamp(-20, 20)  # clamp 防溢出

        exp_neg = torch.exp(-log_f_cum)  # (B, L, mem) — 归一化因子
        exp_pos = torch.exp(log_f_cum)   # (B, L, mem) — 还原因子

        # M_t = Σ_{s≤t} decay(t,s) * k_s ⊗ v_s  (并行)
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)       # (B, L, mem, D)
        weighted_kv = exp_neg.unsqueeze(-1) * kv
        cum_kv = torch.cumsum(weighted_kv, dim=1)
        M = exp_pos.unsqueeze(-1) * cum_kv            # (B, L, mem, D)

        # z_t = Σ_{s≤t} decay(t,s) * k_s
        weighted_k = exp_neg * k
        cum_k = torch.cumsum(weighted_k, dim=1)
        z = exp_pos * cum_k                            # (B, L, mem)

        numerator = torch.einsum('blm,blmd->bld', q, M)
        denominator = torch.einsum('blm,blm->bl', q, z).unsqueeze(-1) + 1e-6

        return self.out_proj(self.norm(numerator / denominator))

    def _sequential_forward(self, x, past_memory, past_z, use_cache):
        """顺序实现（用于增量缓存）。"""
        B, L, D = x.shape
        q = F.elu(self.q_proj(x), alpha=1.0) + 1
        k = F.elu(self.k_proj(x), alpha=1.0) + 1
        v = self.v_proj(x)
        q = q * self.scale
        forget = torch.sigmoid(self.forget_gate(x))
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
            q_t, k_t, v_t = q[:, t], k[:, t], v[:, t]
            f_t = forget[:, t].unsqueeze(-1)
            num = torch.bmm(q_t.unsqueeze(1), memory).squeeze(1)
            den = torch.bmm(q_t.unsqueeze(1), z).squeeze(1)
            outputs.append(num / (den + 1e-6))
            memory = f_t * memory + torch.bmm(k_t.unsqueeze(2), v_t.unsqueeze(1))
            z = f_t * z + k_t.unsqueeze(2)
        out = self.out_proj(self.norm(torch.stack(outputs, dim=1)))
        if use_cache:
            return out, memory, z
        return out


class EpisodicMemory(nn.Module):
    """情景记忆：可学习的记忆槽位，通过交叉注意力交互。

    维护固定数量的记忆槽位，通过训练学习存储有用的模式。
    使用交叉注意力机制让输入 token 检索相关记忆。

    架构:
        读取: CrossAttention(input → memory_slots)
        更新: 通过梯度缓慢调整槽位内容

    Args:
        d_model: 模型维度
        num_slots: 记忆槽位数量
        num_heads: 交叉注意力头数
    """

    def __init__(self, d_model: int, num_slots: int = 32, num_heads: int = 4):
        super().__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 可学习的记忆槽位
        self.memory_slots = nn.Parameter(
            torch.randn(1, num_slots, d_model) * 0.02
        )

        # 交叉注意力投影
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # 记忆更新门控
        self.update_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        memory = self.memory_slots.expand(B, -1, -1)  # (B, slots, D)

        # 交叉注意力：input queries → memory keys/values
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(B, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, self.num_slots, self.num_heads, self.head_dim).transpose(1, 2)

        # SDPA: 自动选择 FlashAttention / Memory-Efficient 后端
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        return self.out_proj(out)


class SemanticMemory(nn.Module):
    """语义记忆：全局长期知识库。

    维护一组全局知识向量，通过双向交叉注意力与输入交互：
    1. 输入从语义记忆中检索相关知识
    2. 语义记忆根据输入生成上下文相关的知识表示

    与情景记忆的区别:
    - 语义记忆存储抽象的、跨样本的通用知识
    - 情景记忆存储具体的、样本内的上下文信息
    - 语义记忆的容量更大，更新更慢

    Args:
        d_model: 模型维度
        num_slots: 知识向量数量
    """

    def __init__(self, d_model: int, num_slots: int = 64):
        super().__init__()
        self.num_slots = num_slots

        # 全局知识向量
        self.knowledge = nn.Parameter(
            torch.randn(1, num_slots, d_model) * 0.02
        )

        # 双向注意力
        self.input_to_memory = nn.Linear(d_model, d_model, bias=False)
        self.memory_to_input = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        knowledge = self.knowledge.expand(B, -1, -1)  # (B, slots, D)

        # 输入 → 知识检索 (SDPA 加速)
        query = self.input_to_memory(x).unsqueeze(1)  # (B, 1, L, D) 作为单头
        k = knowledge.unsqueeze(1)  # (B, 1, slots, D)
        v = knowledge.unsqueeze(1)  # (B, 1, slots, D)
        retrieved = F.scaled_dot_product_attention(
            query, k, v, dropout_p=0.0, is_causal=False,
        ).squeeze(1)  # (B, L, D)

        # 门控融合
        gate = torch.sigmoid(self.gate(x))
        output = gate * retrieved + (1 - gate) * x

        return self.out_proj(output)


class MemoryController(nn.Module):
    """记忆控制器：动态协调三层记忆的读写。

    根据输入内容决定从哪层记忆中读取以及各层的权重：
    - 需要即时上下文 → 偏重工作记忆
    - 需要历史模式 → 偏重情景记忆
    - 需要通用知识 → 偏重语义记忆
    """

    def __init__(self, d_model: int):
        super().__init__()
        bottleneck = max(d_model // 4, 32)
        self.controller = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, 3),  # 三层记忆的权重
        )

    def forward(
        self,
        working_out: torch.Tensor,
        episodic_out: torch.Tensor,
        semantic_out: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # 基于输入计算记忆权重
        weights = F.softmax(self.controller(x), dim=-1)  # (B, L, 3)

        stacked = torch.stack(
            [working_out, episodic_out, semantic_out], dim=-1
        )  # (B, L, D, 3)
        fused = (stacked * weights.unsqueeze(-2)).sum(dim=-1)

        return fused


class HierarchicalMemorySystem(nn.Module):
    """分层记忆系统：统一接口。

    整合三层记忆和记忆控制器，提供统一的前向接口。

    Args:
        d_model: 模型维度
        working_dim: 工作记忆维度
        episodic_slots: 情景记忆槽位数
        semantic_slots: 语义记忆槽位数
        num_heads: 情景记忆注意力头数
    """

    def __init__(
        self,
        d_model: int,
        working_dim: int = 64,
        episodic_slots: int = 32,
        semantic_slots: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()

        self.working = WorkingMemory(d_model, working_dim)
        self.episodic = EpisodicMemory(d_model, episodic_slots, num_heads)
        self.semantic = SemanticMemory(d_model, semantic_slots)
        self.controller = MemoryController(d_model)

    def forward(
        self,
        x: torch.Tensor,
        past_working_memory: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            past_working_memory: (past_memory, past_z) 仅 WorkingMemory 有状态
            use_cache: 若 True 返回 (output, (new_memory, new_z))
        """
        if past_working_memory is not None:
            past_mem, past_z = past_working_memory
        else:
            past_mem = past_z = None
        if use_cache:
            w_out, new_mem, new_z = self.working(
                x, past_memory=past_mem, past_z=past_z, use_cache=True
            )
        else:
            w_out = self.working(
                x, past_memory=past_mem, past_z=past_z, use_cache=False
            )
        e_out = self.episodic(x)
        s_out = self.semantic(x)
        out = self.controller(w_out, e_out, s_out, x)
        if use_cache:
            return out, (new_mem, new_z)
        return out
