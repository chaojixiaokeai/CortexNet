from __future__ import annotations

"""
CortexNet 核心构建块 (Core Building Blocks)

将所有组件组合成 CortexBlock——CortexNet 架构的基本重复单元。

每个 CortexBlock 包含三条并行处理路径和一个 MoE 前馈层：
  ┌──────────────── CortexBlock ────────────────┐
  │                                               │
  │  Input                                        │
  │    │                                          │
  │    ├──► RMSNorm ──┬──► Multi-Scale SSM ──┐   │
  │    │              │                       │   │
  │    │              ├──► Sparse Attention ──┤   │
  │    │              │                       │   │
  │    │              └──► Synaptic Memory ───┤   │
  │    │                                      │   │
  │    │              ┌── Adaptive Fusion ◄───┘   │
  │    │              │                            │
  │    ├──── + ◄──────┘  (残差连接)                │
  │    │                                          │
  │    ├──► RMSNorm ──► MoE FFN ──┐              │
  │    │                           │              │
  │    └──── + ◄──────────────────┘              │
  │         │          (残差连接)                  │
  │       Output                                  │
  └───────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import CortexNetConfig
    from .ssm import MultiScaleSSM
    from .attention import SelectiveSparseAttention
    from .memory import SynapticMemory
    from .routing import MixtureOfExperts
except ImportError:
    # 兼容脚本式导入: `from blocks import CortexBlockV3`
    from cortexnet.config import CortexNetConfig
    from cortexnet.ssm import MultiScaleSSM
    from cortexnet.attention import SelectiveSparseAttention
    from cortexnet.memory import SynapticMemory
    from cortexnet.routing import MixtureOfExperts


class RMSNorm(nn.Module):
    """均方根层归一化 (Root Mean Square Layer Normalization)。

    相比标准 LayerNorm，RMSNorm 不需要计算均值和偏移，
    只进行缩放归一化，计算更高效，效果相当。

    数学公式：
        RMSNorm(x) = x / √(mean(x²) + ε) × γ

    Args:
        dim: 归一化的维度
        eps: 防止除零的小常数
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 采用 FP32 归一化提高半精度推理稳定性（与主流 LLM 实现一致）
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = x_fp32 * rms
        return (out.to(input_dtype) * self.weight.to(input_dtype))


class AdaptiveFusionGate(nn.Module):
    """自适应融合门控 (Adaptive Fusion Gate)。

    动态学习如何平衡 SSM、注意力和记忆三条路径的贡献。
    不同 token、不同上下文可能需要不同的路径组合：
    - 局部模式：SSM 权重更高
    - 需要远程参考：注意力权重更高
    - 需要上下文适应：记忆权重更高

    使用瓶颈结构减少参数量：
        Concat(3D) → Linear(3D, D//2) → SiLU → Linear(D//2, 3) → Softmax

    Args:
        d_model: 模型维度
        num_paths: 路径数量（默认 3）
    """

    def __init__(self, d_model: int, num_paths: int = 3):
        super().__init__()
        bottleneck = max(d_model // 4, 32)
        self.gate = nn.Sequential(
            nn.Linear(d_model * num_paths, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, num_paths),
        )

    def forward(self, *path_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *path_outputs: 多个路径输出，各为 (batch, seq_len, d_model)
        Returns:
            fused: (batch, seq_len, d_model)
        """
        # 拼接所有路径的输出用于门控计算
        concat = torch.cat(path_outputs, dim=-1)  # (B, L, num_paths * D)
        gate_weights = F.softmax(
            self.gate(concat), dim=-1
        )  # (B, L, num_paths)

        # 堆叠 + 加权求和
        stacked = torch.stack(
            path_outputs, dim=-1
        )  # (B, L, D, num_paths)
        fused = (stacked * gate_weights.unsqueeze(-2)).sum(
            dim=-1
        )  # (B, L, D)

        return fused


class CortexBlock(nn.Module):
    """CortexNet 核心构建块。

    每个块包含三条并行处理路径：
      1. 多尺度 SSM：在多个时间尺度上捕获序列模式 — O(n)
      2. 选择性稀疏注意力：聚焦最重要的 token — O(n·k)
      3. 突触记忆：快速上下文适应 — O(n)

    路径输出通过自适应融合门控动态合并，
    然后经过混合专家前馈网络进行高效的非线性变换。

    Args:
        config: CortexNet 配置
        layer_idx: 当前层索引（用于调试）
    """

    def __init__(self, config: CortexNetConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx

        # 预归一化
        self.norm1 = RMSNorm(config.hidden_size, config.norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.norm_eps)

        # 三条并行处理路径
        self.ssm = MultiScaleSSM(
            d_model=config.hidden_size,
            num_scales=config.num_scales,
            state_size=config.ssm_state_size,
            expand_factor=config.ssm_expand_factor,
        )

        self.attention = SelectiveSparseAttention(
            d_model=config.hidden_size,
            num_heads=config.num_heads,
            top_k_ratio=config.top_k_ratio,
            k_mode=config.attention_k_mode,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )

        self.memory = SynapticMemory(
            d_model=config.hidden_size,
            memory_dim=config.memory_dim,
            decay_init=config.memory_decay_init,
        )

        # 自适应融合
        self.fusion = AdaptiveFusionGate(config.hidden_size, num_paths=3)

        # 混合专家前馈网络
        self.moe = MixtureOfExperts(
            d_model=config.hidden_size,
            d_ff=config.expert_ff_dim,
            num_experts=config.num_experts,
            num_active=config.num_active_experts,
            aux_loss_weight=config.moe_aux_loss_weight,
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        # ═══ 多路径处理 + 残差 ═══
        residual = x
        x_norm = self.norm1(x)

        # 三条路径并行处理
        ssm_out = self.ssm(x_norm)  # O(n) 复杂度
        attn_out = self.attention(x_norm)  # O(n·k) 复杂度
        mem_out = self.memory(x_norm)  # O(n) 复杂度

        # 自适应融合
        fused = self.fusion(ssm_out, attn_out, mem_out)
        x = residual + self.dropout(fused)

        # ═══ MoE FFN + 残差 ═══
        residual = x
        x = residual + self.dropout(self.moe(self.norm2(x)))

        return x

    def get_aux_loss(self) -> torch.Tensor:
        """获取 MoE 路由的辅助损失。"""
        return self.moe.aux_loss


# ═══════════════════════════════════════════════════════════════
#                      CortexBlock V2 — 进化版
# ═══════════════════════════════════════════════════════════════

try:
    from .hierarchical_memory import HierarchicalMemorySystem
    from .graph_reasoning import GraphReasoningModule
    from .meta_learning import MetaLearningAdapter, TaskAdaptiveController
    from .routing import CollaborativeMoE
except ImportError:
    from cortexnet.hierarchical_memory import HierarchicalMemorySystem
    from cortexnet.graph_reasoning import GraphReasoningModule
    from cortexnet.meta_learning import MetaLearningAdapter, TaskAdaptiveController
    from cortexnet.routing import CollaborativeMoE


class CortexBlockV2(nn.Module):
    """CortexNet V2 进化版构建块。

    在 V1 的基础上增加了 4 项重大升级：

    ┌──────────────── CortexBlock V2 ────────────────────────────┐
    │                                                             │
    │  Input                                                      │
    │    │                                                        │
    │    ├──► RMSNorm ──┬──► Multi-Scale SSM (并行扫描) ──┐     │
    │    │              ├──► Selective Sparse Attention ────┤     │
    │    │              ├──► Hierarchical Memory (3层) ─────┤     │
    │    │              └──► Graph Reasoning (多步推理) ────┤     │
    │    │                                                  │     │
    │    │              ┌── Adaptive Fusion (4路) ◄─────────┘     │
    │    │ (residual)   │                                         │
    │    ├──── + ◄──────┤                                         │
    │    │              │                                         │
    │    │              └── Meta-Learning Adapter (FiLM)          │
    │    │                                                        │
    │    ├──► RMSNorm ──► Collaborative MoE FFN ──┐              │
    │    │ (residual)                               │              │
    │    └──── + ◄────────────────────────────────┘              │
    │         │                                                   │
    │         └──► Task Adaptive Controller                       │
    │              │                                              │
    │            Output                                           │
    └─────────────────────────────────────────────────────────────┘

    V2 新增:
      1. 分层记忆系统替代单一突触记忆
      2. 图推理模块实现多步关系推理
      3. 元学习适配器实现快速任务适应
      4. 协作式 MoE 实现专家间知识共享
      5. 任务自适应控制器实现隐式任务切换

    Args:
        config: CortexNet 配置
        layer_idx: 层索引
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx

        # 预归一化
        self.norm1 = RMSNorm(config.hidden_size, config.norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.norm_eps)

        # ═══ 四条并行处理路径 ═══

        # 路径1: 多尺度 SSM（含分块并行扫描）
        self.ssm = MultiScaleSSM(
            d_model=config.hidden_size,
            num_scales=config.num_scales,
            state_size=config.ssm_state_size,
            expand_factor=config.ssm_expand_factor,
        )

        # 路径2: 选择性稀疏注意力
        self.attention = SelectiveSparseAttention(
            d_model=config.hidden_size,
            num_heads=config.num_heads,
            top_k_ratio=config.top_k_ratio,
            k_mode=config.attention_k_mode,
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )

        # 路径3: 分层记忆系统（V2 新增）
        self.memory = HierarchicalMemorySystem(
            d_model=config.hidden_size,
            working_dim=getattr(config, "memory_dim", 64),
            episodic_slots=getattr(config, "episodic_slots", 32),
            semantic_slots=getattr(config, "semantic_slots", 64),
            num_heads=min(4, config.num_heads),
        )

        # 路径4: 图推理模块（V2 新增）
        self.graph_reasoning = GraphReasoningModule(
            d_model=config.hidden_size,
            num_neighbors=getattr(config, "graph_neighbors", 16),
            num_iterations=getattr(config, "graph_iterations", 2),
        )

        # ═══ 自适应融合（4 路径） ═══
        self.fusion = AdaptiveFusionGate(config.hidden_size, num_paths=4)

        # ═══ 元学习适配器（V2 新增） ═══
        self.meta_adapter = MetaLearningAdapter(config.hidden_size)

        # ═══ 协作式 MoE（V2 升级） ═══
        self.moe = CollaborativeMoE(
            d_model=config.hidden_size,
            d_ff=config.expert_ff_dim,
            num_experts=config.num_experts,
            num_active=config.num_active_experts,
            aux_loss_weight=config.moe_aux_loss_weight,
        )

        # ═══ 任务自适应控制器（V2 新增） ═══
        self.task_controller = TaskAdaptiveController(
            config.hidden_size,
            num_modes=getattr(config, "num_task_modes", 4),
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        # ═══ 四路并行处理 + 残差 ═══
        residual = x
        x_norm = self.norm1(x)

        ssm_out = self.ssm(x_norm)
        attn_out = self.attention(x_norm)
        mem_out = self.memory(x_norm)
        graph_out = self.graph_reasoning(x_norm)

        # 自适应融合（4 路径）
        fused = self.fusion(ssm_out, attn_out, mem_out, graph_out)

        # 元学习适配（FiLM 调制）
        fused = self.meta_adapter(fused)

        x = residual + self.dropout(fused)

        # ═══ 协作式 MoE FFN + 残差 ═══
        residual = x
        x = residual + self.dropout(self.moe(self.norm2(x)))

        # ═══ 任务自适应控制 ═══
        x = self.task_controller(x)

        return x

    def get_aux_loss(self) -> torch.Tensor:
        """获取辅助损失。"""
        return self.moe.aux_loss


# ═══════════════════════════════════════════════════════════════
#                CortexBlock V3 — 终极进化
# ═══════════════════════════════════════════════════════════════

from typing import Optional, Tuple, Any

import torch.utils.checkpoint as ckpt

try:
    from .causal_reasoning import CausalReasoningModule
    from .self_evolution import DynamicPathController
    from .multi_agent import MultiAgentSystem
    from .adversarial import AdversarialShield
except ImportError:
    from cortexnet.causal_reasoning import CausalReasoningModule
    from cortexnet.self_evolution import DynamicPathController
    from cortexnet.multi_agent import MultiAgentSystem
    from cortexnet.adversarial import AdversarialShield


class MixtureOfDepths(nn.Module):
    """Mixture of Depths: token 级自适应层跳过。

    学习一个路由器，决定每个 token 是否需要完整的 block 处理。
    「简单」token 跳过完整计算，只做轻量残差；
    「困难」token 接受全部 5 路径处理。
    整体减少平均计算量而不损失建模能力。

    参考: Raposo et al., "Mixture-of-Depths" (2024)

    Args:
        d_model: 模型维度
        capacity: 每层处理的 token 比例 (0~1)
    """

    def __init__(self, d_model: int, capacity: float = 0.5):
        super().__init__()
        self.capacity = capacity
        self.router = nn.Linear(d_model, 1, bias=False)
        # 轻量旁路（跳过 token 不是纯恒等，而是经过简单变换保持表达力）
        self.bypass = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
        )
        nn.init.eye_(self.bypass[0].weight)  # 初始化为恒等，渐进式学习

    def forward(
        self,
        x: torch.Tensor,
        block_fn,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            block_fn: callable，接受 (B, L, D) 返回 (B, L, D) 的完整 block 函数
        Returns:
            output: (B, L, D)
        """
        B, L, D = x.shape
        k = max(1, int(L * self.capacity))

        if k >= L:
            return block_fn(x)

        scores = self.router(x).squeeze(-1)  # (B, L)
        _, top_idx = scores.topk(k, dim=-1, sorted=True)  # (B, k)
        top_idx_sorted, _ = top_idx.sort(dim=-1)

        # Gather 选中 + 未选中 token（连续内存）
        idx_exp = top_idx_sorted.unsqueeze(-1).expand(-1, -1, D)
        x_selected = x.gather(1, idx_exp)  # (B, k, D)

        # 完整处理选中 token
        out_selected = block_fn(x_selected)

        # 构建未选中 token 索引（连续 gather，避免 bool mask）
        selected_mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        selected_mask.scatter_(1, top_idx_sorted, True)

        # 全量 bypass（先初始化为原始 x，再覆盖选中位置）
        output = self.bypass(x)  # (B, L, D) — 所有 token 经过轻量 bypass
        # 选中 token 覆盖 bypass 结果
        output.scatter_(1, idx_exp, out_selected)

        return output


class CortexBlockV3(nn.Module):
    """CortexNet V3 终极进化构建块。

    在 V2 基础上新增 4 项突破性能力：

    ┌────────────────── CortexBlock V3 ──────────────────────────┐
    │                                                             │
    │  Input ──► Adversarial Shield (对抗防御) ──┐               │
    │                                             │               │
    │  ├──► RMSNorm ──┬──► SSM ──────────────────┐│              │
    │  │              ├──► Sparse Attention ──────┤│              │
    │  │              ├──► Hierarchical Memory ───┤│              │
    │  │              ├──► Graph Reasoning ────────┤              │
    │  │              └──► Causal Reasoning ──────┤│ (5条路径)   │
    │  │                                          ││              │
    │  │  Dynamic Path Controller ──► 路径开关 ───┤│              │
    │  │  (自我进化: 动态激活/禁用路径)           ││              │
    │  │                                          ││              │
    │  │              ┌── Adaptive Fusion (5路) ◄─┘│              │
    │  │ (residual)   │                             │              │
    │  ├──── + ◄──────┤ Meta Adapter                │              │
    │  │                                             │              │
    │  ├──► RMSNorm ──► Collaborative MoE ──┐       │              │
    │  │ (residual)                          │       │              │
    │  └──── + ◄────────────────────────────┘       │              │
    │       │                                        │              │
    │       └──► Multi-Agent Coordinator ────────────┘              │
    │            (多智能体协作决策)                                 │
    │            │                                                  │
    │            └──► Task Controller ──► Output                   │
    └──────────────────────────────────────────────────────────────┘

    V3 新增:
      1. 因果推理 — 第5条处理路径，理解因果而非仅相关性
      2. 自我进化 — 动态路径开关，输入驱动的架构适应
      3. 多智能体 — 多专家协作决策
      4. 对抗防御 — 三层防护保障安全性
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_gradient_checkpointing = getattr(config, "use_gradient_checkpointing", False)
        D = config.hidden_size

        # 预归一化
        self.norm1 = RMSNorm(D, config.norm_eps)
        self.norm2 = RMSNorm(D, config.norm_eps)

        # ═══ 对抗防御 (V3) ═══
        self.adversarial_shield = AdversarialShield(D)

        # ═══ Mixture of Depths (可选) ═══
        use_mod = getattr(config, "use_mixture_of_depths", False)
        mod_cap = getattr(config, "mod_capacity", 0.5)
        self.mixture_of_depths = MixtureOfDepths(D, mod_cap) if use_mod else None

        # ═══ 五条并行处理路径 ═══
        self.ssm = MultiScaleSSM(
            d_model=D, num_scales=config.num_scales,
            state_size=config.ssm_state_size, expand_factor=config.ssm_expand_factor,
        )
        self.attention = SelectiveSparseAttention(
            d_model=D, num_heads=config.num_heads,
            top_k_ratio=config.top_k_ratio, k_mode=config.attention_k_mode,
            max_seq_len=config.max_seq_len, rope_theta=config.rope_theta,
            dropout=config.dropout,
            sliding_window_size=getattr(config, "sliding_window_size", 0),
        )
        self.memory = HierarchicalMemorySystem(
            d_model=D, working_dim=getattr(config, "memory_dim", 64),
            episodic_slots=getattr(config, "episodic_slots", 32),
            semantic_slots=getattr(config, "semantic_slots", 64),
            num_heads=min(4, config.num_heads),
        )
        self.graph_reasoning = GraphReasoningModule(
            d_model=D,
            num_neighbors=getattr(config, "graph_neighbors", 16),
            num_iterations=getattr(config, "graph_iterations", 2),
        )
        # 第5条路径: 因果推理 (V3 新增)
        self.causal_reasoning = CausalReasoningModule(
            d_model=D,
            num_heads=min(4, config.num_heads),
            num_counterfactuals=getattr(config, "num_counterfactuals", 4),
        )

        # ═══ 自我进化: 动态路径控制 (V3) ═══
        self.path_controller = DynamicPathController(D, num_paths=5)

        # ═══ 5路自适应融合 ═══
        self.fusion = AdaptiveFusionGate(D, num_paths=5)

        # ═══ 元学习适配器 ═══
        self.meta_adapter = MetaLearningAdapter(D)

        # ═══ 协作式 MoE ═══
        self.moe = CollaborativeMoE(
            d_model=D, d_ff=config.expert_ff_dim,
            num_experts=config.num_experts, num_active=config.num_active_experts,
            aux_loss_weight=config.moe_aux_loss_weight,
        )

        # ═══ 多智能体协作 (V3) ═══
        self.multi_agent = MultiAgentSystem(
            d_model=D,
            num_agents=getattr(config, "num_agents", 4),
        )

        # ═══ 任务自适应控制 ═══
        self.task_controller = TaskAdaptiveController(
            D, num_modes=getattr(config, "num_task_modes", 4),
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_cache: Optional[Tuple[Any, Any, Any]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[Tuple[Any, Any, Any]]]:
        """Forward with optional cache for incremental decoding.

        Args:
            x: (batch, seq_len, hidden_size)
            past_cache: (ssm_state, (mem, z), (K, V, top_k)) 或 None
            use_cache: 若 True 返回 (output, new_cache)
        """
        x = self.adversarial_shield.defend_input(x)

        # Mixture of Depths：选择哪些 token 需要完整处理
        if self.mixture_of_depths is not None and not use_cache:
            return self.mixture_of_depths(x, lambda z: self._full_forward(z, None, False))

        return self._full_forward(x, past_cache, use_cache)

    def _full_forward(
        self,
        x: torch.Tensor,
        past_cache: Optional[Tuple[Any, Any, Any]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[Tuple[Any, Any, Any]]]:
        """完整的 block 前向，支持梯度检查点。"""
        residual = x
        x_norm = self.norm1(x)
        path_gates = self.path_controller(x_norm)

        past_ssm, past_mem, past_kv = (None, None, None)
        if past_cache is not None:
            past_ssm, past_mem, past_kv = past_cache

        # ═══ 五路并行处理（路径感知跳过 + 梯度检查点）═══
        _SKIP_THRESH = 0.0 if self.training else 0.01  # eval 时跳过近零路径

        def _compute_paths(xn, pg, p_ssm, p_mem, p_kv, uc):
            zero = torch.zeros_like(xn)
            n_ssm = n_kv = n_mem = None

            # 路径感知跳过：门控 ≈ 0 的路径直接输出零，不做计算
            if pg[:, 0].max() > _SKIP_THRESH:
                s_res = self.ssm(xn, past_state=p_ssm, use_cache=uc)
                s_out = (s_res[0] if uc and isinstance(s_res, tuple) else s_res) * pg[:, 0:1].unsqueeze(1)
                n_ssm = s_res[1] if uc and isinstance(s_res, tuple) else None
            else:
                s_out = zero

            if pg[:, 1].max() > _SKIP_THRESH:
                a_res = self.attention(xn, past_key_value=p_kv, use_cache=uc)
                a_out = (a_res[0] if uc and isinstance(a_res, tuple) else a_res) * pg[:, 1:2].unsqueeze(1)
                n_kv = a_res[1] if uc and isinstance(a_res, tuple) else None
            else:
                a_out = zero

            if pg[:, 2].max() > _SKIP_THRESH:
                m_res = self.memory(xn, past_working_memory=p_mem, use_cache=uc)
                m_out = (m_res[0] if uc and isinstance(m_res, tuple) else m_res) * pg[:, 2:3].unsqueeze(1)
                n_mem = m_res[1] if uc and isinstance(m_res, tuple) else None
            else:
                m_out = zero

            g_out = self.graph_reasoning(xn) * pg[:, 3:4].unsqueeze(1) if pg[:, 3].max() > _SKIP_THRESH else zero
            c_out = self.causal_reasoning(xn) * pg[:, 4:5].unsqueeze(1) if pg[:, 4].max() > _SKIP_THRESH else zero

            fused = self.fusion(s_out, a_out, m_out, g_out, c_out)
            fused = self.meta_adapter(fused)
            return fused, n_ssm, n_mem, n_kv

        if self.use_gradient_checkpointing and self.training and not use_cache:
            fused, new_ssm, new_mem, new_kv = ckpt.checkpoint(
                _compute_paths, x_norm, path_gates,
                past_ssm, past_mem, past_kv, use_cache,
                use_reentrant=False,
            )
        else:
            fused, new_ssm, new_mem, new_kv = _compute_paths(
                x_norm, path_gates, past_ssm, past_mem, past_kv, use_cache
            )

        fused = self.adversarial_shield.defend_features(fused)
        x = residual + self.dropout(fused)

        residual = x
        x = residual + self.dropout(self.moe(self.norm2(x)))
        x = self.multi_agent(x)
        x = self.task_controller(x)

        if use_cache:
            return x, (new_ssm, new_mem, new_kv)
        return x

    def get_aux_loss(self) -> torch.Tensor:
        return self.moe.aux_loss
