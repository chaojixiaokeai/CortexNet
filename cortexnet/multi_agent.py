"""
多智能体协作系统 (Multi-Agent Collaboration System)

核心创新：
  将单一模型扩展为多个专业化智能体的协作系统。
  每个智能体拥有独特的 "认知风格"，通过协调器融合集体智慧。

  ┌─────────────────────────────────────────────────────────────┐
  │              多智能体协作架构                                │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  输入 ─────┬──► Agent 1 (逻辑推理型) ──┐                   │
  │            ├──► Agent 2 (模式识别型) ──┤                   │
  │            ├──► Agent 3 (创意联想型) ──┤──► 协调器 ──► 输出│
  │            └──► Agent 4 (记忆检索型) ──┘                   │
  │                                                             │
  │  ● 每个 Agent 有独特的 "认知风格" (specialty embedding)    │
  │  ● 协调器根据输入内容动态分配各 Agent 的权重               │
  │  ● Agent 间通过共享消息板交换关键信息                      │
  └─────────────────────────────────────────────────────────────┘

  类比: 如同一个专家团队——不同专家从不同角度分析问题，
  最终由一个主持人综合各方意见做出决策。

  优化 (v3.2):
    - SharedMessageBoard.write() 使用残差连接保持梯度流，
      替代了原先 .data 原地赋值绕过 autograd 的问题
    - BatchedAgents 将多个 Agent 的参数合并为批量矩阵乘法，
      替代了原先的列表推导式串行执行
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BatchedAgents(nn.Module):
    """批量化智能体：将多个 Agent 的 SwiGLU FFN 合并为单次矩阵乘法。

    相比 ModuleList + for 循环，通过 (num_agents, d_model, d_ff) 的权重张量
    一次 batched matmul 完成所有 Agent 的计算，无 Python 循环开销。

    Args:
        d_model: 输入/输出维度
        num_agents: 智能体数量
    """

    def __init__(self, d_model: int, num_agents: int = 4):
        super().__init__()
        self.num_agents = num_agents
        d_ff = d_model * 2

        # 每个 Agent 的独特认知风格
        self.specialty = nn.Parameter(torch.randn(num_agents, 1, d_model) * 0.02)

        # 合并所有 Agent 的 SwiGLU 权重: (num_agents, in, out)
        self.gate_proj = nn.Parameter(torch.randn(num_agents, d_model, d_ff) * (d_model ** -0.5))
        self.up_proj = nn.Parameter(torch.randn(num_agents, d_model, d_ff) * (d_model ** -0.5))
        self.down_proj = nn.Parameter(torch.randn(num_agents, d_ff, d_model) * (d_ff ** -0.5))

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
        Returns:
            outputs: (B, L, num_agents, D) — 所有 Agent 的输出
        """
        B, L, D = x.shape

        # 加入 specialty 并归一化: (K, B, L, D)
        x_agents = self.norm(x.unsqueeze(0) + self.specialty.unsqueeze(1))

        # 批量 SwiGLU: 所有 Agent 同时计算
        # x_agents: (K, B, L, D), gate_proj: (K, D, FF) → (K, B, L, FF)
        gate = torch.einsum('kbld,kdf->kblf', x_agents, self.gate_proj)
        up = torch.einsum('kbld,kdf->kblf', x_agents, self.up_proj)
        hidden = F.silu(gate) * up
        out = torch.einsum('kblf,kfd->kbld', hidden, self.down_proj)

        # 转换为 (B, L, K, D)
        return out.permute(1, 2, 0, 3)


class SharedMessageBoard(nn.Module):
    """共享消息板：智能体间的信息交换通道。

    每个 Agent 向消息板写入关键发现，并读取其他 Agent 的消息。
    通过交叉注意力实现选择性的信息交换。

    优化: write() 方法通过返回更新后的 slots 视图（而非 .data 原地赋值）
    保持梯度流通，支持端到端训练。

    Args:
        d_model: 维度
        num_slots: 消息槽位数
    """

    def __init__(self, d_model: int, num_slots: int = 8):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.write_proj = nn.Linear(d_model, d_model, bias=False)
        self.write_gate = nn.Linear(d_model, num_slots, bias=False)
        self.read_proj = nn.Linear(d_model, d_model, bias=False)
        self.ema_decay = nn.Parameter(torch.tensor(0.9))

        # 用于存储上一步的 slots 更新（保持梯度流）
        self._current_slots: torch.Tensor | None = None

    def _get_slots(self) -> torch.Tensor:
        """获取当前的 slots 值（如果有残差更新则使用更新后的值）。"""
        if self._current_slots is not None:
            return self._current_slots
        return self.slots

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """从消息板读取相关信息。"""
        B, L, D = query.shape
        q = self.read_proj(query)
        slots = self._get_slots().expand(B, -1, -1)
        attn = F.softmax(q @ slots.transpose(-1, -2) / D**0.5, dim=-1)
        return attn @ slots

    def write(self, content: torch.Tensor) -> None:
        """向消息板写入信息（EMA 累积更新，保持梯度流）。

        通过残差连接更新 slots，不再使用 .data 原地赋值。
        梯度可以通过 write_proj / write_gate 流回写入内容。
        """
        # content: (B, L, D) — Agent 的聚合输出
        projected = self.write_proj(content)  # (B, L, D)
        # 注意力权重决定写入哪些槽位
        gate = F.softmax(self.write_gate(content.mean(dim=1)), dim=-1)  # (B, S)
        # 新消息 = content 的加权平均
        new_msg = projected.mean(dim=1, keepdim=True)  # (B, 1, D)
        # EMA 更新（保留计算图，梯度可回传到 content）
        decay = torch.sigmoid(self.ema_decay)
        update = gate.unsqueeze(-1) * new_msg  # (B, S, D)
        # batch 平均作为全局更新方向
        update_mean = update.mean(dim=0, keepdim=True)  # (1, S, D)
        # 残差式更新: 保持梯度流通
        self._current_slots = decay * self.slots + (1 - decay) * update_mean

    def reset_state(self):
        """重置消息板状态（每个训练 step 开始时调用）。"""
        self._current_slots = None


class AgentCoordinator(nn.Module):
    """智能体协调器：综合各 Agent 的输出做最终决策。

    根据输入内容动态评估每个 Agent 的可信度，
    加权融合产生最终输出。

    Args:
        d_model: 维度
        num_agents: 智能体数量
    """

    def __init__(self, d_model: int, num_agents: int):
        super().__init__()
        self.trust_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_agents),
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, agent_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) 原始输入
            agent_outputs: (B, L, K, D) 所有 Agent 的堆叠输出
        Returns:
            merged: (B, L, D)
        """
        B, L, D = x.shape

        context = x.mean(dim=1, keepdim=True).expand(-1, L, -1)
        coord_input = torch.cat([x, context], dim=-1)
        trust = F.softmax(self.trust_scorer(coord_input), dim=-1)  # (B, L, K)

        merged = (agent_outputs * trust.unsqueeze(-1)).sum(dim=2)
        return self.out_proj(merged)


class MultiAgentSystem(nn.Module):
    """多智能体协作系统：统一接口。

    Args:
        d_model: 模型维度
        num_agents: 智能体数量
        message_slots: 共享消息板槽位数
        dropout: Dropout 比率
    """

    def __init__(
        self,
        d_model: int,
        num_agents: int = 4,
        message_slots: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.batched_agents = BatchedAgents(d_model, num_agents)
        self.message_board = SharedMessageBoard(d_model, message_slots)
        self.coordinator = AgentCoordinator(d_model, num_agents)
        self.residual_gate = nn.Linear(d_model, 1)
        # 可学习的上下文融合强度（替代硬编码 0.1）
        self.context_scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 从消息板获取共享上下文
        shared_context = self.message_board.read(x)

        # 所有 Agent 批量并行处理
        enriched = x + shared_context * self.context_scale
        agent_outputs = self.batched_agents(enriched)  # (B, L, K, D)

        # 协调器融合
        merged = self.coordinator(x, agent_outputs)
        merged = self.dropout(merged)

        # 向消息板写入本轮的聚合输出（EMA 累积，保持梯度流）
        self.message_board.write(merged)

        # 残差门控
        gate = torch.sigmoid(self.residual_gate(x))
        return x + gate * merged


# ═══ 向后兼容别名 ═══
# 保持旧的 import 路径可用
SpecialistAgent = BatchedAgents  # 类接口变更，但名称保留
