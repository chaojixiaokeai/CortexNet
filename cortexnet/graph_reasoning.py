"""
图推理模块 (Graph Reasoning Module)

核心创新：
  将 token 序列视为图结构，通过消息传递神经网络 (MPNN)
  进行多步关系推理。与注意力机制的关键区别：

  ┌─────────────────────────────────────────────────────┐
  │           注意力 vs 图推理                           │
  ├─────────────────────────────────────────────────────┤
  │ 注意力:                                             │
  │  • 单步信息聚合                                     │
  │  • 全局/稀疏的软关联                                │
  │  • 无显式关系建模                                   │
  │                                                     │
  │ 图推理:                                             │
  │  • 多步迭代推理（信息逐步传播）                     │
  │  • 结构化的邻域关系                                 │
  │  • 显式边特征建模 token 间关系                      │
  │  • 门控线性单元保持推理记忆                         │
  └─────────────────────────────────────────────────────┘

  通过 K 步消息传递，信息可以在图中传播 K 跳距离，
  实现类似 "思维链" 的逐步推理能力。

  复杂度: O(L · k · D) per iteration，其中 k = 邻居数

  优化 (v3.2):
    - 用门控线性单元 (GLU) 替代 GRUCell，完全消除顺序依赖，
      所有节点可并行更新。GRU 的 (B*L, D) 独立 cell 调用本身
      无序列依赖，但 GLU 更轻量且编译器友好
    - 添加了 dropout 支持
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GraphReasoningModule(nn.Module):
    """图推理模块：基于消息传递的关系推理。

    架构流程（每次迭代）：
        1. 构建稀疏图（局部 + 跨步邻居）
        2. 计算边特征（节点对的关系表示）
        3. 生成消息（边特征 × 邻居特征）
        4. 聚合消息（加权求和）
        5. 门控线性单元更新节点状态（替代 GRU，更易并行化）

    多次迭代使信息在图中逐步传播，实现多跳推理。

    Args:
        d_model: 节点特征维度
        num_neighbors: 每个节点的邻居数
        num_iterations: 消息传递迭代次数
        edge_dim: 边特征维度
        dropout: Dropout 比率
    """

    def __init__(
        self,
        d_model: int,
        num_neighbors: int = 16,
        num_iterations: int = 2,
        edge_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_neighbors = num_neighbors
        self.num_iterations = num_iterations

        # 边特征计算：从节点对中提取关系
        self.edge_encoder = nn.Sequential(
            nn.Linear(d_model * 2, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )

        # 消息生成：边特征调制邻居信息
        self.message_fn = nn.Sequential(
            nn.Linear(d_model + edge_dim, d_model),
            nn.GELU(),
        )

        # 消息注意力权重
        self.attn_proj = nn.Linear(edge_dim, 1)

        # 节点更新: 门控线性单元 (GLU)
        # 替代 GRU，完全可并行，不需要顺序执行
        self.node_update = nn.Linear(d_model * 2, d_model * 2)  # 输出一半是gate，一半是value
        self.update_norm = nn.LayerNorm(d_model)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _build_neighbor_indices(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """构建混合邻居索引：局部 + 跨步。

        不使用 O(L²) 的全局相似度计算，而是使用：
        - 局部邻居：捕获相邻 token 的关系
        - 跨步邻居：捕获长距离 token 的关系

        Returns:
            neighbors: (1, L, k) 邻居索引
        """
        k = min(self.num_neighbors, seq_len - 1)
        if k <= 0:
            return torch.zeros(1, seq_len, 1, dtype=torch.long, device=device)

        k_local = max(k // 2, 1)
        k_stride = k - k_local

        indices = torch.arange(seq_len, device=device)

        # 局部邻居
        local_offsets = torch.arange(
            -(k_local // 2), k_local // 2 + 1, device=device
        )
        local_offsets = local_offsets[local_offsets != 0][:k_local]
        if len(local_offsets) < k_local:
            extra = torch.arange(1, k_local - len(local_offsets) + 1, device=device)
            local_offsets = torch.cat([local_offsets, extra])
        local_nb = (indices.unsqueeze(1) + local_offsets.unsqueeze(0)).clamp(
            0, seq_len - 1
        )  # (L, k_local)

        # 跨步邻居
        if k_stride > 0 and seq_len > k_local + 1:
            stride = max(1, seq_len // (k_stride + 1))
            stride_positions = torch.arange(0, seq_len, stride, device=device)
            if len(stride_positions) > k_stride:
                stride_positions = stride_positions[:k_stride]
            else:
                pad = torch.zeros(
                    k_stride - len(stride_positions),
                    dtype=torch.long,
                    device=device,
                )
                stride_positions = torch.cat([stride_positions, pad])
            stride_nb = stride_positions.unsqueeze(0).expand(
                seq_len, -1
            )  # (L, k_stride)
            neighbors = torch.cat([local_nb, stride_nb], dim=-1)  # (L, k)
        else:
            neighbors = local_nb  # (L, k_local)

        return neighbors.unsqueeze(0)  # (1, L, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) 节点特征
        Returns:
            output: (batch, seq_len, d_model) 更新后的节点特征
        """
        B, L, D = x.shape

        if L <= 1:
            # 单 token 时无法构建图，但仍通过投影和归一化保持表达力
            return self.out_proj(self.norm(x))

        # 构建邻居索引
        neighbors = self._build_neighbor_indices(L, x.device)  # (1, L, k)
        k = neighbors.shape[-1]
        neighbors = neighbors.expand(B, -1, -1)  # (B, L, k)

        h = x  # 初始节点状态

        for iteration in range(self.num_iterations):
            # 收集邻居特征
            nb_flat = neighbors.reshape(B, -1)  # (B, L*k)
            nb_expanded = nb_flat.unsqueeze(-1).expand(-1, -1, D)  # (B, L*k, D)
            h_neighbors = h.gather(1, nb_expanded).view(B, L, k, D)  # (B, L, k, D)

            # 计算边特征
            h_expanded = h.unsqueeze(2).expand(-1, -1, k, -1)  # (B, L, k, D)
            edge_input = torch.cat([h_expanded, h_neighbors], dim=-1)  # (B,L,k,2D)
            edge_features = self.edge_encoder(edge_input)  # (B, L, k, edge_dim)

            # 生成消息
            msg_input = torch.cat([h_neighbors, edge_features], dim=-1)
            messages = self.message_fn(msg_input)  # (B, L, k, D)

            # 注意力加权聚合
            attn_logits = self.attn_proj(edge_features).squeeze(-1)  # (B, L, k)
            attn_weights = F.softmax(attn_logits, dim=-1)
            agg_messages = (messages * attn_weights.unsqueeze(-1)).sum(
                dim=2
            )  # (B, L, D)

            # 门控线性单元节点更新（全并行，替代 GRU）
            update_input = torch.cat([h, agg_messages], dim=-1)  # (B, L, 2D)
            update_raw = self.node_update(update_input)  # (B, L, 2D)
            gate_val, value = update_raw.chunk(2, dim=-1)  # 各 (B, L, D)
            gate_val = torch.sigmoid(gate_val)
            h = gate_val * self.update_norm(value) + (1 - gate_val) * h

        output = self.out_proj(self.norm(h))
        return self.dropout(output)
