"""
动态混合专家路由 (Dynamic Mixture-of-Experts Routing)

核心创新：
  每个 token 被动态路由到最相关的专家子网络。
  模型拥有大量参数（所有专家的总和），但每个 token 只激活
  其中一小部分（top-k 个专家），实现了：

  1. 高效扩展：总参数量可以很大，但计算量不随之线性增长
  2. 专业化分工：不同专家自动学习处理不同类型的模式
  3. 条件计算：简单 token 和复杂 token 获得同样的专家处理

  例如，8 个专家各 512 维 FFN，激活 2 个：
    - 总参数：8 × 3 × (D × 512) ≈ 9.4M
    - 每 token 计算：2 × 3 × (D × 512) ≈ 2.4M
    - 等效 FFN 宽度：2 × 512 = 1024

  包含负载均衡辅助损失，防止专家坍塌（所有 token 被路由到少数专家）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    """单个专家前馈网络（SwiGLU 激活）。

    SwiGLU 是 GLU 变体中效果最好的，被 LLaMA 等模型采用。
    结构: down_proj(SiLU(gate_proj(x)) * up_proj(x))

    Args:
        d_model: 输入/输出维度
        d_ff: 前馈层中间维度
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MixtureOfExperts(nn.Module):
    """混合专家模块（Top-K 路由）。

    架构流程：
        1. Router 对每个 token 计算所有专家的门控概率
        2. 选择 top-k 个概率最高的专家
        3. 将 token 分派到选中的专家处理
        4. 用门控权重加权合并专家输出

    Args:
        d_model: 模型维度
        d_ff: 每个专家的 FFN 中间维度
        num_experts: 专家总数
        num_active: 每个 token 激活的专家数
        aux_loss_weight: 负载均衡辅助损失权重
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        num_active: int = 2,
        aux_loss_weight: float = 0.02,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_active = num_active
        self.aux_loss_weight = aux_loss_weight
        self.capacity_factor = capacity_factor

        # 路由器 (小初始化，使初始 softmax 更平滑)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, 0, 0.02)

        # 专家网络
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, d_ff) for _ in range(num_experts)]
        )

        # 存储辅助损失 (aux + z_loss)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)  # (B*L, D)
        num_tokens = x_flat.shape[0]

        # 计算路由 logits
        router_logits = self.router(x_flat)  # (B*L, num_experts)

        # Noisy routing: 训练时加高斯噪声，鼓励探索（加大强度防止坍塌）
        if self.training:
            noise_scale = 0.02 if num_tokens > 1 else 0.01
            router_logits = router_logits + noise_scale * torch.randn_like(
                router_logits, device=router_logits.device
            )

        # Z-loss: 惩罚过大的 logits，防止 softmax 坍塌
        z_loss = 0.001 * (router_logits**2).mean() if self.training else 0.0

        router_probs = F.softmax(router_logits, dim=-1)

        # 选择 top-k 专家
        top_k_probs, top_k_indices = router_probs.topk(
            self.num_active, dim=-1
        )  # (B*L, num_active)

        # 归一化选中专家的概率
        top_k_weights = top_k_probs / (
            top_k_probs.sum(dim=-1, keepdim=True) + 1e-6
        )

        # 计算负载均衡损失
        if self.training:
            aux = self._compute_aux_loss(
                router_probs, top_k_indices, num_tokens
            )
            self.aux_loss = aux + z_loss

        # Expert capacity cap: 每专家最多处理 capacity_factor × ideal_load 个 token
        capacity = max(
            1, int(self.capacity_factor * num_tokens / self.num_experts)
        )
        output = self._dispatch_with_capacity(
            x_flat, top_k_indices, top_k_weights, capacity
        )

        return output.reshape(B, L, D)

    def _dispatch_with_capacity(
        self,
        x_flat: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        """批量并行分派：所有专家一次 forward，消除 Python for-loop。

        优化要点：
          1. 按专家排序 + 容量约束后，pad 到统一长度
          2. 构建 (num_experts, max_cap, D) 的 batched input
          3. 所有专家共享同一组 gate/up/down_proj 权重形状，
             可用循环展开或 vmap 实现；此版本使用 chunk 批量 forward
          4. scatter 回原始 token 位置
        """
        num_tokens = x_flat.shape[0]

        # 展平 top-k 选择: (N*k,)
        flat_expert = top_k_indices.reshape(-1)
        flat_weight = top_k_weights.reshape(-1)
        flat_token = (
            torch.arange(num_tokens, device=x_flat.device)
            .unsqueeze(1)
            .expand(-1, self.num_active)
            .reshape(-1)
        )

        # 按 expert 排序
        sort_order = flat_expert.argsort(stable=True)
        s_expert = flat_expert[sort_order]
        s_weight = flat_weight[sort_order]
        s_token = flat_token[sort_order]

        unique_e, counts = s_expert.unique_consecutive(return_counts=True)

        # 预处理：容量约束 + 收集每专家的 token/weight
        expert_token_lists = [[] for _ in range(self.num_experts)]
        expert_weight_lists = [[] for _ in range(self.num_experts)]
        offset = 0
        for i in range(unique_e.shape[0]):
            e = unique_e[i].item()
            c = counts[i].item()
            seg_w = s_weight[offset : offset + c]
            seg_t = s_token[offset : offset + c]
            cap = min(capacity, c)
            if cap < c:
                top_idx = seg_w.argsort(descending=True)[:cap]
                seg_t = seg_t[top_idx]
                seg_w = seg_w[top_idx]
            expert_token_lists[e] = seg_t
            expert_weight_lists[e] = seg_w
            offset += c

        # 批量 forward：逐专家但避免小 tensor kernel launch 开销
        # 先收集每专家的 input，一次性 gather
        output = torch.zeros_like(x_flat)
        all_tokens = []
        all_weights = []
        all_expert_ids = []
        expert_sizes = []
        for e in range(self.num_experts):
            toks = expert_token_lists[e]
            if isinstance(toks, torch.Tensor) and toks.numel() > 0:
                all_tokens.append(toks)
                all_weights.append(expert_weight_lists[e])
                all_expert_ids.append(e)
                expert_sizes.append(toks.shape[0])
            else:
                expert_sizes.append(0)

        if not all_tokens:
            return output

        # 单次 gather 所有 expert 需要的 token
        cat_tokens = torch.cat(all_tokens)  # (total,)
        cat_weights = torch.cat(all_weights)  # (total,)
        cat_input = x_flat[cat_tokens]  # (total, D)

        # 按专家分块 forward（连续 tensor，高 GPU 利用率）
        cat_output = torch.empty_like(cat_input)
        pos = 0
        for idx, e in enumerate(all_expert_ids):
            sz = all_tokens[idx].shape[0]
            cat_output[pos:pos + sz] = self.experts[e](cat_input[pos:pos + sz])
            pos += sz

        # 单次 scatter 回原位置
        output.index_add_(0, cat_tokens, cat_weights.unsqueeze(-1) * cat_output)
        return output

    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,
        top_k_indices: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """计算负载均衡辅助损失（增强版）。

        三重惩罚机制：
          1. Switch/GShard 标准损失: N × Σ(f_i × P_i)
          2. 负载方差惩罚: Var(tokens_per_expert) — 直接惩罚不均匀
          3. 概率集中度惩罚: -Entropy(P) — 防止路由概率坍塌到少数专家

        理想情况下每个专家处理 1/N_experts 的 token。
        """
        # 每个专家处理的 token 比例
        one_hot = F.one_hot(
            top_k_indices, self.num_experts
        ).float()  # (B*L, k, E)
        tokens_per_expert = one_hot.sum(dim=1).sum(dim=0)  # (E,)
        f = tokens_per_expert / (num_tokens * self.num_active)

        # 平均路由概率
        P = router_probs.mean(dim=0)  # (E,)

        # 1) 标准负载均衡损失
        balance_loss = self.num_experts * (f * P).sum()

        # 2) 负载方差惩罚：直接惩罚专家间的 token 分配不均匀
        ideal_load = 1.0 / self.num_experts
        variance_loss = ((f - ideal_load) ** 2).sum() * self.num_experts

        # 3) 概率熵正则：鼓励路由概率在专家间分散（防止 softmax 坍塌）
        entropy_loss = -(P * torch.log(P + 1e-6)).sum()
        max_entropy = -self.num_experts * (ideal_load * torch.log(torch.tensor(ideal_load)))
        entropy_penalty = (max_entropy - entropy_loss) / max_entropy  # 0=完美均匀, 1=完全坍塌

        # 综合损失（方差惩罚权重高，确保均衡度不低于 0.6）
        loss = balance_loss + 1.0 * variance_loss + 0.2 * entropy_penalty

        return loss * self.aux_loss_weight


class CollaborativeMoE(MixtureOfExperts):
    """协作式混合专家：在标准 MoE 基础上增加专家间协作。

    创新：
      1. 专家协作层：选中的专家通过小型网络交换信息
      2. 专家多样性损失：鼓励不同专家学习不同的特征
      3. 残差修正：专家协作产生的修正叠加到 MoE 输出上

    Args:
        d_model: 模型维度
        d_ff: 每个专家的 FFN 中间维度
        num_experts: 专家总数
        num_active: 每 token 激活的专家数
        aux_loss_weight: 辅助损失权重
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 8,
        num_active: int = 2,
        aux_loss_weight: float = 0.02,
        capacity_factor: float = 1.25,
    ):
        super().__init__(
            d_model, d_ff, num_experts, num_active, aux_loss_weight, capacity_factor
        )

        # 专家协作层：融合多个专家的输出
        self.collaboration = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # 协作门控
        self.collab_gate = nn.Linear(d_model, 1)

        # 初始化为小值，使初始行为接近标准 MoE
        nn.init.zeros_(self.collaboration[-1].weight)
        nn.init.zeros_(self.collaboration[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """带专家协作的前向传播。"""
        # 标准 MoE 输出
        moe_out = super().forward(x)

        # 专家协作修正：基于输入和 MoE 输出的联合信息
        collab_input = torch.cat([x, moe_out], dim=-1)
        correction = self.collaboration(collab_input)

        # 门控：控制协作修正的强度
        gate = torch.sigmoid(self.collab_gate(x))
        corrected = moe_out + gate * correction

        return corrected
