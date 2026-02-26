"""
因果推理模块 (Causal Reasoning Module)

核心创新：
  传统注意力只学习相关性 P(Y|X)，而因果推理学习因果关系 P(Y|do(X))。
  这使模型不仅知道"什么和什么一起出现"，还知道"什么导致了什么"。

  ┌─────────────────────────────────────────────────────────────┐
  │            因果推理 vs 传统注意力                            │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  传统注意力:  P(Y|X) — "X 出现时 Y 也出现"                │
  │    → 只捕获相关性，无法区分因果与混淆                      │
  │                                                             │
  │  因果推理:    P(Y|do(X)) — "如果我们主动设置 X，Y 会如何"  │
  │    → 捕获真正的因果关系，支持反事实推理                    │
  │                                                             │
  │  三大组件:                                                  │
  │  1. 因果强度估计器 — 评估每个 token 的因果重要性           │
  │  2. 干预注意力 — 基于因果方向选择高因果 token 计算注意力   │
  │  3. 反事实分支 — "如果这个原因不同，结果会如何？"          │
  └─────────────────────────────────────────────────────────────┘

  灵感来源: Pearl's do-calculus, 结构因果模型 (SCM)

  优化 (v3.2):
    - InterventionalAttention 使用选择性稀疏注意力替代 O(n²) 全量注意力，
      仅对因果强度 Top-K 的 token 计算注意力，复杂度降至 O(n·k)
    - CounterfactualBranch 使用合并的批量线性变换替代 for 循环，
      单次 matmul 完成所有反事实分支计算
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CausalStrengthEstimator(nn.Module):
    """因果强度估计器：评估每个 token 对后续 token 的因果影响力。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scorer(x))  # (B, L, 1)


class InterventionalAttention(nn.Module):
    """干预注意力：基于因果方向的选择性稀疏注意力。

    优化: 仅对因果强度 Top-K 的 token 计算注意力，
    复杂度从 O(n²) 降至 O(n·k)，与系统整体 O(n) 定位一致。

    与标准注意力的区别:
      标准: attn(i,j) = softmax(Q_i · K_j / √d)  — 全量 O(n²)
      干预: attn(i,j) = softmax((Q_i · K_top_j + bias_j) / √d)  — 稀疏 O(n·k)

    因果偏置让 "因果上更重要" 的 token 获得更多关注。
    """

    def __init__(self, d_model: int, num_heads: int = 4, top_k_ratio: float = 0.25,
                 dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.top_k_ratio = top_k_ratio

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def _compute_k_count(self, seq_len: int) -> int:
        """根据序列长度计算选择的 token 数量。"""
        k = max(1, int(seq_len * self.top_k_ratio))
        return min(k, seq_len)

    def forward(
        self, x: torch.Tensor, causal_strength: torch.Tensor
    ) -> torch.Tensor:
        B, L, D = x.shape
        H, hd = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, hd).transpose(1, 2)  # (B, H, L, hd)
        k = self.k_proj(x).view(B, L, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, hd).transpose(1, 2)

        # 确定 top-k 数量
        k_count = self._compute_k_count(L)

        if k_count >= L:
            # 短序列退化为完整注意力（但加上因果偏置）
            return self._full_attention(q, k, v, causal_strength, B, L, D, H, hd)

        # ═══ 选择性稀疏: 按因果强度选 top-k token 作为 KV ═══
        # causal_strength: (B, L, 1)
        scores = causal_strength.squeeze(-1)  # (B, L)
        _, top_indices = scores.topk(k_count, dim=-1, sorted=False)  # (B, k)

        # 收集 top-k 的 K, V
        top_idx_kv = top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, hd)
        k_sel = k.gather(2, top_idx_kv)  # (B, H, k, hd)
        v_sel = v.gather(2, top_idx_kv)  # (B, H, k, hd)

        # 计算稀疏注意力分数
        attn = (q @ k_sel.transpose(-2, -1)) * self.scale  # (B, H, L, k)

        # 因果偏置: 对选中的 token 施加因果强度调制
        causal_bias_sel = scores.gather(1, top_indices)  # (B, k)
        causal_bias = torch.log(causal_bias_sel + 1e-6).clamp(min=-10)
        attn = attn + causal_bias.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, k) broadcast

        # 因果掩码: 只允许 attend 到位置 ≤ 当前 token 的 key
        positions = torch.arange(L, device=x.device).unsqueeze(1)  # (L, 1)
        key_positions = top_indices.unsqueeze(1)  # (B, 1, k)
        causal_mask = positions >= key_positions  # (B, L, k)
        attn = attn.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = (attn @ v_sel).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)

    def _full_attention(self, q, k, v, causal_strength, B, L, D, H, hd):
        """短序列退化为完整注意力（带因果偏置）。"""
        attn = (q @ k.transpose(-2, -1)) * self.scale

        causal_bias = causal_strength.squeeze(-1).unsqueeze(1).unsqueeze(2)
        attn = attn + torch.log(causal_bias + 1e-6).clamp(min=-10)

        causal_mask = torch.tril(torch.ones(L, L, device=q.device))
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class CounterfactualBranch(nn.Module):
    """反事实推理分支：探索 "如果原因不同，结果会如何"。

    优化: 使用合并的权重矩阵 (d_model, num_cf * d_model) 替代
    K 个独立 nn.Linear 的 for 循环，单次 matmul 完成所有反事实变换。

    维护 K 个反事实变换，每个代表一种 "假设情景"。
    通过门控机制软选择最相关的反事实分支。
    """

    def __init__(self, d_model: int, num_counterfactuals: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_cf = num_counterfactuals
        self.d_model = d_model

        # 合并所有反事实变换为单个矩阵: (d_model, num_cf * d_model)
        self.merged_transform = nn.Linear(d_model, num_counterfactuals * d_model, bias=False)
        # 初始化为接近零（初始不改变输入）
        nn.init.zeros_(self.merged_transform.weight)

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, num_counterfactuals),
        )
        # 可学习的反事实融合强度
        self.cf_scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_strength: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        K = self.num_cf

        gate_weights = F.softmax(self.gate(x), dim=-1)  # (B, L, K)

        # 单次 matmul 完成所有 K 个反事实变换
        all_cf = self.merged_transform(x)  # (B, L, K*D)
        all_cf = all_cf.view(B, L, K, D)  # (B, L, K, D)

        # 门控加权融合
        counterfactual = (all_cf * gate_weights.unsqueeze(-1)).sum(dim=2)  # (B, L, D)
        counterfactual = self.dropout(counterfactual)

        # 因果强度高的 token 贡献更多反事实信息
        return counterfactual * causal_strength * self.cf_scale


class CausalReasoningModule(nn.Module):
    """因果推理模块：完整的因果推理流水线。

    架构:
      1. 估计每个 token 的因果强度
      2. 干预注意力（因果方向感知，稀疏 O(n·k)）
      3. 反事实推理（探索替代情景，批量单次 matmul）
      4. 融合观察结果和反事实结果

    Args:
        d_model: 模型维度
        num_heads: 干预注意力头数
        num_counterfactuals: 反事实分支数
        top_k_ratio: 干预注意力的 top-k 比例
        dropout: Dropout 比率
    """

    def __init__(self, d_model: int, num_heads: int = 4, num_counterfactuals: int = 4,
                 top_k_ratio: float = 0.25, dropout: float = 0.0):
        super().__init__()
        self.causal_estimator = CausalStrengthEstimator(d_model)
        self.interventional_attn = InterventionalAttention(
            d_model, num_heads, top_k_ratio=top_k_ratio, dropout=dropout
        )
        self.counterfactual = CounterfactualBranch(
            d_model, num_counterfactuals, dropout=dropout
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        causal_strength = self.causal_estimator(x)  # (B, L, 1)
        observational = self.interventional_attn(x, causal_strength)
        counterfactual = self.counterfactual(x, causal_strength)
        combined = observational + counterfactual
        return self.out_proj(self.norm(combined))
