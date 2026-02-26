"""
元学习自适应模块 (Meta-Learning Adaptation Module)

核心创新：
  使网络能够根据输入上下文快速调整自身行为，
  无需梯度更新即可适应新模式和新任务。

  灵感来源：
  - FiLM (Feature-wise Linear Modulation) — 特征级调制
  - MAML (Model-Agnostic Meta-Learning) — 快速适应
  - HyperNetworks — 用网络生成网络参数

  ┌─────────────────────────────────────────────────────┐
  │              元学习自适应机制                        │
  ├─────────────────────────────────────────────────────┤
  │                                                     │
  │  输入序列 ──► 上下文编码器 ──► 全局上下文向量        │
  │                                    │                │
  │                          ┌─────────┴────────┐      │
  │                          ▼                   ▼      │
  │                    γ (缩放因子)         β (偏移因子) │
  │                          │                   │      │
  │                          └─────────┬────────┘      │
  │                                    ▼                │
  │              FiLM 调制: y = γ ⊙ x + β              │
  │                                                     │
  │  效果：每个样本/任务获得定制化的特征变换            │
  └─────────────────────────────────────────────────────┘

  额外创新：
  - 层级自适应：不同层可以有不同的适应策略
  - 注意力池化：比均值池化更精确的上下文提取
  - 残差调制：保证训练稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    """上下文编码器：从输入序列提取全局上下文向量。

    使用注意力池化代替简单的均值池化，
    使模型能够关注序列中最有信息量的部分。

    Args:
        d_model: 输入维度
    """

    def __init__(self, d_model: int):
        super().__init__()
        # 注意力池化
        self.attn_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn_proj = nn.Linear(d_model, d_model, bias=False)

        # 上下文变换
        self.context_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            context: (batch, d_model)
        """
        B, L, D = x.shape

        # 注意力池化
        query = self.attn_query.expand(B, -1, -1)  # (B, 1, D)
        keys = self.attn_proj(x)  # (B, L, D)

        attn = F.softmax(
            (query @ keys.transpose(-1, -2)) / D**0.5, dim=-1
        )  # (B, 1, L)
        pooled = (attn @ x).squeeze(1)  # (B, D)

        return self.context_transform(pooled)


class MetaLearningAdapter(nn.Module):
    """元学习自适应层 (FiLM Conditioning)。

    根据全局上下文向量生成特征调制参数 (γ, β)，
    对层输出进行 token 级的缩放和偏移。

    数学原理：
        context = ContextEncoder(input)
        γ = W_γ · context + 1    (初始化为恒等映射)
        β = W_β · context
        output = γ ⊙ x + β

    Args:
        d_model: 特征维度
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.context_encoder = ContextEncoder(d_model)

        # FiLM 参数生成器
        self.gamma_gen = nn.Linear(d_model, d_model)
        self.beta_gen = nn.Linear(d_model, d_model)

        # 初始化为近似恒等映射
        nn.init.zeros_(self.gamma_gen.weight)
        nn.init.zeros_(self.gamma_gen.bias)
        nn.init.zeros_(self.beta_gen.weight)
        nn.init.zeros_(self.beta_gen.bias)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) 待调制的特征
            context: (batch, d_model) 可选的外部上下文
        Returns:
            modulated: (batch, seq_len, d_model) 调制后的特征
        """
        if context is None:
            context = self.context_encoder(x)  # (B, D)

        gamma = 1 + self.gamma_gen(context).unsqueeze(1)  # (B, 1, D)
        beta = self.beta_gen(context).unsqueeze(1)  # (B, 1, D)

        return x * gamma + beta


class TaskAdaptiveController(nn.Module):
    """任务自适应控制器：根据输入推断任务类型并调整策略。

    通过一个小型分类器推断当前输入属于哪种"任务模式"，
    然后为每种模式提供不同的处理策略。

    这实现了一种隐式的任务切换机制，使模型能够
    根据上下文自动在不同的处理模式间切换。

    Args:
        d_model: 特征维度
        num_modes: 任务模式数量
    """

    def __init__(self, d_model: int, num_modes: int = 4):
        super().__init__()
        self.num_modes = num_modes
        self.context_encoder = ContextEncoder(d_model)
        # 可学习的调制强度（替代硬编码 0.1）
        self.adapt_scale = nn.Parameter(torch.tensor(0.1))

        # 模式分类器
        self.mode_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_modes),
        )

        # 合并所有模式的变换到单个 Linear（消除 Python for-loop）
        self.mode_transform_merged = nn.Linear(d_model, d_model * num_modes, bias=False)
        nn.init.zeros_(self.mode_transform_merged.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            adapted: (batch, seq_len, d_model)
        """
        context = self.context_encoder(x)  # (B, D)

        # 推断任务模式
        mode_logits = self.mode_classifier(context)  # (B, num_modes)
        mode_weights = F.softmax(mode_logits, dim=-1)  # (B, num_modes)

        # 批量变换：单次 Linear + reshape + tanh（替代 num_modes 次串行 forward）
        all_transforms = self.mode_transform_merged(context)  # (B, D*num_modes)
        transforms = torch.tanh(all_transforms.view(-1, self.num_modes, context.shape[-1]))  # (B, num_modes, D)

        # 加权融合
        adaptation = (transforms * mode_weights.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # 应用调制（残差形式）
        return x + x * adaptation.unsqueeze(1) * self.adapt_scale  # 可学习的调制强度
