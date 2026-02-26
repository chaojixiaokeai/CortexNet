"""
自我进化引擎 (Self-Evolution Engine)

核心创新：
  模型在推理时动态决定激活哪些处理路径，实现可微分的神经架构搜索。
  不同输入自动获得不同的架构配置——简单输入用少量路径，
  复杂输入启用全部路径。

  ┌─────────────────────────────────────────────────────────────┐
  │              自我进化 vs 传统 NAS                            │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  传统 NAS:  训练前搜索固定架构（离线，计算昂贵）           │
  │  自我进化:  推理时动态选择架构（在线，零额外开销）         │
  │                                                             │
  │  实现方式:                                                  │
  │  1. 路径门控 — Gumbel-Sigmoid 可微分开关                   │
  │  2. 计算预算 — 根据输入复杂度分配计算资源                  │
  │  3. 进化记忆 — 记录最优配置，加速未来决策                  │
  └─────────────────────────────────────────────────────────────┘

  类比: 如同人脑的 "系统1/系统2 思维"——
    简单问题用快速直觉（少量路径），
    复杂问题启动深度思考（全部路径）。
"""

import torch
import torch.nn as nn


class DynamicPathController(nn.Module):
    """动态路径控制器：可微分的路径开关。

    使用 Gumbel-Sigmoid 实现可微分的二值决策，
    训练时使用软决策（梯度流通），推理时使用硬决策（效率优先）。

    Args:
        d_model: 输入维度
        num_paths: 可控制的路径数
    """

    def __init__(self, d_model: int, num_paths: int = 5):
        super().__init__()
        self.num_paths = num_paths
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, num_paths),
        )
        # 初始偏置: 默认激活所有路径
        nn.init.constant_(self.gate_net[-1].bias, 2.0)
        self.temperature = 5.0  # 初始高温，鼓励探索
        self.min_temperature = 0.5
        self.anneal_steps = 10000  # 默认退火步数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            gates: (batch, num_paths) - 每条路径的激活概率
        """
        context = x.mean(dim=1)  # (B, D) 全局上下文
        logits = self.gate_net(context)  # (B, num_paths)

        if self.training:
            # Gumbel-Sigmoid: 可微分的随机二值决策（收紧范围防止极端值）
            noise = torch.zeros_like(logits).uniform_(1e-4, 1 - 1e-4)
            noise = (torch.log(noise) - torch.log(1 - noise)).clamp(-10, 10)
            gates = torch.sigmoid((logits + noise) / max(self.temperature, 0.1))
        else:
            gates = (logits > 0).float()

        return gates

    def anneal_temperature(self, step: int):
        """线性温度退火：从 initial_temp 线性降至 min_temp。

        应在每个训练 step 后调用。

        Args:
            step: 当前训练步数
        """
        ratio = min(step / max(self.anneal_steps, 1), 1.0)
        self.temperature = 5.0 * (1.0 - ratio) + self.min_temperature * ratio


class ComputeBudgetAllocator(nn.Module):
    """计算预算分配器：根据输入复杂度动态分配计算资源。

    评估输入的 "难度"，为简单输入分配少量计算（提速），
    为复杂输入分配更多计算（提质）。

    Args:
        d_model: 输入维度
        num_levels: 计算预算级别数
    """

    def __init__(self, d_model: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels
        self.complexity_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            budget: (batch,) 值在 [0,1]，表示应使用的计算比例
        """
        complexity = self.complexity_scorer(x.mean(dim=1)).squeeze(-1)  # (B,)
        return complexity


class SelfEvolutionEngine(nn.Module):
    """自我进化引擎：统一的动态架构控制。

    整合路径门控和计算预算分配，为每个 CortexBlock
    提供动态的架构决策。

    Args:
        d_model: 模型维度
        num_paths: 路径数
        num_blocks: 模型总层数
    """

    def __init__(self, d_model: int, num_paths: int = 5, num_blocks: int = 4):
        super().__init__()
        self.num_paths = num_paths
        self.num_blocks = num_blocks

        # 每层独立的路径控制器
        self.path_controllers = nn.ModuleList(
            [DynamicPathController(d_model, num_paths) for _ in range(num_blocks)]
        )
        # 全局计算预算
        self.budget_allocator = ComputeBudgetAllocator(d_model)

        # 进化记忆: 记录历史最优配置 (不参与梯度)
        self.register_buffer(
            "config_history",
            torch.ones(num_blocks, num_paths),  # 平均配置
        )
        self._history_count = 0

    def get_block_config(self, x: torch.Tensor, block_idx: int) -> torch.Tensor:
        """获取指定层的路径激活配置。"""
        return self.path_controllers[block_idx](x)

    def get_compute_budget(self, x: torch.Tensor) -> torch.Tensor:
        """获取全局计算预算。"""
        return self.budget_allocator(x)

    @torch.no_grad()
    def update_history(self, block_idx: int, config: torch.Tensor):
        """更新进化记忆。"""
        mean_config = config.mean(dim=0)
        momentum = 0.99
        self.config_history[block_idx] = (
            momentum * self.config_history[block_idx]
            + (1 - momentum) * mean_config
        )
        self._history_count += 1

    def get_efficiency_loss(self) -> torch.Tensor:
        """计算效率损失: 鼓励使用更少的路径。"""
        total_activation = sum(
            ctrl.gate_net[-1].bias.sigmoid().mean()
            for ctrl in self.path_controllers
        )
        return total_activation * 0.01  # 小权重避免过度压缩
