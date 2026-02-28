"""
CortexNet 训练工具 (Training Utilities)

提供梯度监控、种子设置、设备选择等训练辅助功能。
"""

from __future__ import annotations

import random
from typing import Dict

import torch
import torch.nn as nn
import numpy as np


class GradientMonitor:
    """梯度监控器：记录训练过程中的梯度统计信息。

    用于诊断梯度消失/爆炸、优化学习率等。

    Usage:
        monitor = GradientMonitor(model)
        loss.backward()
        stats = monitor.get_stats()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks = []
        self._grad_stats: Dict[str, Dict[str, float]] = {}
        self._register_hooks()

    def _register_hooks(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, n=name: self._record_grad(n, grad)
                )
                self._hooks.append(hook)

    def _record_grad(self, name: str, grad: torch.Tensor):
        self._grad_stats[name] = {
            "mean": grad.mean().item(),
            "std": grad.std().item(),
            "max": grad.max().item(),
            "min": grad.min().item(),
            "norm": grad.norm().item(),
        }

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """获取最近一次 backward 的梯度统计。"""
        return dict(self._grad_stats)

    def get_summary(self) -> Dict[str, float]:
        """获取汇总统计。"""
        if not self._grad_stats:
            return {}
        norms = [s["norm"] for s in self._grad_stats.values()]
        return {
            "grad_norm_mean": sum(norms) / len(norms),
            "grad_norm_max": max(norms),
            "grad_norm_min": min(norms),
            "num_params_tracked": len(norms),
        }

    def remove_hooks(self):
        """移除所有已注册的钩子。"""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()


def check_gradients_finite(model: nn.Module) -> bool:
    """检查模型所有参数的梯度是否都是有限值（无 NaN/Inf）。

    Args:
        model: 要检查的模型

    Returns:
        True 如果所有梯度都有限（或无梯度），False 如果存在 NaN/Inf。
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                return False
    return True


def set_seed(seed: int = 42):
    """设置全局随机种子，确保可复现性。

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_best_device() -> torch.device:
    """自动选择最佳可用计算设备。

    优先级: CUDA GPU > Apple MPS > CPU

    Returns:
        最佳可用设备的 torch.device 对象
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    total_steps: int = 10000,
    min_lr_ratio: float = 0.1,
    betas: tuple = (0.9, 0.95),
):
    """创建 AdamW 优化器 + 余弦退火调度器。

    典型的 LLM 训练配置：
      - AdamW (β₁=0.9, β₂=0.95)
      - 线性 warmup → 余弦退火
      - 最终学习率 = min_lr_ratio × 初始学习率

    Args:
        model: 目标模型
        lr: 初始学习率
        weight_decay: L2 正则权重（不应用于 bias/norm）
        warmup_steps: warmup 步数
        total_steps: 总训练步数
        min_lr_ratio: 最终学习率与初始学习率的比值
        betas: Adam 的 β 参数

    Returns:
        (optimizer, scheduler) 元组
    """
    import math

    # 分组参数：bias 和 LayerNorm/RMSNorm 不做 weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or "bias" in name or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas)

    # 余弦退火 + 线性 warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def safe_clip_grad_norm_(
    model: torch.nn.Module,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """安全的梯度裁剪：先检查 NaN，再执行 clip。

    Args:
        model: 目标模型
        max_norm: 梯度范数上限
        norm_type: 范数类型（默认 L2）

    Returns:
        裁剪前的梯度总范数
    """
    # 检查 NaN/Inf 梯度
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            param.grad.zero_()  # 用零替代 NaN 梯度

    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm, norm_type=norm_type,
    )
    return float(total_norm)


class PerformanceTimer:
    """高性能计时器，支持 CUDA 同步。

    用于准确测量代码块的执行时间。

    Usage:
        with PerformanceTimer("Forward Pass") as timer:
            output = model(input)
        print(f"Time: {timer.elapsed:.4f}s")
    """

    def __init__(self, name: str = "Task", sync_cuda: bool = True):
        self.name = name
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        if self.sync_cuda:
            torch.cuda.synchronize()
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        if self.sync_cuda:
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start_time
