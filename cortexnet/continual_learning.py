"""
连续学习模块 (Continual Learning Module)

核心创新：
  解决神经网络在学习新任务时"灾难性遗忘"旧知识的问题。
  实现两种互补的反遗忘策略：

  1. 弹性权重巩固 (EWC, Elastic Weight Consolidation)
     - 计算 Fisher 信息矩阵，识别对旧任务重要的参数
     - 在训练新任务时，惩罚对这些重要参数的大幅修改
     - 效果：旧知识被"保护"，新知识在不重要的参数空间中学习

  2. 渐进式记忆回放 (Progressive Memory Replay)
     - 维护一个经验缓冲区，存储旧任务的关键样本
     - 在训练新任务时，混合旧样本进行回放
     - 效果：周期性"复习"旧知识，防止遗忘

  类比人类学习：
    - EWC ≈ 大脑对重要神经连接的保护机制
    - 回放 ≈ 人类的睡眠巩固和复习
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class ElasticWeightConsolidation:
    """弹性权重巩固 (EWC)。

    通过 Fisher 信息矩阵估计每个参数对已学任务的重要性，
    在学习新任务时添加正则化项以保护重要参数。

    数学原理：
        L_total = L_new_task + λ · Σ_i F_i · (θ_i - θ*_i)²

    其中 F_i 是第 i 个参数的 Fisher 信息（重要性）,
    θ*_i 是在旧任务上的最优参数值。

    Args:
        model: CortexNet 模型
        lambda_ewc: EWC 正则化强度
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self._consolidated = False
        self._num_consolidations = 0

    def consolidate(self, dataloader, num_samples: int = 200):
        """计算 Fisher 信息并保存当前最优参数。

        在完成一个任务的训练后调用，将当前知识"巩固"。

        Args:
            dataloader: 当前任务的数据加载器
            num_samples: 用于估计 Fisher 的样本数
        """
        self.model.train()  # 需要梯度计算
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break

            input_ids, labels = batch
            input_ids = input_ids.to(
                next(self.model.parameters()).device
            )
            labels = labels.to(next(self.model.parameters()).device)

            self.model.zero_grad()
            output = self.model(input_ids, labels=labels)

            # 使用对数似然的梯度平方作为 Fisher 近似
            output["loss"].backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2)

            count += 1

        # 平均 Fisher 信息
        for n in fisher:
            fisher[n] /= max(count, 1)

        # 如果之前已巩固，使用在线更新（累积 Fisher）
        if self._consolidated:
            for n in fisher:
                if n in self.fisher:
                    fisher[n] = (
                        self.fisher[n] * self._num_consolidations
                        + fisher[n]
                    ) / (self._num_consolidations + 1)

        self.fisher = fisher
        self.optimal_params = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        self._consolidated = True
        self._num_consolidations += 1

    def penalty(self) -> torch.Tensor:
        """计算 EWC 正则化惩罚。

        在训练新任务时添加到损失函数中。

        Returns:
            penalty: 标量张量
        """
        if not self._consolidated:
            device = next(self.model.parameters()).device
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(
            0.0, device=next(self.model.parameters()).device
        )
        for n, p in self.model.named_parameters():
            if n in self.fisher and p.requires_grad:
                loss = loss + (
                    self.fisher[n] * (p - self.optimal_params[n]).pow(2)
                ).sum()

        return self.lambda_ewc * loss


class ProgressiveMemoryReplay:
    """渐进式记忆回放。

    维护一个经验缓冲区，在训练新任务时混合旧样本。
    使用 reservoir sampling 确保缓冲区均匀覆盖所有旧任务。

    Args:
        buffer_size: 缓冲区最大样本数
        replay_ratio: 每批中旧样本的比例
    """

    def __init__(self, buffer_size: int = 5000, replay_ratio: float = 0.3):
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.buffer_inputs = []
        self.buffer_labels = []
        self._count = 0

    def add_samples(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ):
        """向缓冲区添加样本（使用 reservoir sampling）。"""
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            if len(self.buffer_inputs) < self.buffer_size:
                self.buffer_inputs.append(input_ids[i].cpu())
                self.buffer_labels.append(labels[i].cpu())
            else:
                # Reservoir sampling
                idx = torch.randint(0, self._count + 1, (1,)).item()
                if idx < self.buffer_size:
                    self.buffer_inputs[idx] = input_ids[i].cpu()
                    self.buffer_labels[idx] = labels[i].cpu()
            self._count += 1

    def get_replay_batch(
        self, batch_size: int, device: torch.device
    ) -> Optional[tuple]:
        """获取回放批次。"""
        if len(self.buffer_inputs) == 0:
            return None

        num_replay = max(1, int(batch_size * self.replay_ratio))
        num_replay = min(num_replay, len(self.buffer_inputs))

        indices = torch.randperm(len(self.buffer_inputs))[:num_replay]

        replay_inputs = torch.stack(
            [self.buffer_inputs[i] for i in indices]
        ).to(device)
        replay_labels = torch.stack(
            [self.buffer_labels[i] for i in indices]
        ).to(device)

        return replay_inputs, replay_labels

    @property
    def size(self) -> int:
        return len(self.buffer_inputs)


class ContinualLearningManager:
    """连续学习管理器：整合 EWC + 记忆回放。

    提供统一接口管理连续学习的各个组件。

    使用方法：
        manager = ContinualLearningManager(model)

        # 任务 1 训练
        for batch in task1_loader:
            loss = model(batch)
            loss = loss + manager.get_regularization_loss()
            loss.backward()

        # 巩固任务 1 的知识
        manager.consolidate_task(task1_loader)

        # 任务 2 训练（自动保护任务 1 的知识）
        for batch in task2_loader:
            loss = model(batch)
            loss = loss + manager.get_regularization_loss()
            replay = manager.get_replay_batch(batch_size)
            if replay:
                replay_loss = model(replay)
                loss = loss + replay_loss * 0.5
            loss.backward()
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 1000.0,
        buffer_size: int = 5000,
        replay_ratio: float = 0.3,
    ):
        self.ewc = ElasticWeightConsolidation(model, lambda_ewc)
        self.replay = ProgressiveMemoryReplay(buffer_size, replay_ratio)
        self.task_count = 0

    def consolidate_task(self, dataloader, num_samples: int = 200):
        """巩固当前任务的知识。"""
        self.ewc.consolidate(dataloader, num_samples)
        self.task_count += 1

    def get_regularization_loss(self) -> torch.Tensor:
        """获取防遗忘正则化损失。"""
        return self.ewc.penalty()

    def add_experience(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ):
        """记录训练样本到经验缓冲区。"""
        self.replay.add_samples(input_ids, labels)

    def get_replay_batch(self, batch_size: int, device: torch.device):
        """获取回放样本。"""
        return self.replay.get_replay_batch(batch_size, device)
