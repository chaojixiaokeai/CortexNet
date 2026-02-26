"""
轻量校准器 (Lightweight Calibrator)

核心功能：
  使用极少量数据（~100 样本，1 epoch）微调 CortexNet 的适配层参数，
  使模型在适配后的行为与原生模型尽可能一致。

关键设计：
  1. 仅优化 <1% 的参数（融合门控 + MoE 路由层）
  2. 冻结所有核心权重（Q/K/V/FFN 等）
  3. 仅使用自回归交叉熵损失（无辅助损失）
  4. 校准结果缓存复用
"""

from __future__ import annotations

import os
import json
import logging
import hashlib
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LightweightCalibrator:
    """轻量校准器。

    Args:
        cortex_model: CortexNet 模型
        model_type: 源模型类型
        cache_dir: 校准参数缓存目录
    """

    def __init__(
        self,
        cortex_model: nn.Module,
        model_type: str = "default",
        cache_dir: Optional[str] = None,
    ):
        self.model = cortex_model
        self.model_type = model_type
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "cortexnet", "calibration"
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def calibrate(
        self,
        calibration_data: Optional[List[Dict[str, torch.Tensor]]] = None,
        n_samples: int = 100,
        lr: float = 1e-5,
        use_cache: bool = True,
    ) -> nn.Module:
        """执行轻量校准。

        Args:
            calibration_data: 校准数据列表，每个元素包含 "input_ids"
            n_samples: 校准样本数（如未提供数据则自动生成）
            lr: 学习率
            use_cache: 是否使用/保存缓存

        Returns:
            校准后的模型
        """
        # 尝试加载缓存
        cache_key = self._get_cache_key()
        if use_cache and self._load_cached(cache_key):
            logger.info("Loaded calibration parameters from cache.")
            return self.model

        # 准备校准数据
        if calibration_data is None:
            calibration_data = self._generate_calibration_data(n_samples)

        if not calibration_data:
            logger.warning("No calibration data available. Skipping calibration.")
            return self.model

        # 冻结核心权重，仅解冻适配层
        trainable_params = self._freeze_core_weights()

        if not trainable_params:
            logger.info("No trainable adaptation parameters found. Skipping calibration.")
            return self.model

        trainable_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Calibration: optimizing {trainable_count:,} / {total_count:,} params "
            f"({100 * trainable_count / max(total_count, 1):.2f}%)"
        )

        # 优化器
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # 1 epoch 快速校准
        self.model.train()
        total_loss = 0.0
        n_steps = 0

        for batch in calibration_data[:n_samples]:
            input_ids = batch["input_ids"]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            labels = input_ids.clone()

            # 清零梯度（在 forward 前，确保无残留计算图引用）
            optimizer.zero_grad(set_to_none=True)

            try:
                # 前向（eval 模式避免 aux_loss 图残留，手动开启梯度）
                self.model.eval()
                with torch.enable_grad():
                    output = self.model(input_ids)
                    logits = output["logits"]

                    # 移位交叉熵
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )

                    # 跳过 NaN/Inf loss（保护训练稳定性）
                    if not loss.isfinite():
                        continue

                    loss.backward()

                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                total_loss += loss.detach().item()
                n_steps += 1
            except RuntimeError:
                # 自动恢复：清除可能残留的梯度
                optimizer.zero_grad(set_to_none=True)
                continue

        avg_loss = total_loss / max(n_steps, 1)
        logger.info(f"Calibration complete: {n_steps} steps, avg loss={avg_loss:.4f}")

        # 保存缓存
        if use_cache:
            self._save_cached(cache_key)

        # 恢复参数梯度状态
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.eval()
        return self.model

    def _freeze_core_weights(self) -> List[nn.Parameter]:
        """冻结核心权重，返回可训练的适配层参数。"""
        # 需要优化的参数名模式（适配层）
        trainable_patterns = [
            "fusion",          # 融合门控
            "router",          # MoE 路由器
            "gate_net",        # 动态路径控制器
            "meta_adapter",    # 元学习适配器
            "task_controller", # 任务控制器
            "path_controller", # 路径控制器
        ]

        trainable_params = []
        for name, param in self.model.named_parameters():
            is_trainable = any(pat in name for pat in trainable_patterns)
            param.requires_grad = is_trainable
            if is_trainable:
                trainable_params.append(param)

        return trainable_params

    def _generate_calibration_data(
        self,
        n_samples: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """生成合成校准数据（当无真实数据时使用）。

        使用随机 token 序列作为校准数据。
        虽然不如真实文本效果好，但足以校准门控/路由参数。
        """
        vocab_size = getattr(self.model.config, 'vocab_size', 32000)
        max_len = min(getattr(self.model.config, 'max_seq_len', 512), 256)

        data = []
        for _ in range(n_samples):
            ids = torch.randint(1, vocab_size, (max_len,), dtype=torch.long)
            data.append({"input_ids": ids})

        logger.info(f"Generated {n_samples} synthetic calibration samples")
        return data

    def _get_cache_key(self) -> str:
        """生成校准缓存的唯一键。"""
        config_str = json.dumps({
            "model_type": self.model_type,
            "hidden_size": getattr(self.model.config, 'hidden_size', 0),
            "num_layers": getattr(self.model.config, 'num_layers', 0),
            "vocab_size": getattr(self.model.config, 'vocab_size', 0),
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _save_cached(self, cache_key: str):
        """保存校准参数到缓存。"""
        cache_path = os.path.join(self.cache_dir, f"calibration_{cache_key}.pt")
        # 仅保存适配层参数
        adapt_state = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if any(pat in name for pat in [
                "fusion", "router", "gate_net",
                "meta_adapter", "task_controller", "path_controller",
            ])
        }
        torch.save(adapt_state, cache_path)
        logger.info(f"Saved calibration cache: {cache_path}")

    def _load_cached(self, cache_key: str) -> bool:
        """从缓存加载校准参数。"""
        cache_path = os.path.join(self.cache_dir, f"calibration_{cache_key}.pt")
        if not os.path.exists(cache_path):
            return False

        try:
            adapt_state = torch.load(cache_path, map_location="cpu", weights_only=True)
            model_state = self.model.state_dict()
            for name, param in adapt_state.items():
                if name in model_state and model_state[name].shape == param.shape:
                    model_state[name] = param
            self.model.load_state_dict(model_state, strict=False)
            return True
        except Exception as e:
            logger.warning(f"Failed to load calibration cache: {e}")
            return False
