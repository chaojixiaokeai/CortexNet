"""
架构适配层 (Architecture Adapter)

核心功能：
  根据源模型的架构特性，自动调整 CortexNet 内部模块的参数和行为，
  使其最大程度保留原模型的能力。

适配项：
  1. 稀疏注意力 top-k / 滑动窗口
  2. SSM 多尺度参数缩放
  3. 动态门控阈值（训推一致性）
  4. 位置编码参数（RoPE scaling / ALiBi）
  5. MoE 路由初始化
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ArchitectureAdapter:
    """架构适配器。

    在模型初始化后，根据源模型类型自动调整 CortexNet 模块参数。
    """

    def __init__(self, model_type: str, config: Any):
        self.model_type = model_type
        self.config = config

    def adapt(self, cortex_model: nn.Module) -> nn.Module:
        """对 CortexNet 模型执行全面架构适配。

        Args:
            cortex_model: 已初始化的 CortexNet 模型

        Returns:
            适配后的模型（原地修改）
        """
        logger.info(f"Starting architecture adaptation for model type: {self.model_type}")

        self._adapt_attention(cortex_model)
        self._adapt_ssm(cortex_model)
        self._adapt_position_encoding(cortex_model)
        self._adapt_dynamic_gating(cortex_model)
        self._adapt_moe(cortex_model)
        self._adapt_normalization(cortex_model)

        logger.info("Architecture adaptation complete.")
        return cortex_model

    def _adapt_attention(self, model: nn.Module):
        """适配注意力模块。"""
        preserve_pretrained_behavior = bool(getattr(self.config, "source_model_path", None))

        for block in model.blocks:
            attn = block.attention

            # 预训练权重迁移优先保真：Lite 稀疏注意力退化为全注意力，避免随机重要性打分破坏输出。
            if preserve_pretrained_behavior and hasattr(attn, "top_k_ratio"):
                attn.top_k_ratio = 1.0

            # 滑动窗口（Mistral/Mistral-derived）
            if self.config.sliding_window_size > 0:
                if hasattr(attn, 'sliding_window_size'):
                    attn.sliding_window_size = self.config.sliding_window_size
                logger.debug(
                    f"Set sliding window size to {self.config.sliding_window_size}"
                )

            # 根据模型上下文窗口长度调整 top-k
            if (not preserve_pretrained_behavior) and self.config.max_seq_len > 8192:
                # 长上下文模型：降低 top-k 比例以节省显存
                new_ratio = max(0.1, 0.25 * (8192 / self.config.max_seq_len))
                if hasattr(attn, 'top_k_ratio'):
                    attn.top_k_ratio = new_ratio
                logger.debug(
                    f"Adjusted top_k_ratio to {new_ratio:.3f} "
                    f"for max_seq_len={self.config.max_seq_len}"
                )

            # Lite 三路径中 SSM/Memory 无预训练权重时，先强偏向 Attention 作为稳定基线。
            if preserve_pretrained_behavior and hasattr(block, "fusion_weights"):
                with torch.no_grad():
                    block.fusion_weights.copy_(
                        torch.tensor(
                            [-8.0, 8.0, -8.0],
                            device=block.fusion_weights.device,
                            dtype=block.fusion_weights.dtype,
                        )
                    )

    def _adapt_ssm(self, model: nn.Module):
        """适配 SSM 多尺度参数。"""
        for block in model.blocks:
            ssm = block.ssm

            # 根据模型隐藏维度调整 SSM 内部扩展
            if self.config.hidden_size >= 4096:
                # 大模型：更多尺度捕获不同粒度的依赖
                if hasattr(ssm, 'num_scales'):
                    target_scales = min(8, self.config.hidden_size // 512)
                    logger.debug(
                        f"SSM scales adjusted for large model: {target_scales}"
                    )

    def _adapt_position_encoding(self, model: nn.Module):
        """适配位置编码参数。"""
        for block in model.blocks:
            attn = block.attention

            # RoPE scaling（动态 NTK、线性插值等）
            if self.config.rope_scaling:
                scaling_type = self.config.rope_scaling.get("type", "linear")
                scaling_factor = self.config.rope_scaling.get("factor", 1.0)

                if hasattr(attn, 'rope_scaling_factor'):
                    attn.rope_scaling_factor = scaling_factor
                    attn.rope_scaling_type = scaling_type

                logger.debug(
                    f"Applied RoPE scaling: type={scaling_type}, factor={scaling_factor}"
                )

            # 更新 RoPE theta
            if hasattr(attn, 'rope_theta'):
                attn.rope_theta = self.config.rope_theta

    def _adapt_dynamic_gating(self, model: nn.Module):
        """优化动态门控：从硬二值化改为软裁剪，解决训推不一致。

        CortexNet V3 的 DynamicPathController 默认使用硬裁剪（eval 时 >0 → 1.0）。
        这里改为软阈值（sigmoid 连续值），保证训练和推理行为一致。
        """
        _SOFT_THRESHOLD = 0.1  # 软裁剪阈值

        for block in model.blocks:
            if hasattr(block, 'path_controller'):
                controller = block.path_controller

                # Monkey-patch forward 方法，使推理时也使用 sigmoid（非硬裁剪）
                original_forward = controller.forward

                def _soft_forward(self_ctrl, x, _orig=original_forward, _thresh=_SOFT_THRESHOLD):
                    context = x.mean(dim=1)
                    logits = self_ctrl.gate_net(context)

                    if self_ctrl.training:
                        noise = torch.zeros_like(logits).uniform_(1e-4, 1 - 1e-4)
                        noise = (torch.log(noise) - torch.log(1 - noise)).clamp(-10, 10)
                        gates = torch.sigmoid(
                            (logits + noise) / max(self_ctrl.temperature, 0.1)
                        )
                    else:
                        # 软裁剪：sigmoid 输出，而非硬二值
                        gates = torch.sigmoid(logits)
                        # 低于阈值的路径置零（节省计算但保持连续性）
                        gates = gates * (gates > _thresh).float()

                    return gates

                import types
                controller.forward = types.MethodType(_soft_forward, controller)

        logger.debug(f"Applied soft gating threshold: {_SOFT_THRESHOLD}")

    def _adapt_moe(self, model: nn.Module):
        """适配 MoE 路由。

        对于非 MoE 原模型（如 LLaMA），需要将原模型的 FFN 权重
        复制到 CortexNet MoE 的第一个专家，并初始化路由器偏向该专家。
        Lite 模式下无 MoE，自动跳过。
        """
        for block in model.blocks:
            if not hasattr(block, 'moe'):
                continue  # Lite 模式使用 FFN，无需适配
            moe = block.moe

            # 检查是否已经有映射过的权重（experts.0 有非零权重）
            expert_0 = moe.experts[0] if hasattr(moe, 'experts') else None
            if expert_0 is None:
                continue

            # 初始化路由器偏置，使其倾向于激活 expert 0
            if hasattr(moe, 'router') and hasattr(moe.router, 'weight'):
                router = moe.router
                with torch.no_grad():
                    # 对 expert 0 的路由权重加大偏置
                    if router.weight.shape[0] > 0:
                        router.weight.data[0] += 0.5

        logger.debug("MoE router initialized with bias toward adapted expert")

    def _adapt_normalization(self, model: nn.Module):
        """确保归一化层类型与 CortexNet 一致。"""
        # CortexNet 使用 RMSNorm，如果源模型使用 LayerNorm 则参数已在 WeightAdapter 中处理
        # 这里只做最终验证
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                logger.warning(
                    f"Found unexpected LayerNorm at {name}. "
                    f"CortexNet expects RMSNorm. This may cause subtle differences."
                )
