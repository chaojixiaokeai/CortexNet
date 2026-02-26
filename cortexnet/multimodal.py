"""
多模态编码器框架 (Multi-Modal Encoder Framework)

核心创新：
  统一的多模态输入处理框架，支持文本、图像、音频等多种模态。
  通过模态特定的编码器将不同类型的输入映射到统一的表示空间，
  然后通过跨模态融合机制建立模态间的语义关联。

  ┌────────────────────────────────────────────────────────┐
  │                多模态处理流水线                         │
  ├────────────────────────────────────────────────────────┤
  │                                                        │
  │  文本 ──► TokenEmbed ──┐                               │
  │                        │                               │
  │  图像 ──► PatchEmbed ──┼──► 模态标记 ──► 拼接 ──► 输出│
  │                        │      │                        │
  │  音频 ──► FrameEmbed ──┘      │                        │
  │                               │                        │
  │                     跨模态融合层 (可选)                 │
  │                                                        │
  └────────────────────────────────────────────────────────┘

  每种模态的编码器将输入转换为 (batch, num_tokens, d_model) 的格式，
  然后通过模态类型嵌入标记来源，最后拼接成统一序列。
"""

import torch
import torch.nn as nn
from typing import Optional


class PatchEmbedding(nn.Module):
    """ViT 风格的图像块嵌入。

    将图像分割为不重叠的块，每个块通过线性投影映射到 d_model 维。
    加入可学习的位置嵌入。

    Args:
        d_model: 输出维度
        image_size: 输入图像尺寸（正方形）
        patch_size: 块大小
        in_channels: 输入通道数
    """

    def __init__(
        self,
        d_model: int,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 块投影（卷积实现）
        self.proj = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

        # 可学习的位置嵌入
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            patches: (batch, num_patches, d_model)
        """
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, D)
        x = x + self.pos_embed[:, : x.shape[1]]
        return self.norm(x)


class AudioEmbedding(nn.Module):
    """音频帧嵌入。

    将 Mel 频谱图通过 1D 卷积编码为帧级表示。
    使用步进卷积进行时间降采样。

    Args:
        d_model: 输出维度
        n_mels: Mel 频率通道数
        frame_stride: 时间降采样步长
    """

    def __init__(
        self, d_model: int, n_mels: int = 80, frame_stride: int = 4
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, d_model, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(
                d_model,
                d_model,
                kernel_size=frame_stride,
                stride=frame_stride,
            ),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_mels, time_frames) Mel 频谱图
        Returns:
            frames: (batch, time_frames/stride, d_model)
        """
        x = self.encoder(x).transpose(1, 2)  # (B, T/stride, D)
        return self.norm(x)


class CrossModalFusion(nn.Module):
    """跨模态融合层。

    通过交叉注意力让不同模态的 token 相互交流，
    建立跨模态的语义关联。

    Args:
        d_model: 特征维度
        num_heads: 注意力头数
    """

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, L1, d_model) 查询模态
            context: (batch, L2, d_model) 上下文模态
        Returns:
            output: (batch, L1, d_model)
        """
        # 交叉注意力
        residual = x
        x = self.norm1(x)
        x, _ = self.cross_attn(x, context, context)
        x = residual + x

        # FFN
        residual = x
        x = residual + self.ffn(self.norm2(x))

        return x


class MultiModalEncoder(nn.Module):
    """统一多模态编码器。

    支持三种模态的输入编码，并提供跨模态融合能力。
    所有模态的输出都映射到相同的 d_model 维空间。

    Args:
        d_model: 统一的表示维度
        vocab_size: 文本词表大小
        image_size: 图像尺寸
        patch_size: 图像块大小
        n_mels: 音频 Mel 通道数
        num_fusion_heads: 跨模态融合注意力头数
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int = 32000,
        image_size: int = 224,
        patch_size: int = 16,
        n_mels: int = 80,
        num_fusion_heads: int = 8,
    ):
        super().__init__()
        self.d_model = d_model

        # 模态特定编码器
        self.text_embed = nn.Embedding(vocab_size, d_model)
        self.image_embed = PatchEmbedding(d_model, image_size, patch_size)
        self.audio_embed = AudioEmbedding(d_model, n_mels)

        # 模态类型嵌入 (text=0, image=1, audio=2)
        self.modality_embed = nn.Embedding(3, d_model)

        # 跨模态融合（可选）
        self.cross_modal_fusion = CrossModalFusion(d_model, num_fusion_heads)

        # 模态投影归一化
        self.text_norm = nn.LayerNorm(d_model)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """编码文本 token。"""
        x = self.text_norm(self.text_embed(tokens))
        mod = self.modality_embed(
            torch.zeros(
                tokens.shape[0],
                tokens.shape[1],
                dtype=torch.long,
                device=tokens.device,
            )
        )
        return x + mod

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像。"""
        patches = self.image_embed(images)
        B, L = patches.shape[:2]
        mod = self.modality_embed(
            torch.ones(B, L, dtype=torch.long, device=images.device)
        )
        return patches + mod

    def encode_audio(self, mel_specs: torch.Tensor) -> torch.Tensor:
        """编码音频 Mel 频谱图。"""
        frames = self.audio_embed(mel_specs)
        B, L = frames.shape[:2]
        mod = self.modality_embed(
            2 * torch.ones(B, L, dtype=torch.long, device=mel_specs.device)
        )
        return frames + mod

    def forward(
        self,
        text: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        fuse_modalities: bool = False,
    ) -> torch.Tensor:
        """统一编码多模态输入。

        将所有模态的编码结果拼接为一个序列。

        Args:
            text: (B, L_text) 文本 token
            images: (B, C, H, W) 图像
            audio: (B, n_mels, T) 音频 Mel 频谱
            fuse_modalities: 是否进行跨模态融合
        Returns:
            embeddings: (B, L_total, d_model)
        """
        parts = []

        if text is not None:
            parts.append(self.encode_text(text))
        if images is not None:
            parts.append(self.encode_image(images))
        if audio is not None:
            parts.append(self.encode_audio(audio))

        if not parts:
            raise ValueError("至少需要提供一种模态的输入")

        # 拼接所有模态
        combined = torch.cat(parts, dim=1)  # (B, L_total, D)

        # 可选的跨模态融合
        if fuse_modalities and len(parts) > 1:
            combined = self.cross_modal_fusion(combined, combined)

        return combined
