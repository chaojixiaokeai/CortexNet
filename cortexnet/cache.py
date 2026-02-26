"""
CortexNet 推理缓存 (CortexNet Inference Cache)

用于自回归生成时的增量解码，避免每步重算完整序列。

缓存层级：
  CortexNetCache
    └── LayerCache (per block)
          ├── ssm_state: Tensor          — SSM hidden state
          ├── memory_state: (Tensor, Tensor) — WorkingMemory (memory, z)
          └── kv_cache: (Tensor, Tensor, Tensor) — Attention (K, V, indices)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class LayerCache:
    """单层 CortexBlock 的缓存。

    Attributes:
        ssm_state: SSM 隐状态 (B, d_inner, N)，可为 None（首次前向）
        memory_state: WorkingMemory 的 (memory, z)，可为 None
        kv_cache: Attention 的 (K, V, top_k_indices)，可为 None
    """

    ssm_state: Optional[torch.Tensor] = None
    memory_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    def as_tuple(self) -> Tuple:
        """转换为 (ssm_state, memory_state, kv_cache) 元组，兼容旧 API。"""
        return (self.ssm_state, self.memory_state, self.kv_cache)

    @staticmethod
    def from_tuple(t: Tuple) -> LayerCache:
        """从旧式元组创建。"""
        if t is None:
            return LayerCache()
        return LayerCache(
            ssm_state=t[0] if len(t) > 0 else None,
            memory_state=t[1] if len(t) > 1 else None,
            kv_cache=t[2] if len(t) > 2 else None,
        )


@dataclass
class CortexNetCache:
    """CortexNet 增量解码缓存（全模型）。

    Attributes:
        layers: 每层的 LayerCache
        position_offset: 已处理的 token 数量
    """

    layers: List[LayerCache] = field(default_factory=list)
    position_offset: int = 0

    @staticmethod
    def init_empty(num_layers: int) -> CortexNetCache:
        """创建空缓存。"""
        return CortexNetCache(
            layers=[LayerCache() for _ in range(num_layers)],
            position_offset=0,
        )

    def as_list(self) -> List[Tuple]:
        """转换为旧式 List[Tuple] 格式，兼容旧 API。"""
        return [lc.as_tuple() for lc in self.layers]

    @staticmethod
    def from_list(lst: List[Tuple]) -> CortexNetCache:
        """从旧式 List[Tuple] 创建。"""
        if lst is None:
            return CortexNetCache()
        return CortexNetCache(
            layers=[LayerCache.from_tuple(t) for t in lst],
        )
