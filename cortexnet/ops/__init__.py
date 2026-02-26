"""
CortexNet 算子子包 (Ops Sub-package)

提供硬件抽象层，支持 NVIDIA GPU、国产 NPU 和 CPU 后端。
"""

from .device_manager import (
    DeviceManager,
    get_best_device_info,
    is_npu_available,
    is_mlu_available,
    get_device_type,
    resolve_device_string,
    resolve_dtype_for_device,
)
from .npu_ops import NPUOperators, get_operators

__all__ = [
    "DeviceManager",
    "get_best_device_info",
    "is_npu_available",
    "is_mlu_available",
    "get_device_type",
    "resolve_device_string",
    "resolve_dtype_for_device",
    "NPUOperators",
    "get_operators",
]
