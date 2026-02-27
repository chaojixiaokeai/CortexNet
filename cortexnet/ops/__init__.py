"""
CortexNet 核心算子包 (Core Operators)
"""

from .device_manager import (
    DeviceManager,
    DeviceInfo,
    get_best_device_info,
    is_npu_available,
    is_mlu_available,
    get_device_type,
    resolve_device_string,
    resolve_dtype_for_device,
)
from .npu_ops import NPUOperators, get_operators
from .ssm_ops import fused_chunk_scan

__all__ = [
    # device_manager
    "DeviceManager",
    "DeviceInfo",
    "get_best_device_info",
    "is_npu_available",
    "is_mlu_available",
    "get_device_type",
    "resolve_device_string",
    "resolve_dtype_for_device",

    # npu_ops
    "NPUOperators",
    "get_operators",

    # ssm_ops
    "fused_chunk_scan",
]
