"""
CortexNet 适配器子包 (Adapter Sub-package)

提供开源大模型到 CortexNet 的自动适配能力：
- 模型识别与注册
- 权重映射
- 架构适配
- 推理接口统一
- 轻量校准
"""

from .model_registry import ModelRegistry, detect_model_type, get_cortexnet_config
from .weight_adapter import WeightAdapter
from .arch_adapter import ArchitectureAdapter
from .inference_adapter import InferenceAdapter
from .calibrator import LightweightCalibrator

__all__ = [
    "ModelRegistry",
    "detect_model_type",
    "get_cortexnet_config",
    "WeightAdapter",
    "ArchitectureAdapter",
    "InferenceAdapter",
    "LightweightCalibrator",
]
