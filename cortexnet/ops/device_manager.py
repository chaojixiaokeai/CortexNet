"""
硬件设备管理器 (Device Manager)

自动检测可用硬件（NVIDIA GPU / 昇腾 NPU / 寒武纪 MLU / Apple MPS / CPU），
提供最优设备选择和配置策略。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Sequence

import torch

logger = logging.getLogger(__name__)
SUPPORTED_DEVICE_TYPES = {"auto", "cpu", "cuda", "mps", "npu", "mlu"}


def _try_import_torch_npu() -> bool:
    """尝试加载 torch_npu，使 torch.npu 能被正确注册。"""
    if hasattr(torch, "npu"):
        return True
    try:
        import torch_npu  # type: ignore  # noqa: F401
        _ = torch_npu
    except Exception as exc:  # pragma: no cover - 依赖环境差异
        logger.debug("torch_npu import failed: %s", exc)
        return False
    return hasattr(torch, "npu")


def _try_import_torch_mlu() -> bool:
    """尝试加载 torch_mlu，使 torch.mlu 能被正确注册。"""
    if hasattr(torch, "mlu"):
        return True
    try:
        import torch_mlu  # type: ignore  # noqa: F401
        _ = torch_mlu
    except Exception as exc:  # pragma: no cover - 依赖环境差异
        logger.debug("torch_mlu import failed: %s", exc)
        return False
    return hasattr(torch, "mlu")


def is_npu_available() -> bool:
    """NPU 是否可用（含 torch_npu 惰性注册）。"""
    _try_import_torch_npu()
    npu = getattr(torch, "npu", None)
    if npu is None:
        return False
    try:
        return bool(npu.is_available())
    except Exception as exc:  # pragma: no cover - 依赖环境差异
        logger.debug("torch.npu.is_available() failed: %s", exc)
        return False


def is_mlu_available() -> bool:
    """MLU 是否可用（含 torch_mlu 惰性注册）。"""
    _try_import_torch_mlu()
    mlu = getattr(torch, "mlu", None)
    if mlu is None:
        return False
    try:
        return bool(mlu.is_available())
    except Exception as exc:  # pragma: no cover - 依赖环境差异
        logger.debug("torch.mlu.is_available() failed: %s", exc)
        return False


def _parse_device_request(requested: Optional[str]) -> Tuple[str, Optional[int]]:
    value = "auto" if requested is None else str(requested).strip().lower()
    if not value:
        value = "auto"
    if value == "gpu":
        value = "cuda"

    if ":" in value:
        device_type, index_raw = value.split(":", 1)
        if not index_raw:
            raise ValueError(f"Invalid device string: {requested}")
        try:
            device_index = int(index_raw)
        except ValueError as exc:
            raise ValueError(f"Invalid device index in '{requested}'") from exc
        if device_index < 0:
            raise ValueError(f"Device index must be >= 0 in '{requested}'")
    else:
        device_type, device_index = value, None

    if device_type not in SUPPORTED_DEVICE_TYPES:
        supported = ", ".join(sorted(SUPPORTED_DEVICE_TYPES))
        raise ValueError(f"Unsupported device '{requested}'. Supported: {supported}")
    return device_type, device_index


def get_device_type(device: Optional[str]) -> str:
    """返回设备类型（去掉索引，如 npu:0 -> npu）。"""
    device_type, _ = _parse_device_request(device)
    if device_type == "auto":
        resolved = resolve_device_string("auto")
        device_type, _ = _parse_device_request(resolved)
    return device_type


def _device_count(device_type: str) -> int:
    if device_type == "cuda":
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if device_type == "npu":
        if not is_npu_available():
            return 0
        npu = getattr(torch, "npu", None)
        if npu is None:
            return 0
        try:
            return int(npu.device_count())
        except Exception:  # pragma: no cover - 依赖环境差异
            return 0
    if device_type == "mlu":
        if not is_mlu_available():
            return 0
        mlu = getattr(torch, "mlu", None)
        if mlu is None:
            return 0
        try:
            return int(mlu.device_count())
        except Exception:  # pragma: no cover - 依赖环境差异
            return 0
    if device_type == "mps":
        return int(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    if device_type == "cpu":
        return 1
    return 0


def _is_device_available(device_type: str) -> bool:
    if device_type == "cuda":
        return torch.cuda.is_available()
    if device_type == "npu":
        return is_npu_available()
    if device_type == "mlu":
        return is_mlu_available()
    if device_type == "mps":
        return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if device_type == "cpu":
        return True
    return False


def _format_device(device_type: str, device_index: Optional[int]) -> str:
    if device_type in {"cpu", "mps"} or device_index is None:
        return device_type
    return f"{device_type}:{device_index}"


def resolve_device_string(
    requested: Optional[str] = "auto",
    *,
    auto_priority: Sequence[str] = ("npu", "cuda", "mlu", "mps", "cpu"),
    allow_fallback: bool = True,
) -> str:
    """统一解析设备字符串，支持 NPU 运行时惰性加载与回退。"""
    device_type, device_index = _parse_device_request(requested)

    if device_type == "auto":
        for candidate in auto_priority:
            if candidate in SUPPORTED_DEVICE_TYPES and candidate != "auto" and _is_device_available(candidate):
                return _format_device(candidate, None)
        return "cpu"

    if not _is_device_available(device_type):
        if allow_fallback:
            fallback = resolve_device_string(
                "auto",
                auto_priority=auto_priority,
                allow_fallback=False,
            )
            logger.warning(
                "Requested device '%s' is unavailable, fallback to '%s'.",
                requested,
                fallback,
            )
            return fallback

        hint = ""
        if device_type == "npu":
            hint = " (hint: install and configure torch_npu + CANN runtime)"
        if device_type == "mlu":
            hint = " (hint: install and configure torch_mlu runtime)"
        raise RuntimeError(f"Requested device '{requested}' is unavailable{hint}.")

    if device_index is not None:
        if device_type in {"cpu", "mps"}:
            logger.warning(
                "Device '%s' does not use index; ignore index %s.",
                device_type,
                device_index,
            )
            device_index = None
        else:
            count = _device_count(device_type)
            if count > 0 and device_index >= count:
                if allow_fallback:
                    logger.warning(
                        "Requested %s index=%s out of range (count=%s), fallback to index=0.",
                        device_type,
                        device_index,
                        count,
                    )
                    device_index = 0
                else:
                    raise RuntimeError(
                        f"Requested {device_type}:{device_index} out of range. "
                        f"Available count={count}."
                    )

    return _format_device(device_type, device_index)


def resolve_dtype_for_device(
    dtype: Optional[str | torch.dtype],
    device: Optional[str],
) -> torch.dtype:
    """按设备统一解析 dtype（含 mps/npu 兼容回退）。"""
    if isinstance(dtype, torch.dtype):
        resolved_dtype = dtype
    else:
        dtype_name = "auto" if dtype is None else str(dtype).lower()
        if dtype_name == "auto":
            base = get_device_type(device)
            return torch.float16 if base in {"cuda", "mps", "npu", "mlu"} else torch.float32
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        resolved_dtype = mapping[dtype_name]

    base = get_device_type(device)
    if base == "mps" and resolved_dtype == torch.bfloat16:
        return torch.float16
    if base == "npu" and resolved_dtype == torch.bfloat16:
        npu = getattr(torch, "npu", None)
        if npu is not None and hasattr(npu, "is_bf16_supported"):
            try:
                if not bool(npu.is_bf16_supported()):
                    return torch.float16
            except Exception:  # pragma: no cover - 依赖环境差异
                return torch.float16
    return resolved_dtype


@dataclass
class DeviceInfo:
    """硬件设备信息。"""
    device_type: str           # "cuda", "npu", "mlu", "mps", "cpu"
    device_name: str           # 设备名称
    device_index: int = 0      # 设备索引
    total_memory_gb: float = 0.0  # 总显存/内存 (GB)
    compute_capability: str = ""  # 计算能力版本
    supports_bf16: bool = False
    supports_fp16: bool = False
    supports_flash_attn: bool = False
    dynamic_graph_support: str = "full"  # "full", "limited", "none"

    @property
    def torch_device(self) -> torch.device:
        if self.device_type == "cpu":
            return torch.device("cpu")
        return torch.device(f"{self.device_type}:{self.device_index}")

    @property
    def optimal_dtype(self) -> torch.dtype:
        if self.supports_bf16:
            return torch.bfloat16
        if self.supports_fp16:
            return torch.float16
        return torch.float32


class DeviceManager:
    """硬件设备管理器。

    自动检测可用硬件并提供最优配置建议。
    """

    def __init__(self):
        self.devices: List[DeviceInfo] = []
        self._detect_devices()

    def _detect_devices(self):
        """检测所有可用硬件设备。"""
        # CUDA GPU
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_memory = getattr(props, "total_memory", getattr(props, "total_mem", 0))
                mem_gb = total_memory / (1024 ** 3) if total_memory else 0.0
                cc = f"{props.major}.{props.minor}"
                supports_bf16 = props.major >= 8  # Ampere+
                supports_flash = props.major >= 8

                self.devices.append(DeviceInfo(
                    device_type="cuda",
                    device_name=props.name,
                    device_index=i,
                    total_memory_gb=round(mem_gb, 1),
                    compute_capability=cc,
                    supports_bf16=supports_bf16,
                    supports_fp16=True,
                    supports_flash_attn=supports_flash,
                        dynamic_graph_support="full",
                ))

        # 昇腾 NPU (torch_npu)
        if is_npu_available():
            npu = getattr(torch, "npu", None)
            count = _device_count("npu")
            if npu is not None:
                for i in range(count):
                    name = f"Ascend NPU {i}"
                    mem_gb = 0.0
                    try:
                        queried_name = npu.get_device_name(i)
                        if queried_name:
                            name = str(queried_name)
                    except Exception:  # pragma: no cover - 依赖环境差异
                        pass
                    try:
                        props = npu.get_device_properties(i)
                        mem = getattr(props, "total_memory", 0)
                        if mem:
                            mem_gb = mem / (1024 ** 3)
                    except Exception:  # pragma: no cover - 依赖环境差异
                        pass

                    supports_bf16 = True
                    if hasattr(npu, "is_bf16_supported"):
                        try:
                            supports_bf16 = bool(npu.is_bf16_supported())
                        except Exception:  # pragma: no cover - 依赖环境差异
                            supports_bf16 = True

                    self.devices.append(DeviceInfo(
                        device_type="npu",
                        device_name=name,
                        device_index=i,
                        total_memory_gb=round(mem_gb, 1),
                        supports_bf16=supports_bf16,
                        supports_fp16=True,
                        supports_flash_attn=False,
                        dynamic_graph_support="limited",
                    ))

        # 寒武纪 MLU (torch_mlu)
        if is_mlu_available():
            count = _device_count("mlu")
            for i in range(count):
                self.devices.append(DeviceInfo(
                    device_type="mlu",
                    device_name=f"Cambricon MLU {i}",
                    device_index=i,
                    supports_bf16=False,
                    supports_fp16=True,
                    supports_flash_attn=False,
                    dynamic_graph_support="limited",
                ))

        # Apple MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.devices.append(DeviceInfo(
                device_type="mps",
                device_name="Apple Silicon GPU",
                device_index=0,
                supports_bf16=False,
                supports_fp16=True,
                supports_flash_attn=False,
                dynamic_graph_support="full",
            ))

        # CPU (always available)
        self.devices.append(DeviceInfo(
            device_type="cpu",
            device_name="CPU",
            device_index=0,
            supports_bf16=True,
            supports_fp16=True,
            supports_flash_attn=False,
            dynamic_graph_support="full",
        ))

    def get_best_device(self) -> DeviceInfo:
        """返回最佳可用设备（按优先级：CUDA > NPU > MLU > MPS > CPU）。"""
        priority = {"cuda": 0, "npu": 1, "mlu": 2, "mps": 3, "cpu": 4}
        sorted_devices = sorted(
            self.devices,
            key=lambda d: (priority.get(d.device_type, 99), -d.total_memory_gb),
        )
        best = sorted_devices[0]
        logger.info(
            f"Best device: {best.device_name} ({best.device_type}:{best.device_index}), "
            f"memory={best.total_memory_gb}GB, dtype={best.optimal_dtype}"
        )
        return best

    def get_optimization_config(self, device: Optional[DeviceInfo] = None) -> Dict[str, Any]:
        """根据设备生成优化配置。"""
        if device is None:
            device = self.get_best_device()

        config = {
            "device": str(device.torch_device),
            "dtype": str(device.optimal_dtype),
            "use_flash_attention": device.supports_flash_attn,
            "use_gradient_checkpointing": device.total_memory_gb < 24,
            "dynamic_path_mode": "batch_static" if device.dynamic_graph_support == "limited" else "dynamic",
        }

        # NPU 特殊配置
        if device.device_type in ("npu", "mlu"):
            config.update({
                "use_static_graph": True,
                "max_batch_size": max(1, int(device.total_memory_gb // 8)),
            })

        return config

    def list_devices(self) -> List[Dict[str, Any]]:
        """列出所有检测到的设备。"""
        return [
            {
                "type": d.device_type,
                "name": d.device_name,
                "index": d.device_index,
                "memory_gb": d.total_memory_gb,
                "dtype": str(d.optimal_dtype),
            }
            for d in self.devices
        ]


def get_best_device_info() -> DeviceInfo:
    """便捷函数：获取最佳设备信息。"""
    return DeviceManager().get_best_device()
