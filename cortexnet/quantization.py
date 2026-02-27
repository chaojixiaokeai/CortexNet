"""
CortexNet 量化封装 (Quantization Wrapper)

支持多种量化策略：
  1. Dynamic INT8: PyTorch 原生动态量化，适用于 CPU 推理加速
  2. FP16/BF16: 半精度推理，适用于 GPU
  3. Weight-Only INT8: 仅量化权重，激活保持原精度
  4. Smooth Quantization: 预处理权重使量化误差更均匀

用法:
    # 动态 INT8 量化
    model = quantize_dynamic(model)

    # 统一接口
    model = QuantizationWrapper(model, strategy="dynamic_int8")
    model = QuantizationWrapper(model, strategy="weight_only_int8")
    model = QuantizationWrapper(model, strategy="fp16")
"""

from __future__ import annotations

import copy
import logging
import os
import warnings
from typing import Dict, List, Optional, Set, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WeightOnlyInt8Linear(nn.Module):
    """权重量化线性层（按输出通道对称 INT8）。"""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = linear.weight.detach().to(torch.float32)
        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        qweight = (weight / scale).round().clamp(-127, 127).to(torch.int8)

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.register_buffer("qweight", qweight)
        self.register_buffer("weight_scale", scale)
        self.register_buffer(
            "bias",
            linear.bias.detach().to(torch.float32).clone()
            if linear.bias is not None
            else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = (self.qweight.float() * self.weight_scale).to(
            device=x.device, dtype=x.dtype
        )
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)


def _state_dict_nbytes(model: nn.Module) -> int:
    total = 0
    for _, tensor in model.state_dict().items():
        if torch.is_tensor(tensor):
            total += tensor.numel() * tensor.element_size()
    return total


def _sum_nested_tensor_bytes(obj: object, visited: Optional[Set[int]] = None) -> int:
    """估算对象中张量底层占用字节（支持 torchao tensor subclass）。"""
    if visited is None:
        visited = set()
    oid = id(obj)
    if oid in visited:
        return 0
    visited.add(oid)

    if torch.is_tensor(obj):
        # torchao tensor subclass: 优先统计其内部量化存储，而非逻辑 float 形状。
        module_name = type(obj).__module__
        if module_name.startswith("torchao"):
            inner = 0
            for attr_name in ("tensor_impl", "original_weight_tensor"):
                if hasattr(obj, attr_name):
                    try:
                        inner += _sum_nested_tensor_bytes(getattr(obj, attr_name), visited)
                    except Exception:
                        pass
            for v in getattr(obj, "__dict__", {}).values():
                inner += _sum_nested_tensor_bytes(v, visited)
            if inner > 0:
                return inner
        return int(obj.numel() * obj.element_size())

    if isinstance(obj, dict):
        total = 0
        for v in obj.values():
            total += _sum_nested_tensor_bytes(v, visited)
        return total
    if isinstance(obj, (list, tuple, set)):
        total = 0
        for v in obj:
            total += _sum_nested_tensor_bytes(v, visited)
        return total
    if hasattr(obj, "__dict__"):
        total = 0
        for v in obj.__dict__.values():
            total += _sum_nested_tensor_bytes(v, visited)
        return total
    return 0


def _state_dict_effective_nbytes(model: nn.Module) -> int:
    total = 0
    for _, tensor in model.state_dict().items():
        total += _sum_nested_tensor_bytes(tensor)
    return total


def _build_torchao_target_model(model: nn.Module, inplace: bool) -> Optional[nn.Module]:
    if inplace:
        return model
    try:
        return copy.deepcopy(model)
    except TypeError as e:
        if "RLock" in str(e):
            logger.warning("torchao deepcopy failed due RLock; trying re-init clone fallback")
            if os.getenv("CORTEXNET_TORCHAO_ALLOW_REINIT_CLONE", "1") != "1":
                return None
            cfg = getattr(model, "config", None)
            model_cls = type(model)
            if cfg is None:
                return None
            try:
                cloned = model_cls(copy.deepcopy(cfg)).eval()
                state = model.state_dict()
                cloned.load_state_dict(state, strict=False)
                return cloned
            except Exception as clone_exc:
                logger.warning("torchao re-init clone fallback failed: %s", clone_exc)
                return None
        raise


def _select_effective_linear_modules(
    model: nn.Module,
) -> Tuple[List[Tuple[str, nn.Linear]], Dict[str, int]]:
    """选择更可能带来收益的 Linear 子图，避免量化无收益或高风险分支。"""
    min_params = int(os.getenv("CORTEXNET_TORCHAO_MIN_LINEAR_PARAMS", "65536"))
    skip_tied_lm_head = os.getenv("CORTEXNET_TORCHAO_SKIP_TIED_LM_HEAD", "1") == "1"
    include_keywords = tuple(
        s.strip().lower()
        for s in os.getenv(
            "CORTEXNET_TORCHAO_INCLUDE",
            "proj,ffn,expert,attention,mlp,fc,dense",
        ).split(",")
        if s.strip()
    )
    exclude_keywords = tuple(
        s.strip().lower()
        for s in os.getenv(
            "CORTEXNET_TORCHAO_EXCLUDE",
            "router,gate,norm,score,bias,aux,calib",
        ).split(",")
        if s.strip()
    )

    embedding_weight_ids = {
        id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)
    }

    selected: List[Tuple[str, nn.Linear]] = []
    skipped: Dict[str, int] = {}

    def _skip(reason: str):
        skipped[reason] = skipped.get(reason, 0) + 1

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        lname = name.lower()
        numel = int(module.weight.numel())
        if numel < min_params:
            _skip("small_linear")
            continue
        if any(k in lname for k in exclude_keywords):
            _skip("exclude_keyword")
            continue
        if include_keywords and not any(k in lname for k in include_keywords):
            _skip("not_included")
            continue

        # tied embedding/lm_head 通常会导致压缩收益差甚至 state_dict 变大，默认跳过。
        if skip_tied_lm_head and id(module.weight) in embedding_weight_ids:
            _skip("tied_embedding_weight")
            continue

        selected.append((name, module))

    return selected, skipped


def _try_torchao_dynamic_int8(model: nn.Module, inplace: bool) -> Optional[nn.Module]:
    """尝试使用 torchao eager quantize_ API（可选依赖，默认选择性量化）。"""
    if os.getenv("CORTEXNET_DISABLE_TORCHAO", "0") == "1":
        return None

    # 屏蔽 torchao 内部的已知 DeprecationWarning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*torchao.dtypes.uintx.dyn_int8_act_int4_wei_cpu_layout.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*torchao.dtypes.uintx.uintx_layout.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*BlockSparseLayout.*",
            category=DeprecationWarning,
        )
        try:
            import torchao.quantization as tq  # type: ignore
        except Exception:
            return None

    quantize_fn = getattr(tq, "quantize_", None)
    if quantize_fn is None:
        return None

    candidates = [
        "int8_dynamic_activation_int8_weight",
        "int8_dynamic_activation_int8_weight_per_channel",
        "int8_dynamic_per_token_weight",
        "Int8DynamicActivationInt8WeightConfig",
        "Int8DynamicActivationIntxWeightConfig",
    ]
    for name in candidates:
        cfg_ctor = getattr(tq, name, None)
        if not callable(cfg_ctor):
            continue
        try:
            target = _build_torchao_target_model(model, inplace=inplace)
            if target is None:
                return None

            if os.getenv("CORTEXNET_TORCHAO_FULL_GRAPH", "0") == "1":
                quantize_fn(target, cfg_ctor())
                setattr(target, "_cortex_quant_torchao_mode", "full_graph")
                logger.info(f"Dynamic quantization backend: torchao ({name}, full_graph)")
            else:
                selected, skipped = _select_effective_linear_modules(target)
                if not selected:
                    logger.warning("torchao selective quantization: no effective Linear modules selected")
                    return None

                ok = 0
                fail = 0
                for module_name, linear in selected:
                    try:
                        quantize_fn(linear, cfg_ctor())
                        ok += 1
                    except Exception as exc:
                        fail += 1
                        logger.debug("torchao selective quantize failed for %s: %s", module_name, exc)

                if ok <= 0:
                    return None

                setattr(target, "_cortex_quant_torchao_mode", "selective_linear")
                setattr(target, "_cortex_quant_torchao_selected", len(selected))
                setattr(target, "_cortex_quant_torchao_quantized", ok)
                setattr(target, "_cortex_quant_torchao_failed", fail)
                logger.info(
                    "Dynamic quantization backend: torchao (%s, selective) "
                    "selected=%s quantized=%s failed=%s skipped=%s",
                    name,
                    len(selected),
                    ok,
                    fail,
                    skipped,
                )
            return target
        except Exception as e:
            logger.debug(f"torchao backend candidate '{name}' failed: {e}")
            continue
    return None


def quantize_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    inplace: bool = False,
    target_modules: Optional[Set[Type[nn.Module]]] = None,
) -> nn.Module:
    """对模型进行动态量化（Linear 层 INT8）。

    适用于推理加速，对 CortexNet 的 Linear 层进行动态量化。
    注意：SSM、自定义 attention 等可能不兼容，建议只对纯 Linear 子模块使用。

    Args:
        model: 待量化模型
        dtype: 量化类型，默认 torch.qint8
        inplace: 是否原地修改
        target_modules: 要量化的模块类型集合，默认 {nn.Linear}
    Returns:
        量化后的模型
    """
    if target_modules is None:
        target_modules = {nn.Linear}

    # 优先 torchao（官方迁移路径），仅在标准 Linear + qint8 场景尝试。
    if dtype == torch.qint8 and target_modules == {nn.Linear} and not inplace:
        baseline_bytes = _state_dict_effective_nbytes(model)
        baseline_logical_bytes = _state_dict_nbytes(model)
        torchao_model = _try_torchao_dynamic_int8(model, inplace=inplace)
        if torchao_model is not None:
            torchao_bytes = _state_dict_effective_nbytes(torchao_model)
            torchao_logical_bytes = _state_dict_nbytes(torchao_model)
            keep_ratio = float(os.getenv("CORTEXNET_TORCHAO_KEEP_RATIO", "0.98"))
            if torchao_bytes <= int(baseline_bytes * keep_ratio):
                setattr(torchao_model, "_cortex_quant_backend", "torchao_dynamic_int8")
                setattr(torchao_model, "_cortex_quant_baseline_bytes", baseline_bytes)
                setattr(torchao_model, "_cortex_quant_result_bytes", torchao_bytes)
                setattr(torchao_model, "_cortex_quant_baseline_logical_bytes", baseline_logical_bytes)
                setattr(torchao_model, "_cortex_quant_result_logical_bytes", torchao_logical_bytes)
                return torchao_model
            logger.warning(
                "torchao quantization rejected due weak effective compression: "
                "baseline=%.3fMB, torchao=%.3fMB (logical baseline=%.3fMB, logical torchao=%.3fMB)",
                baseline_bytes / (1024 ** 2),
                torchao_bytes / (1024 ** 2),
                baseline_logical_bytes / (1024 ** 2),
                torchao_logical_bytes / (1024 ** 2),
            )

    # 回退到 torch.ao（兼容旧环境），并屏蔽已知弃用告警。
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="torch.ao.quantization is deprecated.*",
            category=DeprecationWarning,
        )
        # macOS/CPU 常见场景下默认 engine=none，需显式设置为 qnnpack。
        if hasattr(torch.backends, "quantized"):
            current_engine = getattr(torch.backends.quantized, "engine", "none")
            supported = list(getattr(torch.backends.quantized, "supported_engines", []))
            if current_engine == "none" and "qnnpack" in supported:
                torch.backends.quantized.engine = "qnnpack"
        try:
            quantized = torch.ao.quantization.quantize_dynamic(
                model,
                target_modules,
                dtype=dtype,
                inplace=inplace,
            )
        except TypeError as e:
            if (not inplace) and "RLock" in str(e):
                # CortexNet 含线程锁对象，deepcopy 路径不可用；回退到原地量化。
                logger.warning(
                    "quantize_dynamic deepcopy failed due RLock; retry with inplace=True"
                )
                quantized = torch.ao.quantization.quantize_dynamic(
                    model,
                    target_modules,
                    dtype=dtype,
                    inplace=True,
                )
            else:
                raise
        setattr(quantized, "_cortex_quant_backend", "torch_ao_dynamic_int8")
        return quantized


def to_fp16(model: nn.Module) -> nn.Module:
    """将模型转换为 FP16（半精度）。"""
    return model.half()


def to_bf16(model: nn.Module) -> nn.Module:
    """将模型转换为 BF16。"""
    return model.to(dtype=torch.bfloat16)


def weight_only_int8(model: nn.Module, inplace: bool = False) -> nn.Module:
    """仅量化权重为 INT8，保持激活为原精度。

    这种方式减少模型显存占用，同时保持更好的精度。
    """
    if inplace:
        target = model
    else:
        try:
            target = copy.deepcopy(model)
        except TypeError as e:
            if "RLock" in str(e):
                logger.warning(
                    "weight_only_int8 deepcopy failed due RLock; retry with inplace=True"
                )
                target = model
            else:
                raise
    replaced = 0

    def _replace(module: nn.Module):
        nonlocal replaced
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, WeightOnlyInt8Linear(child))
                replaced += 1
            else:
                _replace(child)

    _replace(target)
    logger.info(f"Weight-only INT8 replaced {replaced} Linear modules")
    setattr(target, "_cortex_quant_backend", "weight_only_int8")
    return target


def smooth_quantization_preprocess(model: nn.Module, alpha: float = 0.5) -> nn.Module:
    """平滑量化预处理：均衡权重和激活的量化难度。

    基于 SmoothQuant 思想，将激活的量化难度转移到权重上，
    使两者都更容易量化。

    Args:
        model: 模型
        alpha: 平滑系数 (0=全部转移到权重, 1=不转移)
    Returns:
        预处理后的模型
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
            weight = module.weight.data
            # 计算每个输出通道的权重范围
            w_max = weight.abs().max(dim=0, keepdim=True).values.clamp(min=1e-8)
            # 平滑因子
            smooth_factor = w_max.pow(alpha)
            # 缩放权重
            module.weight.data = weight / smooth_factor
            # 记录逆缩放因子（推理时需对输入应用）
            module.register_buffer('smooth_scale', smooth_factor.squeeze(0))
    return model


class QuantizationWrapper(nn.Module):
    """量化包装器：统一多种量化策略接口。

    Args:
        model: 待量化模型
        strategy: 量化策略
            - "dynamic_int8": 动态 INT8 量化
            - "weight_only_int8": 仅权重 INT8
            - "fp16": FP16 半精度
            - "bf16": BF16 半精度
            - "smooth_int8": 平滑量化 + 动态 INT8
            - None/"none": 不量化
        smooth_alpha: 平滑量化的 alpha 系数
    """

    STRATEGIES = {"dynamic_int8", "weight_only_int8", "fp16", "bf16", "smooth_int8", "none", None}

    def __init__(
        self,
        model: nn.Module,
        strategy: Optional[str] = None,
        smooth_alpha: float = 0.5,
        # 向后兼容旧接口
        use_fp16: Optional[bool] = None,
        use_dynamic_int8: bool = False,
    ):
        super().__init__()

        # 向后兼容旧接口
        if use_dynamic_int8:
            strategy = "dynamic_int8"
        elif use_fp16 is True:
            strategy = "fp16"
        elif use_fp16 is None and strategy is None and torch.cuda.is_available():
            strategy = "fp16"

        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown quantization strategy: '{strategy}'. "
                f"Supported: {self.STRATEGIES - {None}}"
            )

        self.strategy = strategy or "none"

        if strategy == "dynamic_int8":
            self.model = quantize_dynamic(model, inplace=False)
        elif strategy == "weight_only_int8":
            self.model = weight_only_int8(model, inplace=False)
        elif strategy == "fp16":
            self.model = model.half()
        elif strategy == "bf16":
            self.model = to_bf16(model)
        elif strategy == "smooth_int8":
            model = smooth_quantization_preprocess(model, alpha=smooth_alpha)
            self.model = quantize_dynamic(model, inplace=False)
        else:
            self.model = model

        self.backend = getattr(self.model, "_cortex_quant_backend", self.strategy)
        logger.info(f"QuantizationWrapper initialized with strategy='{self.strategy}'")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
