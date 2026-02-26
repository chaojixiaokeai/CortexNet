"""
模型注册表与自动识别 (Model Registry & Auto-Detection)

核心功能：
  1. 从 HuggingFace config.json 自动识别模型类型
  2. 将 HuggingFace 配置转换为 CortexNetConfig
  3. 维护支持模型的注册表

支持的模型族：
  LLaMA 2/3, Qwen 1.5/2, Mistral, Baichuan 1/2, ChatGLM 3/4,
  Phi 2/3, Yi, DeepSeek, InternLM, Gemma, CodeLlama, Falcon
"""

from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """模型族元信息。"""
    family: str                           # 模型族标识 (如 "llama")
    display_name: str                     # 显示名称
    architectures: List[str]              # HuggingFace architectures 字段匹配
    model_type_patterns: List[str]        # config.json 中 model_type 匹配模式
    weight_format: str                    # 权重命名格式族 ("llama", "qwen", "glm" 等)
    default_rope_theta: float = 10000.0   # 默认 RoPE theta
    supports_gqa: bool = False            # 是否使用 GQA
    default_norm_type: str = "rmsnorm"    # 默认归一化类型
    position_encoding: str = "rope"       # 位置编码类型


# ═══════════════════════════════════════════════════════════════
#                   模型注册表
# ═══════════════════════════════════════════════════════════════

MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "llama": ModelInfo(
        family="llama",
        display_name="LLaMA / LLaMA 2 / LLaMA 3",
        architectures=["LlamaForCausalLM"],
        model_type_patterns=["llama"],
        weight_format="llama",
        default_rope_theta=10000.0,
        supports_gqa=True,
        default_norm_type="rmsnorm",
    ),
    "qwen2": ModelInfo(
        family="qwen2",
        display_name="Qwen 1.5 / Qwen 2",
        architectures=["Qwen2ForCausalLM", "QWenLMHeadModel"],
        model_type_patterns=["qwen2", "qwen"],
        weight_format="qwen",
        default_rope_theta=10000.0,
        supports_gqa=True,
        default_norm_type="rmsnorm",
    ),
    "qwen3": ModelInfo(
        family="qwen3",
        display_name="Qwen 3",
        architectures=["Qwen3ForCausalLM"],
        model_type_patterns=["qwen3", "qwen"],
        weight_format="qwen",
        default_rope_theta=1000000.0,
        supports_gqa=True,
        default_norm_type="rmsnorm",
    ),
    "mistral": ModelInfo(
        family="mistral",
        display_name="Mistral / Mixtral",
        architectures=["MistralForCausalLM", "MixtralForCausalLM"],
        model_type_patterns=["mistral", "mixtral"],
        weight_format="llama",
        default_rope_theta=10000.0,
        supports_gqa=True,
        default_norm_type="rmsnorm",
    ),
    "baichuan": ModelInfo(
        family="baichuan",
        display_name="Baichuan 1 / Baichuan 2",
        architectures=["BaichuanForCausalLM", "BaiChuanForCausalLM"],
        model_type_patterns=["baichuan"],
        weight_format="baichuan",
        default_rope_theta=10000.0,
        supports_gqa=False,
        default_norm_type="rmsnorm",
    ),
    "chatglm": ModelInfo(
        family="chatglm",
        display_name="ChatGLM 3 / GLM-4",
        architectures=["ChatGLMModel", "ChatGLMForConditionalGeneration"],
        model_type_patterns=["chatglm"],
        weight_format="glm",
        default_rope_theta=10000.0,
        supports_gqa=True,
        default_norm_type="rmsnorm",
        position_encoding="rope",
    ),
    "phi": ModelInfo(
        family="phi",
        display_name="Phi 2 / Phi 3",
        architectures=["PhiForCausalLM", "Phi3ForCausalLM"],
        model_type_patterns=["phi", "phi3"],
        weight_format="phi",
        default_rope_theta=10000.0,
        supports_gqa=True,
        default_norm_type="layernorm",
    ),
    "yi": ModelInfo(
        family="yi",
        display_name="Yi",
        architectures=["YiForCausalLM"],
        model_type_patterns=["yi"],
        weight_format="llama",
        default_rope_theta=5000000.0,
        supports_gqa=True,
        default_norm_type="rmsnorm",
    ),
    "deepseek": ModelInfo(
        family="deepseek",
        display_name="DeepSeek / DeepSeek V2",
        architectures=["DeepSeekForCausalLM", "DeepseekV2ForCausalLM"],
        model_type_patterns=["deepseek"],
        weight_format="llama",
        default_rope_theta=10000.0,
        supports_gqa=True,
        default_norm_type="rmsnorm",
    ),
    "internlm": ModelInfo(
        family="internlm",
        display_name="InternLM / InternLM 2",
        architectures=["InternLMForCausalLM", "InternLM2ForCausalLM"],
        model_type_patterns=["internlm", "internlm2"],
        weight_format="llama",
        default_rope_theta=10000.0,
        supports_gqa=True,
        default_norm_type="rmsnorm",
    ),
    "gemma": ModelInfo(
        family="gemma",
        display_name="Gemma / Gemma 2",
        architectures=["GemmaForCausalLM", "Gemma2ForCausalLM"],
        model_type_patterns=["gemma", "gemma2"],
        weight_format="gemma",
        default_rope_theta=10000.0,
        supports_gqa=True,
        default_norm_type="rmsnorm",
    ),
    "falcon": ModelInfo(
        family="falcon",
        display_name="Falcon",
        architectures=["FalconForCausalLM", "RWForCausalLM"],
        model_type_patterns=["falcon"],
        weight_format="falcon",
        default_rope_theta=10000.0,
        supports_gqa=True,
        default_norm_type="layernorm",
    ),
    "codelama": ModelInfo(
        family="codelama",
        display_name="Code Llama",
        architectures=["LlamaForCausalLM"],
        model_type_patterns=["llama"],
        weight_format="llama",
        default_rope_theta=1000000.0,
        supports_gqa=True,
        default_norm_type="rmsnorm",
    ),
}


class ModelRegistry:
    """模型注册表管理器。"""

    @staticmethod
    def list_supported() -> List[str]:
        """返回所有支持的模型族名称。"""
        return list(MODEL_REGISTRY.keys())

    @staticmethod
    def get_info(family: str) -> Optional[ModelInfo]:
        """获取指定模型族的信息。"""
        return MODEL_REGISTRY.get(family)

    @staticmethod
    def register(family: str, info: ModelInfo):
        """注册新的模型族。"""
        MODEL_REGISTRY[family] = info
        logger.info(f"Registered model family: {family} ({info.display_name})")


def _load_hf_config(model_path: str) -> Dict[str, Any]:
    """加载 HuggingFace config.json。"""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No config.json found at {model_path}. "
            "Please provide a valid HuggingFace model directory."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_model_type(model_path: str) -> str:
    """从 HuggingFace 模型目录自动识别模型类型。

    识别逻辑（按优先级）：
      1. config.json 中的 architectures 字段精确匹配
      2. config.json 中的 model_type 字段模式匹配
      3. 模型目录名称启发式匹配

    Args:
        model_path: 模型目录路径

    Returns:
        模型族标识符 (如 "llama", "qwen2", "mistral")

    Raises:
        ValueError: 无法识别模型类型
    """
    hf_config = _load_hf_config(model_path)

    # 策略1：architectures 字段精确匹配
    architectures = hf_config.get("architectures", [])
    for family, info in MODEL_REGISTRY.items():
        for arch in architectures:
            if arch in info.architectures:
                logger.info(f"Detected model type '{family}' via architecture: {arch}")
                return family

    # 策略2：model_type 字段模式匹配
    model_type = hf_config.get("model_type", "").lower()
    for family, info in MODEL_REGISTRY.items():
        for pattern in info.model_type_patterns:
            if pattern in model_type:
                logger.info(f"Detected model type '{family}' via model_type: {model_type}")
                return family

    # 策略3：目录名称启发式
    dir_name = os.path.basename(model_path).lower()
    for family in MODEL_REGISTRY:
        if family in dir_name:
            logger.info(f"Detected model type '{family}' via directory name: {dir_name}")
            return family

    raise ValueError(
        f"Cannot detect model type from {model_path}. "
        f"Supported model families: {list(MODEL_REGISTRY.keys())}. "
        f"Found architectures={architectures}, model_type='{model_type}'."
    )


def get_cortexnet_config(model_path: str, model_type: Optional[str] = None):
    """将 HuggingFace 配置转换为 CortexNetConfig。

    自动从 HuggingFace config.json 提取所有必要参数，
    并映射到 CortexNet 的配置格式。

    Args:
        model_path: 模型目录路径
        model_type: 模型类型（可选，自动检测）

    Returns:
        生成的 CortexNetConfig
    """
    try:
        from ..config import CortexNetConfig
    except ImportError:
        # 兼容脚本式导入: `from adapter.model_registry import ...`
        from cortexnet.config import CortexNetConfig

    hf_config = _load_hf_config(model_path)

    if model_type is None:
        model_type = detect_model_type(model_path)

    model_info = MODEL_REGISTRY.get(model_type)

    # 提取通用参数（不同模型的命名可能不同）
    vocab_size = hf_config.get("vocab_size", 32000)
    hidden_size = hf_config.get("hidden_size", 4096)
    num_layers = hf_config.get(
        "num_hidden_layers",
        hf_config.get("num_layers", 32),
    )
    num_heads = hf_config.get(
        "num_attention_heads",
        hf_config.get("num_heads", 32),
    )
    num_kv_heads = hf_config.get(
        "num_key_value_heads",
        hf_config.get("multi_query_group_num", num_heads),
    )
    intermediate_size = hf_config.get(
        "intermediate_size",
        hf_config.get("ffn_hidden_size", hidden_size * 4),
    )
    max_seq_len = hf_config.get(
        "max_position_embeddings",
        hf_config.get("seq_length", 8192),
    )
    rope_theta = hf_config.get(
        "rope_theta",
        model_info.default_rope_theta if model_info else 10000.0,
    )
    rope_scaling = hf_config.get("rope_scaling")
    norm_eps = hf_config.get(
        "rms_norm_eps",
        hf_config.get("layer_norm_epsilon", 1e-6),
    )
    tie_word_embeddings = hf_config.get("tie_word_embeddings", True)
    use_qk_norm = bool(model_type in {"qwen2", "qwen3"})

    # 滑动窗口（Mistral 等）
    sliding_window = hf_config.get("sliding_window", 0) or 0

    # 为兼容已有大模型推理，默认不放大 FFN 参数量：
    # 将 MoE 退化为单专家，相当于普通前馈层，保证参数规模与源模型同量级。
    num_experts = 1
    num_active_experts = 1
    expert_ff_dim = intermediate_size

    # 构建 CortexNetConfig
    config = CortexNetConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        use_qk_norm=use_qk_norm,
        max_seq_len=max_seq_len,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        norm_eps=norm_eps,
        dropout=0.0,  # 推理时关闭 dropout
        # SSM 参数根据模型规模自适应
        num_scales=min(4, max(2, hidden_size // 1024)),
        ssm_state_size=16,
        ssm_expand_factor=2,
        # 注意力参数
        top_k_ratio=0.25,
        attention_k_mode="ratio",
        sliding_window_size=sliding_window,
        # 记忆参数
        memory_dim=min(128, hidden_size // 4),
        memory_decay_init=0.95,
        # MoE 参数（兼容模式下为单专家）
        expert_ff_dim=expert_ff_dim,
        num_experts=num_experts,
        num_active_experts=num_active_experts,
        moe_aux_loss_weight=0.02,
        # 适配器元数据
        model_type=model_type,
        source_model_path=model_path,
        intermediate_size=intermediate_size,
        tie_word_embeddings=tie_word_embeddings,
        # Lite 模式（默认）：SSM + Attention + Memory + FFN，参数高效
        compatibility_mode=False,
        lite=True,
        # Qwen/LLaMA 等 GQA 模型保持原生 KV 头维度，不做全头扩展
        expand_gqa_weights=False,
        # SSM 低秩参数化，降低额外参数开销
        compat_ssm_rank=min(256, max(64, hidden_size // 16)),
    )

    logger.info(
        f"Created CortexNetConfig for {model_type}: "
        f"hidden={hidden_size}, layers={num_layers}, heads={num_heads}, "
        f"kv_heads={num_kv_heads}, vocab={vocab_size}"
    )

    return config
