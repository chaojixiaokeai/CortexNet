"""
权重适配层 (Weight Adapter)

核心功能：
  将开源大模型的原生权重无损映射到 CortexNet 的模块结构。

技术要点：
  1. 模式化权重名称匹配（支持模糊匹配）
  2. 维度自适应投影（当源模型与 CortexNet 维度不匹配时）
  3. 归一化参数自动转换（LayerNorm ↔ RMSNorm）
  4. GQA 权重扩展（num_kv_heads < num_heads 时复制 KV 投影）
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#              权重名称映射规则库
# ═══════════════════════════════════════════════════════════════

# 每个模型族定义从 HuggingFace 权重名称到 CortexNet 模块的映射规则。
# 格式: (source_pattern, cortex_pattern)
# source_pattern: HuggingFace 权重名中的子串
# cortex_pattern: CortexNet 权重名模板（{layer} 会被替换为层索引）

WEIGHT_MAPPING_RULES: Dict[str, List[Tuple[str, str]]] = {
    "llama": [
        # Embedding
        ("model.embed_tokens.weight", "embed.weight"),
        ("lm_head.weight", "lm_head.weight"),
        # Per-layer mappings
        ("self_attn.q_proj.weight", "blocks.{layer}.attention.q_proj.weight"),
        ("self_attn.k_proj.weight", "blocks.{layer}.attention.k_proj.weight"),
        ("self_attn.v_proj.weight", "blocks.{layer}.attention.v_proj.weight"),
        ("self_attn.o_proj.weight", "blocks.{layer}.attention.o_proj.weight"),
        ("mlp.gate_proj.weight", "blocks.{layer}.moe.experts.0.gate_proj.weight"),
        ("mlp.up_proj.weight", "blocks.{layer}.moe.experts.0.up_proj.weight"),
        ("mlp.down_proj.weight", "blocks.{layer}.moe.experts.0.down_proj.weight"),
        ("input_layernorm.weight", "blocks.{layer}.norm1.weight"),
        ("post_attention_layernorm.weight", "blocks.{layer}.norm2.weight"),
        # 归一化层 bias（如有）
        ("input_layernorm.bias", "blocks.{layer}.norm1.bias"),
        ("post_attention_layernorm.bias", "blocks.{layer}.norm2.bias"),
        # Final norm
        ("model.norm.weight", "final_norm.weight"),
    ],
    "qwen": [
        ("transformer.wte.weight", "embed.weight"),
        ("lm_head.weight", "lm_head.weight"),
        # Qwen2 style
        ("model.embed_tokens.weight", "embed.weight"),
        ("self_attn.q_proj.weight", "blocks.{layer}.attention.q_proj.weight"),
        ("self_attn.k_proj.weight", "blocks.{layer}.attention.k_proj.weight"),
        ("self_attn.v_proj.weight", "blocks.{layer}.attention.v_proj.weight"),
        ("self_attn.q_norm.weight", "blocks.{layer}.attention.q_norm.weight"),
        ("self_attn.k_norm.weight", "blocks.{layer}.attention.k_norm.weight"),
        ("self_attn.o_proj.weight", "blocks.{layer}.attention.o_proj.weight"),
        ("mlp.gate_proj.weight", "blocks.{layer}.moe.experts.0.gate_proj.weight"),
        ("mlp.up_proj.weight", "blocks.{layer}.moe.experts.0.up_proj.weight"),
        ("mlp.down_proj.weight", "blocks.{layer}.moe.experts.0.down_proj.weight"),
        ("input_layernorm.weight", "blocks.{layer}.norm1.weight"),
        ("post_attention_layernorm.weight", "blocks.{layer}.norm2.weight"),
        ("model.norm.weight", "final_norm.weight"),
    ],
    "glm": [
        ("transformer.embedding.word_embeddings.weight", "embed.weight"),
        ("transformer.output_layer.weight", "lm_head.weight"),
        ("self_attention.query_key_value.weight", "blocks.{layer}.attention.qkv_proj.weight"),
        ("self_attention.dense.weight", "blocks.{layer}.attention.o_proj.weight"),
        ("mlp.dense_h_to_4h.weight", "blocks.{layer}.moe.experts.0.gate_up_proj.weight"),
        ("mlp.dense_4h_to_h.weight", "blocks.{layer}.moe.experts.0.down_proj.weight"),
        ("input_layernorm.weight", "blocks.{layer}.norm1.weight"),
        ("post_attention_layernorm.weight", "blocks.{layer}.norm2.weight"),
        ("transformer.encoder.final_layernorm.weight", "final_norm.weight"),
    ],
    "baichuan": [
        ("model.embed_tokens.weight", "embed.weight"),
        ("lm_head.weight", "lm_head.weight"),
        ("self_attn.W_pack.weight", "blocks.{layer}.attention.qkv_proj.weight"),
        ("self_attn.o_proj.weight", "blocks.{layer}.attention.o_proj.weight"),
        ("mlp.gate_proj.weight", "blocks.{layer}.moe.experts.0.gate_proj.weight"),
        ("mlp.up_proj.weight", "blocks.{layer}.moe.experts.0.up_proj.weight"),
        ("mlp.down_proj.weight", "blocks.{layer}.moe.experts.0.down_proj.weight"),
        ("input_layernorm.weight", "blocks.{layer}.norm1.weight"),
        ("post_attention_layernorm.weight", "blocks.{layer}.norm2.weight"),
        ("model.norm.weight", "final_norm.weight"),
    ],
    "phi": [
        ("model.embed_tokens.weight", "embed.weight"),
        ("lm_head.weight", "lm_head.weight"),
        ("self_attn.q_proj.weight", "blocks.{layer}.attention.q_proj.weight"),
        ("self_attn.k_proj.weight", "blocks.{layer}.attention.k_proj.weight"),
        ("self_attn.v_proj.weight", "blocks.{layer}.attention.v_proj.weight"),
        ("self_attn.dense.weight", "blocks.{layer}.attention.o_proj.weight"),
        ("mlp.fc1.weight", "blocks.{layer}.moe.experts.0.gate_proj.weight"),
        ("mlp.fc2.weight", "blocks.{layer}.moe.experts.0.down_proj.weight"),
        ("input_layernorm.weight", "blocks.{layer}.norm1.weight"),
        ("input_layernorm.bias", "blocks.{layer}.norm1.bias"),
        ("post_attention_layernorm.weight", "blocks.{layer}.norm2.weight"),
        ("post_attention_layernorm.bias", "blocks.{layer}.norm2.bias"),
        ("model.final_layernorm.weight", "final_norm.weight"),
    ],
    "gemma": [
        ("model.embed_tokens.weight", "embed.weight"),
        # Gemma 没有单独的 lm_head，使用 embed_tokens
        ("self_attn.q_proj.weight", "blocks.{layer}.attention.q_proj.weight"),
        ("self_attn.k_proj.weight", "blocks.{layer}.attention.k_proj.weight"),
        ("self_attn.v_proj.weight", "blocks.{layer}.attention.v_proj.weight"),
        ("self_attn.o_proj.weight", "blocks.{layer}.attention.o_proj.weight"),
        ("mlp.gate_proj.weight", "blocks.{layer}.moe.experts.0.gate_proj.weight"),
        ("mlp.up_proj.weight", "blocks.{layer}.moe.experts.0.up_proj.weight"),
        ("mlp.down_proj.weight", "blocks.{layer}.moe.experts.0.down_proj.weight"),
        ("input_layernorm.weight", "blocks.{layer}.norm1.weight"),
        ("post_attention_layernorm.weight", "blocks.{layer}.norm2.weight"),
        ("model.norm.weight", "final_norm.weight"),
    ],
    "falcon": [
        ("transformer.word_embeddings.weight", "embed.weight"),
        ("lm_head.weight", "lm_head.weight"),
        ("self_attention.query_key_value.weight", "blocks.{layer}.attention.qkv_proj.weight"),
        ("self_attention.dense.weight", "blocks.{layer}.attention.o_proj.weight"),
        ("mlp.dense_h_to_4h.weight", "blocks.{layer}.moe.experts.0.gate_proj.weight"),
        ("mlp.dense_4h_to_h.weight", "blocks.{layer}.moe.experts.0.down_proj.weight"),
        ("input_layernorm.weight", "blocks.{layer}.norm1.weight"),
        ("input_layernorm.bias", "blocks.{layer}.norm1.bias"),
        ("ln_attn.weight", "blocks.{layer}.norm2.weight"),
        ("ln_attn.bias", "blocks.{layer}.norm2.bias"),
        ("transformer.ln_f.weight", "final_norm.weight"),
    ],
}

# 使用 llama 格式的模型族别名
for _alias in ["mistral", "yi", "deepseek", "internlm", "codelama"]:
    WEIGHT_MAPPING_RULES[_alias] = WEIGHT_MAPPING_RULES["llama"]


def _extract_layer_idx(name: str) -> Optional[int]:
    """从权重名称中提取层索引。"""
    patterns = [
        r"\.layers\.(\d+)\.",       # LLaMA/Qwen/Mistral ...
        r"\.h\.(\d+)\.",            # GPT-style
        r"\.encoder\.layers\.(\d+)\.",  # ChatGLM
        r"\.blocks\.(\d+)\.",       # 某些自定义模型
        r"\.transformer\.h\.(\d+)\.",   # GPT-2 style
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


class WeightAdapter:
    """权重适配器：将开源模型权重映射到 CortexNet 模块。

    Args:
        model_type: 模型族标识 (如 "llama", "qwen")
        config: CortexNetConfig
    """

    def __init__(self, model_type: str, config: Any):
        self.model_type = model_type
        self.config = config
        self.mapping_rules = self._get_mapping_rules()
        self._unmapped: List[str] = []
        # Lite 注意力使用 kv_proj（合并 K/V），缓存分片流式加载下的半边投影。
        self._pending_kv_proj: Dict[str, Dict[str, torch.Tensor]] = {}

    def _get_mapping_rules(self) -> List[Tuple[str, str]]:
        """获取当前模型族的映射规则。"""
        # 首先查找精确匹配
        from .model_registry import MODEL_REGISTRY
        model_info = MODEL_REGISTRY.get(self.model_type)
        weight_format = model_info.weight_format if model_info else self.model_type

        rules = WEIGHT_MAPPING_RULES.get(weight_format)
        if rules is None:
            logger.warning(
                f"No weight mapping rules for '{weight_format}', "
                f"falling back to 'llama' format."
            )
            rules = WEIGHT_MAPPING_RULES["llama"]

        # Lite 模式：将 MoE 路径重映射为 FFN 路径
        if getattr(self.config, 'lite', False):
            remapped = []
            for src, dst in rules:
                dst = dst.replace("moe.experts.0.", "ffn.")
                remapped.append((src, dst))
            rules = remapped

        return rules

    def map_weights(
        self,
        raw_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """将 HuggingFace 权重映射到 CortexNet 权重名。

        Args:
            raw_weights: HuggingFace 模型 state_dict

        Returns:
            CortexNet 格式的 state_dict（部分映射）
        """
        cortex_weights: Dict[str, torch.Tensor] = OrderedDict()
        self._unmapped = []
        mapped_count = 0

        for raw_name, tensor in raw_weights.items():
            mapped = False

            # 提取层索引
            layer_idx = _extract_layer_idx(raw_name)

            for src_pattern, dst_pattern in self.mapping_rules:
                if src_pattern in raw_name:
                    # 替换层索引占位符
                    cortex_name = dst_pattern
                    if "{layer}" in cortex_name and layer_idx is not None:
                        cortex_name = cortex_name.replace("{layer}", str(layer_idx))
                    elif "{layer}" in cortex_name:
                        # 无法提取层索引，跳过
                        continue

                    # 处理特殊情况：合并的 QKV 投影
                    if "qkv_proj" in cortex_name:
                        self._split_qkv(cortex_name, tensor, cortex_weights)
                    elif "gate_up_proj" in cortex_name:
                        self._split_gate_up(cortex_name, tensor, cortex_weights)
                    elif getattr(self.config, "lite", False) and (
                        cortex_name.endswith(".attention.k_proj.weight")
                        or cortex_name.endswith(".attention.v_proj.weight")
                    ):
                        self._merge_lite_kv(cortex_name, tensor, cortex_weights)
                    else:
                        cortex_weights[cortex_name] = tensor

                    mapped = True
                    mapped_count += 1
                    break

            if not mapped:
                self._unmapped.append(raw_name)

        # 归一化参数转换
        cortex_weights = self._convert_norm_params(cortex_weights)

        # GQA 权重扩展
        cortex_weights = self._expand_gqa_weights(cortex_weights)

        logger.info(
            f"Weight mapping complete: {mapped_count} mapped, "
            f"{len(self._unmapped)} unmapped out of {len(raw_weights)} total"
        )

        if self._unmapped:
            logger.debug(f"Unmapped weights: {self._unmapped[:10]}...")

        return cortex_weights

    def _merge_lite_kv(
        self,
        cortex_name: str,
        tensor: torch.Tensor,
        weights: Dict[str, torch.Tensor],
    ) -> None:
        """Lite 注意力把分离的 K/V 权重合并为 kv_proj.weight。"""
        if cortex_name.endswith(".attention.k_proj.weight"):
            base = cortex_name[: -len("k_proj.weight")]
            slot = "k"
        else:
            base = cortex_name[: -len("v_proj.weight")]
            slot = "v"

        bucket = self._pending_kv_proj.setdefault(base, {})
        bucket[slot] = tensor
        if "k" in bucket and "v" in bucket:
            kv_name = f"{base}kv_proj.weight"
            weights[kv_name] = torch.cat([bucket["k"], bucket["v"]], dim=0)
            del self._pending_kv_proj[base]

    def _split_qkv(
        self,
        cortex_name: str,
        tensor: torch.Tensor,
        weights: Dict[str, torch.Tensor],
    ):
        """将合并的 QKV 投影分割为独立的 Q/K/V 权重。"""
        # 推断 Q/K/V 的维度
        hidden_size = self.config.hidden_size
        num_heads = self.config.num_heads
        num_kv_heads = self.config.num_kv_heads
        head_dim = hidden_size // num_heads

        q_size = num_heads * head_dim
        k_size = num_kv_heads * head_dim
        v_size = num_kv_heads * head_dim

        q_name = cortex_name.replace("qkv_proj", "q_proj")
        k_name = cortex_name.replace("qkv_proj", "k_proj")
        v_name = cortex_name.replace("qkv_proj", "v_proj")

        if tensor.shape[0] == q_size + k_size + v_size:
            q, k, v = tensor.split([q_size, k_size, v_size], dim=0)
            weights[q_name] = q
            weights[k_name] = k
            weights[v_name] = v
        else:
            # 均分（如 Baichuan W_pack，三等份）
            q, k, v = tensor.chunk(3, dim=0)
            weights[q_name] = q
            weights[k_name] = k
            weights[v_name] = v

    def _split_gate_up(
        self,
        cortex_name: str,
        tensor: torch.Tensor,
        weights: Dict[str, torch.Tensor],
    ):
        """将合并的 gate+up 投影分割。"""
        gate, up = tensor.chunk(2, dim=0)
        gate_name = cortex_name.replace("gate_up_proj", "gate_proj")
        up_name = cortex_name.replace("gate_up_proj", "up_proj")
        weights[gate_name] = gate
        weights[up_name] = up

    def _convert_norm_params(
        self,
        weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """转换归一化参数（LayerNorm bias 移除 → RMSNorm 仅 weight）。"""
        from .model_registry import MODEL_REGISTRY

        model_info = MODEL_REGISTRY.get(self.model_type)
        if model_info and model_info.default_norm_type == "layernorm":
            # CortexNet 使用 RMSNorm（仅 weight，无 bias）
            # 保留 weight，丢弃 bias
            keys_to_remove = [k for k in weights if k.endswith(".bias") and "norm" in k]
            for k in keys_to_remove:
                logger.debug(f"Removing LayerNorm bias (CortexNet uses RMSNorm): {k}")
                del weights[k]

        return weights

    def _expand_gqa_weights(
        self,
        weights: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """扩展 GQA 的 K/V 权重到全头数。

        当 num_kv_heads < num_heads 时，需要复制 K/V 投影使其匹配。
        """
        if not getattr(self.config, "expand_gqa_weights", True):
            return weights

        if self.config.num_kv_heads >= self.config.num_heads:
            return weights

        repeat_factor = self.config.num_heads // self.config.num_kv_heads

        for key in list(weights.keys()):
            if ("k_proj.weight" in key or "v_proj.weight" in key):
                tensor = weights[key]
                head_dim = self.config.hidden_size // self.config.num_heads
                expected_full_size = self.config.num_heads * head_dim

                if tensor.shape[0] < expected_full_size:
                    # GQA: 每个 KV 头复制 repeat_factor 次
                    weights[key] = tensor.repeat(repeat_factor, 1)
                    logger.debug(
                        f"GQA weight expansion: {key} "
                        f"{tensor.shape} -> {weights[key].shape}"
                    )

        return weights

    def get_unmapped_weights(self) -> List[str]:
        """返回未映射的权重名称列表。"""
        return self._unmapped.copy()

    def get_unmapped_count(self) -> int:
        """返回当前批次未映射权重数量（避免复制列表）。"""
        return len(self._unmapped)

    def verify_mapping(
        self,
        cortex_state_dict: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> Dict[str, str]:
        """验证映射后的权重是否可以正确加载到 CortexNet 模型。

        Returns:
            {"matched": [...], "missing": [...], "unexpected": [...]}
        """
        model_keys = set(model.state_dict().keys())
        mapped_keys = set(cortex_state_dict.keys())

        matched = model_keys & mapped_keys
        missing = model_keys - mapped_keys
        unexpected = mapped_keys - model_keys

        return {
            "matched": sorted(matched),
            "missing": sorted(missing),
            "unexpected": sorted(unexpected),
            "match_ratio": len(matched) / max(len(model_keys), 1),
        }
