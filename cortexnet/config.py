"""
CortexNet 配置模块 (Configuration Module)

定义 CortexNet 架构和训练的所有超参数。
使用 dataclass 实现类型安全的配置管理。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CortexNetConfig:
    """CortexNet 架构配置。

    控制模型的规模、结构和行为。所有 CortexNet 模块共享此配置。

    Attributes:
        vocab_size: 词汇表大小
        hidden_size: 隐藏层维度 (d_model)
        num_layers: CortexBlock 层数
        num_heads: 注意力头数
        num_scales: SSM 多尺度数
        ssm_state_size: SSM 状态维度
        ssm_expand_factor: SSM 内部扩展倍数
        top_k_ratio: 稀疏注意力 top-k 比例
        attention_k_mode: top-k 计算模式 ("ratio", "sqrt", "log", "fixed")
        max_seq_len: 最大序列长度
        rope_theta: RoPE 频率基数
        dropout: Dropout 比率
        memory_dim: 记忆模块维度
        memory_decay_init: 记忆衰减初始值
        expert_ff_dim: 专家 FFN 中间维度
        num_experts: 专家总数
        num_active_experts: 每个 token 激活的专家数
        moe_aux_loss_weight: MoE 辅助损失权重
        moe_capacity_factor: MoE 容量因子
        norm_eps: 归一化层 epsilon
    """

    # ═══ 基础架构参数 ═══
    vocab_size: int = 32000
    hidden_size: int = 2560
    num_layers: int = 32
    num_heads: int = 8
    max_seq_len: int = 8192
    dropout: float = 0.0
    norm_eps: float = 1e-6

    # ═══ SSM 参数 ═══
    num_scales: int = 4
    ssm_state_size: int = 16
    ssm_expand_factor: int = 2

    # ═══ 注意力参数 ═══
    top_k_ratio: float = 0.25
    attention_k_mode: str = "ratio"
    rope_theta: float = 10000.0
    sliding_window_size: int = 0

    # ═══ 记忆参数 ═══
    memory_dim: int = 64
    memory_decay_init: float = 0.95
    episodic_slots: int = 32
    semantic_slots: int = 64

    # ═══ MoE 参数 ═══
    expert_ff_dim: int = 1024
    num_experts: int = 8
    num_active_experts: int = 2
    moe_aux_loss_weight: float = 0.02
    moe_capacity_factor: float = 1.25  # MoE 路由容量因子

    # ═══ V2/V3 扩展参数 ═══
    graph_neighbors: int = 16
    graph_iterations: int = 2
    num_task_modes: int = 4
    num_counterfactuals: int = 4
    num_agents: int = 4
    use_gradient_checkpointing: bool = False
    use_mixture_of_depths: bool = False
    mod_capacity: float = 0.5
    causal_top_k_ratio: float = 0.25  # 因果推理干预注意力的 top-k 比例

    # ═══ 适配器参数（新增：用于开源模型加载） ═══
    model_type: str = "cortexnet"  # 源模型类型标识
    source_model_path: Optional[str] = None  # HuggingFace 模型路径
    auto_calibrate: bool = True  # 是否在加载后自动校准
    intermediate_size: int = 0  # 源模型中间层维度（如有）
    num_kv_heads: int = 0  # GQA 中的 KV 头数（0 = 与 num_heads 相同）
    use_qk_norm: bool = False  # 注意力中是否启用 per-head Q/K RMSNorm（Qwen2/3）
    rope_scaling: Optional[dict] = None  # RoPE 缩放配置
    tie_word_embeddings: bool = True  # 是否绑定嵌入和输出权重
    compatibility_mode: bool = False  # 兼容大模型的轻量 V3 路径
    lite: bool = True  # Lite 模式：仅 SSM+Attention+Memory+FFN（参数减少 60-70%）
    ssm_decode_after: int = 0  # SSM 纯解码阈值（>0 时，超过此 token 数后仅用 SSM）
    expand_gqa_weights: bool = True  # 是否将 KV 投影扩展到全头（旧结构兼容）
    compat_ssm_rank: int = 256  # 兼容模式 SSM 低秩维度（控制增量参数量）
    fusion_long_context_threshold: int = 2048  # 长序列时启用 SSM 最低占比的阈值
    fusion_long_context_ssm_ratio: float = 0.35  # 长序列时 SSM 最低占比
    mapped_cache_enabled: bool = False  # 是否启用映射后权重缓存（可选，默认关闭）
    mapped_cache_dir: Optional[str] = None  # 映射缓存目录（None 使用默认目录）
    mapped_cache_force_refresh: bool = False  # 是否强制忽略缓存并重建
    mapped_cache_auto_enable_with_lazy: bool = True  # lazy_device_load 时自动启用映射缓存
    mapped_cache_fast_init_on_hit: bool = True  # 命中映射缓存时跳过额外自定义重初始化
    lazy_device_load: bool = False  # 是否启用惰性上设备（from_pretrained 快速返回）
    lazy_cpu_fallback: bool = True  # 惰性阶段是否允许 CPU 兜底推理
    lazy_background_warmup: bool = True  # 首次推理后是否后台预热到目标设备
    lazy_start_warmup_on_load: bool = True  # from_pretrained 返回前立即启动后台预热线程
    lazy_disable_on_cache_hit: bool = True  # 命中映射缓存时自动关闭 lazy，优先首 token 体验

    def __post_init__(self):
        # ═══ 默认值推导 ═══
        if self.num_kv_heads == 0:
            self.num_kv_heads = self.num_heads
        if self.intermediate_size == 0:
            self.intermediate_size = self.expert_ff_dim

        # ═══ 参数验证 ═══
        self._validate()

    def _validate(self):
        """全面的参数验证，确保配置合法且防止隐蔽错误。"""
        # 基础维度检查
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) 必须能被 num_heads ({self.num_heads}) 整除"
            )
        if self.num_kv_heads > self.num_heads:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) 不能超过 num_heads ({self.num_heads})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) 必须能被 num_kv_heads ({self.num_kv_heads}) 整除 (GQA 要求)"
            )

        # MoE 参数检查
        if self.num_active_experts > self.num_experts:
            raise ValueError(
                f"num_active_experts ({self.num_active_experts}) 不能超过 num_experts ({self.num_experts})"
            )
        if self.moe_capacity_factor <= 0:
            raise ValueError(
                f"moe_capacity_factor ({self.moe_capacity_factor}) 必须为正数"
            )

        # 注意力参数检查
        if not (0 < self.top_k_ratio <= 1.0):
            raise ValueError(
                f"top_k_ratio ({self.top_k_ratio}) 必须在 (0, 1.0] 范围内"
            )
        _valid_k_modes = {"ratio", "sqrt", "log", "fixed"}
        if self.attention_k_mode not in _valid_k_modes:
            raise ValueError(
                f"attention_k_mode ('{self.attention_k_mode}') 必须是 {_valid_k_modes} 之一"
            )

        # 因果推理参数检查
        if not (0 < self.causal_top_k_ratio <= 1.0):
            raise ValueError(
                f"causal_top_k_ratio ({self.causal_top_k_ratio}) 必须在 (0, 1.0] 范围内"
            )

        # 正数检查
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) 必须为正整数")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers ({self.num_layers}) 必须为正整数")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len ({self.max_seq_len}) 必须为正整数")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size ({self.vocab_size}) 必须为正整数")

        # 范围检查
        if not (0 <= self.dropout < 1.0):
            raise ValueError(
                f"dropout ({self.dropout}) 必须在 [0, 1.0) 范围内"
            )

        # 混合精度检查
        if self.fusion_long_context_ssm_ratio < 0 or self.fusion_long_context_ssm_ratio > 1:
            raise ValueError(
                f"fusion_long_context_ssm_ratio ({self.fusion_long_context_ssm_ratio}) 必须在 [0, 1] 范围内"
            )

        # 软警告
        if self.hidden_size < 64:
            logger.warning(f"hidden_size={self.hidden_size} 过小，可能影响模型表达能力")
        if self.num_experts > 1 and self.num_active_experts < 1:
            logger.warning("num_active_experts < 1，MoE 路由可能无效")

    @classmethod
    def from_dict(cls, d: dict) -> "CortexNetConfig":
        """从字典创建配置，忽略未知字段。"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class TrainingConfig:
    """训练超参数配置。

    Attributes:
        learning_rate: 学习率
        weight_decay: 权重衰减
        num_epochs: 训练轮数
        batch_size: 批大小
        gradient_accumulation_steps: 梯度累积步数
        max_grad_norm: 梯度裁剪最大范数
        warmup_steps: 学习率预热步数
        eval_interval: 评估间隔步数
        save_interval: 保存间隔步数
        mixed_precision: 混合精度类型 ("no", "fp16", "bf16")
    """

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    mixed_precision: str = "no"
    seed: int = 42
    log_interval: int = 10
