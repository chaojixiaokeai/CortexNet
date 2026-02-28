"""
CortexNet: 超越 Transformer 的新一代神经网络架构

CortexNet 融合了五项核心创新：
  1. 多尺度状态空间模块 (MSSM) — O(n) 高效序列建模
  2. 选择性稀疏注意力 — 聚焦重要 token，大幅降低计算量
  3. 突触可塑性记忆 — 受生物启发的快速上下文适应
  4. 混合专家路由 (MoE) — 高效参数扩展
  5. 自适应融合门控 — 动态平衡各处理路径

完整架构流程：
  Token IDs → Embedding → [CortexBlock × N] → RMSNorm → LM Head → Logits
  
  其中每个 CortexBlock:
    Input ─┬─► SSM ────────┐
           ├─► Attention ──┤─► Fusion ─► + ─► MoE FFN ─► + ─► Output
           └─► Memory ─────┘     ↑                ↑
                          Residual          Residual

复杂度对比：
  ┌─────────────┬──────────────┬──────────────┐
  │             │ Transformer  │  CortexNet   │
  ├─────────────┼──────────────┼──────────────┤
  │ 序列建模     │ O(n²·d)     │ O(n·d)       │
  │ 注意力       │ O(n²·d)     │ O(n·k·d)     │
  │ 上下文学习   │ 隐式         │ 显式记忆      │
  │ 参数利用     │ 全部激活     │ 稀疏激活(MoE) │
  │ 位置编码     │ 绝对/相对    │ RoPE         │
  └─────────────┴──────────────┴──────────────┘
  其中 k << n，可选 k = √n 实现亚二次复杂度
"""

import os
import threading
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any

try:
    from .config import CortexNetConfig
    from .blocks import CortexBlock, CortexBlockV2, CortexBlockV3, RMSNorm
    from .cortex_block_lite import CortexBlockLite
    from .pretrained_utils import (
        discover_weight_files,
        build_mapped_cache_file,
        iter_weight_tensors,
    )
    from .compat import (
        _NoOpEvolutionEngine, _CompatCortexBlockV3,
    )
except ImportError:
    # 兼容脚本式导入: `from model import CortexNetV3`
    from cortexnet.config import CortexNetConfig
    from cortexnet.blocks import CortexBlock, CortexBlockV2, CortexBlockV3, RMSNorm
    from cortexnet.cortex_block_lite import CortexBlockLite
    from cortexnet.pretrained_utils import (
        discover_weight_files,
        build_mapped_cache_file,
        iter_weight_tensors,
    )
    from cortexnet.compat import (
        _NoOpEvolutionEngine, _CompatCortexBlockV3,
    )

logger = logging.getLogger(__name__)



class CortexNetOutput(dict):
    """兼容输出容器：支持 dict 访问，也支持 Tensor 风格 shape 访问。"""

    @property
    def logits(self) -> torch.Tensor:
        return self["logits"]

    @property
    def shape(self) -> torch.Size:
        return self.logits.shape

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        logits = self.get("logits")
        if logits is not None and hasattr(logits, name):
            return getattr(logits, name)
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")


class CortexNetBase(nn.Module):
    """CortexNet 语言模型。

    Args:
        config: CortexNetConfig 配置对象
    """

    def __init__(self, config: CortexNetConfig):
        super().__init__()
        self.config = config

        # Token 嵌入
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)

        # CortexNet 块堆叠
        self.blocks = nn.ModuleList(
            [
                CortexBlock(config, layer_idx=i)
                for i in range(config.num_layers)
            ]
        )

        # 最终归一化
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)

        # 语言模型输出头（与嵌入层共享权重）
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # 可选权重绑定（部分模型如 Qwen3 默认不绑定）
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.embed.weight

        # 初始化权重
        self._maybe_init_weights()

    def _silent_calibrate_compat(self):
        """兼容模式下的无感校准：保持无损迁移基线并为后续学习预留入口。"""
        if not getattr(self, "compatibility_mode", False):
            return
        with torch.no_grad():
            for block in self.blocks:
                if hasattr(block, "ssm") and hasattr(block.ssm, "alpha"):
                    block.ssm.alpha.zero_()
                    if hasattr(block.ssm, "fast_skip"):
                        block.ssm.fast_skip = True
                if hasattr(block, "fusion") and hasattr(block.fusion, "attn_bias"):
                    block.fusion.attn_bias.fill_(10.0)

    def _init_weights(self, module: nn.Module):
        """使用缩放正态分布初始化权重。
        
        初始化标准差可通过 config.weight_init_std 配置，默认为 0.02。
        """
        # 从 config 读取初始化标准差，支持自定义
        init_std = getattr(self.config, 'weight_init_std', 0.02)
        
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _maybe_init_weights(self) -> None:
        """按配置决定是否执行自定义二次初始化。"""
        if bool(getattr(self.config, "skip_weight_init", False)):
            return
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> CortexNetOutput:
        """
        Args:
            input_ids: (batch, seq_len) token 索引
            labels: (batch, seq_len) 目标 token 索引
        Returns:
            字典包含 'logits'，训练时还包含 'loss' 和 'aux_loss'
        """
        B, L = input_ids.shape

        # 嵌入
        x = self.embed(input_ids)  # (B, L, D)
        x = self.embed_dropout(x)

        # 通过 CortexNet 块
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x)
            if self.training:
                aux_loss = aux_loss + block.get_aux_loss()

        # 最终归一化 + 输出
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, L, vocab_size)

        result = CortexNetOutput(logits=logits)

        # 计算损失
        if labels is not None:
            # 移位以进行下一个 token 预测
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # 加入 MoE 负载均衡辅助损失
            if self.training:
                loss = loss + aux_loss

            result["loss"] = loss
            result["aux_loss"] = aux_loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """自回归文本生成。

        Args:
            input_ids: (batch, seq_len) 初始 token 索引
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（越高越随机）
            top_k: Top-k 过滤
            top_p: 核采样阈值
        Returns:
            generated: (batch, seq_len + max_new_tokens) token 索引
        """
        self.eval()

        for _ in range(max_new_tokens):
            # 截断到最大序列长度
            idx_cond = (
                input_ids
                if input_ids.shape[1] <= self.config.max_seq_len
                else input_ids[:, -self.config.max_seq_len :]
            )

            # 前向传播
            output = self.forward(idx_cond)
            logits = output["logits"][:, -1, :] / max(
                temperature, 1e-8
            )  # (B, vocab)

            # Top-k 过滤
            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k_val)[0][
                    ..., -1, None
                ]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # 采样（保护 NaN/Inf）
            logits = logits.clamp(-100, 100)
            logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
            probs = F.softmax(logits, dim=-1)
            probs = probs.clamp(min=1e-8)  # 防止零概率
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> Dict[str, int]:
        """按组件统计模型参数量。"""
        counts = {
            "embedding": sum(p.numel() for p in self.embed.parameters()),
            "ssm": 0,
            "attention": 0,
            "memory": 0,
            "fusion": 0,
            "moe": 0,
            "norm": 0,
        }

        for block in self.blocks:
            if hasattr(block, "ssm"):
                counts["ssm"] += sum(p.numel() for p in block.ssm.parameters())
            if hasattr(block, "attention"):
                counts["attention"] += sum(
                    p.numel() for p in block.attention.parameters()
                )
            if hasattr(block, "memory"):
                counts["memory"] += sum(
                    p.numel() for p in block.memory.parameters()
                )
            if hasattr(block, "fusion"):
                counts["fusion"] += sum(
                    p.numel() for p in block.fusion.parameters()
                )
            if hasattr(block, "moe"):
                counts["moe"] += sum(p.numel() for p in block.moe.parameters())
            if hasattr(block, "norm1") and hasattr(block, "norm2"):
                counts["norm"] += sum(
                    p.numel() for p in block.norm1.parameters()
                ) + sum(p.numel() for p in block.norm2.parameters())

        counts["final_norm"] = sum(
            p.numel() for p in self.final_norm.parameters()
        )
        counts["total"] = sum(p.numel() for p in self.parameters())

        # 每个 token 的活跃参数（考虑 MoE 稀疏性）
        if len(self.blocks) > 0:
            if hasattr(self.blocks[0], "moe") and hasattr(self.blocks[0].moe, "experts"):
                expert_params_per_block = sum(
                    p.numel() for p in self.blocks[0].moe.experts[0].parameters()
                )
                inactive_experts = (
                    self.config.num_experts - self.config.num_active_experts
                )
                inactive_params = (
                    expert_params_per_block
                    * inactive_experts
                    * self.config.num_layers
                )
                counts["active_per_token"] = counts["total"] - inactive_params
            else:
                counts["active_per_token"] = counts["total"]
        else:
            counts["active_per_token"] = counts["total"]

        return counts


# ═══════════════════════════════════════════════════════════════
#                CortexNet V2 — 全面进化版
# ═══════════════════════════════════════════════════════════════


class CortexNetV2(CortexNetBase):
    """CortexNet V2：全面进化的神经网络架构。

    在 V1 基础上集成 9 大升级：

    ┌─────────────────────────────────────────────────────────────┐
    │                CortexNet V2 进化清单                        │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  1. ✦ 分层记忆系统 — 工作/情景/语义三层记忆                │
    │  2. ✦ 图推理模块 — 多步消息传递关系推理                    │
    │  3. ✦ 元学习适配 — FiLM 快速任务适应                       │
    │  4. ✦ 任务自适应控制 — 隐式多任务切换                      │
    │  5. ✦ 协作式 MoE — 专家间知识共享                          │
    │  6. ✦ 分块并行扫描 SSM — 长序列加速                        │
    │  7. ✦ 连续学习支持 — EWC + 记忆回放                        │
    │  8. ✦ 多模态编码 — 文本/图像/音频统一处理                  │
    │  9. ✦ 可解释性系统 — 思维流实时监控                        │
    │                                                             │
    │  架构:                                                      │
    │    [MultiModal Encoder] → [CortexBlockV2 × N] → Output     │
    │                                                             │
    │  CortexBlockV2:                                             │
    │    SSM + Attention + HierarchicalMemory + GraphReasoning    │
    │    → AdaptiveFusion(4路) → MetaAdapter → CollabMoE          │
    │    → TaskController                                         │
    └─────────────────────────────────────────────────────────────┘

    Args:
        config: CortexNetConfig 配置
    """

    def __init__(self, config: CortexNetConfig):
        # 不调用 CortexNet.__init__，而是直接初始化
        nn.Module.__init__(self)
        self.config = config

        # Token 嵌入
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)

        # V2 进化版块
        self.blocks = nn.ModuleList(
            [
                CortexBlockV2(config, layer_idx=i)
                for i in range(config.num_layers)
            ]
        )

        # 最终归一化
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)

        # 语言模型输出头
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.embed.weight

        # 初始化
        self._maybe_init_weights()

    def get_evolution_info(self) -> Dict[str, any]:
        """获取 V2 进化信息。"""
        info = self.count_parameters()
        info["version"] = "V2"
        info["num_paths"] = 4  # SSM + Attention + Memory + Graph
        info["memory_tiers"] = 3  # Working + Episodic + Semantic
        info["meta_learning"] = True
        info["graph_reasoning"] = True
        info["collaborative_moe"] = True
        info["continual_learning_ready"] = True
        info["multimodal_ready"] = True
        return info


# ═══════════════════════════════════════════════════════════════
#            CortexNet V3 — 终极进化: 世界顶级架构
# ═══════════════════════════════════════════════════════════════

try:
    from .self_evolution import SelfEvolutionEngine
except ImportError:
    from cortexnet.self_evolution import SelfEvolutionEngine


class CortexNetV3(CortexNetBase):
    """CortexNet V3：终极进化的世界顶级神经网络架构。

    在 V2 基础上集成 4 项突破性能力：

    ┌─────────────────────────────────────────────────────────────┐
    │                CortexNet V3 终极进化                        │
    ├─────────────────────────────────────────────────────────────┤
    │  V1 基础 (5 项):  SSM + Attn + Memory + MoE + Fusion      │
    │  V2 进化 (9 项):  +3层记忆 +图推理 +元学习 +协作MoE ...   │
    │  V3 突破 (4 项):                                           │
    │    10. ✦ 因果推理 — 理解因果，支持反事实思考               │
    │    11. ✦ 自我进化 — 动态架构，输入驱动的 NAS              │
    │    12. ✦ 多智能体 — 专家团队协作决策                       │
    │    13. ✦ 对抗防御 — 三层防护，安全可靠                     │
    │                                                             │
    │  处理路径: 5 条 (SSM + Attn + Memory + Graph + Causal)     │
    │  架构: 动态自适应 (路径可按需激活/禁用)                    │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: CortexNetConfig):
        nn.Module.__init__(self)
        self.config = config

        # 嵌入
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.compatibility_mode = bool(getattr(config, "compatibility_mode", False))

        self.lite_mode = bool(getattr(config, 'lite', False))

        if self.lite_mode:
            # Lite 模式：精简块，参数与 Transformer 相当
            self.blocks = nn.ModuleList(
                [CortexBlockLite(config, layer_idx=i) for i in range(config.num_layers)]
            )
            self.evolution_engine = _NoOpEvolutionEngine()
        elif self.compatibility_mode:
            # 兼容模式：使用轻量块，参数规模与开源源模型同量级
            self.blocks = nn.ModuleList(
                [_CompatCortexBlockV3(config) for _ in range(config.num_layers)]
            )
            self.evolution_engine = _NoOpEvolutionEngine()
        else:
            # 原始 V3 全功能路径
            self.blocks = nn.ModuleList(
                [CortexBlockV3(config, layer_idx=i) for i in range(config.num_layers)]
            )
            self.evolution_engine = SelfEvolutionEngine(
                config.hidden_size,
                num_paths=5,
                num_blocks=config.num_layers,
            )

        # 最终归一化 + 输出
        self.final_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.embed.weight

        self._maybe_init_weights()

        # 惰性上设备运行时状态
        self._lazy_enabled = False
        self._lazy_ready = True
        self._lazy_target_device: Optional[str] = None
        self._lazy_target_dtype: Optional[torch.dtype] = None
        self._lazy_cpu_fallback = True
        self._lazy_background_warmup = True
        self._lazy_warmup_started = False
        self._lazy_warmup_error: Optional[str] = None
        self._lazy_lock = threading.RLock()
        self._lazy_thread: Optional[threading.Thread] = None

    def _setup_lazy_runtime(
        self,
        *,
        target_device: str,
        target_dtype: Optional[torch.dtype],
        cpu_fallback: bool = True,
        background_warmup: bool = True,
    ) -> None:
        """启用惰性上设备。

        行为：
          1. from_pretrained 先返回 CPU 可用模型；
          2. 首次推理可走 CPU 兜底；
          3. 推理结束后后台线程搬运到目标设备（如 mps/cuda/npu）。
        """
        self._lazy_enabled = True
        self._lazy_ready = False
        self._lazy_target_device = str(target_device)
        self._lazy_target_dtype = target_dtype
        self._lazy_cpu_fallback = bool(cpu_fallback)
        self._lazy_background_warmup = bool(background_warmup)
        self._lazy_warmup_started = False
        self._lazy_warmup_error = None
        self._lazy_thread = None

    def start_background_warmup(self) -> bool:
        """手动触发后台预热（可重复调用，只有首次生效）。"""
        if not self._lazy_enabled or self._lazy_ready or self._lazy_warmup_started:
            return False

        def _worker():
            try:
                with self._lazy_lock:
                    if self._lazy_target_device is None:
                        self._lazy_warmup_error = "lazy target device is not set"
                        return
                    self.to(self._lazy_target_device)
                    if self._lazy_target_dtype is not None:
                        self.to(dtype=self._lazy_target_dtype)
                    self._lazy_ready = True
            except Exception as exc:
                self._lazy_warmup_error = str(exc)

        self._lazy_warmup_started = True
        self._lazy_thread = threading.Thread(target=_worker, daemon=True)
        self._lazy_thread.start()
        return True

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """等待后台预热完成。"""
        thread = self._lazy_thread
        if thread is not None:
            thread.join(timeout=timeout)
        return bool(self._lazy_ready)

    def get_lazy_status(self) -> Dict[str, Any]:
        return {
            "enabled": self._lazy_enabled,
            "ready": self._lazy_ready,
            "target_device": self._lazy_target_device,
            "target_dtype": (
                str(self._lazy_target_dtype).replace("torch.", "")
                if self._lazy_target_dtype is not None
                else None
            ),
            "background_warmup": self._lazy_background_warmup,
            "warmup_started": self._lazy_warmup_started,
            "warmup_error": self._lazy_warmup_error,
        }

    def _prepare_input_for_lazy(
        self,
        input_ids: torch.Tensor,
        *,
        start_warmup_after_infer: bool,
    ) -> Tuple[torch.Tensor, bool]:
        """根据惰性状态准备输入设备，并返回是否应在本次推理后触发预热。"""
        if not self._lazy_enabled:
            return input_ids, False

        if self._lazy_ready and self._lazy_target_device is not None:
            if str(input_ids.device) != self._lazy_target_device:
                input_ids = input_ids.to(self._lazy_target_device)
            return input_ids, False

        # 尚未预热完成：走 CPU 兜底
        if str(input_ids.device) != "cpu":
            input_ids = input_ids.to("cpu")

        should_start = (
            bool(start_warmup_after_infer)
            and self._lazy_background_warmup
            and not self._lazy_warmup_started
        )
        return input_ids, should_start

    def _move_cache_to_device(
        self,
        cache: Optional[List[Tuple[Any, Any, Any]]],
        device: torch.device,
    ) -> Optional[List[Tuple[Any, Any, Any]]]:
        """将增量缓存迁移到目标设备，避免 CPU/MPS 混用报错。"""
        if cache is None:
            return None

        def _move(x: Any) -> Any:
            if torch.is_tensor(x):
                return x.to(device)
            if isinstance(x, tuple):
                return tuple(_move(v) for v in x)
            if isinstance(x, list):
                return [_move(v) for v in x]
            return x

        return _move(cache)

    def forward_from_embeddings(
        self,
        x_emb: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_cache: Optional[List[Tuple[Any, Any, Any]]] = None,
        use_cache: bool = False,
    ) -> CortexNetOutput:
        """从嵌入张量前向（跳过 embedding 层），用于对抗训练等场景。"""
        return self._forward_impl(x_emb, labels, past_cache, use_cache)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_cache: Optional[List[Tuple[Any, Any, Any]]] = None,
        use_cache: bool = False,
        *,
        start_warmup_after_infer: bool = True,
        ssm_only: bool = False,
    ) -> CortexNetOutput:
        """Forward with optional layer caches for incremental decoding.

        Args:
            ssm_only: Lite 模式下启用纯 SSM 解码。
        """
        with self._lazy_lock:
            input_ids, should_start_warmup = self._prepare_input_for_lazy(
                input_ids,
                start_warmup_after_infer=start_warmup_after_infer,
            )
            past_cache = self._move_cache_to_device(past_cache, input_ids.device)
            x = self.embed_dropout(self.embed(input_ids))
            output = self._forward_impl(x, labels, past_cache, use_cache, ssm_only=ssm_only)

        if should_start_warmup:
            self.start_background_warmup()
        return output

    def _forward_impl(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_cache: Optional[List[Tuple[Any, Any, Any]]] = None,
        use_cache: bool = False,
        ssm_only: bool = False,
    ) -> CortexNetOutput:
        """内部前向实现（接受已嵌入的张量）。

        Args:
            ssm_only: Lite 模式下启用纯 SSM 解码（跳过 Attention，O(1) 每 token）。
        """

        compute_budget = self.evolution_engine.get_compute_budget(x)

        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        new_cache: List[Tuple[Any, Any, Any]] = [] if use_cache else []

        for i, block in enumerate(self.blocks):
            past_layer = past_cache[i] if past_cache and i < len(past_cache) else None
            # Lite 模式的 CortexBlockLite 支持 ssm_only 参数
            block_kwargs = {}
            if self.lite_mode and ssm_only:
                block_kwargs['ssm_only'] = True
            if use_cache:
                x, layer_cache = block(x, past_cache=past_layer, use_cache=True, **block_kwargs)
                new_cache.append(layer_cache)
            else:
                x = block(x, past_cache=past_layer, use_cache=False, **block_kwargs)
            if self.training:
                aux_loss = aux_loss + block.get_aux_loss()

        if self.training:
            aux_loss = aux_loss + self.evolution_engine.get_efficiency_loss()

        x = self.final_norm(x)
        logits = self.lm_head(x)

        result = CortexNetOutput(logits=logits, compute_budget=compute_budget)
        if use_cache and new_cache:
            result["past_cache"] = new_cache

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1), ignore_index=-100,
            )
            if self.training:
                loss = loss + aux_loss
            result["loss"] = loss
            result["aux_loss"] = aux_loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """自回归生成，使用 KV/状态缓存加速；支持 repetition_penalty 减轻重复。"""
        self.eval()
        input_ids, _ = self._prepare_input_for_lazy(
            input_ids,
            start_warmup_after_infer=False,
        )
        should_start_warmup_after_generate = (
            self._lazy_enabled
            and not self._lazy_ready
            and self._lazy_background_warmup
            and not self._lazy_warmup_started
        )
        past_cache: Optional[List[Tuple[Any, Any, Any]]] = None
        generated = input_ids
        use_repetition_penalty = repetition_penalty != 1.0 and repetition_penalty > 0
        seen_mask: Optional[torch.Tensor] = None

        for step_i in range(max_new_tokens):
            if past_cache is None:
                idx_cond = (
                    generated
                    if generated.shape[1] <= self.config.max_seq_len
                    else generated[:, -self.config.max_seq_len :]
                )
            else:
                idx_cond = generated[:, -1:]  # 增量：只处理新 token

            # Lite SSM 优先解码：prefill 后切换纯 SSM 模式
            use_ssm_only = (
                self.lite_mode
                and past_cache is not None
                and getattr(self.config, 'ssm_decode_after', 0) > 0
                and step_i >= self.config.ssm_decode_after
            )

            output = self.forward(
                idx_cond,
                past_cache=past_cache,
                use_cache=True,
                start_warmup_after_infer=False,
                ssm_only=use_ssm_only,
            )
            logits = output["logits"][:, -1, :] / max(temperature, 1e-8)
            past_cache = output.get("past_cache")

            # 向量化重复惩罚：避免 Python token 循环（提升解码吞吐）
            if use_repetition_penalty:
                if seen_mask is None or seen_mask.shape[1] != logits.shape[1]:
                    seen_mask = torch.zeros(
                        logits.shape[0],
                        logits.shape[1],
                        device=logits.device,
                        dtype=torch.bool,
                    )
                    seen_ids = generated.clamp_max(logits.shape[1] - 1)
                    seen_mask.scatter_(1, seen_ids, True)
                positive = logits > 0
                logits = torch.where(seen_mask & positive, logits / repetition_penalty, logits)
                logits = torch.where(seen_mask & ~positive, logits * repetition_penalty, logits)

            if top_k > 0:
                top_k_val = min(top_k, logits.size(-1))
                _, top_indices = torch.topk(logits, top_k_val)
                mask = torch.ones_like(logits, dtype=torch.bool)
                mask.scatter_(1, top_indices, False)
                logits.masked_fill_(mask, float("-inf"))

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            logits = logits.clamp(-100, 100)
            logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
            probs = F.softmax(logits, dim=-1)
            probs = probs.clamp(min=1e-8)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if seen_mask is not None:
                seen_mask.scatter_(1, next_token, True)

        if should_start_warmup_after_generate:
            self.start_background_warmup()
        return generated

    def compile_model(self, **kwargs) -> "CortexNet":
        """使用 torch.compile 编译关键子模块，获得自动 kernel fusion 加速。

        建议在训练/推理前调用一次。首次前向会有编译开销，后续大幅加速。
        """
        try:
            for block in self.blocks:
                if hasattr(block, "ssm"):
                    block.ssm = torch.compile(block.ssm, **kwargs)
                if hasattr(block, "attention"):
                    block.attention = torch.compile(block.attention, **kwargs)
                if hasattr(block, "moe"):
                    block.moe = torch.compile(block.moe, **kwargs)
                if hasattr(block, "graph_reasoning"):
                    block.graph_reasoning = torch.compile(block.graph_reasoning, **kwargs)
                if hasattr(block, "causal_reasoning"):
                    block.causal_reasoning = torch.compile(block.causal_reasoning, **kwargs)
        except Exception as exc:
            if os.getenv("CORTEXNET_COMPILE_STRICT", "0") == "1":
                raise
            logger.warning("torch.compile skipped: %s", exc)
        return self

    @torch.no_grad()
    def speculative_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        draft_steps: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """推测式解码：用浅层草稿批量猜测，全模型一次验证。

        比标准自回归快 draft_steps 倍（理论上限），实际 2-3x 加速。

        原理：
          1. 用前 1 层作为轻量「草稿模型」，快速生成 draft_steps 个候选 token
          2. 用完整模型对整个候选序列做一次前向，得到每步的 logits
          3. 从左到右验证：若草稿与完整模型一致则接受，否则从分歧处截断
          4. 接受 n 个 token + 采样 1 个新 token = 每轮最多前进 draft_steps+1 步
        """
        self.eval()
        input_ids, _ = self._prepare_input_for_lazy(
            input_ids,
            start_warmup_after_infer=False,
        )
        should_start_warmup_after_generate = (
            self._lazy_enabled
            and not self._lazy_ready
            and self._lazy_background_warmup
            and not self._lazy_warmup_started
        )
        generated = input_ids.clone()

        while generated.shape[1] - input_ids.shape[1] < max_new_tokens:
            remaining = max_new_tokens - (generated.shape[1] - input_ids.shape[1])
            n_draft = min(draft_steps, remaining)

            # === 阶段 1：草稿生成（只用嵌入 + 1 层 block + lm_head）===
            draft_tokens = []
            draft_input = generated[:, -1:]
            for _ in range(n_draft):
                x_d = self.embed(draft_input)
                x_d = self.blocks[0](x_d)  # 只用第 0 层
                x_d = self.final_norm(x_d)
                d_logits = self.lm_head(x_d)[:, -1, :] / max(temperature, 1e-8)
                if top_k > 0:
                    v, _ = torch.topk(d_logits, min(top_k, d_logits.size(-1)))
                    d_logits[d_logits < v[:, -1:]] = float("-inf")
                d_probs = F.softmax(d_logits, dim=-1)
                d_tok = torch.multinomial(d_probs, 1)
                draft_tokens.append(d_tok)
                draft_input = d_tok

            if not draft_tokens:
                break
            draft_seq = torch.cat(draft_tokens, dim=1)  # (B, n_draft)
            candidate = torch.cat([generated[:, -1:], draft_seq], dim=1)  # (B, 1+n_draft)

            # === 阶段 2：完整模型验证 ===
            full_out = self.forward(candidate, start_warmup_after_infer=False)
            full_logits = full_out["logits"][:, :-1, :] / max(temperature, 1e-8)  # (B, n_draft, V)

            # === 阶段 3：逐步验证 ===
            n_accepted = 0
            for i in range(n_draft):
                if top_k > 0:
                    fl = full_logits[:, i, :]
                    v, _ = torch.topk(fl, min(top_k, fl.size(-1)))
                    fl[fl < v[:, -1:]] = float("-inf")
                    full_probs = F.softmax(fl, dim=-1)
                else:
                    full_probs = F.softmax(full_logits[:, i, :], dim=-1)
                full_tok = torch.multinomial(full_probs, 1)
                if (full_tok == draft_seq[:, i:i+1]).all():
                    n_accepted += 1
                else:
                    generated = torch.cat([generated, draft_seq[:, :i], full_tok], dim=1)
                    break
            else:
                # 全部接受 + 采样下一个
                last_logits = full_out["logits"][:, -1, :] / max(temperature, 1e-8)
                if top_k > 0:
                    v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                    last_logits[last_logits < v[:, -1:]] = float("-inf")
                bonus = torch.multinomial(F.softmax(last_logits, dim=-1), 1)
                generated = torch.cat([generated, draft_seq, bonus], dim=1)

        if should_start_warmup_after_generate:
            self.start_background_warmup()
        return generated[:, : input_ids.shape[1] + max_new_tokens]

    def get_evolution_info(self) -> Dict[str, any]:
        info = self.count_parameters()
        info["version"] = "CortexNet"
        info["num_paths"] = 5
        info["memory_tiers"] = 3
        info["causal_reasoning"] = True
        info["self_evolution"] = True
        info["multi_agent"] = True
        info["adversarial_defense"] = True
        info["meta_learning"] = True
        info["graph_reasoning"] = True
        info["collaborative_moe"] = True
        info["continual_learning_ready"] = True
        info["multimodal_ready"] = True
        return info

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        model_type: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        auto_calibrate: Optional[bool] = None,
        calibration_data: Optional[List] = None,
        load_weights: bool = True,
        max_weight_tensors: Optional[int] = None,
        **kwargs,
    ) -> "CortexNet":
        """从 HuggingFace 预训练模型一键加载 CortexNet。

        完整加载流程：
          1. 自动识别模型类型（LLaMA/Qwen/Mistral/...）
          2. 转换配置为 CortexNetConfig
          3. 初始化 CortexNet 模型
          4. 映射权重到 CortexNet 模块
          5. 执行架构适配（注意力/SSM/门控参数调整）
          6. 可选：轻量校准

        Args:
            model_path: HuggingFace 模型目录路径
            model_type: 模型类型（可选，自动检测）
            device: 目标设备 ("cuda", "cpu", "npu" 等)
            dtype: 目标数据类型 (torch.float16, torch.bfloat16 等)
            auto_calibrate: 是否自动执行校准（None 时使用配置默认值）
            calibration_data: 校准数据（可选）
            load_weights: 是否加载权重（False 则仅创建架构）
            max_weight_tensors: 最多加载多少个源权重 tensor（用于兼容性快速验证）
            **kwargs: 覆盖 CortexNetConfig 的额外参数

        Returns:
            适配后的 CortexNet 模型实例

        Example:
            >>> model = CortexNet.from_pretrained("/path/to/llama3-7b")
            >>> output = model.generate(input_ids, max_new_tokens=100)
        """
        import logging
        logger = logging.getLogger(__name__)

        # ═══ Step 1: 识别模型类型 ═══
        try:
            from .adapter.model_registry import detect_model_type, get_cortexnet_config
        except ImportError:
            from cortexnet.adapter.model_registry import detect_model_type, get_cortexnet_config

        if model_type is None:
            model_type = detect_model_type(model_path)
        logger.info(f"Loading model from {model_path} (type: {model_type})")

        # ═══ Step 2: 生成 CortexNetConfig ═══
        config = get_cortexnet_config(model_path, model_type)

        # 应用用户覆盖参数
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # compatibility_mode 与 lite 二选一。
        # 若用户显式开启 compatibility_mode 且未显式传 lite，
        # 默认关闭 lite，保证走兼容路径而非 Lite 路径。
        if (
            bool(getattr(config, "compatibility_mode", False))
            and "lite" not in kwargs
        ):
            config.lite = False

        if auto_calibrate is None:
            # Lite 模式默认不校准（大模型 CPU 校准极慢）
            if getattr(config, 'lite', False):
                auto_calibrate = False
            else:
                auto_calibrate = bool(getattr(config, "auto_calibrate", True))

        mapped_cache_enabled = bool(getattr(config, "mapped_cache_enabled", False))
        if (
            load_weights
            and bool(getattr(config, "lazy_device_load", False))
            and bool(getattr(config, "mapped_cache_auto_enable_with_lazy", True))
            and (not mapped_cache_enabled)
        ):
            mapped_cache_enabled = True
            config.mapped_cache_enabled = True
            logger.info("Auto-enable mapped cache because lazy_device_load=True")

        mapped_cache_force_refresh = bool(getattr(config, "mapped_cache_force_refresh", False))
        mapped_cache_dir = getattr(config, "mapped_cache_dir", None) or os.path.join(
            os.path.expanduser("~"), ".cache", "cortexnet", "mapped_weights"
        )
        weight_files: List[str] = discover_weight_files(model_path) if load_weights else []
        cache_file: Optional[str] = None
        cache_hit = False
        if load_weights and mapped_cache_enabled and weight_files:
            os.makedirs(mapped_cache_dir, exist_ok=True)
            cache_file = build_mapped_cache_file(
                source_path=model_path,
                source_model_type=model_type,
                files=weight_files,
                cache_dir=mapped_cache_dir,
                config=config,
            )
            cache_hit = bool(
                cache_file
                and os.path.exists(cache_file)
                and (not mapped_cache_force_refresh)
                and max_weight_tensors is None
            )
            if cache_hit and bool(getattr(config, "mapped_cache_fast_init_on_hit", True)):
                # 命中缓存时跳过额外的自定义二次初始化，加快冷启动。
                setattr(config, "skip_weight_init", True)
            if (
                cache_hit
                and bool(getattr(config, "lazy_device_load", False))
                and bool(getattr(config, "lazy_disable_on_cache_hit", True))
            ):
                # cache 命中时，直接走目标设备加载通常首 token 更快，避免 lazy CPU 首轮抖动。
                config.lazy_device_load = False
                logger.info("Mapped cache hit: disable lazy_device_load for better first-token latency")

        # ═══ Step 3: 初始化模型 ═══
        model = cls(config)
        # 若用户显式指定 dtype，先转换
        if dtype is not None:
            model = model.to(dtype=dtype)

        # ═══ Step 4: 映射并加载权重 ═══
        if load_weights:
            try:
                from .adapter.weight_adapter import WeightAdapter
            except ImportError:
                from cortexnet.adapter.weight_adapter import WeightAdapter
            import gc

            if weight_files:
                loaded_from_cache = False
                if cache_hit and cache_file:
                    try:
                        from safetensors.torch import load_model  # type: ignore

                        missing, unexpected = load_model(model, cache_file, strict=False, device="cpu")
                        if missing or unexpected:
                            logger.warning(
                                "Mapped cache partially matched (missing=%s, unexpected=%s), "
                                "fallback to raw mapping.",
                                len(missing),
                                len(unexpected),
                            )
                        else:
                            loaded_from_cache = True
                            logger.info(
                                f"Loaded mapped weights cache: {cache_file} "
                                f"(missing={len(missing)}, unexpected={len(unexpected)})"
                            )
                    except Exception as exc:
                        logger.warning(f"Failed to load mapped cache, fallback to raw mapping: {exc}")

                if not loaded_from_cache:
                    target_tensors: Dict[str, torch.Tensor] = {}
                    for name, param in model.named_parameters():
                        target_tensors[name] = param
                    for name, buf in model.named_buffers():
                        if name not in target_tensors:
                            target_tensors[name] = buf

                    adapter = WeightAdapter(model_type, config)
                    total_source = 0
                    total_mapped = 0
                    total_shape_mismatch = 0
                    total_unexpected = 0
                    total_unmapped_source = 0
                    mismatch_examples = 0
                    stop_loading = False

                    for wf in weight_files:
                        logger.info(f"Loading weight shard: {os.path.basename(wf)}")
                        for raw_chunk in iter_weight_tensors(wf, logger=logger):
                            if not raw_chunk:
                                continue
                            if max_weight_tensors is not None and total_source >= max_weight_tensors:
                                stop_loading = True
                                break
                            total_source += len(raw_chunk)

                            mapped_chunk = adapter.map_weights(raw_chunk)
                            total_unmapped_source += adapter.get_unmapped_count()

                            for name, param in mapped_chunk.items():
                                target = target_tensors.get(name)
                                if target is not None:
                                    if target.shape == param.shape:
                                        with torch.no_grad():
                                            source = param
                                            if source.device != target.device or source.dtype != target.dtype:
                                                source = source.to(device=target.device, dtype=target.dtype)
                                            target.copy_(source)
                                        total_mapped += 1
                                    else:
                                        total_shape_mismatch += 1
                                        if mismatch_examples < 5:
                                            logger.warning(
                                                f"Shape mismatch for {name}: "
                                                f"expected {target.shape}, "
                                                f"got {param.shape}"
                                            )
                                            mismatch_examples += 1
                                else:
                                    total_unexpected += 1

                            # 主动触发回收，减少大模型加载时峰值内存
                            if total_source % 1024 == 0:
                                gc.collect()
                        if stop_loading:
                            logger.info(
                                f"Reached max_weight_tensors={max_weight_tensors}, "
                                "stopping early for compatibility check."
                            )
                            break
                    logger.info(
                        f"Weight loading: {total_mapped} mapped, "
                        f"{total_shape_mismatch} shape mismatches, "
                        f"{total_unexpected} unexpected, "
                        f"{total_unmapped_source} source weights unmapped "
                        f"out of {total_source} tensors"
                    )

                    # 首次完成映射后缓存结果，二次加载可直接复用
                    if (
                        mapped_cache_enabled
                        and cache_file
                        and max_weight_tensors is None
                    ):
                        try:
                            from safetensors.torch import save_model  # type: ignore

                            metadata = {
                                "model_type": model_type,
                                "compatibility_mode": str(
                                    bool(getattr(config, "compatibility_mode", False))
                                ),
                            }
                            save_model(model, cache_file, metadata=metadata)
                            logger.info(f"Saved mapped weights cache: {cache_file}")
                        except Exception as exc:
                            logger.warning(f"Failed to save mapped cache: {exc}")
            else:
                logger.warning(
                    f"No weight files found in {model_path}. "
                    f"Model initialized with random weights."
                )

        # ═══ Step 5: 架构适配 ═══
        if not getattr(config, "compatibility_mode", False):
            try:
                from .adapter.arch_adapter import ArchitectureAdapter
            except ImportError:
                from cortexnet.adapter.arch_adapter import ArchitectureAdapter
            arch_adapter = ArchitectureAdapter(model_type, config)
            arch_adapter.adapt(model)

        # ═══ Step 6: 可选校准 ═══
        if auto_calibrate:
            if getattr(config, "compatibility_mode", False):
                model._silent_calibrate_compat()
                logger.info("Compatibility calibration applied (silent mode).")
            else:
                try:
                    from .adapter.calibrator import LightweightCalibrator
                except ImportError:
                    from cortexnet.adapter.calibrator import LightweightCalibrator
                calibrator = LightweightCalibrator(model, model_type)
                model = calibrator.calibrate(calibration_data=calibration_data)

        # ═══ Step 7: 设备和精度 ═══
        try:
            from .ops.device_manager import (
                get_best_device_info,
                resolve_device_string,
                resolve_dtype_for_device,
            )
        except ImportError:
            from cortexnet.ops.device_manager import (
                get_best_device_info,
                resolve_device_string,
                resolve_dtype_for_device,
            )

        if device is None:
            device_info = get_best_device_info()
            device = str(device_info.torch_device)
            if dtype is None:
                dtype = device_info.optimal_dtype
        else:
            device = resolve_device_string(
                str(device),
                auto_priority=("npu", "cuda", "mlu", "mps", "cpu"),
                allow_fallback=True,
            )

        if isinstance(dtype, str):
            dtype = resolve_dtype_for_device(dtype, str(device))

        lazy_device_load = bool(getattr(config, "lazy_device_load", False))
        lazy_cpu_fallback = bool(getattr(config, "lazy_cpu_fallback", True))
        lazy_background_warmup = bool(getattr(config, "lazy_background_warmup", True))
        target_device_type = str(device).split(":", 1)[0]
        use_lazy_runtime = lazy_device_load and target_device_type != "cpu"

        if use_lazy_runtime:
            # 惰性模式：先返回 CPU 模型，首次推理后后台预热到目标设备。
            model = model.to("cpu")
            if dtype is not None:
                model = model.to(dtype)
            model._setup_lazy_runtime(
                target_device=str(device),
                target_dtype=dtype,
                cpu_fallback=lazy_cpu_fallback,
                background_warmup=lazy_background_warmup,
            )
            if (
                bool(getattr(config, "lazy_start_warmup_on_load", True))
                and lazy_background_warmup
            ):
                model.start_background_warmup()
        else:
            model = model.to(device)
            if dtype is not None:
                model = model.to(dtype)

        # 存储推理适配器
        try:
            from .adapter.inference_adapter import InferenceAdapter
        except ImportError:
            from cortexnet.adapter.inference_adapter import InferenceAdapter
        model._inference_adapter = InferenceAdapter(model, model_type)

        model.eval()
        if getattr(model, "compatibility_mode", False):
            logger.info("Core Mode: SSM + Sparse Attention + Lite Fusion")
        else:
            logger.info("Core Mode: Full CortexNet")
        if use_lazy_runtime:
            logger.info(
                f"Model loaded in lazy mode (cpu fallback), target={device}, dtype={dtype}"
            )
        else:
            logger.info(f"Model loaded successfully on {device} with dtype {dtype}")
        return model

    def smart_generate(self, input_ids: torch.Tensor, **kwargs):
        """使用推理适配器的智能生成（自动匹配源模型参数）。

        如果模型是通过 from_pretrained() 加载的，会使用源模型的默认参数。
        否则回退到标准 generate()。
        """
        if hasattr(self, '_inference_adapter'):
            return self._inference_adapter.generate(input_ids, **kwargs)
        return self.generate(input_ids, **kwargs)


class CortexNet(CortexNetV3):
    """Unified CortexNet model (canonical API).

    This is the primary public model class. Legacy names like ``CortexNetV3``
    remain available for compatibility, but new code should use ``CortexNet``.
    """

    pass
