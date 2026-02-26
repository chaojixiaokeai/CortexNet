"""CortexNet public package API."""

from __future__ import annotations

from .config import CortexNetConfig, TrainingConfig
from .model import CortexNet, CortexNetBase, CortexNetV2 as _CortexNetV2, CortexNetV3 as _CortexNetV3
from .blocks import CortexBlock, CortexBlockV2 as _CortexBlockV2, CortexBlockV3 as _CortexBlockV3, RMSNorm, AdaptiveFusionGate
from .ssm import MultiScaleSSM
from .attention import SelectiveSparseAttention
from .memory import SynapticMemory
from .routing import MixtureOfExperts, ExpertFFN, CollaborativeMoE
from .cache import CortexNetCache
from .transformer_baseline import TransformerLM
from .quantization import quantize_dynamic, QuantizationWrapper
from .hierarchical_memory import (
    HierarchicalMemorySystem,
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    MemoryController,
)
from .graph_reasoning import GraphReasoningModule
from .meta_learning import MetaLearningAdapter, TaskAdaptiveController, ContextEncoder
from .multimodal import MultiModalEncoder, PatchEmbedding, AudioEmbedding, CrossModalFusion
from .continual_learning import (
    ElasticWeightConsolidation,
    ProgressiveMemoryReplay,
    ContinualLearningManager,
)
from .interpretability import ThoughtFlowMonitor
from .causal_reasoning import CausalReasoningModule, InterventionalAttention, CounterfactualBranch
from .self_evolution import SelfEvolutionEngine, DynamicPathController, ComputeBudgetAllocator
from .multi_agent import MultiAgentSystem, SpecialistAgent, AgentCoordinator, SharedMessageBoard
from .adversarial import AdversarialShield, AdversarialTrainer, InputShield, FeatureShield
from .training_utils import GradientMonitor, check_gradients_finite, set_seed, get_best_device
from .adapter import (
    WeightAdapter,
    ArchitectureAdapter,
    InferenceAdapter,
    LightweightCalibrator,
    ModelRegistry,
    detect_model_type,
    get_cortexnet_config,
)
from .ops import (
    DeviceManager,
    get_best_device_info,
    is_npu_available,
    is_mlu_available,
    get_device_type,
    resolve_device_string,
    resolve_dtype_for_device,
    NPUOperators,
    get_operators,
)
from .distributed import (
    setup_distributed,
    cleanup_distributed,
    wrap_fsdp,
    wrap_ddp,
    get_rank,
    get_world_size,
    is_main_process,
)

_HAS_DATA = False
try:
    from .data import (
        SimpleTokenizer,
        MiniMindTokenizer,
        TextCorpusDataset,
        StreamingDataset,
        ConversationDataset,
        PretrainDataset,
        CodeCompletionDataset,
        CodeGenerationDataset,
        download_wikitext2,
        download_minimind_data,
    )
    _DATA_EXPORTS = (
        SimpleTokenizer,
        MiniMindTokenizer,
        TextCorpusDataset,
        StreamingDataset,
        ConversationDataset,
        PretrainDataset,
        CodeCompletionDataset,
        CodeGenerationDataset,
        download_wikitext2,
        download_minimind_data,
    )
    _HAS_DATA = True
except ImportError:
    _DATA_EXPORTS = ()
    pass

# Legacy compatibility aliases (not part of the default public API list).
CortexNetV2 = _CortexNetV2
CortexNetV3 = _CortexNetV3
CortexBlockV2 = _CortexBlockV2
CortexBlockV3 = _CortexBlockV3

__version__ = "3.2.1"

__all__ = [
    "CortexNet",
    "CortexNetBase",
    "CortexNetConfig",
    "TrainingConfig",
    "CortexBlock",
    "RMSNorm",
    "AdaptiveFusionGate",
    "MultiScaleSSM",
    "SelectiveSparseAttention",
    "SynapticMemory",
    "MixtureOfExperts",
    "ExpertFFN",
    "CollaborativeMoE",
    "CortexNetCache",
    "TransformerLM",
    "quantize_dynamic",
    "QuantizationWrapper",
    "HierarchicalMemorySystem",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "MemoryController",
    "GraphReasoningModule",
    "MetaLearningAdapter",
    "TaskAdaptiveController",
    "ContextEncoder",
    "MultiModalEncoder",
    "PatchEmbedding",
    "AudioEmbedding",
    "CrossModalFusion",
    "ElasticWeightConsolidation",
    "ProgressiveMemoryReplay",
    "ContinualLearningManager",
    "ThoughtFlowMonitor",
    "CausalReasoningModule",
    "InterventionalAttention",
    "CounterfactualBranch",
    "SelfEvolutionEngine",
    "DynamicPathController",
    "ComputeBudgetAllocator",
    "MultiAgentSystem",
    "SpecialistAgent",
    "AgentCoordinator",
    "SharedMessageBoard",
    "AdversarialShield",
    "AdversarialTrainer",
    "InputShield",
    "FeatureShield",
    "GradientMonitor",
    "check_gradients_finite",
    "set_seed",
    "get_best_device",
    "WeightAdapter",
    "ArchitectureAdapter",
    "InferenceAdapter",
    "LightweightCalibrator",
    "ModelRegistry",
    "detect_model_type",
    "get_cortexnet_config",
    "DeviceManager",
    "get_best_device_info",
    "is_npu_available",
    "is_mlu_available",
    "get_device_type",
    "resolve_device_string",
    "resolve_dtype_for_device",
    "NPUOperators",
    "get_operators",
    "setup_distributed",
    "cleanup_distributed",
    "wrap_fsdp",
    "wrap_ddp",
    "get_rank",
    "get_world_size",
    "is_main_process",
]

if _HAS_DATA:
    __all__.extend(
        [
            "SimpleTokenizer",
            "MiniMindTokenizer",
            "TextCorpusDataset",
            "StreamingDataset",
            "ConversationDataset",
            "PretrainDataset",
            "CodeCompletionDataset",
            "CodeGenerationDataset",
            "download_wikitext2",
            "download_minimind_data",
        ]
    )
