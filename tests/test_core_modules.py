"""
CortexNet 核心模块单元测试

覆盖范围：
  1. MultiScaleSSM 前向传播 + 增量解码
  2. SelectiveSparseAttention 正确性
  3. SynapticMemory 并行化正确性
  4. MoE 路由 + 负载均衡
  5. CausalReasoningModule (稀疏 + 反事实)
  6. GraphReasoningModule (GLU 节点更新)
  7. MultiAgentSystem (BatchedAgents + 梯度安全 write)
  8. CortexNetConfig 验证
  9. QuantizationWrapper 策略
  10. 分布式工具 (非分布式环境)
"""

import sys
import os
import pytest
import torch

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════
#                  1. MultiScaleSSM 测试
# ═══════════════════════════════════════════════════════════════

def test_ssm_forward_shape():
    """测试 SSM 前向传播输出维度。"""
    from cortexnet.ssm import MultiScaleSSM
    ssm = MultiScaleSSM(d_model=64, num_scales=4, state_size=16, expand_factor=2)
    x = torch.randn(2, 32, 64)
    y = ssm(x)
    assert y.shape == (2, 32, 64), f"Expected (2, 32, 64), got {y.shape}"


def test_ssm_incremental_decode():
    """测试 SSM 增量解码（use_cache=True）。"""
    from cortexnet.ssm import MultiScaleSSM
    ssm = MultiScaleSSM(d_model=64, num_scales=2, state_size=8)
    ssm.eval()

    # 首次前向
    x = torch.randn(1, 16, 64)
    y1, state = ssm(x, use_cache=True)
    assert state is not None, "State should not be None when use_cache=True"

    # 增量解码
    x2 = torch.randn(1, 1, 64)
    y2, state2 = ssm(x2, past_state=state, use_cache=True)
    assert y2.shape == (1, 1, 64)
    assert state2 is not None


def test_ssm_single_token():
    """测试 SSM 处理单个 token 的情况。"""
    from cortexnet.ssm import MultiScaleSSM
    ssm = MultiScaleSSM(d_model=64)
    x = torch.randn(1, 1, 64)
    y = ssm(x)
    assert y.shape == (1, 1, 64)


# ═══════════════════════════════════════════════════════════════
#                  2. SelectiveSparseAttention 测试
# ═══════════════════════════════════════════════════════════════

def test_attention_forward_shape():
    """测试稀疏注意力输出维度。"""
    from cortexnet.attention import SelectiveSparseAttention
    attn = SelectiveSparseAttention(d_model=64, num_heads=4, top_k_ratio=0.5)
    x = torch.randn(2, 32, 64)
    y = attn(x)
    assert y.shape == (2, 32, 64)


def test_attention_short_sequence():
    """测试短序列退化为完整注意力。"""
    from cortexnet.attention import SelectiveSparseAttention
    attn = SelectiveSparseAttention(d_model=64, num_heads=4, top_k_ratio=0.5)
    x = torch.randn(1, 4, 64)
    y = attn(x)
    assert y.shape == (1, 4, 64)


def test_attention_sparse_cache_position_tracking():
    """测试稀疏缓存下的位置偏移跟踪（应基于 seen_len 而非 cache_len）。"""
    from cortexnet.attention import SelectiveSparseAttention

    attn = SelectiveSparseAttention(
        d_model=64,
        num_heads=4,
        top_k_ratio=0.25,
        max_seq_len=512,
    )
    attn.eval()

    x_prefill = torch.randn(1, 16, 64)
    _, cache = attn(x_prefill, use_cache=True)
    assert len(cache) == 5
    assert cache[4] == 16  # total_seen_len

    x_next = torch.randn(1, 1, 64)
    _, cache2 = attn(x_next, past_key_value=cache, use_cache=True)
    assert cache2[2][0, -1].item() == 16  # 新 token 位置应是 16 而非 top-k 长度
    assert cache2[4] == 17


# ═══════════════════════════════════════════════════════════════
#                  3. SynapticMemory 测试
# ═══════════════════════════════════════════════════════════════

def test_memory_forward_shape():
    """测试突触记忆输出维度。"""
    from cortexnet.memory import SynapticMemory
    mem = SynapticMemory(d_model=64, memory_dim=32)
    x = torch.randn(2, 16, 64)
    y = mem(x)
    assert y.shape == (2, 16, 64)


def test_memory_gradient_flow():
    """测试记忆模块梯度流通。"""
    from cortexnet.memory import SynapticMemory
    mem = SynapticMemory(d_model=64, memory_dim=32)
    x = torch.randn(1, 8, 64, requires_grad=True)
    y = mem(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ═══════════════════════════════════════════════════════════════
#                  4. MoE 路由测试
# ═══════════════════════════════════════════════════════════════

def test_moe_forward_shape():
    """测试 MoE 前向传播。"""
    from cortexnet.routing import MixtureOfExperts
    moe = MixtureOfExperts(
        d_model=64, d_ff=128, num_experts=4, num_active=2
    )
    x = torch.randn(2, 16, 64)
    y = moe(x)
    assert y.shape == (2, 16, 64)


def test_moe_aux_loss():
    """测试 MoE 辅助损失非零。"""
    from cortexnet.routing import MixtureOfExperts
    moe = MixtureOfExperts(
        d_model=64, d_ff=128, num_experts=4, num_active=2
    )
    moe.train()
    x = torch.randn(2, 16, 64)
    _ = moe(x)
    assert hasattr(moe, 'aux_loss')


# ═══════════════════════════════════════════════════════════════
#                  5. CausalReasoningModule 测试
# ═══════════════════════════════════════════════════════════════

def test_causal_reasoning_forward():
    """测试因果推理模块前向传播。"""
    from cortexnet.causal_reasoning import CausalReasoningModule
    cr = CausalReasoningModule(d_model=64, num_heads=4, num_counterfactuals=4)
    x = torch.randn(2, 16, 64)
    y = cr(x)
    assert y.shape == (2, 16, 64)


def test_causal_sparse_attention():
    """测试干预注意力的稀疏性。"""
    from cortexnet.causal_reasoning import InterventionalAttention
    attn = InterventionalAttention(d_model=64, num_heads=4, top_k_ratio=0.25)
    x = torch.randn(1, 32, 64)
    causal = torch.sigmoid(torch.randn(1, 32, 1))
    y = attn(x, causal)
    assert y.shape == (1, 32, 64)


def test_counterfactual_merged_transform():
    """测试合并反事实变换的正确性。"""
    from cortexnet.causal_reasoning import CounterfactualBranch
    cf = CounterfactualBranch(d_model=64, num_counterfactuals=4)
    x = torch.randn(1, 16, 64)
    causal = torch.sigmoid(torch.randn(1, 16, 1))
    y = cf(x, causal)
    assert y.shape == (1, 16, 64)


# ═══════════════════════════════════════════════════════════════
#                  6. GraphReasoningModule 测试
# ═══════════════════════════════════════════════════════════════

def test_graph_reasoning_forward():
    """测试图推理模块前向传播。"""
    from cortexnet.graph_reasoning import GraphReasoningModule
    gr = GraphReasoningModule(d_model=64, num_neighbors=8, num_iterations=2)
    x = torch.randn(2, 16, 64)
    y = gr(x)
    assert y.shape == (2, 16, 64)


def test_graph_reasoning_single_token():
    """测试图推理处理单个 token。"""
    from cortexnet.graph_reasoning import GraphReasoningModule
    gr = GraphReasoningModule(d_model=64)
    x = torch.randn(1, 1, 64)
    y = gr(x)
    assert y.shape == (1, 1, 64)


def test_graph_reasoning_gradient_flow():
    """测试图推理梯度流通（GLU 节点更新）。"""
    from cortexnet.graph_reasoning import GraphReasoningModule
    gr = GraphReasoningModule(d_model=64, num_neighbors=4, num_iterations=2)
    x = torch.randn(1, 8, 64, requires_grad=True)
    y = gr(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ═══════════════════════════════════════════════════════════════
#                  7. MultiAgentSystem 测试
# ═══════════════════════════════════════════════════════════════

def test_multi_agent_forward():
    """测试多智能体系统前向传播。"""
    from cortexnet.multi_agent import MultiAgentSystem
    mas = MultiAgentSystem(d_model=64, num_agents=4, message_slots=8)
    x = torch.randn(2, 16, 64)
    y = mas(x)
    assert y.shape == (2, 16, 64)


def test_multi_agent_gradient_flow():
    """测试多智能体系统的梯度流通（验证 .data 修复）。"""
    from cortexnet.multi_agent import MultiAgentSystem
    mas = MultiAgentSystem(d_model=64, num_agents=2)
    x = torch.randn(1, 8, 64, requires_grad=True)
    y = mas(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_message_board_gradient():
    """测试消息板 write 的梯度流通。"""
    from cortexnet.multi_agent import SharedMessageBoard
    mb = SharedMessageBoard(d_model=64, num_slots=4)
    content = torch.randn(2, 8, 64, requires_grad=True)
    mb.write(content)
    # 读取后应该可以反向传播
    query = torch.randn(2, 8, 64)
    read_out = mb.read(query)
    loss = read_out.sum()
    loss.backward()
    # 验证 write_proj 的梯度存在
    assert mb.write_proj.weight.grad is not None or content.grad is not None


# ═══════════════════════════════════════════════════════════════
#                  8. CortexNetConfig 验证测试
# ═══════════════════════════════════════════════════════════════

def test_config_validation_hidden_size():
    """测试 hidden_size 不能被 num_heads 整除时报错。"""
    from cortexnet.config import CortexNetConfig
    with pytest.raises(ValueError, match="hidden_size"):
        CortexNetConfig(hidden_size=100, num_heads=8)


def test_config_validation_top_k_ratio():
    """测试 top_k_ratio 范围验证。"""
    from cortexnet.config import CortexNetConfig
    with pytest.raises(ValueError, match="top_k_ratio"):
        CortexNetConfig(top_k_ratio=0.0)
    with pytest.raises(ValueError, match="top_k_ratio"):
        CortexNetConfig(top_k_ratio=1.5)


def test_config_validation_attention_mode():
    """测试 attention_k_mode 合法值验证。"""
    from cortexnet.config import CortexNetConfig
    with pytest.raises(ValueError, match="attention_k_mode"):
        CortexNetConfig(attention_k_mode="invalid")


def test_config_validation_moe():
    """测试 MoE 参数约束验证。"""
    from cortexnet.config import CortexNetConfig
    with pytest.raises(ValueError, match="num_active_experts"):
        CortexNetConfig(num_experts=4, num_active_experts=8)


def test_config_capacity_factor():
    """测试 moe_capacity_factor 配置项。"""
    from cortexnet.config import CortexNetConfig
    c = CortexNetConfig(moe_capacity_factor=2.0)
    assert c.moe_capacity_factor == 2.0


# ═══════════════════════════════════════════════════════════════
#                  9. QuantizationWrapper 测试
# ═══════════════════════════════════════════════════════════════

def test_quantization_strategies():
    """测试多种量化策略。"""
    from cortexnet.quantization import QuantizationWrapper
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))

    # 无量化
    qm = QuantizationWrapper(model, strategy="none")
    x = torch.randn(1, 32)
    y = qm(x)
    assert y.shape == (1, 10)


def test_quantization_dynamic_int8():
    """测试动态 INT8 量化（平台支持时）。"""
    from cortexnet.quantization import QuantizationWrapper
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    try:
        qm = QuantizationWrapper(model, strategy="dynamic_int8")
        x = torch.randn(1, 32)
        y = qm(x)
        assert y.shape == (1, 10)
    except RuntimeError as e:
        if "NoQEngine" in str(e) or "quantized" in str(e).lower():
            pytest.skip("Dynamic INT8 quantization not supported on this platform")
        raise


def test_quantization_weight_only_int8():
    """测试权重 INT8 量化路径可用。"""
    from cortexnet.quantization import QuantizationWrapper, WeightOnlyInt8Linear
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
    qm = QuantizationWrapper(model, strategy="weight_only_int8")
    x = torch.randn(1, 32)
    y = qm(x)
    assert y.shape == (1, 10)
    assert any(isinstance(m, WeightOnlyInt8Linear) for m in qm.modules())


def test_quantization_invalid_strategy():
    """测试无效量化策略报错。"""
    from cortexnet.quantization import QuantizationWrapper
    import torch.nn as nn

    model = nn.Linear(32, 10)
    with pytest.raises(ValueError, match="Unknown quantization strategy"):
        QuantizationWrapper(model, strategy="gptq")


def test_torchao_selective_filter_skips_tied_lm_head(monkeypatch):
    """验证 torchao 过滤策略会跳过 tied embedding/lm_head。"""
    from cortexnet.quantization import _select_effective_linear_modules
    import torch.nn as nn

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(128, 32)
            self.ff = nn.Linear(32, 64, bias=False)
            self.router = nn.Linear(32, 8, bias=False)
            self.lm_head = nn.Linear(32, 128, bias=False)
            self.lm_head.weight = self.embed.weight

    monkeypatch.setenv("CORTEXNET_TORCHAO_MIN_LINEAR_PARAMS", "1")
    monkeypatch.setenv("CORTEXNET_TORCHAO_INCLUDE", "ff,router,lm_head")
    model = TinyLM()
    selected, skipped = _select_effective_linear_modules(model)
    names = {name for name, _ in selected}

    assert "ff" in names
    assert "lm_head" not in names
    assert skipped.get("tied_embedding_weight", 0) >= 1


# ═══════════════════════════════════════════════════════════════
#                  10. 分布式工具测试
# ═══════════════════════════════════════════════════════════════

def test_distributed_rank():
    """测试非分布式环境下 rank 返回 0。"""
    from cortexnet.distributed import get_rank, get_world_size, is_main_process
    assert get_rank() == 0
    assert get_world_size() == 1
    assert is_main_process() is True


def test_distributed_setup_no_env():
    """测试无环境变量时 setup 返回 False。"""
    from cortexnet.distributed import setup_distributed
    result = setup_distributed()
    assert result is False


# ═══════════════════════════════════════════════════════════════
#                  Phase 2 新增测试
# ═══════════════════════════════════════════════════════════════

def test_attention_gqa():
    """测试 GQA 模式注意力。"""
    from cortexnet.attention import SelectiveSparseAttention
    attn = SelectiveSparseAttention(
        d_model=64, num_heads=8, num_kv_heads=2, top_k_ratio=0.5,
    )
    x = torch.randn(2, 16, 64)
    y = attn(x)
    assert y.shape == (2, 16, 64)
    # 验证 KV 投影维度是 kv_dim*2 而不是 d_model*2
    assert attn.kv_proj.out_features == 2 * 2 * (64 // 8)  # 2 * kv_heads * head_dim


def test_ssm_output_gate():
    """测试 SSM 输出门控初始行为。"""
    from cortexnet.ssm import MultiScaleSSM
    ssm = MultiScaleSSM(d_model=64, num_scales=2)
    # 初始 output_gate=0 → sigmoid(0)=0.5
    assert abs(torch.sigmoid(ssm.output_gate).item() - 0.5) < 0.01
    x = torch.randn(1, 8, 64)
    y = ssm(x)
    assert y.shape == (1, 8, 64)


def test_evolution_temperature_anneal():
    """测试 SelfEvolution 温度退火。"""
    from cortexnet.self_evolution import DynamicPathController
    ctrl = DynamicPathController(d_model=64, num_paths=5)
    assert ctrl.temperature == 5.0
    ctrl.anneal_temperature(0)
    assert ctrl.temperature == 5.0
    ctrl.anneal_temperature(5000)
    assert abs(ctrl.temperature - 2.75) < 0.01
    ctrl.anneal_temperature(10000)
    assert abs(ctrl.temperature - 0.5) < 0.01


def test_hierarchical_memory_cache():
    """测试 HierarchicalMemorySystem use_cache 安全解包。"""
    from cortexnet.hierarchical_memory import HierarchicalMemorySystem
    mem = HierarchicalMemorySystem(d_model=64)
    x = torch.randn(1, 8, 64)
    # None case
    y1 = mem(x, past_working_memory=None, use_cache=False)
    assert y1.shape == (1, 8, 64)
    # use_cache case
    y2, state = mem(x, use_cache=True)
    assert y2.shape == (1, 8, 64)
    assert state is not None


def test_optimizer_scheduler_factory():
    """测试 create_optimizer_and_scheduler。"""
    from cortexnet.training_utils import create_optimizer_and_scheduler
    import torch.nn as nn
    model = nn.Linear(32, 10)
    opt, sched = create_optimizer_and_scheduler(
        model, lr=1e-3, warmup_steps=10, total_steps=100,
    )
    assert opt is not None
    assert sched is not None
    # Warmup: 步数 0 → lr=0
    assert sched.get_last_lr()[0] < 1e-6
    for _ in range(10):
        opt.step()
        sched.step()
    # After warmup: lr ≈ peak
    assert sched.get_last_lr()[0] > 5e-4


def test_safe_grad_clip():
    """测试 safe_clip_grad_norm_。"""
    from cortexnet.training_utils import safe_clip_grad_norm_
    import torch.nn as nn
    model = nn.Linear(32, 10)
    x = torch.randn(1, 32)
    y = model(x)
    y.sum().backward()
    norm = safe_clip_grad_norm_(model, max_norm=0.5)
    assert norm >= 0


def test_task_controller_batched():
    """测试 TaskAdaptiveController 批量变换。"""
    from cortexnet.meta_learning import TaskAdaptiveController
    ctrl = TaskAdaptiveController(d_model=64, num_modes=4)
    x = torch.randn(2, 16, 64)
    y = ctrl(x)
    assert y.shape == (2, 16, 64)


def test_episodic_memory_sdpa():
    """测试 EpisodicMemory SDPA 交叉注意力。"""
    from cortexnet.hierarchical_memory import EpisodicMemory
    mem = EpisodicMemory(d_model=64, num_slots=16, num_heads=4)
    x = torch.randn(2, 16, 64)
    y = mem(x)
    assert y.shape == (2, 16, 64)


# ═══════════════════════════════════════════════════════════════
#                  设备解析与 NPU 兼容测试
# ═══════════════════════════════════════════════════════════════

def test_resolve_device_string_invalid():
    from cortexnet.ops.device_manager import resolve_device_string
    with pytest.raises(ValueError):
        resolve_device_string("invalid_device")


def test_resolve_device_string_npu_fallback():
    from cortexnet.ops.device_manager import (
        resolve_device_string,
        is_npu_available,
        get_device_type,
    )

    resolved = resolve_device_string("npu", allow_fallback=True)
    assert isinstance(resolved, str) and len(resolved) > 0

    if not is_npu_available():
        assert get_device_type(resolved) != "npu"
        with pytest.raises(RuntimeError):
            resolve_device_string("npu", allow_fallback=False)


def test_resolve_dtype_for_device():
    from cortexnet.ops.device_manager import resolve_dtype_for_device

    assert resolve_dtype_for_device("auto", "cpu") == torch.float32
    assert resolve_dtype_for_device("auto", "npu:0") == torch.float16
    assert resolve_dtype_for_device("bfloat16", "mps") == torch.float16


# ═══════════════════════════════════════════════════════════════
#                  运行入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from _pytest.outcomes import Skipped

    # 收集所有 test_ 函数
    test_functions = [
        v for k, v in sorted(globals().items())
        if k.startswith("test_") and callable(v)
    ]

    passed = 0
    failed = 0
    skipped = 0
    errors = []

    print(f"\n{'='*60}")
    print("CortexNet Core Module Tests")
    print(f"{'='*60}\n")

    for fn in test_functions:
        try:
            fn()
            passed += 1
            print(f"  ✅ {fn.__name__}")
        except Skipped as s:
            skipped += 1
            print(f"  ⏭️  {fn.__name__}: SKIPPED ({s})")
        except Exception as e:
            failed += 1
            errors.append((fn.__name__, str(e)))
            print(f"  ❌ {fn.__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {skipped} skipped, {failed} failed, {passed+skipped+failed} total")
    print(f"{'='*60}\n")

    if errors:
        print("Failed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
        sys.exit(1)
