"""
CortexNet 综合测试套件

测试范围：
  1. 配置系统 (CortexNetConfig)
  2. 权重适配器 (WeightAdapter)
  3. 模型注册表 (ModelRegistry)
  4. 架构适配 (ArchitectureAdapter)
  5. 推理适配 (InferenceAdapter)
  6. 校准器 (LightweightCalibrator)
  7. 设备管理器 (DeviceManager)
  8. NPU 算子 (NPUOperators)
  9. 完整 from_pretrained 流程
  10. CortexNet 前向传播
"""

import os
import sys
import json
import tempfile

import torch

# 确保包可导入
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# ═══════════════════════════════════════════════════════════════
#                  1. 配置系统测试
# ═══════════════════════════════════════════════════════════════

def test_config_defaults():
    """测试 CortexNetConfig 默认值创建。"""
    from cortexnet.config import CortexNetConfig
    config = CortexNetConfig()
    hidden_default = CortexNetConfig.__dataclass_fields__["hidden_size"].default
    layers_default = CortexNetConfig.__dataclass_fields__["num_layers"].default
    assert config.vocab_size == 32000
    assert config.hidden_size == hidden_default
    assert config.num_layers == layers_default
    assert config.num_heads == 8
    assert config.num_kv_heads == 8  # __post_init__ should set to num_heads
    assert config.norm_eps == 1e-6
    print("✅ test_config_defaults passed")


def test_config_custom():
    """测试自定义配置参数。"""
    from cortexnet.config import CortexNetConfig
    config = CortexNetConfig(
        vocab_size=65536,
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
    )
    assert config.vocab_size == 65536
    assert config.hidden_size == 4096
    assert config.num_kv_heads == 8
    print("✅ test_config_custom passed")


def test_config_from_dict():
    """测试从字典创建配置（忽略未知字段）。"""
    from cortexnet.config import CortexNetConfig
    d = {
        "vocab_size": 128000,
        "hidden_size": 2048,
        "unknown_field": "should_be_ignored",
    }
    config = CortexNetConfig.from_dict(d)
    assert config.vocab_size == 128000
    assert config.hidden_size == 2048
    print("✅ test_config_from_dict passed")


def test_training_config():
    """测试 TrainingConfig 默认值。"""
    from cortexnet.config import TrainingConfig
    tc = TrainingConfig()
    assert tc.learning_rate == 3e-4
    assert tc.batch_size == 8
    assert tc.seed == 42
    print("✅ test_training_config passed")


# ═══════════════════════════════════════════════════════════════
#                  2. 模型注册表测试
# ═══════════════════════════════════════════════════════════════

def test_model_registry_list():
    """测试模型注册表列表。"""
    from cortexnet.adapter.model_registry import ModelRegistry
    families = ModelRegistry.list_supported()
    assert "llama" in families
    assert "qwen2" in families
    assert "mistral" in families
    assert len(families) >= 10
    print(f"✅ test_model_registry_list passed ({len(families)} families)")


def test_detect_model_type():
    """测试模型类型自动检测。"""
    from cortexnet.adapter.model_registry import detect_model_type

    # 创建模拟的 config.json
    with tempfile.TemporaryDirectory() as tmpdir:
        # LLaMA 模型
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config, f)

        detected = detect_model_type(tmpdir)
        assert detected == "llama", f"Expected 'llama', got '{detected}'"

    # Qwen 模型
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 2048,
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config, f)

        detected = detect_model_type(tmpdir)
        assert detected == "qwen2", f"Expected 'qwen2', got '{detected}'"

    # Qwen3 模型
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": 4096,
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config, f)

        detected = detect_model_type(tmpdir)
        assert detected == "qwen3", f"Expected 'qwen3', got '{detected}'"

    # Mistral 模型
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "hidden_size": 4096,
            "sliding_window": 4096,
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config, f)

        detected = detect_model_type(tmpdir)
        assert detected == "mistral"

    print("✅ test_detect_model_type passed (llama, qwen2, qwen3, mistral)")


def test_get_cortexnet_config():
    """测试 HuggingFace 配置转换为 CortexNetConfig。"""
    from cortexnet.adapter.model_registry import get_cortexnet_config

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "vocab_size": 128256,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "max_position_embeddings": 8192,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1e-5,
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(hf_config, f)

        config = get_cortexnet_config(tmpdir, "llama")
        assert config.vocab_size == 128256
        assert config.hidden_size == 4096
        assert config.num_layers == 32
        assert config.num_heads == 32
        assert config.num_kv_heads == 8
        assert config.intermediate_size == 14336
        assert config.rope_theta == 500000.0
        assert config.model_type == "llama"

    print("✅ test_get_cortexnet_config passed")


# ═══════════════════════════════════════════════════════════════
#                  3. 权重适配器测试
# ═══════════════════════════════════════════════════════════════

def test_weight_adapter_mapping():
    """测试权重映射规则。"""
    from cortexnet.config import CortexNetConfig
    from cortexnet.adapter.weight_adapter import WeightAdapter

    config = CortexNetConfig(
        hidden_size=256,
        num_heads=4,
        num_kv_heads=4,
        num_layers=2,
        lite=False,
    )

    adapter = WeightAdapter("llama", config)

    # 模拟 LLaMA 权重
    raw_weights = {
        "model.embed_tokens.weight": torch.randn(32000, 256),
        "lm_head.weight": torch.randn(32000, 256),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(256, 256),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(256, 256),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(256, 256),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(256, 256),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(1024, 256),
        "model.layers.0.mlp.up_proj.weight": torch.randn(1024, 256),
        "model.layers.0.mlp.down_proj.weight": torch.randn(256, 1024),
        "model.layers.0.input_layernorm.weight": torch.randn(256),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(256),
        "model.norm.weight": torch.randn(256),
    }

    cortex_weights = adapter.map_weights(raw_weights)

    assert "embed.weight" in cortex_weights
    assert "lm_head.weight" in cortex_weights
    assert "blocks.0.attention.q_proj.weight" in cortex_weights
    assert "blocks.0.attention.k_proj.weight" in cortex_weights
    assert "blocks.0.attention.v_proj.weight" in cortex_weights
    assert "blocks.0.norm1.weight" in cortex_weights
    assert "final_norm.weight" in cortex_weights

    print(f"✅ test_weight_adapter_mapping passed ({len(cortex_weights)} weights mapped)")


def test_weight_adapter_gqa():
    """测试 GQA 权重扩展。"""
    from cortexnet.config import CortexNetConfig
    from cortexnet.adapter.weight_adapter import WeightAdapter

    config = CortexNetConfig(
        hidden_size=256,
        num_heads=8,
        num_kv_heads=2,  # GQA: 4x expansion
        num_layers=1,
        lite=False,
    )

    adapter = WeightAdapter("llama", config)

    raw_weights = {
        "model.layers.0.self_attn.k_proj.weight": torch.randn(64, 256),  # 2 heads * 32 dim
        "model.layers.0.self_attn.v_proj.weight": torch.randn(64, 256),
    }

    cortex_weights = adapter.map_weights(raw_weights)

    # GQA 扩展后应为 8 heads * 32 dim = 256
    k_weight = cortex_weights.get("blocks.0.attention.k_proj.weight")
    assert k_weight is not None
    assert k_weight.shape[0] == 256, f"Expected 256, got {k_weight.shape[0]}"

    print("✅ test_weight_adapter_gqa passed")


def test_weight_adapter_qkv_split():
    """测试合并 QKV 投影分割。"""
    from cortexnet.config import CortexNetConfig
    from cortexnet.adapter.weight_adapter import WeightAdapter

    config = CortexNetConfig(
        hidden_size=256,
        num_heads=4,
        num_kv_heads=4,
        num_layers=1,
    )

    adapter = WeightAdapter("baichuan", config)

    # Baichuan 使用合并的 W_pack (Q+K+V)
    raw_weights = {
        "model.layers.0.self_attn.W_pack.weight": torch.randn(768, 256),  # 3 * 256
    }

    cortex_weights = adapter.map_weights(raw_weights)

    assert "blocks.0.attention.q_proj.weight" in cortex_weights
    assert "blocks.0.attention.k_proj.weight" in cortex_weights
    assert "blocks.0.attention.v_proj.weight" in cortex_weights
    assert cortex_weights["blocks.0.attention.q_proj.weight"].shape[0] == 256

    print("✅ test_weight_adapter_qkv_split passed")


# ═══════════════════════════════════════════════════════════════
#                  4. 设备管理器测试
# ═══════════════════════════════════════════════════════════════

def test_device_manager():
    """测试设备管理器。"""
    from cortexnet.ops.device_manager import DeviceManager

    dm = DeviceManager()
    devices = dm.list_devices()
    assert len(devices) >= 1  # 至少有 CPU
    assert any(d["type"] == "cpu" for d in devices)

    best = dm.get_best_device()
    assert best.device_type in ("cuda", "npu", "mlu", "mps", "cpu")

    opt_config = dm.get_optimization_config(best)
    assert "device" in opt_config
    assert "dtype" in opt_config

    print(f"✅ test_device_manager passed (best: {best.device_name})")


# ═══════════════════════════════════════════════════════════════
#                  5. NPU 算子测试
# ═══════════════════════════════════════════════════════════════

def test_npu_operators():
    """测试 NPU 算子（CPU 回退模式）。"""
    from cortexnet.ops.npu_ops import NPUOperators

    ops = NPUOperators(backend="cpu")

    # MoE 路由
    router_logits = torch.randn(16, 8)  # 16 tokens, 8 experts
    weights, indices = ops.moe_route(router_logits, num_active=2)
    assert weights.shape == (16, 2)
    assert indices.shape == (16, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(16), atol=1e-5)

    print("✅ test_npu_operators passed")


# ═══════════════════════════════════════════════════════════════
#                  6. CortexNet 前向传播测试
# ═══════════════════════════════════════════════════════════════

def test_cortexnet_forward():
    """测试 CortexNet 前向传播。"""
    from cortexnet.config import CortexNetConfig
    from cortexnet import CortexNet

    config = CortexNetConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=256,
        num_scales=2,
        ssm_state_size=8,
        ssm_expand_factor=2,
        expert_ff_dim=256,
        num_experts=4,
        num_active_experts=2,
        memory_dim=32,
    )

    model = CortexNet(config)
    model.eval()

    input_ids = torch.randint(0, 1000, (2, 32))

    with torch.no_grad():
        output = model(input_ids)

    assert "logits" in output
    assert output["logits"].shape == (2, 32, 1000)
    print(f"✅ test_cortexnet_forward passed (logits shape: {output['logits'].shape})")


def test_cortexnet_generate():
    """测试 CortexNet 生成。"""
    from cortexnet.config import CortexNetConfig
    from cortexnet import CortexNet

    config = CortexNetConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        max_seq_len=128,
        num_scales=2,
        ssm_state_size=8,
        expert_ff_dim=128,
        num_experts=2,
        num_active_experts=1,
        memory_dim=16,
    )

    model = CortexNet(config)
    input_ids = torch.randint(0, 100, (1, 5))

    generated = model.generate(input_ids, max_new_tokens=10)
    assert generated.shape == (1, 15)  # 5 input + 10 generated
    print(f"✅ test_cortexnet_generate passed (generated shape: {generated.shape})")


# ═══════════════════════════════════════════════════════════════
#                  7. from_pretrained 流程测试
# ═══════════════════════════════════════════════════════════════

def test_from_pretrained_mock():
    """测试 from_pretrained 使用模拟权重文件。"""
    from cortexnet import CortexNet

    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建模拟 config.json
        hf_config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 256,
            "max_position_embeddings": 256,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(hf_config, f)

        # 加载（不含权重文件 — 将使用随机权重）
        model = CortexNet.from_pretrained(
            tmpdir,
            device="cpu",
            dtype=torch.float32,
            load_weights=True,  # will warn about no weight files
        )

        assert model is not None
        assert hasattr(model, '_inference_adapter')

        # 测试 smart_generate
        input_ids = torch.randint(0, 1000, (1, 5))
        generated = model.smart_generate(input_ids, max_new_tokens=5)
        assert generated.shape[1] == 10  # 5 + 5

    print("✅ test_from_pretrained_mock passed")


def test_from_pretrained_compatibility_disables_lite_by_default():
    """compatibility_mode=True 且未显式传 lite 时应关闭 lite。"""
    from cortexnet import CortexNet

    with tempfile.TemporaryDirectory() as tmpdir:
        hf_config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 256,
            "max_position_embeddings": 256,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5,
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(hf_config, f)

        model = CortexNet.from_pretrained(
            tmpdir,
            device="cpu",
            dtype=torch.float32,
            load_weights=False,
            compatibility_mode=True,
        )
        assert model.compatibility_mode is True
        assert model.lite_mode is False

    print("✅ test_from_pretrained_compatibility_disables_lite_by_default passed")


# ═══════════════════════════════════════════════════════════════
#                  8. 推理适配器测试
# ═══════════════════════════════════════════════════════════════

def test_inference_adapter():
    """测试推理适配器。"""
    from cortexnet.config import CortexNetConfig
    from cortexnet import CortexNet
    from cortexnet.adapter.inference_adapter import InferenceAdapter

    config = CortexNetConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        max_seq_len=128,
        num_scales=2,
        ssm_state_size=8,
        expert_ff_dim=128,
        num_experts=2,
        num_active_experts=1,
        memory_dim=16,
    )

    model = CortexNet(config)
    adapter = InferenceAdapter(model, "llama")

    # 检查默认参数
    assert adapter.default_params["temperature"] == 0.6
    assert adapter.default_params["top_p"] == 0.9

    # 测试生成
    input_ids = torch.randint(0, 100, (1, 5))
    output = adapter.generate(input_ids, max_new_tokens=5)
    assert output.shape[1] == 10

    print("✅ test_inference_adapter passed")


# ═══════════════════════════════════════════════════════════════
#                  9. 校准器测试
# ═══════════════════════════════════════════════════════════════

def test_calibrator():
    """测试轻量校准器。"""
    from cortexnet.config import CortexNetConfig
    from cortexnet import CortexNet
    from cortexnet.adapter.calibrator import LightweightCalibrator

    config = CortexNetConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        max_seq_len=64,
        num_scales=2,
        ssm_state_size=8,
        expert_ff_dim=128,
        num_experts=2,
        num_active_experts=1,
        memory_dim=16,
    )

    model = CortexNet(config)

    with tempfile.TemporaryDirectory() as cache_dir:
        calibrator = LightweightCalibrator(model, "llama", cache_dir=cache_dir)

        # 生成少量校准数据
        cal_data = [
            {"input_ids": torch.randint(0, 100, (32,))}
            for _ in range(5)
        ]

        # 执行校准
        model = calibrator.calibrate(
            calibration_data=cal_data,
            n_samples=5,
            use_cache=True,
        )

        # 验证缓存已保存
        cache_files = os.listdir(cache_dir)
        assert len(cache_files) > 0, "Calibration cache should be saved"

    print("✅ test_calibrator passed")


# ═══════════════════════════════════════════════════════════════
#                  运行所有测试
# ═══════════════════════════════════════════════════════════════

def run_all_tests():
    """运行所有测试。"""
    print("=" * 60)
    print("  CortexNet 适配器综合测试")
    print("=" * 60)

    tests = [
        # 配置
        test_config_defaults,
        test_config_custom,
        test_config_from_dict,
        test_training_config,
        # 注册表
        test_model_registry_list,
        test_detect_model_type,
        test_get_cortexnet_config,
        # 权重适配
        test_weight_adapter_mapping,
        test_weight_adapter_gqa,
        test_weight_adapter_qkv_split,
        # 设备
        test_device_manager,
        # NPU 算子
        test_npu_operators,
        # 模型
        test_cortexnet_forward,
        test_cortexnet_generate,
        # from_pretrained
        test_from_pretrained_mock,
        test_from_pretrained_compatibility_disables_lite_by_default,
        # 推理适配
        test_inference_adapter,
        # 校准
        test_calibrator,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"❌ {test_fn.__name__} FAILED: {e}")

    print()
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
