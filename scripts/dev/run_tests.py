#!/usr/bin/env python3
"""
CortexNet 综合测试运行器

从包根目录运行，确保相对导入正确工作。
用法: python run_tests.py
"""

import os
import sys
import json
import tempfile
import argparse

# 将项目根目录加入 sys.path，确保可直接导入 cortexnet 包
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch


def _make_hf_dir(config_dict):
    """创建临时 HuggingFace 模型目录。"""
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(config_dict, f)
    return tmpdir


def _cleanup(tmpdir):
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
#                     测试用例
# ═══════════════════════════════════════════════════════════════

def test_config_defaults():
    from cortexnet.config import CortexNetConfig
    c = CortexNetConfig()
    hidden_default = CortexNetConfig.__dataclass_fields__["hidden_size"].default
    assert c.vocab_size == 32000
    assert c.hidden_size == hidden_default
    assert c.num_kv_heads == 8
    print("✅ test_config_defaults")

def test_config_custom():
    from cortexnet.config import CortexNetConfig
    c = CortexNetConfig(vocab_size=65536, hidden_size=4096, num_kv_heads=8)
    assert c.vocab_size == 65536
    assert c.num_kv_heads == 8
    print("✅ test_config_custom")

def test_config_from_dict():
    from cortexnet.config import CortexNetConfig
    c = CortexNetConfig.from_dict({"vocab_size": 128000, "unknown_field": True})
    assert c.vocab_size == 128000
    print("✅ test_config_from_dict")

def test_training_config():
    from cortexnet.config import TrainingConfig
    t = TrainingConfig()
    assert t.learning_rate == 3e-4
    print("✅ test_training_config")

def test_model_registry():
    from cortexnet.adapter.model_registry import ModelRegistry
    families = ModelRegistry.list_supported()
    assert "llama" in families and "qwen2" in families and len(families) >= 10
    print(f"✅ test_model_registry ({len(families)} families)")

def test_detect_model_type():
    from cortexnet.adapter.model_registry import detect_model_type
    for arch, mt, expected in [
        ("LlamaForCausalLM", "llama", "llama"),
        ("Qwen2ForCausalLM", "qwen2", "qwen2"),
        ("MistralForCausalLM", "mistral", "mistral"),
    ]:
        d = _make_hf_dir({"architectures": [arch], "model_type": mt, "hidden_size": 128})
        try:
            assert detect_model_type(d) == expected
        finally:
            _cleanup(d)
    print("✅ test_detect_model_type")

def test_get_cortexnet_config():
    from cortexnet.adapter.model_registry import get_cortexnet_config
    d = _make_hf_dir({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "vocab_size": 128256, "hidden_size": 4096,
        "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 8, "intermediate_size": 14336,
        "max_position_embeddings": 8192, "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
    })
    try:
        c = get_cortexnet_config(d, "llama")
        assert c.vocab_size == 128256 and c.hidden_size == 4096 and c.num_kv_heads == 8
        assert c.rope_theta == 500000.0
    finally:
        _cleanup(d)
    print("✅ test_get_cortexnet_config")

def test_weight_adapter_mapping():
    from cortexnet.config import CortexNetConfig
    from cortexnet.adapter.weight_adapter import WeightAdapter
    config = CortexNetConfig(hidden_size=256, num_heads=4, num_kv_heads=4, num_layers=2)
    adapter = WeightAdapter("llama", config)
    raw = {
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
    cw = adapter.map_weights(raw)
    assert "embed.weight" in cw and "blocks.0.attention.q_proj.weight" in cw
    assert "final_norm.weight" in cw
    print(f"✅ test_weight_adapter_mapping ({len(cw)} weights)")

def test_weight_adapter_gqa():
    from cortexnet.config import CortexNetConfig
    from cortexnet.adapter.weight_adapter import WeightAdapter
    config = CortexNetConfig(
        hidden_size=256,
        num_heads=8,
        num_kv_heads=2,
        num_layers=1,
        lite=False,
    )
    adapter = WeightAdapter("llama", config)
    raw = {
        "model.layers.0.self_attn.k_proj.weight": torch.randn(64, 256),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(64, 256),
    }
    cw = adapter.map_weights(raw)
    assert cw["blocks.0.attention.k_proj.weight"].shape[0] == 256
    print("✅ test_weight_adapter_gqa")

def test_weight_adapter_qkv_split():
    from cortexnet.config import CortexNetConfig
    from cortexnet.adapter.weight_adapter import WeightAdapter
    config = CortexNetConfig(hidden_size=256, num_heads=4, num_kv_heads=4, num_layers=1)
    adapter = WeightAdapter("baichuan", config)
    raw = {"model.layers.0.self_attn.W_pack.weight": torch.randn(768, 256)}
    cw = adapter.map_weights(raw)
    assert "blocks.0.attention.q_proj.weight" in cw
    assert "blocks.0.attention.k_proj.weight" in cw
    assert "blocks.0.attention.v_proj.weight" in cw
    print("✅ test_weight_adapter_qkv_split")

def test_pretrained_utils_discover_index():
    from cortexnet.pretrained_utils import discover_weight_files
    with tempfile.TemporaryDirectory() as tmpdir:
        shard1 = os.path.join(tmpdir, "model-00001-of-00002.safetensors")
        shard2 = os.path.join(tmpdir, "model-00002-of-00002.safetensors")
        with open(shard1, "wb") as f:
            f.write(b"a")
        with open(shard2, "wb") as f:
            f.write(b"b")
        with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {},
                    "weight_map": {
                        "a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors",
                    },
                },
                f,
            )
        found = discover_weight_files(tmpdir)
        assert found == [shard1, shard2]
    print("✅ test_pretrained_utils_discover_index")

def test_pretrained_utils_iter_bin_nested_state_dict():
    from cortexnet.pretrained_utils import iter_weight_tensors
    with tempfile.TemporaryDirectory() as tmpdir:
        wf = os.path.join(tmpdir, "pytorch_model.bin")
        torch.save({"state_dict": {"w": torch.randn(2, 2), "meta": "x"}}, wf)
        chunks = list(iter_weight_tensors(wf))
        merged = {}
        for chunk in chunks:
            merged.update(chunk)
        assert "w" in merged and torch.is_tensor(merged["w"])
    print("✅ test_pretrained_utils_iter_bin_nested_state_dict")

def test_device_manager():
    from cortexnet.ops.device_manager import DeviceManager
    dm = DeviceManager()
    devs = dm.list_devices()
    assert len(devs) >= 1 and any(d["type"] == "cpu" for d in devs)
    best = dm.get_best_device()
    assert best.device_type in ("cuda", "npu", "mlu", "mps", "cpu")
    print(f"✅ test_device_manager (best: {best.device_name})")

def test_npu_operators():
    from cortexnet.ops.npu_ops import NPUOperators
    ops = NPUOperators(backend="cpu")
    w, idx = ops.moe_route(torch.randn(16, 8), num_active=2)
    assert w.shape == (16, 2) and idx.shape == (16, 2)
    print("✅ test_npu_operators")

def test_cortexnet_forward():
    from cortexnet.config import CortexNetConfig
    from cortexnet import CortexNet
    config = CortexNetConfig(
        vocab_size=1000, hidden_size=128, num_layers=2, num_heads=4,
        max_seq_len=256, num_scales=2, ssm_state_size=8,
        expert_ff_dim=256, num_experts=4, num_active_experts=2, memory_dim=32,
    )
    m = CortexNet(config)
    m.eval()
    with torch.no_grad():
        out = m(torch.randint(0, 1000, (2, 32)))
    assert out["logits"].shape == (2, 32, 1000)
    print("✅ test_cortexnet_forward")

def test_cortexnet_generate():
    from cortexnet.config import CortexNetConfig
    from cortexnet import CortexNet
    config = CortexNetConfig(
        vocab_size=100, hidden_size=64, num_layers=1, num_heads=2,
        max_seq_len=128, num_scales=2, ssm_state_size=8,
        expert_ff_dim=128, num_experts=2, num_active_experts=1, memory_dim=16,
    )
    m = CortexNet(config)
    gen = m.generate(torch.randint(0, 100, (1, 5)), max_new_tokens=10)
    assert gen.shape == (1, 15)
    print("✅ test_cortexnet_generate")

def test_from_pretrained_mock():
    from cortexnet import CortexNet
    d = _make_hf_dir({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "vocab_size": 500, "hidden_size": 64, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 2,
        "intermediate_size": 128, "max_position_embeddings": 128,
        "rope_theta": 10000.0, "rms_norm_eps": 1e-5,
    })
    try:
        m = CortexNet.from_pretrained(d, device="cpu", dtype=torch.float32)
        assert m is not None and hasattr(m, '_inference_adapter')
        gen = m.smart_generate(torch.randint(0, 500, (1, 4)), max_new_tokens=3)
        assert gen.shape[1] == 7
    finally:
        _cleanup(d)
    print("✅ test_from_pretrained_mock")

def test_inference_adapter():
    from cortexnet.config import CortexNetConfig
    from cortexnet import CortexNet
    from cortexnet.adapter.inference_adapter import InferenceAdapter
    config = CortexNetConfig(
        vocab_size=100, hidden_size=64, num_layers=1, num_heads=2,
        max_seq_len=128, num_scales=2, ssm_state_size=8,
        expert_ff_dim=128, num_experts=2, num_active_experts=1, memory_dim=16,
    )
    m = CortexNet(config)
    ia = InferenceAdapter(m, "llama")
    assert ia.default_params["temperature"] == 0.6
    gen = ia.generate(torch.randint(0, 100, (1, 5)), max_new_tokens=5)
    assert gen.shape[1] == 10
    print("✅ test_inference_adapter")

def test_calibrator():
    from cortexnet.config import CortexNetConfig
    from cortexnet import CortexNet
    from cortexnet.adapter.calibrator import LightweightCalibrator
    config = CortexNetConfig(
        vocab_size=100, hidden_size=64, num_layers=1, num_heads=2,
        max_seq_len=64, num_scales=2, ssm_state_size=8,
        expert_ff_dim=128, num_experts=2, num_active_experts=1, memory_dim=16,
    )
    m = CortexNet(config)
    cache_dir = tempfile.mkdtemp()
    try:
        cal = LightweightCalibrator(m, "llama", cache_dir=cache_dir)
        cal_data = [{"input_ids": torch.randint(0, 100, (32,))} for _ in range(5)]
        m = cal.calibrate(calibration_data=cal_data, n_samples=5, use_cache=True)
        assert len(os.listdir(cache_dir)) > 0
    finally:
        _cleanup(cache_dir)
    print("✅ test_calibrator")


# ═══════════════════════════════════════════════════════════════
#                     测试运行器
# ═══════════════════════════════════════════════════════════════

ALL_TESTS = [
    test_config_defaults, test_config_custom, test_config_from_dict, test_training_config,
    test_model_registry, test_detect_model_type, test_get_cortexnet_config,
    test_weight_adapter_mapping, test_weight_adapter_gqa, test_weight_adapter_qkv_split,
    test_pretrained_utils_discover_index, test_pretrained_utils_iter_bin_nested_state_dict,
    test_device_manager, test_npu_operators,
    test_cortexnet_forward, test_cortexnet_generate,
    test_from_pretrained_mock,
    test_inference_adapter, test_calibrator,
]

def main() -> int:
    parser = argparse.ArgumentParser(description="CortexNet dev regression runner")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tests and exit",
    )
    parser.add_argument(
        "--match",
        default=None,
        help="Run only tests whose name contains this substring",
    )
    args = parser.parse_args()

    selected = ALL_TESTS
    if args.match:
        selected = [fn for fn in ALL_TESTS if args.match in fn.__name__]

    if args.list:
        for fn in selected:
            print(fn.__name__)
        return 0

    print("=" * 60)
    print("  CortexNet 适配器综合测试")
    print("=" * 60)

    passed = failed = 0
    errors = []

    for fn in selected:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            errors.append((fn.__name__, traceback.format_exc()))
            print(f"❌ {fn.__name__}: {e}")

    print()
    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {len(selected)} total")
    print("=" * 60)
    if errors:
        print("\nError details:")
        for name, tb in errors:
            print(f"\n--- {name} ---")
            print(tb)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
