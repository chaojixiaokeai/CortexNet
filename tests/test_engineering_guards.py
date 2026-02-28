"""Engineering guard tests for refactor-safe behavior."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from cortexnet import CortexNet, CortexNetConfig
from cortexnet.pretrained_utils import (
    build_mapped_cache_file,
    discover_weight_files,
    iter_weight_tensors,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class _DummyConfig:
    compatibility_mode = True
    expand_gqa_weights = True
    compat_ssm_rank = 128
    hidden_size = 256
    num_layers = 2
    num_heads = 4
    num_kv_heads = 4
    intermediate_size = 512
    tie_word_embeddings = True


def _run_script(path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.pop("CORTEXNET_MODEL_PATH", None)
    return subprocess.run(
        [sys.executable, str(path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )


def test_discover_weight_files_priority(tmp_path: Path) -> None:
    (tmp_path / "pytorch_model-00001.bin").write_bytes(b"bin")
    (tmp_path / "model-00001.safetensors").write_bytes(b"safe_alt")
    (tmp_path / "weights.safetensors").write_bytes(b"safe")

    found = discover_weight_files(str(tmp_path))
    assert found
    assert all(p.endswith(".safetensors") for p in found)
    assert (tmp_path / "weights.safetensors").as_posix() in found


def test_discover_weight_files_from_index(tmp_path: Path) -> None:
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"a")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"b")
    (tmp_path / "weights.safetensors").write_bytes(b"fallback")
    (tmp_path / "model.safetensors.index.json").write_text(
        (
            '{"metadata":{},"weight_map":{"a":"model-00001-of-00002.safetensors",'
            '"b":"model-00002-of-00002.safetensors"}}'
        ),
        encoding="utf-8",
    )

    found = discover_weight_files(str(tmp_path))
    assert found == [
        (tmp_path / "model-00001-of-00002.safetensors").as_posix(),
        (tmp_path / "model-00002-of-00002.safetensors").as_posix(),
    ]


def test_build_mapped_cache_file_stable(tmp_path: Path) -> None:
    wf = tmp_path / "weights.safetensors"
    wf.write_bytes(b"content")

    cfg = _DummyConfig()
    f1 = build_mapped_cache_file(
        source_path=str(tmp_path),
        source_model_type="qwen3",
        files=[str(wf)],
        cache_dir=str(tmp_path),
        config=cfg,
    )
    f2 = build_mapped_cache_file(
        source_path=str(tmp_path),
        source_model_type="qwen3",
        files=[str(wf)],
        cache_dir=str(tmp_path),
        config=cfg,
    )
    assert f1 is not None
    assert f1 == f2
    assert f1.endswith(".safetensors")


def test_iter_weight_tensors_bin(tmp_path: Path) -> None:
    wf = tmp_path / "pytorch_model.bin"
    torch.save({"w": torch.randn(2, 2), "meta": "x"}, wf)

    chunks = list(iter_weight_tensors(str(wf)))
    assert chunks
    merged = {}
    for chunk in chunks:
        merged.update(chunk)
    assert "w" in merged
    assert torch.is_tensor(merged["w"])


def test_iter_weight_tensors_bin_nested_state_dict(tmp_path: Path) -> None:
    wf = tmp_path / "pytorch_model.bin"
    torch.save({"state_dict": {"w": torch.randn(2, 2), "meta": "x"}}, wf)

    chunks = list(iter_weight_tensors(str(wf)))
    assert chunks
    merged = {}
    for chunk in chunks:
        merged.update(chunk)
    assert "w" in merged
    assert torch.is_tensor(merged["w"])


def test_compile_model_strict_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = CortexNetConfig(vocab_size=128, hidden_size=64, num_layers=1, num_heads=2, lite=False)
    model = CortexNet(cfg).eval()

    def _raise_compile(module, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("compile boom")

    monkeypatch.setattr(torch, "compile", _raise_compile)
    monkeypatch.setenv("CORTEXNET_COMPILE_STRICT", "1")
    with pytest.raises(RuntimeError):
        model.compile_model()

    monkeypatch.setenv("CORTEXNET_COMPILE_STRICT", "0")
    assert model.compile_model() is model


def test_script_requires_model_path_one_click() -> None:
    path = REPO_ROOT / "scripts" / "dev" / "one_click_cortexnet.py"
    result = _run_script(path)
    assert result.returncode == 2
    assert "模型目录" in result.stdout or "model path" in result.stdout.lower()


def test_script_requires_model_path_coldstart() -> None:
    path = REPO_ROOT / "scripts" / "benchmarks" / "benchmark_coldstart.py"
    result = _run_script(path)
    assert result.returncode == 2
    assert "Missing --model-path" in result.stderr


def test_script_requires_model_path_long_context() -> None:
    path = REPO_ROOT / "scripts" / "benchmarks" / "benchmark_long_context.py"
    result = _run_script(path)
    assert result.returncode == 2
    assert "Missing --model-path" in result.stderr


def test_config_validation_hidden_size_divisible() -> None:
    """Test that hidden_size must be divisible by num_heads."""
    with pytest.raises(ValueError):
        CortexNetConfig(hidden_size=100, num_heads=3)  # 100 not divisible by 3


def test_config_default_values() -> None:
    """Test default configuration values."""
    cfg = CortexNetConfig()
    assert cfg.vocab_size == 32000
    assert cfg.hidden_size == 2560
    assert cfg.num_layers == 32
    assert cfg.dropout == 0.0
    assert cfg.norm_eps == 1e-6


def test_model_forward_shape() -> None:
    """Test model forward pass produces correct output shape."""
    cfg = CortexNetConfig(vocab_size=128, hidden_size=64, num_layers=1, num_heads=2)
    model = CortexNet(cfg).eval()
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, cfg.vocab_size)


def test_model_generate_basic() -> None:
    """Test basic text generation functionality."""
    cfg = CortexNetConfig(vocab_size=128, hidden_size=64, num_layers=1, num_heads=2)
    model = CortexNet(cfg).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=5, temperature=1.0, top_p=1.0)
    assert output.shape[1] > input_ids.shape[1]  # Generated more tokens


def test_cortexblock_lite_mode() -> None:
    """Test CortexBlockLite mode for resource-constrained environments."""
    cfg = CortexNetConfig(vocab_size=128, hidden_size=64, num_layers=1, num_heads=2, lite=True)
    model = CortexNet(cfg).eval()
    assert model.config.lite is True
    input_ids = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (1, 5, cfg.vocab_size)
