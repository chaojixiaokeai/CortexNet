"""Public API contract tests for CortexNet."""

from __future__ import annotations

import subprocess
import sys

import torch

import cortexnet
from cortexnet import CortexNet, CortexNetConfig


def test_public_api_version() -> None:
    assert isinstance(cortexnet.__version__, str)
    assert cortexnet.__version__


def test_canonical_model_name_and_aliases() -> None:
    cfg = CortexNetConfig(vocab_size=128, hidden_size=64, num_layers=1, num_heads=2, lite=False)

    model = CortexNet(cfg)
    model_v3 = cortexnet.CortexNetV3(cfg)

    assert model.__class__.__name__ == "CortexNet"
    assert isinstance(model, cortexnet.CortexNetV3)
    assert isinstance(model_v3, cortexnet.CortexNetV3)


def test_forward_smoke_shape() -> None:
    cfg = CortexNetConfig(vocab_size=128, hidden_size=64, num_layers=1, num_heads=2, lite=False)
    model = CortexNet(cfg).eval()

    input_ids = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        out = model(input_ids)

    assert "logits" in out
    assert out["logits"].shape == (2, 6, cfg.vocab_size)


def test_module_cli_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cortexnet", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert cortexnet.__version__ in result.stdout.strip()
