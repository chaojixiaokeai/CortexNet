"""Utilities for pretrained weight discovery and mapped-cache bookkeeping."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from typing import Any, Dict, Iterator, List, Optional

import torch


def discover_weight_files(path: str) -> List[str]:
    """Discover source weight files with deterministic priority."""
    patterns = ("*.safetensors", "pytorch_model*.bin", "model*.safetensors")
    for pattern in patterns:
        files = sorted(glob.glob(os.path.join(path, pattern)))
        if files:
            return files
    return []


def build_mapped_cache_file(
    *,
    source_path: str,
    source_model_type: str,
    files: List[str],
    cache_dir: str,
    config: Any,
) -> Optional[str]:
    """Build deterministic mapped-cache filename from source weight fingerprint."""
    if not files:
        return None

    weight_fingerprints = []
    for weight_file in files:
        stat = os.stat(weight_file)
        weight_fingerprints.append(
            {
                "name": os.path.basename(weight_file),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )

    cache_payload = {
        "model_path": os.path.abspath(source_path),
        "model_type": source_model_type,
        "compatibility_mode": bool(getattr(config, "compatibility_mode", False)),
        "expand_gqa_weights": bool(getattr(config, "expand_gqa_weights", True)),
        "compat_ssm_rank": int(getattr(config, "compat_ssm_rank", 256)),
        "hidden_size": int(getattr(config, "hidden_size", 0)),
        "num_layers": int(getattr(config, "num_layers", 0)),
        "num_heads": int(getattr(config, "num_heads", 0)),
        "num_kv_heads": int(getattr(config, "num_kv_heads", 0)),
        "intermediate_size": int(getattr(config, "intermediate_size", 0)),
        "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", True)),
        "weights": weight_fingerprints,
    }

    cache_key = hashlib.sha256(
        json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return os.path.join(cache_dir, f"{source_model_type}_{cache_key}.safetensors")


def iter_weight_tensors(
    weight_file: str,
    *,
    logger: Any = None,
) -> Iterator[Dict[str, torch.Tensor]]:
    """Stream tensors from a weight shard to reduce peak memory."""
    if weight_file.endswith(".safetensors"):
        try:
            from safetensors.torch import safe_open  # type: ignore
        except ImportError:
            if logger is not None:
                logger.warning(
                    "safetensors not installed. Install with: pip install safetensors"
                )
            return

        with safe_open(weight_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                yield {key: f.get_tensor(key)}
        return

    if weight_file.endswith(".bin"):
        state = torch.load(weight_file, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            for key, value in state.items():
                if torch.is_tensor(value):
                    yield {key: value}
