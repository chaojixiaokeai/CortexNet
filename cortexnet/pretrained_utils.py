"""Utilities for pretrained weight discovery and mapped-cache bookkeeping."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from typing import Any, Dict, Iterator, List, Optional

import torch


def _discover_from_index(path: str, index_name: str) -> List[str]:
    """Resolve shard files from a HuggingFace-style index json."""
    index_path = os.path.join(path, index_name)
    if not os.path.isfile(index_path):
        return []

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        return []

    seen: set[str] = set()
    files: List[str] = []
    for shard_name in sorted(set(weight_map.values())):
        if not isinstance(shard_name, str):
            continue
        shard_file = os.path.join(path, shard_name)
        if os.path.isfile(shard_file) and shard_file not in seen:
            seen.add(shard_file)
            files.append(shard_file)
    return files


def discover_weight_files(path: str) -> List[str]:
    """Discover source weight files with deterministic priority."""
    if not os.path.isdir(path):
        return []

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        files = _discover_from_index(path, index_name)
        if files:
            return files

    patterns = ("*.safetensors", "pytorch_model*.bin", "model*.bin")
    for pattern in patterns:
        files = [
            file
            for file in sorted(glob.glob(os.path.join(path, pattern)))
            if os.path.isfile(file)
        ]
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
        try:
            state = torch.load(weight_file, map_location="cpu", weights_only=True)
        except TypeError:
            # torch<2.0 may not support weights_only; fallback keeps compatibility.
            state = torch.load(weight_file, map_location="cpu")

        state_dict = state if isinstance(state, dict) else None
        if isinstance(state_dict, dict) and not any(torch.is_tensor(v) for v in state_dict.values()):
            for nested_key in ("state_dict", "model", "module"):
                nested = state_dict.get(nested_key)
                if isinstance(nested, dict):
                    state_dict = nested
                    break

        if isinstance(state_dict, dict):
            for key in list(state_dict.keys()):
                value = state_dict.pop(key)
                if torch.is_tensor(value):
                    yield {key: value}
            return

        if logger is not None:
            logger.warning("Unsupported checkpoint format for %s", weight_file)
