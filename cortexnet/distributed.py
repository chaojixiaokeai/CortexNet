"""
CortexNet 分布式训练支持 (Distributed Training)

提供 FSDP (Fully Sharded Data Parallel) 策略和基础分布式工具，
为大规模训练做准备。

用法:
    from cortexnet.distributed import setup_distributed, wrap_fsdp, cleanup_distributed

    setup_distributed()
    model = CortexNet(config)
    model = wrap_fsdp(model, config)
    # ... training loop ...
    cleanup_distributed()
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def setup_distributed(backend: str = "nccl") -> bool:
    """初始化分布式训练环境。

    自动检测是否在分布式环境中运行（通过环境变量），
    如果是则初始化 process group。

    Args:
        backend: 通信后端 ("nccl" for GPU, "gloo" for CPU)

    Returns:
        True 如果成功初始化分布式环境
    """
    if torch.distributed.is_initialized():
        logger.info("Distributed already initialized")
        return True

    # 检查环境变量
    rank = os.environ.get("RANK")
    world_size = os.environ.get("WORLD_SIZE")
    local_rank = os.environ.get("LOCAL_RANK")

    if rank is None or world_size is None:
        logger.info("Not in distributed environment, skipping initialization")
        return False

    rank = int(rank)
    world_size = int(world_size)
    local_rank = int(local_rank) if local_rank is not None else rank

    # 自动选择后端
    if backend == "nccl" and not torch.cuda.is_available():
        backend = "gloo"
        logger.info("CUDA not available, falling back to gloo backend")

    # 设置 GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    logger.info(
        f"Distributed initialized: rank={rank}/{world_size}, "
        f"local_rank={local_rank}, backend={backend}"
    )
    return True


def cleanup_distributed():
    """清理分布式训练环境。"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        logger.info("Distributed process group destroyed")


def get_rank() -> int:
    """获取当前进程的 rank（非分布式返回 0）。"""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """获取总进程数（非分布式返回 1）。"""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def is_main_process() -> bool:
    """是否为主进程。"""
    return get_rank() == 0


def wrap_fsdp(
    model: nn.Module,
    config: Optional[Any] = None,
    mixed_precision: Optional[str] = None,
    activation_checkpointing: bool = False,
) -> nn.Module:
    """使用 FSDP 包装模型。

    自动为 CortexNet 的不同组件选择合适的分片策略。

    Args:
        model: CortexNet 模型实例
        config: CortexNetConfig（可选，用于读取配置）
        mixed_precision: "fp16", "bf16", 或 None
        activation_checkpointing: 是否启用激活检查点

    Returns:
        FSDP 包装后的模型
    """
    if not torch.distributed.is_initialized():
        logger.warning("Distributed not initialized, returning model as-is")
        return model

    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import (
            size_based_auto_wrap_policy,
        )
    except ImportError:
        logger.error("FSDP not available in this PyTorch version")
        return model

    # 混合精度策略
    mp_policy = None
    if mixed_precision == "fp16":
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    elif mixed_precision == "bf16":
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    # 自动分片策略：按参数数量决定分片粒度
    min_params = 1_000_000  # 超过 1M 参数的子模块独立分片
    auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=min_params)

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
    )

    # 激活检查点
    if activation_checkpointing:
        try:
            from torch.distributed.fsdp import apply_activation_checkpointing
            # 对所有 CortexBlock 启用激活检查点
            apply_activation_checkpointing(model)
            logger.info("Activation checkpointing enabled")
        except ImportError:
            logger.warning("Activation checkpointing not available")

    logger.info(f"Model wrapped with FSDP (mixed_precision={mixed_precision})")
    return model


def wrap_ddp(
    model: nn.Module,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """使用 DDP (DistributedDataParallel) 包装模型。

    对于不需要 FSDP 级别分片的场景，使用更简单的 DDP。

    Args:
        model: 模型
        find_unused_parameters: 是否查找未使用参数

    Returns:
        DDP 包装后的模型
    """
    if not torch.distributed.is_initialized():
        logger.warning("Distributed not initialized, returning model as-is")
        return model

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        model = model.to(local_rank)

    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        find_unused_parameters=find_unused_parameters,
    )

    logger.info(f"Model wrapped with DDP (device={local_rank})")
    return model
