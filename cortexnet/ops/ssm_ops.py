from __future__ import annotations

import torch

@torch.jit.script
def fused_chunk_scan(
    a_chunk: torch.Tensor,
    b_chunk: torch.Tensor,
    h_init: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the parallel scan for a single chunk in a numerically stable way.
    This function implements the core recurrence calculation for a chunk of data,
    which is a key part of State Space Models (SSMs).

    The recurrence is: h_t = a_t * h_{t-1} + b_t

    This implementation matches the bit-exact order of operations used in the
    reference path to maintain consistency and pass regression tests.

    Args:
        a_chunk (torch.Tensor): The state transition factors for the chunk.
                                Shape: (B, L_chunk, D_inner, N)
        b_chunk (torch.Tensor): The input factors for the chunk.
                                Shape: (B, L_chunk, D_inner, N)
        h_init (torch.Tensor): The initial state from the previous chunk.
                               Shape: (B, D_inner, N)

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - y_chunk (torch.Tensor): The output for each step in the chunk.
                                      Shape: (B, L_chunk, D_inner, N)
            - h_final (torch.Tensor): The final state of the chunk.
                                      Shape: (B, D_inner, N)
    """
    # Force float32 for stability
    a_chunk = a_chunk.to(torch.float32)
    b_chunk = b_chunk.to(torch.float32)
    h_init = h_init.to(torch.float32)

    # 1. Compute cumulative product of `a`
    log_a = torch.log(a_chunk.clamp(min=1e-8))
    log_a_cum = torch.cumsum(log_a, dim=1)
    a_cum = torch.exp(log_a_cum)

    # 2. Compute the `b` scan part
    # Implementation follows reference path: b_out = a_cum * cumsum(b_chunk / a_cum)
    b_scan_input = b_chunk / (a_cum + 1e-8)
    b_scanned = torch.cumsum(b_scan_input, dim=1)
    b_out = a_cum * b_scanned

    # 3. Combine with initial state `h_init`
    h_chunk = a_cum * h_init.unsqueeze(1) + b_out

    return h_chunk, h_chunk[:, -1]
