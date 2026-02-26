#!/usr/bin/env python3
"""
CortexNet Lite vs Transformer 性能基准对比

运行: python benchmark_lite.py
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保可从脚本目录直接运行
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from cortexnet.config import CortexNetConfig
from cortexnet import CortexNet
from cortexnet.blocks import RMSNorm
from cortexnet.attention import precompute_rope_freqs, apply_rope


# ====== Transformer 基线 ======
class TransformerBlock(nn.Module):
    def __init__(self, d, nH, dff, max_len, rope_theta):
        super().__init__()
        self.nH = nH
        self.hd = d // nH
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, dff, bias=False)
        self.up_proj = nn.Linear(d, dff, bias=False)
        self.down_proj = nn.Linear(dff, d, bias=False)
        self.register_buffer(
            "rf", precompute_rope_freqs(self.hd, max_len, rope_theta), persistent=False
        )

    def forward(self, x):
        B, L, D = x.shape
        res = x
        xn = self.norm1(x)
        q = self.q_proj(xn).view(B, L, self.nH, self.hd).transpose(1, 2)
        k = self.k_proj(xn).view(B, L, self.nH, self.hd).transpose(1, 2)
        v = self.v_proj(xn).view(B, L, self.nH, self.hd).transpose(1, 2)
        q = apply_rope(q, self.rf)
        k = apply_rope(k, self.rf)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = res + self.o_proj(out.transpose(1, 2).contiguous().view(B, L, D))
        res = x
        xn = self.norm2(x)
        return res + self.down_proj(F.silu(self.gate_proj(xn)) * self.up_proj(xn))


class TransformerBaseline(nn.Module):
    def __init__(self, V, d, nL, nH, dff, ml, rt):
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, nH, dff, ml, rt) for _ in range(nL)]
        )
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return {"logits": self.head(self.norm(x))}


def run_benchmark():
    print("=" * 70)
    print("  🧠 CortexNet Lite vs ⚡ Transformer 基准对比")
    print("=" * 70)

    # 配置
    V, D, nL, nH, nKV, DFF, ML = 32000, 1024, 8, 8, 4, 2816, 2048

    # ====== Build Transformer ======
    t0 = time.time()
    model_tf = TransformerBaseline(V, D, nL, nH, DFF, ML, 10000.0)
    t_tf_init = time.time() - t0
    p_tf = sum(p.numel() for p in model_tf.parameters())

    # ====== Build CortexNet Lite ======
    config_lite = CortexNetConfig(
        vocab_size=V, hidden_size=D, num_layers=nL, num_heads=nH,
        num_kv_heads=nKV, intermediate_size=DFF, max_seq_len=ML, lite=True,
    )
    t0 = time.time()
    model_lite = CortexNet(config_lite)
    t_lite_init = time.time() - t0
    p_lite = sum(p.numel() for p in model_lite.parameters())

    # ====== Build CortexNet Full ======
    config_full = CortexNetConfig(
        vocab_size=V, hidden_size=D, num_layers=nL, num_heads=nH,
        num_kv_heads=nKV, intermediate_size=DFF, max_seq_len=ML, lite=False,
    )
    t0 = time.time()
    model_full = CortexNet(config_full)
    t_full_init = time.time() - t0
    p_full = sum(p.numel() for p in model_full.parameters())

    # ====== Speed Benchmark ======
    ids = torch.randint(0, V, (1, 64))
    model_tf.eval()
    model_lite.eval()

    with torch.no_grad():
        # Warmup
        model_tf(ids)
        model_lite(ids)

        N = 30
        t0 = time.time()
        for _ in range(N):
            model_tf(ids)
        t_tf_fwd = (time.time() - t0) / N * 1000

        t0 = time.time()
        for _ in range(N):
            model_lite(ids)
        t_lite_fwd = (time.time() - t0) / N * 1000

    # ====== 参数分布 ======
    groups = {}
    for name, p in model_lite.named_parameters():
        parts = name.split(".")
        key = parts[2] if parts[0] == "blocks" and len(parts) >= 3 else parts[0]
        groups[key] = groups.get(key, 0) + p.numel()

    # ====== Results ======
    print()
    print("┌─────────────────────────────────────────────────────────┐")
    print("│                    参数量对比                           │")
    print("├──────────────────┬──────────┬─────────┬────────────────┤")
    print("│ 模型             │ 参数量    │ 倍率    │ 初始化时间     │")
    print("├──────────────────┼──────────┼─────────┼────────────────┤")
    print(f"│ ⚡ Transformer    │ {p_tf/1e6:>6.1f}M  │ 1.00x   │ {t_tf_init:.2f}s           │")
    print(f"│ 🧠 CortexNet Lite│ {p_lite/1e6:>6.1f}M  │ {p_lite/p_tf:.2f}x   │ {t_lite_init:.2f}s           │")
    print(f"│ 🏋️ CortexNet Full│ {p_full/1e6:>6.1f}M  │ {p_full/p_tf:.2f}x   │ {t_full_init:.2f}s           │")
    print("└──────────────────┴──────────┴─────────┴────────────────┘")

    print()
    print("┌─────────────────────────────────────────────────────────┐")
    print("│                  推理速度对比 (seq=64)                  │")
    print("├──────────────────┬──────────┬──────────────────────────┤")
    print(f"│ ⚡ Transformer    │ {t_tf_fwd:>5.1f}ms  │ O(n²) 注意力             │")
    print(f"│ 🧠 CortexNet Lite│ {t_lite_fwd:>5.1f}ms  │ SSM O(1) + 稀疏注意力    │")
    print("└──────────────────┴──────────┴──────────────────────────┘")

    print()
    print("CortexNet Lite 参数分布:")
    for k, v in sorted(groups.items(), key=lambda x: -x[1]):
        pct = 100 * v / p_lite
        if pct > 1:
            bar = "█" * int(pct / 2)
            print(f"  {k:>12s}: {v/1e6:>6.1f}M ({pct:>4.1f}%) {bar}")

    print()
    print("CortexNet 相比 Transformer 的独特优势:")
    print("  ✅ SSM O(1) 增量解码 (Transformer 是 O(n))")
    print("  ✅ Synaptic Memory 快速权重系统")
    print("  ✅ 三路径自适应融合门控")
    print("  ✅ 稀疏注意力 (top-k) 减少无效计算")
    print("  ✅ GQA 支持减少 KV cache 2-4x")
    print()
    print(f"  参数开销: 仅多 +{(p_lite-p_tf)/1e6:.1f}M ({(p_lite/p_tf-1)*100:.0f}%) 换取以上全部能力")
    print(f"  相比 Full 版: 参数节省 {(1-p_lite/p_full)*100:.0f}%")


if __name__ == "__main__":
    run_benchmark()
