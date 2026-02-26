from __future__ import annotations

import time

import gradio as gr
import torch

from cortexnet import CortexNet, CortexNetConfig, __version__


def run_smoke(seq_len: int, iters: int) -> str:
    cfg = CortexNetConfig(
        vocab_size=4096,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=max(128, seq_len),
        lite=True,
    )

    model = CortexNet(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, seq_len))

    with torch.no_grad():
        _ = model(input_ids)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            out = model(input_ids)
    dt = (time.perf_counter() - t0) / max(iters, 1)

    msg = (
        f"CortexNet version: {__version__}\n"
        f"Average forward latency: {dt * 1000:.4f} ms\n"
        f"Logits shape: {tuple(out['logits'].shape)}"
    )
    return msg


DESCRIPTION = """
# CortexNet Space Demo

This Space runs a lightweight smoke benchmark of CortexNet to verify runtime health.

- Source: https://github.com/chaojixiaokeai/CortexNet
- PyPI: https://pypi.org/project/cortexnet/
"""


with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)
    seq_len = gr.Slider(minimum=8, maximum=256, value=64, step=8, label="Sequence Length")
    iters = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Iterations")
    run_btn = gr.Button("Run Smoke Benchmark")
    output = gr.Textbox(label="Result", lines=8)
    run_btn.click(fn=run_smoke, inputs=[seq_len, iters], outputs=[output])


if __name__ == "__main__":
    demo.launch()
