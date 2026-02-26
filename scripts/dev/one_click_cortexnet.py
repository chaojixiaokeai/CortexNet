#!/usr/bin/env python3
"""
一键运行 CortexNet + Qwen3-8B

默认流程：
  1) 跑回归测试 tests/test_all.py
  2) 启动 chat_qwen3_cortexnet.py

用法：
  python one_click_cortexnet.py
  python one_click_cortexnet.py --prompt "你好"
  python one_click_cortexnet.py --skip-tests --prompt "一句话介绍当前架构"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="一键测试并启动 CortexNet 聊天")
    parser.add_argument("--skip-tests", action="store_true", help="跳过回归测试")
    parser.add_argument("--prompt", default=None, help="单轮提问（设置后执行一次并退出）")
    parser.add_argument(
        "--model-path",
        default="/Users/pengjiajun/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
        help="本地模型目录路径",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto/cpu/mps/cuda/npu/mlu（支持索引，如 npu:0）",
    )
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--thinking", action="store_true", help="显示思考摘要")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tests_py = os.path.join(repo_root, "tests", "test_all.py")
    chat_py = os.path.join(repo_root, "scripts", "chat", "chat_qwen3_cortexnet.py")

    if not os.path.exists(chat_py):
        print(f"[错误] 找不到聊天脚本: {chat_py}")
        return 2
    if not args.skip_tests:
        if not os.path.exists(tests_py):
            print(f"[错误] 找不到测试脚本: {tests_py}")
            return 2
        print("[一键] Step 1/2: 运行回归测试...", flush=True)
        ret = subprocess.run([sys.executable, tests_py], cwd=repo_root)
        if ret.returncode != 0:
            print("[一键] 测试失败，已停止。")
            return ret.returncode

    print("[一键] Step 2/2: 启动聊天...", flush=True)
    cmd = [
        sys.executable,
        chat_py,
        "--model-path",
        args.model_path,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.thinking:
        cmd.append("--thinking")
    if args.no_stream:
        cmd.append("--no-stream")
    if args.prompt is not None:
        cmd.extend(["--prompt", args.prompt])

    ret = subprocess.run(cmd, cwd=repo_root)
    return ret.returncode


if __name__ == "__main__":
    raise SystemExit(main())
