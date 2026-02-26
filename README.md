# CortexNet

[![CI](https://github.com/chaojixiaokeai/CortexNet/actions/workflows/ci.yml/badge.svg)](https://github.com/chaojixiaokeai/CortexNet/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/cortexnet.svg)](https://pypi.org/project/cortexnet/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)

## Language / 语言

- 中文文档入口: `docs/README.md`（Chinese + English full docs index）
- English quick docs: `docs/en/QUICKSTART_AND_USAGE.md`
- Chinese quick docs: `docs/zh-CN/QUICKSTART_AND_USAGE.md`

`CortexNet` 是一个面向语言建模与推理场景的神经网络架构实现，核心思路是将多尺度状态空间建模、选择性稀疏注意力、记忆系统、条件路由与可选高级推理模块组合到同一框架中。

`CortexNet` is a unified neural architecture implementation for language modeling and reasoning, combining multi-scale SSM, selective sparse attention, memory, conditional routing, and optional advanced reasoning modules in one framework.

本仓库已经完成以下整理：

- 统一对外主模型命名为 `CortexNet`（不再以 `*V3` 作为主入口）
- 代码按 `pip` 可发布标准重构到 `cortexnet/` 包目录
- 基准/聊天/评测脚本统一归档到 `scripts/`
- 清理无用导入、冗余变量和缓存文件
- 补充架构文档和模块映射文档，方便开源协作

## Installation

```bash
pip install -e .
```

或构建并安装 wheel：

```bash
python -m pip install build
python -m build
pip install dist/*.whl
```

## CLI

```bash
python -m cortexnet --version
python -m cortexnet --smoke-test
```

## Quick Start

```python
import torch
from cortexnet import CortexNet, CortexNetConfig

config = CortexNetConfig(
    vocab_size=32000,
    hidden_size=512,
    num_layers=4,
    num_heads=8,
    max_seq_len=2048,
)

model = CortexNet(config).eval()
input_ids = torch.randint(0, config.vocab_size, (1, 16))

with torch.no_grad():
    out = model(input_ids)
print(out["logits"].shape)
```

## Package Layout

```text
cortexnet/
  adapter/      # 开源模型识别、权重映射、架构适配、推理适配、校准
  ops/          # 设备与算子抽象（CPU/CUDA/MPS/NPU/MLU）
  model.py      # 主模型与 from_pretrained
  blocks.py     # 核心 block 组合
  ...
scripts/
  benchmarks/   # 性能与量化基准
  chat/         # 聊天脚本
  eval/         # 能力评测脚本
  dev/          # 一键流程与测试运行器
tests/          # 回归与单元测试
```

## Test

```bash
python -m pytest -q
```

## Docs

- 文档中心（中英文完整文档）：`docs/README.md`
- 架构说明：`ARCHITECTURE.md`
- 模块职责图：`MODULE_MAP.md`
- 变更记录：`CHANGELOG.md`
- 贡献指南：`CONTRIBUTING.md`
- 支持与问题分流：`SUPPORT.md`
- 安全策略：`SECURITY.md`
- 示例代码：`examples/README.md`
- 发布基准报告：`docs/reports/README.md`

## Compatibility Notes

- 对外推荐只使用 `CortexNet`。
- 历史命名（如 `CortexNetV2` / `CortexNetV3`）仍保留为兼容别名，避免旧代码立即失效。

## Development Commands

```bash
make install-dev
make lint
make test-all
make check
```
