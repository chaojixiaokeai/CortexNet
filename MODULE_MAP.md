# Module Map

## Core Runtime

| File | Responsibility |
|---|---|
| `cortexnet/model.py` | 主模型实现、生成、`from_pretrained`、lazy 设备策略 |
| `cortexnet/config.py` | 模型与训练配置定义、参数校验 |
| `cortexnet/blocks.py` | 模块级组合（SSM/Attention/Memory/Fusion/MoE） |
| `cortexnet/attention.py` | 稀疏注意力、RoPE、缓存解码 |
| `cortexnet/ssm.py` | 多尺度状态空间扫描与增量状态 |
| `cortexnet/memory.py` | 突触记忆模块 |
| `cortexnet/routing.py` | MoE 与协作路由 |
| `cortexnet/cache.py` | 推理缓存结构 |
| `cortexnet/compat.py` | 历史兼容组件（用于 compatibility mode） |
| `cortexnet/cortex_block_lite.py` | Lite 路径 block |

## Capability Extensions

| File | Responsibility |
|---|---|
| `cortexnet/hierarchical_memory.py` | 分层记忆系统 |
| `cortexnet/graph_reasoning.py` | 图推理模块 |
| `cortexnet/meta_learning.py` | 元学习适配与任务控制 |
| `cortexnet/multimodal.py` | 多模态编码器 |
| `cortexnet/continual_learning.py` | 连续学习（EWC + replay） |
| `cortexnet/causal_reasoning.py` | 因果/反事实模块 |
| `cortexnet/self_evolution.py` | 动态路径与预算控制 |
| `cortexnet/multi_agent.py` | 多智能体协作 |
| `cortexnet/adversarial.py` | 对抗鲁棒性模块 |
| `cortexnet/interpretability.py` | 监控与可解释性 |

## Adaptation Layer

| File | Responsibility |
|---|---|
| `cortexnet/adapter/model_registry.py` | 外部模型识别与配置转换 |
| `cortexnet/adapter/weight_adapter.py` | 权重映射 |
| `cortexnet/adapter/arch_adapter.py` | 架构行为适配 |
| `cortexnet/adapter/inference_adapter.py` | 推理参数策略 |
| `cortexnet/adapter/calibrator.py` | 轻量校准 |

## Platform / Infra

| File | Responsibility |
|---|---|
| `cortexnet/ops/device_manager.py` | 设备选择与 dtype 解析 |
| `cortexnet/ops/npu_ops.py` | NPU/后端算子抽象 |
| `cortexnet/distributed.py` | 分布式训练工具 |
| `cortexnet/training_utils.py` | 训练辅助工具 |
| `cortexnet/quantization.py` | 量化与压缩工具 |
| `cortexnet/transformer_baseline.py` | Transformer 对照基线 |

## Scripts (Non-Library)

| File | Responsibility |
|---|---|
| `scripts/benchmarks/*.py` | 性能、长上下文、冷启动、量化基准 |
| `scripts/eval/eval_ability_compare.py` | 能力评测 |
| `scripts/chat/chat_qwen3_cortexnet.py` | 聊天交互脚本 |
| `scripts/dev/one_click_cortexnet.py` | 一键流程脚本 |
| `scripts/dev/run_tests.py` | 脚本化回归测试运行器 |

## Dead/Redundant Code Cleanup Done

- 已移除：未使用导入、未使用局部变量、无效 `f-string`、错误路径注入逻辑
- 已隔离：脚本代码从库代码中拆分到 `scripts/`
- 已标准化：测试与脚本统一使用 `cortexnet.*` 导入

