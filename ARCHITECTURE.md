# CortexNet Architecture

## 1. High-Level Pipeline

```text
Token IDs
  -> Embedding + Dropout
  -> [Cortex Block x N]
       - SSM path
       - Sparse Attention path
       - Memory path
       - (optional) reasoning/evolution/agent/defense paths
       - Adaptive Fusion
       - FFN / MoE
  -> Final RMSNorm
  -> LM Head
  -> Logits
```

## 2. Core Components

- `cortexnet/model.py`
  - 主模型入口 `CortexNet`
  - `from_pretrained`：自动识别外部模型、构造配置、映射权重、设备部署
  - 支持 lazy 设备预热、缓存映射权重、生成策略

- `cortexnet/blocks.py`
  - 基础并行路径块（SSM + Attention + Memory + Fusion + MoE）
  - 兼容扩展块（历史 V2/V3 路径仍保留为兼容层）

- `cortexnet/ssm.py`
  - 多尺度状态空间扫描（支持增量缓存）
  - 优先走 PyTorch，检测到 Triton 时启用对应优化路径

- `cortexnet/attention.py`
  - 选择性稀疏注意力（Top-K / 滑动窗口 / RoPE / GQA）
  - 兼顾全序列与增量解码缓存

- `cortexnet/memory.py`
  - 突触记忆建模，用于快速上下文适配

- `cortexnet/routing.py`
  - 条件计算路由（MoE / 协作专家）
  - 包含负载均衡辅助损失与容量控制

## 3. Extended Capability Modules

- `hierarchical_memory.py`: 分层记忆（工作/情景/语义）
- `graph_reasoning.py`: 图结构消息传递推理
- `meta_learning.py`: 任务自适应与上下文编码
- `multimodal.py`: 文本/图像/音频统一编码
- `continual_learning.py`: EWC + 回放机制
- `causal_reasoning.py`: 干预注意力与反事实分支
- `self_evolution.py`: 动态路径预算与进化控制
- `multi_agent.py`: 多智能体协作与消息板
- `adversarial.py`: 输入/特征/输出防御与对抗训练
- `interpretability.py`: 思维流监控与可解释性统计

## 4. Adaptation and Runtime Layer

- `cortexnet/adapter/`
  - `model_registry.py`: 外部模型识别与配置转换
  - `weight_adapter.py`: 权重映射与形状适配
  - `arch_adapter.py`: 架构行为微调
  - `inference_adapter.py`: 推理参数策略
  - `calibrator.py`: 小样本轻量校准

- `cortexnet/ops/`
  - 设备检测与优选（cpu/cuda/mps/npu/mlu）
  - 算子抽象与后端兼容接口

## 5. Code Organization Rules (Current)

- `cortexnet/` 只放可发布库代码
- `scripts/` 只放可执行脚本（基准、评测、交互）
- `tests/` 只放测试
- 对外主 API 固定为 `CortexNet`

## 6. Cleanup Summary

已完成的主要清理动作：

- 删除无用导入、未使用局部变量、重复/无效路径逻辑
- 清理缓存垃圾文件（`__pycache__`, `.DS_Store`）
- 修复脚本在新目录结构下的路径错误
- 统一测试和脚本导入为 `cortexnet.*`

