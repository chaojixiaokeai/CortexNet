# CortexNet 技术架构书

版本：3.2.1  
文档定位：工程实现视角的完整架构说明  
目标读者：模型工程师、平台工程师、研究工程师、开源贡献者

---

## 1. 文档目标与范围

本架构书回答以下核心问题：

1. CortexNet 的系统边界与分层是什么。
2. 主模型从输入到输出的完整数据流如何组织。
3. 各核心模块（SSM、稀疏注意力、记忆、路由、适配器、设备抽象）如何协作。
4. 为什么采用当前目录与工程组织方式，如何支持开源维护。
5. 线上推理与离线训练分别如何部署与验证。

不在本文件展开的内容：

- 具体学术推导细节（见白皮书）
- 特定业务场景 Prompt 设计
- 第三方推理框架深度集成（如 vLLM/TensorRT-LLM）

---

## 2. 架构设计原则

CortexNet 的工程设计遵循以下原则：

- 可解释分层：模块职责清晰，目录层级与运行时分层一致。
- 默认可用：`pip install` 后可直接实例化与 smoke test。
- 向后兼容：对外主类统一为 `CortexNet`，保留历史别名兼容迁移。
- 渐进增强：Lite / Compatibility / Full 三种路径共存，按资源与目标切换。
- 可验证优先：每次改动能通过静态检查、单测、打包检查、脚本检查闭环。

---

## 3. 系统分层与目录映射

```text
应用层（scripts/*）
  ├─ benchmarks   性能、冷启动、长上下文、量化
  ├─ chat         交互式对话脚本
  ├─ eval         能力评估脚本
  └─ dev          回归与一键运行脚本

核心库层（cortexnet/*）
  ├─ model.py     主模型入口、生成、from_pretrained
  ├─ blocks.py    多路径 block 组合
  ├─ attention.py / ssm.py / memory.py / routing.py
  ├─ *_reasoning / *_learning / adversarial / interpretability
  ├─ adapter/*    外部模型识别、权重映射、架构适配、推理适配、校准
  └─ ops/*        设备与后端算子抽象

保障层（tests/* + .github/*）
  ├─ 单元/回归测试
  ├─ CI / CodeQL / Dependabot
  └─ 发布与质量门禁
```

---

## 4. 主模型数据流

### 4.1 训练/前向路径

```text
input_ids
  -> embedding + dropout
  -> N x CortexBlock
      -> norm
      -> SSM path
      -> Sparse Attention path
      -> Memory path
      -> (optional) graph/causal/agent/adversarial path
      -> Adaptive Fusion
      -> FFN or MoE
  -> final norm
  -> lm_head
  -> logits (+ loss/aux_loss in training)
```

### 4.2 生成路径

- 支持增量缓存（`past_cache`）
- 支持 `top_k`、`top_p`、温度、重复惩罚
- Lite 路径支持 SSM-only 解码阈值切换
- 可选 lazy 设备预热，降低冷启动体感

### 4.3 预训练模型迁移路径（`from_pretrained`）

1. 模型类型识别（`adapter/model_registry.py`）
2. 目标配置生成（`CortexNetConfig`）
3. 模型构建（`CortexNet`）
4. 权重映射与加载（`adapter/weight_adapter.py`）
5. 架构适配（`adapter/arch_adapter.py`）
6. 可选轻量校准（`adapter/calibrator.py`）
7. 设备与精度就绪（`ops/device_manager.py`）

---

## 5. 核心模块说明

### 5.1 状态空间模块（SSM）

- 文件：`cortexnet/ssm.py`
- 作用：线性复杂度序列建模，兼顾长依赖
- 特性：多尺度、增量状态缓存、Triton 可用时自动优化

### 5.2 选择性稀疏注意力

- 文件：`cortexnet/attention.py`
- 作用：在可控复杂度下保留关键信息交互
- 特性：Top-K、滑窗、RoPE、GQA、增量缓存

### 5.3 记忆与条件计算

- `memory.py`：突触记忆
- `hierarchical_memory.py`：分层记忆（工作/情景/语义）
- `routing.py`：MoE 路由、容量与均衡策略

### 5.4 进阶能力模块

- 因果推理：`causal_reasoning.py`
- 图推理：`graph_reasoning.py`
- 自我进化控制：`self_evolution.py`
- 多智能体协作：`multi_agent.py`
- 对抗鲁棒：`adversarial.py`
- 可解释监控：`interpretability.py`

---

## 6. 运行模式

CortexNet 主要支持三类运行形态：

1. Lite：参数高效，优先速度与部署简洁。
2. Compatibility：迁移已有大模型时的保真路径。
3. Full：启用更多进阶能力模块，适合研究验证。

建议策略：

- 资源受限或在线服务：Lite/Compatibility
- 架构实验与能力探索：Full

---

## 7. 平台与设备抽象

- 文件：`cortexnet/ops/device_manager.py`
- 支持：`cpu/cuda/mps/npu/mlu`
- 能力：设备优选、dtype 解析、统一接口 fallback

这样做的价值：

- 降低上层脚本分支判断复杂度
- 保证不同环境下的最小可运行路径

---

## 8. 测试与质量门禁

### 8.1 本地质量链路

- `make lint`：语法/静态检查
- `make test-all`：`pytest` + 脚本回归
- `make check`：构建与分发校验

### 8.2 CI 与安全

- CI：`.github/workflows/ci.yml`
- 发布：`.github/workflows/publish.yml`
- 安全扫描：`.github/workflows/codeql.yml`
- 依赖升级：`.github/dependabot.yml`

---

## 9. 可扩展点与二次开发建议

常见扩展入口：

- 新模型族适配：扩展 `ModelRegistry` + `WeightAdapter`
- 新设备后端：扩展 `ops/device_manager.py` 与 `ops/npu_ops.py`
- 新能力路径：在 `blocks.py` 中新增路径并更新融合策略
- 新评测任务：扩展 `scripts/eval/`

扩展约束建议：

- 保持 `CortexNet` 公共 API 稳定
- 新增能力必须附带最小回归测试
- 不在核心路径引入不可控全局状态

---

## 10. 已知边界与风险

- 某些第三方库（如 `torchao`）存在 deprecation warning，不影响当前可用性。
- 真实业务吞吐与延迟仍需在目标硬件上复测，不建议直接引用开发机结果。
- 兼容路径映射依赖源模型命名规范，异常权重命名需新增映射规则。

---

## 11. 维护建议

- 每个发布版本同步更新：`CHANGELOG.md`、`CITATION.cff`、`pyproject.toml`。
- 每次合入主干前至少通过：`make test-all && make check`。
- 保持文档与代码同频演进，新增模块必须补到 `MODULE_MAP.md` 与 `docs/`。

