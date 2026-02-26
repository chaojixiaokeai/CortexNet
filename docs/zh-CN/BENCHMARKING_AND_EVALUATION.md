# CortexNet 基准与评测方法

版本：3.2.1

---

## 1. 目的

本文件定义 CortexNet 的评测目标、指标口径与脚本使用方式，确保不同环境下结果可比较、可复现。

---

## 2. 评测维度

1. 延迟：首 token 延迟、总生成时长。
2. 吞吐：tokens/s。
3. 资源：峰值 RSS、设备占用。
4. 稳定性：连续运行波动、异常率。
5. 质量：规则检查通过率、对照模型文本相似度。

---

## 3. 脚本清单

### 3.1 推理对比

- `scripts/benchmarks/benchmark_cortexnet_vs_transformer.py`

对比项：

1. 首 token 延迟
2. 解码速度
3. 峰值内存
4. 固定问题回答对照

### 3.2 长上下文

- `scripts/benchmarks/benchmark_long_context.py`

对比不同 context 长度下的延迟、吞吐和内存曲线。

### 3.3 冷启动

- `scripts/benchmarks/benchmark_coldstart.py`

关注首次加载、首次推理和缓存命中后的性能差异。

### 3.4 量化

- `scripts/benchmarks/benchmark_quantization.py`

对比多种量化策略在速度、内存与精度代理指标上的变化。

### 3.5 Lite 对照

- `scripts/benchmarks/benchmark_lite.py`

对比 Lite / Full / Transformer 路径差异，适合快速本地评估。

### 3.6 能力评测

- `scripts/eval/eval_ability_compare.py`

输出固定问题回答、规则化检查分数与跨引擎相似度。

---

## 4. 标准执行建议

### 4.1 环境固定

1. 固定 Python、PyTorch、驱动版本。
2. 固定设备与电源策略（避免自动降频干扰）。
3. 固定随机种子。

### 4.2 执行顺序

1. 预热脚本
2. 冷启动评测
3. 常规推理对比
4. 长上下文评测
5. 质量评测

### 4.3 结果保存

建议统一输出 JSON，按日期和 commit SHA 归档。

---

## 5. 指标解释

1. `first_token_latency_s`：交互体验核心指标。
2. `decode_tokens_per_s`：持续生成效率指标。
3. `peak_rss_gb`：主机内存压力指标。
4. `checks_pass_rate`：规则化质量代理指标。
5. `similarity`：与对照输出的文本相似度（非真实性指标）。

---

## 6. 示例命令

```bash
python scripts/benchmarks/benchmark_cortexnet_vs_transformer.py \
  --model-path "/path/to/model" \
  --device auto \
  --dtype auto \
  --save-json benchmark_compare.json

python scripts/benchmarks/benchmark_long_context.py \
  --model-path "/path/to/model" \
  --context-lengths 512,1024,2048,4096 \
  --save-json benchmark_long_context.json

python scripts/benchmarks/benchmark_coldstart.py \
  --model-path "/path/to/model" \
  --save-json benchmark_coldstart.json

python scripts/eval/eval_ability_compare.py \
  --model-path "/path/to/model" \
  --engine cortex
```

---

## 7. 结果解读建议

1. 不仅看平均值，也看 P95 与方差。
2. 冷启动与热启动分开报告。
3. 质量评测至少保留同一题集版本，避免口径漂移。
4. 同步记录关键配置（compatibility_mode、lazy、dtype）。

---

## 8. 失真来源与控制

常见失真来源：

1. 不同 run 的设备负载不同。
2. tokenizer 或数据预处理开销混入主指标。
3. 缓存状态不一致（首轮与多轮混算）。
4. 未固定采样参数导致输出路径变化。

控制建议：

1. 独立进程评测。
2. 提前声明 warmup 策略。
3. 结果文件附带完整参数快照。

---

## 9. CI 集成建议

可在 CI 中引入“轻量基准守门”：

1. 小规模输入下验证功能与性能不出现明显回退。
2. 超过阈值时触发告警。
3. 报告附带 commit 信息方便回溯。

---

## 10. 结论模板

发布前建议输出统一结论模板：

1. 环境信息
2. 指标摘要（冷启动/热启动）
3. 质量摘要
4. 相比基线变化
5. 是否满足上线阈值
