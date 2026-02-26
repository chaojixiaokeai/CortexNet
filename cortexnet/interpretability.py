"""
可解释性与监控系统 (Interpretability & Monitoring System)

核心创新：
  实时追踪模型内部的"思维流"，让模型的决策过程透明可解释。

  ┌─────────────────────────────────────────────────────────┐
  │              CortexNet 思维流监控                        │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │  ┌─── 路径利用率分析 ──────────────────────────────┐   │
  │  │  追踪 SSM / Attention / Memory / GNN 的使用比例  │   │
  │  │  了解模型在不同层、不同 token 上偏好哪条路径     │   │
  │  └──────────────────────────────────────────────────┘   │
  │                                                         │
  │  ┌─── 专家路由可视化 ──────────────────────────────┐   │
  │  │  追踪哪些 token 被路由到哪些专家                 │   │
  │  │  检测专家负载均衡和专业化程度                     │   │
  │  └──────────────────────────────────────────────────┘   │
  │                                                         │
  │  ┌─── 注意力重要性分析 ────────────────────────────┐   │
  │  │  追踪哪些 token 被选为"重要 token"               │   │
  │  │  可视化稀疏注意力的选择模式                       │   │
  │  └──────────────────────────────────────────────────┘   │
  │                                                         │
  │  ┌─── 记忆系统状态 ───────────────────────────────┐    │
  │  │  监控工作记忆/情景记忆/语义记忆的活跃度          │   │
  │  │  追踪记忆控制器的分配策略                         │   │
  │  └──────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any
from collections import defaultdict


class ThoughtFlowMonitor:
    """思维流监控器：追踪模型内部的信息处理路径。

    通过 forward hook 机制捕获模型的内部状态，
    无需修改模型代码即可实现完整的可解释性。

    使用方法：
        monitor = ThoughtFlowMonitor(model)
        monitor.start_monitoring()

        output = model(input_ids)

        report = monitor.get_report()
        monitor.stop_monitoring()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List[Any] = []
        self.data: Dict[str, Any] = defaultdict(dict)
        self._monitoring = False

    def start_monitoring(self):
        """开始监控（注册 forward hooks）。"""
        self.stop_monitoring()  # 清除旧 hooks
        self.data = defaultdict(dict)
        self._monitoring = True
        self._register_hooks()

    def stop_monitoring(self):
        """停止监控（移除所有 hooks）。"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._monitoring = False

    def _register_hooks(self):
        """注册 forward hooks 到关键组件。"""
        # 检查模型是否有 blocks 属性
        blocks = getattr(self.model, "blocks", [])

        for layer_idx, block in enumerate(blocks):
            # 捕获融合门控权重
            fusion = getattr(block, "fusion", None)
            if fusion is not None:
                gate_module = getattr(fusion, "gate", None)
                if gate_module is not None:
                    self.hooks.append(
                        gate_module.register_forward_hook(
                            self._make_fusion_hook(layer_idx)
                        )
                    )

            # 捕获 MoE 路由决策
            moe = getattr(block, "moe", None)
            if moe is not None:
                router = getattr(moe, "router", None)
                if router is not None:
                    self.hooks.append(
                        router.register_forward_hook(
                            self._make_routing_hook(layer_idx)
                        )
                    )

    def _make_fusion_hook(self, layer_idx: int):
        def hook(module, input, output):
            # output 是 gate 的输出: (B, L, num_paths)
            if isinstance(output, torch.Tensor):
                weights = F.softmax(output.detach(), dim=-1)
                self.data["fusion_weights"][layer_idx] = (
                    weights.cpu()
                )
        return hook

    def _make_routing_hook(self, layer_idx: int):
        def hook(module, input, output):
            # output 是 router logits: (B*L, num_experts)
            if isinstance(output, torch.Tensor):
                probs = F.softmax(output.detach(), dim=-1)
                self.data["routing_probs"][layer_idx] = probs.cpu()
        return hook

    def get_report(self) -> Dict[str, Any]:
        """生成可解释性报告。"""
        report = {}

        # 路径利用率
        if "fusion_weights" in self.data:
            report["pathway_utilization"] = (
                self._analyze_pathway_utilization()
            )

        # 专家负载
        if "routing_probs" in self.data:
            report["expert_load"] = self._analyze_expert_load()

        return report

    def _analyze_pathway_utilization(self) -> Dict[str, List[float]]:
        """分析各层的路径利用率。"""
        path_names = ["SSM", "Attention", "Memory", "GraphReasoning"]
        utilization = {name: [] for name in path_names}

        for layer_idx in sorted(self.data["fusion_weights"].keys()):
            weights = self.data["fusion_weights"][layer_idx]
            mean_weights = weights.mean(dim=(0, 1))  # (num_paths,)

            for i, name in enumerate(path_names):
                if i < len(mean_weights):
                    utilization[name].append(mean_weights[i].item())

        return utilization

    def _analyze_expert_load(self) -> Dict[str, Any]:
        """分析专家负载分布。"""
        expert_loads = {}

        for layer_idx in sorted(self.data["routing_probs"].keys()):
            probs = self.data["routing_probs"][layer_idx]
            mean_probs = probs.mean(dim=0)  # (num_experts,)
            expert_loads[f"layer_{layer_idx}"] = {
                "mean_prob": mean_probs.tolist(),
                "max_prob": mean_probs.max().item(),
                "min_prob": mean_probs.min().item(),
                "balance_ratio": (
                    mean_probs.min() / mean_probs.max()
                ).item()
                if mean_probs.max() > 0
                else 0,
            }

        return expert_loads

    def print_summary(self):
        """打印监控摘要。"""
        report = self.get_report()

        print("\n  ╔═══ CortexNet 思维流报告 ═══╗\n")

        if "pathway_utilization" in report:
            print("  ◆ 路径利用率（各层平均）:")
            for path, values in report["pathway_utilization"].items():
                if values:
                    avg = sum(values) / len(values)
                    bar = "█" * int(avg * 40)
                    print(f"    {path:>14s}: {avg:.1%} {bar}")

        if "expert_load" in report:
            print("\n  ◆ 专家负载均衡:")
            for layer, stats in report["expert_load"].items():
                balance = stats.get("balance_ratio", 0)
                status = "✓ 均衡" if balance > 0.5 else "⚠ 不均衡"
                print(
                    f"    {layer}: balance={balance:.2f} {status}"
                )

        print("\n  ╚═══════════════════════════╝")
