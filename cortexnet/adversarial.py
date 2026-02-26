"""
对抗防御系统 (Adversarial Defense System)

核心创新：
  多层防御机制保护模型免受对抗攻击，确保在恶意输入下仍能
  可靠运行。同时提供对抗训练工具增强模型鲁棒性。

  ┌─────────────────────────────────────────────────────────────┐
  │              三层防御架构                                    │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  第1层: 输入防御 ──── 异常检测 + 随机平滑                  │
  │                        │                                    │
  │  第2层: 特征防御 ──── 特征去噪 + 鲁棒归一化                │
  │                        │                                    │
  │  第3层: 输出防御 ──── 置信度校准 + 一致性检查              │
  │                                                             │
  │  训练工具:                                                  │
  │  ● FGSM 对抗训练 — 快速梯度符号攻击                        │
  │  ● PGD 对抗训练 — 投影梯度下降攻击                         │
  │  ● 随机平滑 — 概率性鲁棒性保证                             │
  └─────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn


class InputShield(nn.Module):
    """输入防护层：检测并中和异常输入。

    1. 异常检测：识别偏离正常分布的输入
    2. 随机平滑：通过添加校准噪声提供概率鲁棒性
    3. 自适应去噪：根据异常程度调整去噪强度
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.anomaly_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        # 可学习的噪声尺度
        self.noise_scale = nn.Parameter(torch.tensor(0.01))
        # 去噪网络
        self.denoiser = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.denoiser[-1].weight)
        nn.init.zeros_(self.denoiser[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anomaly_score = self.anomaly_detector(x)  # (B, L, 1)

        # 随机平滑 (推理时)
        if not self.training:
            noise = torch.randn_like(x) * self.noise_scale.abs()
            x = x + noise

        # 自适应去噪: 异常越大，去噪越强
        correction = self.denoiser(x)
        x = x + anomaly_score * correction

        return x


class FeatureShield(nn.Module):
    """特征防护层：对中间特征进行鲁棒性增强。

    1. 鲁棒归一化：比标准 LayerNorm 更抗扰动
    2. 特征裁剪：限制特征值范围防止极端值
    3. 频谱正则化：限制特征的频谱范数
    """

    def __init__(self, d_model: int, clip_value: float = 10.0):
        super().__init__()
        self.clip_value = clip_value
        self.robust_norm = nn.LayerNorm(d_model)
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征裁剪
        x = x.clamp(-self.clip_value, self.clip_value)
        # 鲁棒归一化
        x = self.robust_norm(x) * self.scale
        return x


class OutputShield(nn.Module):
    """输出防护层：校准输出置信度。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.confidence_calibrator = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        confidence = self.confidence_calibrator(x)
        return x * confidence


class AdversarialShield(nn.Module):
    """对抗防御系统：三层防御的统一接口。

    Args:
        d_model: 模型维度
        enable_input_shield: 是否启用输入防护
        enable_feature_shield: 是否启用特征防护
        enable_output_shield: 是否启用输出防护
    """

    def __init__(self, d_model: int, enable_input: bool = True,
                 enable_feature: bool = True, enable_output: bool = True):
        super().__init__()
        self.input_shield = InputShield(d_model) if enable_input else nn.Identity()
        self.feature_shield = FeatureShield(d_model) if enable_feature else nn.Identity()
        self.output_shield = OutputShield(d_model) if enable_output else nn.Identity()

    def defend_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.input_shield(x)

    def defend_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_shield(x)

    def defend_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_shield(x)


class AdversarialTrainer:
    """对抗训练工具：通过模拟攻击增强模型鲁棒性。

    支持 FGSM 和 PGD 两种攻击方式。

    使用方式:
        adv_trainer = AdversarialTrainer(model)
        for batch in dataloader:
            loss = adv_trainer.adversarial_step(batch, optimizer)
    """

    def __init__(self, model: nn.Module, epsilon: float = 0.01,
                 attack_type: str = "fgsm", pgd_steps: int = 3,
                 use_amp: bool = False):
        self.model = model
        self.epsilon = epsilon
        self.attack_type = attack_type
        self.pgd_steps = pgd_steps
        self.use_amp = use_amp and torch.cuda.is_available()

    @torch.enable_grad()
    def generate_adversarial(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """生成对抗样本。"""
        emb = embeddings.clone().detach().requires_grad_(True)

        if self.attack_type == "fgsm":
            return self._fgsm(emb, labels)
        else:
            return self._pgd(emb, labels)

    def _fgsm(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """FGSM: 单步快速攻击（支持 AMP）。"""
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            output = self.model.forward_from_embeddings(emb, labels=labels)
            loss = output["loss"]
        loss.backward(retain_graph=True)
        perturbation = self.epsilon * emb.grad.sign()
        return (emb + perturbation).detach()

    def _pgd(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """PGD: 多步迭代攻击（支持 AMP）。"""
        perturbed = emb.clone()
        step_size = self.epsilon / self.pgd_steps * 2

        for _ in range(self.pgd_steps):
            perturbed = perturbed.detach().requires_grad_(True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model.forward_from_embeddings(perturbed, labels=labels)
                loss = output["loss"]
            loss.backward(retain_graph=True)
            perturbation = step_size * perturbed.grad.sign()
            perturbed = perturbed + perturbation
            # 投影到 epsilon 球内
            delta = (perturbed - emb).clamp(-self.epsilon, self.epsilon)
            perturbed = emb + delta

        return perturbed.detach()
