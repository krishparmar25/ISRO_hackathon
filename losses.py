from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LunarIceLoss(nn.Module):
    """Composite Tversky + Focal + BCE loss for severe class imbalance."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 2.0,
        w_tversky: float = 0.5,
        w_focal: float = 0.3,
        w_bce: float = 0.2,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w_tversky = w_tversky
        self.w_focal = w_focal
        self.w_bce = w_bce

    def tversky_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        tp = (pred * target).sum()
        fp = (pred * (1.0 - target)).sum()
        fn = ((1.0 - pred) * target).sum()
        score = (tp + eps) / (tp + self.alpha * fp + self.beta * fn + eps)
        return 1.0 - score

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        return ((1.0 - pt) ** self.gamma * bce).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tv = self.tversky_loss(pred, target)
        fl = self.focal_loss(pred, target)
        bce = F.binary_cross_entropy(pred, target)
        return self.w_tversky * tv + self.w_focal * fl + self.w_bce * bce

