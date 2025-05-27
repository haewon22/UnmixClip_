import torch
import torch.nn as nn
import torch.nn.functional as F

class MFILoss(nn.Module):
    def __init__(self, lambda_=0.2):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, t_prime):          # t_prime : (2N, D) or (N,D)
        vocab = t_prime.shape[0]          # 2N
        t_norm = t_prime[:vocab // 2, :]  # 양(positive) 텍스트만 사용
        S = t_norm @ t_norm.t()           # (N, N)

        diag = torch.diagonal(S)
        collapse = (diag - 1).pow(2).sum()
        off_diag = (S.pow(2).sum() - diag.pow(2).sum())

        return collapse + self.lambda_ * off_diag


class AsymmetricLoss(nn.Module):               
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float | None = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    @staticmethod
    def _reduce(loss, reduction: str = "mean"):
        return loss.mean() if reduction == "mean" else loss.sum()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B, N]  (raw scores: 긍정 로짓)
        targets : [B, N]  (0/1 float)
        """
        targets = targets.float()
        xs_pos  = torch.sigmoid(logits)
        xs_neg  = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 기본 BCE
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # focal modulation
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt    = xs_pos * targets + xs_neg * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss *= (1 - pt).pow(gamma)

        return -self._reduce(loss, reduction="mean")