import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────
# ★ MFILoss : 배치 차원 없이 (2N, D) 입력   ← (저자 Q3)
# ─────────────────────────────────────────────────────────────
class MFILoss(nn.Module):
    def __init__(self, lambda_=0.2):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, t_prime):          # t_prime : (2N, D) or (N,D)
        t_norm = F.normalize(t_prime, dim=-1)         # (2N,D)
        S = t_norm @ t_norm.t()                       # (2N,2N)

        diag = torch.diagonal(S)
        collapse = (diag - 1).pow(2).sum()            # Σ(S_ii−1)²
        off_diag = (S.pow(2).sum() - (diag.pow(2)).sum())

        return collapse + self.lambda_ * off_diag


# ─────────────────────────────────────────────────────────────
# AsymmetricLoss - DualCoOp 스타일 (reduction='mean') ★
# ─────────────────────────────────────────────────────────────
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=2.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.g_neg, self.g_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logit_pos, logit_neg, target):
        """
        logit_pos / logit_neg : [B, N]   (output from model)
        target                : [B, N]   (0/1)
        """
        # 확률
        p_pos = torch.sigmoid(logit_pos)          # + 프롬프트
        p_neg = torch.sigmoid(logit_neg)          # − 프롬프트 (1-p_pos 아님!)

        # asymmetric clipping (논문 δ=0.05)
        if self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1)

        # CE
        loss_pos = target * torch.log(p_pos.clamp(min=self.eps))
        loss_neg = (1 - target) * torch.log(p_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # focusing
        pt = p_pos * target + p_neg * (1 - target)
        gamma = self.g_pos * target + self.g_neg * (1 - target)
        loss *= (1 - pt).pow(gamma)

        return -loss.mean()   # ★ batch mean (저자 Q5)