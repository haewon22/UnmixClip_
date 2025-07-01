# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MFILoss(nn.Module):
    def __init__(self, lambda_=0.2, HNS=False, beta=1.0, eps=1e-6):
        super().__init__()
        self.lambda_ = lambda_
        self.HNS = HNS
        self.beta = beta
        self.eps = eps

    def forward(self, t_prime):                      
        vocab = t_prime.shape[0]                     
        t_norm = F.normalize(t_prime, p=2, dim=1)    
        S = t_norm @ t_norm.t()                      

        diag = torch.diagonal(S)
        collapse = (diag - 1).pow(2).sum()            

        if self.HNS:
            mask = ~torch.eye(vocab, device=S.device, dtype=torch.bool)
            neg_sims = S[mask].view(vocab, vocab-1)               
            mean_neg = neg_sims.mean(dim=1, keepdim=True)        
            reweight = self.beta * neg_sims / (mean_neg + self.eps)
            hard_neg_loss = (reweight * neg_sims.pow(2)).sum()
            return collapse + self.lambda_ * hard_neg_loss
        else:
            off_diag = (S - torch.diag_embed(diag)).pow(2).sum() 
            return collapse + self.lambda_ * off_diag


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=2.0, gamma_pos=1.0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, HNS=False):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.HNS = HNS

    @staticmethod
    def _reduce(loss, reduction: str = "mean"):
        return loss.mean() if reduction == "mean" else loss.sum()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        xs_pos = torch.sigmoid(logits).clamp(min=self.eps, max=1-self.eps)
        xs_neg = (1.0 - xs_pos)

        p = torch.sigmoid(logits).clamp(self.eps, 1-self.eps)
        if self.clip:
            p_pos = (p - self.clip).clamp(min=0.0)
        else:
            p_pos = p
        p_neg = 1 - p_pos

        los_pos = targets * torch.log(xs_pos)
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        if self.HNS:
            with torch.no_grad():
                weight_neg = xs_neg / (xs_neg.mean(dim=1, keepdim=True) + self.eps)
            los_neg = los_neg * weight_neg

        loss = los_pos + los_neg

        # focal modulation
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt    = xs_pos * targets + xs_neg * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss  = loss * (1 - pt).pow(gamma)

        return -self._reduce(loss, reduction="mean")
