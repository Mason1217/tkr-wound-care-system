import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import register_loss

@register_loss("FocalLoss")
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean", device="cpu"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.device = device

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float, device=device).view(-1)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha.to(device).view(-1)
        else:
            self.alpha = torch.tensor([alpha], dtype=torch.float, device=device).view(-1)

    def forward(self, logits, targets):
        targets = targets.to(logits.device)
        
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
                
            if self.alpha.numel() == 1:
                at = self.alpha
            else:
                if targets.ndim == 1:
                    # Hard label
                    at = self.alpha[targets.long()]
                elif targets.ndim == 2 and targets.shape[1] == logits.shape[1]:
                    # Mixup case: soft label
                    at = (targets.float() * self.alpha).sum(dim=-1)
                elif targets.ndim == 2 and targets.shape[1] == 1:
                    # One-hot hard label
                    at = self.alpha[targets.long().squeeze(-1)]
                else:
                    at = 1.0
        else:
            at = 1.0
        
        at = at.view(-1) if isinstance(at, torch.Tensor) else at
        pt = pt.view(-1)
        ce_loss = ce_loss.view(-1)

        focal_loss = at * (1 - pt)**self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss