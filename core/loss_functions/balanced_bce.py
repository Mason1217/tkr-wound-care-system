import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .registry import register_loss

@register_loss("BalancedBCELoss")
class BalancedBCELoss(nn.Module):
    def __init__(self, cls_freq_list, tau=1.0, reduction='mean', device='cpu'):
        super(BalancedBCELoss, self).__init__()
        
        if cls_freq_list is None:
            raise ValueError("cls_freq_list must be provided for BalancedBCELoss")

        cls_freq_list = np.array(cls_freq_list)
        total_samples = np.sum(cls_freq_list)

        if total_samples == 0:
            total_samples = 1 
            
        pi = torch.tensor(cls_freq_list / total_samples, dtype=torch.float32, device=device)
        
        eps = 1e-6 
        pi = torch.clamp(pi, min=eps, max=1-eps)
        
        C = len(cls_freq_list)
        self.bias = torch.log(pi) - torch.log(1 - pi) + torch.log(torch.tensor(C - 1.0))
        self.bias = self.bias.to(device)
        
        self.tau = tau
        self.reduction = reduction

    def forward(self, logits, targets):
        if self.bias.device != logits.device:
            self.bias = self.bias.to(logits.device)

        adjusted_logits = logits + self.tau * self.bias
        
        if targets.dim() == 2 and targets.shape[1] == logits.shape[1]:
            targets_one_hot = targets.float()
        else:
            num_classes = logits.size(1)
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
            
        loss = F.binary_cross_entropy_with_logits(
            adjusted_logits, 
            targets_one_hot, 
            reduction=self.reduction
        )
        return loss