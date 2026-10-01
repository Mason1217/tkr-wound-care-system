import torch
import torch.nn as nn
import numpy as np

class BaseMixAugment(nn.Module):
    def __init__(self, alpha: float = 1.0, prob: float = 0.5, num_classes: int = 5):
        super().__init__()

        self.alpha = alpha
        self.prob = prob
        self.num_classes = num_classes

    def _to_tensor(self, labels: torch.Tensor) -> torch.Tensor:
        '''Ensure the format of labels is one-hot (soft label)'''

        if labels.ndim == 1:
            onehot = torch.zeros(labels.size(0), self.num_classes, device=labels.device, dtype=torch.float)
            onehot.scatter_(1, labels.unsqueeze(1), 1.0)

            return onehot
        
        return labels.float()
    
    def forward(self, images: torch.Tensor, metadata: torch.Tensor, labels: torch.Tensor):
        """
        Returns:
            tuple: images, metadata, labels
        """
        labels = self._to_tensor(labels)

        if np.random.rand() > self.prob:
            return images, metadata, labels
        
        return self._apply_mix(images, metadata, labels)
    
    def _apply_mix(self, images, metadata, labels):
        raise NotImplementedError