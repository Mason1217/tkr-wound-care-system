import torch
import numpy as np

from .base_mix import BaseMixAugment

class Mixup(BaseMixAugment):
    def __init__(self, alpha: float = 0.4, prob: float = 0.5, num_classes: int = 5):
        super().__init__(alpha=alpha, prob=prob, num_classes=num_classes)

    def _apply_mix(self, images: torch.Tensor, metadata: torch.Tensor, labels: torch.Tensor):
        batch_size = images.size(0)
        device = images.device

        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        index = torch.randperm(batch_size, device=device)

        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_metadata = lam * metadata + (1 - lam) * metadata[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return mixed_images, mixed_metadata, mixed_labels