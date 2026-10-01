import torch
import numpy as np

from .base_mix import BaseMixAugment

class CutMix(BaseMixAugment):
    def __init__(self, alpha: float = 1.0, prob: float = 0.5, num_classes: int = 5):
        super().__init__(alpha=alpha, prob=prob, num_classes=num_classes)

    def _apply_mix(self, images: torch.Tensor, metadata: torch.Tensor, labels: torch.Tensor):
        batch_size = images.size(0)
        device = images.device

        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        index = torch.randperm(batch_size, device=device)

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        actual_lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2))) # based on area ratio

        mixed_images = images.clone()
        mixed_images[:, :, bbx1: bbx2, bby1: bby2] = images[index, :, bbx1: bbx2, bby1: bby2]

        mixed_metadata = actual_lam * metadata + (1 - actual_lam) * metadata[index]
        mixed_labels = actual_lam * labels + (1 - actual_lam) * labels[index]

        return mixed_images, mixed_metadata, mixed_labels

    def _rand_bbox(self, size, lam):
        W, H = size[2], size[3]

        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2