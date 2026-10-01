import torch
import torch.nn as nn

class TKRFoldEnsemble(nn.Module):
    '''
    Ensemble one or more models and compute the average logits while inference.
    '''
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor, feature_list: list[torch.Tensor] | None = None, temperatures: list[float] | float | None = None) -> torch.Tensor:
        '''
        Args:
            images: shape (B, C, H, W)
            features: shape (B, tabular_dim) or None for image_only mode
            temperature: for calibration
        Returns:
            avg_probs: shape (B, num_classes)
        '''
        all_probs = []

        for i, model in enumerate(self.models):
            logits = model(images, self._get_features(i, feature_list))

            if isinstance(temperatures, list) and i < len(temperatures): t = temperatures[i]
            elif isinstance(temperatures, float): t = temperatures
            else: t = 1.0

            probs = torch.softmax(logits / t, dim=1) # using temperature scaling
            all_probs.append(probs)

        # Shape (K, B, num_classes) -> (B, num_classes)
        avg_probs = torch.stack(all_probs).mean(dim=0)

        return avg_probs
    
    def _get_features(self, idx: int, feature_list: list[torch.Tensor] | None) -> torch.Tensor | None:
        num_features = len(feature_list) if feature_list is not None else 0

        if num_features == 0 or feature_list is None: return None
        elif num_features == 1: return feature_list[0]
        
        return feature_list[idx]