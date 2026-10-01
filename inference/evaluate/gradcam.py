import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer):
        '''
        Args:
            target_layer: target conv layer (usually last one)
        '''
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_category_idx: int = None):
        '''
        Args:
            input_tensor: (1, C, H, W)
            target_category_idx(int): if None, use class with highest prediction score
        '''
        self.model.eval()
        logits = self.model(input_tensor)

        if target_category_idx is None:
            target_category_idx = torch.argmax(logits, dim=1).item()
        
        self.model.zero_grad()

        target = logits[0, target_category_idx]
        target.backward()

        # Generate CAM
        # gradients: (1, C, H, W) -> Global Average Pooling -> (1, C, 1, 1)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        activations = self.activations.detach()

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap.cpu().numpy()
        heatmap /= np.max(heatmap) + 1e-8

        return heatmap, target_category_idx