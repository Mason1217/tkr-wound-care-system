from abc import ABC, abstractmethod
import torch
import cv2
import numpy as np

from .gradcam import GradCAM

class BaseHeatmapVisualizer(ABC):
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.attention_weights = None

    @abstractmethod
    def generate_heatmap(self, img_tensor: torch.Tensor, original_img: np.ndarray) -> np.ndarray:
        pass

    def _pad_numpy_to_square(self, img: np.ndarray, padding_value: tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
        
        h, w = img.shape[:2]
        max_wh = max(w, h)
        p_left, p_top = (max_wh - w) // 2, (max_wh - h) // 2
        p_right, p_bottom = max_wh - w - p_left, max_wh - h - p_top
        
        padded_img = cv2.copyMakeBorder(
            img, 
            p_top, p_bottom, p_left, p_right, 
            cv2.BORDER_CONSTANT, 
            value=padding_value
        )
        return padded_img


class DINOv2HeatmapVisualizer(BaseHeatmapVisualizer):
    def generate_heatmap(self, img_tensor: torch.Tensor, original_img: np.ndarray) -> np.ndarray:
        
        self.attention_weights = None

        with torch.inference_mode():
            outputs = self.model.backbone(img_tensor, output_attentions=True)
            attentions = outputs.attentions
            self.attention_weights = attentions[-1].cpu()
        
        return self._get_heatmap(original_img)
    
    def _get_heatmap(self, original_img: np.ndarray) -> np.ndarray:

        original_img = self._pad_numpy_to_square(original_img)
        sq_size = original_img.shape[0]

        if self.attention_weights is None: # shape: (num_attention_head, tokens, tokens) (e.g. (6, 257, 257))
            raise RuntimeError("[Heatmap]\t [Error]\t Hook failed to capture attention weights.")

        attn = self.attention_weights[0]
        attn = torch.mean(attn, dim=0)
        cls_attn = attn[0, 1:] # take cls token (first 0) and remove attention score about itself (second 1:), shape = (256,)

        grid_size = int(np.sqrt(cls_attn.size(0))) # e.g. image size = 224 * 224 and patch size is 16, so grid size = 224 / 14 = 16
        cls_attn = cls_attn.reshape(grid_size, grid_size).numpy() # shape = (16, 16)

        # Normalization and color
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8) # Normalization
        cls_attn = np.uint8(255 * cls_attn)
        heatmap = cv2.resize(cls_attn, (sq_size, sq_size))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        superimposed_img = heatmap * 0.4 + original_img * 0.6
        return superimposed_img.astype(np.uint8)

class CNNHeatmapVisualizer(BaseHeatmapVisualizer):
    def __init__(self, model: torch.nn.Module, target_layer_name: str = "layer4"):
        super().__init__(model)
        self.target_layer_name = target_layer_name
        
        # Locate target layer
        target_layer = self._get_target_layer()
        self.grad_cam_engine  = self._ini_gradcam(target_layer)

    def _get_target_layer(self):
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, self.target_layer_name):
            return getattr(self.model.backbone, self.target_layer_name)
        elif hasattr(self.model, self.target_layer_name):
            return getattr(self.model, self.target_layer_name)
        else:
            raise ValueError(f"[Heatmap]\t [Error]\t Target layer '{self.target_layer_name}' not found in the model {self.model.__class__.__name__}.")

    def _ini_gradcam(self, target_layer) -> GradCAM:
        if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, self.target_layer_name):
            return GradCAM(self.model.backbone, target_layer)
        elif hasattr(self.model, self.target_layer_name):
            return GradCAM(self.model, target_layer)
        else:
            raise ValueError(f"[Heatmap]\t [Error]\t Target layer '{self.target_layer_name}' not found in the model {self.model.__class__.__name__}.")


    def generate_heatmap(self, img_tensor: torch.Tensor, original_img: np.ndarray, target_class: int = None) -> np.ndarray:
        with torch.enable_grad():
            img_tensor = img_tensor.requires_grad_()
            heatmap, _ = self.grad_cam_engine(img_tensor, target_category_idx=target_class)

        return self._get_heatmap(original_img, heatmap)

    def _get_heatmap(self, original_img: np.ndarray, cam_heatmap: np.ndarray) -> np.ndarray:
        original_img = self._pad_numpy_to_square(original_img)
        sq_size = original_img.shape[0]

        # Convert the format of image and apply ColorMap (values in cam_heatmap are already between 0~1)
        cam_heatmap = np.uint8(255 * cam_heatmap)
        cam_heatmap = cv2.resize(cam_heatmap, (sq_size, sq_size))
        cam_heatmap = cv2.applyColorMap(cam_heatmap, cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

        # Overlap heatmap and original image
        superimposed_img = cam_heatmap * 0.4 + original_img * 0.6
        return superimposed_img.astype(np.uint8)

def get_heatmap_visualizer(model_name: str, model: torch.nn.Module, target_layer_name: str = "layer4") -> BaseHeatmapVisualizer:

    if "DINOv2" in model_name or "ViT" in model_name:
        print(f"[Heatmap]\t Using DINOv2HeatmapVisualizer (for vit) | model name: {model_name}")
        return DINOv2HeatmapVisualizer(model)
    elif "ResNet" in model_name or "CNN" in model_name:
        print(f"[Heatmap]\t Using CNNHeatmapVisualizer (for cnn) | model name: {model_name}")
        return CNNHeatmapVisualizer(model, target_layer_name=target_layer_name)
    else:
        raise NotImplementedError(f"[Heatmap]\t [Error]\t Heatmap visualizer for {model_name} is not implemented.")
        