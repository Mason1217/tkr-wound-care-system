import torch.nn as nn
from omegaconf import DictConfig

from utils import filter_valid_args
from .registry import MODEL_REGISTRY
from .vit import ViT
from .timm_model import TimmModel
from .dino import DINOv2Classifier

def get_model(cfg: DictConfig) -> nn.Module:
    model_name = cfg.name
    yaml_params = cfg.get("params", {}) 

    defaults = {}
    target_class_name = None # Key for searching in MODEL_REGISTRY

    if model_name == "ViTB32":
        target_class_name = "ViT"
        defaults = {
            "backbone_name": "google/vit-base-patch32-224-in21k",
            "classifier_hidden_dim": 128,
        }
    
    elif model_name == "ViTB16":
        target_class_name = "ViT"
        defaults = {
            "backbone_name": "google/vit-base-patch16-224-in21k",
            "classifier_hidden_dim": 256,
        }
    
    elif model_name == "EfficientNet":
        target_class_name = "EfficientNet"
        defaults = {
            "backbone_name": "tf_efficientnetv2_s.in21k_ft_in1k",
            "dropout_rate": 0.2,
            "drop_path_rate": 0.1,
        }

    elif model_name == "MobileNetV3":
        target_class_name = "MobileNetV3"
        defaults = {
            "backbone_name": "mobilenetv3_large_100",
            "dropout_rate": 0.2,
            "drop_path_rate": 0.0,
        }
    
    elif model_name == "ResNet18":
        target_class_name = "ResNet18"
        defaults = {
            "backbone_name": "resnet18.a1_in1k",
            "dropout_rate": 0.2,
            "drop_path_rate": 0.0,
        }

    elif model_name == "ResNet50":
        target_class_name = "ResNet50"
        defaults = {
            "backbone_name": "resnet50.a1_in1k",
            "dropout_rate": 0.2,
            "drop_path_rate": 0.0,
        }

    elif model_name == "ViTB32FusionClassifier":
        target_class_name = "ViTFusionClassifier"
        defaults = {
            "backbone_name": "google/vit-base-patch32-224-in21k",
            "classifier_hidden_dim": 256,
        }
    
    elif model_name == "ViTB16FusionClassifier":
        target_class_name = "ViTFusionClassifier"
        defaults = {
            "backbone_name": "google/vit-base-patch16-224-in21k",
            "classifier_hidden_dim": 256,
        }

    elif model_name == "ViTS16FusionClassifier":
        target_class_name = "ViTFusionClassifier"
        defaults = {
            "backbone_name": "WinKawaks/vit-small-patch16-224",
            "classifier_hidden_dim": 256,
        }

    elif model_name == "ResNet50FusionClassifier":
        target_class_name = "TimmModelFusionClassifier"
        defaults = {
            "backbone_name": "resnet50.a1_in1k",
            "classifier_hidden_dim": 256,
            "dropout_rate": 0.2,
            "drop_path_rate": 0.0,
        }
        
    elif model_name == "ResNet18FusionClassifier":
        target_class_name = "TimmModelFusionClassifier"
        defaults = {
            "backbone_name": "resnet18.a1_in1k",
            "classifier_hidden_dim": 256,
            "dropout_rate": 0.2,
            "drop_path_rate": 0.0,
        }

    elif model_name == "DINOv2":
        target_class_name = "DINOv2Classifier"
        defaults = {
            "backbone_name": "assets/dinov2-small-local",
            "freeze_backbone": True,
        }

    elif model_name == "DINOv2FusionClassifier":
        target_class_name = "DINOv2FusionClassifier"
        defaults = {
            "backbone_name": "assets/dinov2-small-local",
        }

    else:
        target_class_name = model_name

    # --- Searching MODEL_REGISTRY ---
    if target_class_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' (mapped to '{target_class_name}') is not in MODEL_REGISTRY. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    
    model_cls = MODEL_REGISTRY[target_class_name]

    # Priority: YAML > Defaults (YAML would override Defaults)
    all_params = {**defaults, **yaml_params}
    final_params = filter_valid_args(model_cls, all_params)

    # --- Instantiate ---
    try:
        model = model_cls(**final_params)
    except TypeError as e:
        raise TypeError(f"Error initializing {model_name}: {e}")

    return model