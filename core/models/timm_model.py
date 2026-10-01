import torch
import torch.nn as nn
import timm
from .registry import register_model

@register_model("ResNet18")
@register_model("MobileNetV3")
@register_model("EfficientNet")
class TimmModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        dropout_rate: float = 0.2,
        drop_path_rate: float = 0.0,
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
            drop_path_rate=drop_path_rate
        )

    def forward(self, images, features=None):
        """
        Args:
            images: (B, C, H, W)
            features: (B, D)
        """
        logits = self.backbone(images)
        return logits

    def get_embedding(self, images, features=None):
        feats = self.backbone.forward_features(images) # shape: [B, C, H, W]
        return self.backbone.forward_head(feats, pre_logits=True) # shape: [B, D]


@register_model("TimmModelFusionClassifier")
class TimmModelFusionClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        tabular_dim: int,
        tabular_hidden_dim: int = 32,
        classifier_hidden_dim: int = 256,
        dropout_rate: float = 0.2,
        drop_path_rate: float = 0.0,
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__()
        print(f"Loading Timm Fusion: {backbone_name} ...")
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0, # make backbone output encoded vector instead of classification result 
            drop_rate=dropout_rate,
            drop_path_rate=drop_path_rate
        )

        self.embed_dim = self.backbone.num_features

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, tabular_hidden_dim),
            nn.ReLU(),
        )

        fusion_dim = self.embed_dim + tabular_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, classifier_hidden_dim),
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(self, images, features):
        if features is None or features.nelement() == 0:
            raise ValueError("Tabular features are missing. This model requires multimodal input.")
            
        img_embeds = self.backbone(images)
        tab_embeds = self.tabular_encoder(features)
        
        fused_features = torch.cat([img_embeds, tab_embeds], dim=1)
        return self.classifier(fused_features)

    def get_embedding(self, images, features):
        if features is None or features.nelement() == 0:
            raise ValueError("Tabular features are missing. This model requires multimodal input.")

        img_embeds = self.backbone(images)
        tab_embeds = self.tabular_encoder(features)

        return torch.cat([img_embeds, tab_embeds], dim=1)