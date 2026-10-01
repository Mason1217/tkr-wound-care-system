import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig
from transformers import AutoModel, AutoConfig
from .registry import register_model

@register_model("DINOv2Classifier")
class DINOv2Classifier(nn.Module):
    def __init__(
            self,
            backbone_name: str = "facebook/dinov2-small",
            num_classes: int = 5,
            use_mlp: bool = False,
            mlp_dim: int = 256,
            use_lora: bool = False,
            lora_cfg: dict = None,
            **kwargs,
    ):
        super().__init__()
        print(f"Loading DINOv2 (HF): {backbone_name} ...")

        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            attn_implementation="eager"
        )
        
        if use_lora and lora_cfg:
            print(f"Applying LoRA: {lora_cfg}")
            peft_config = LoraConfig(
                r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("alpha", 16),
                target_modules=lora_cfg.get("target_modules", ["query", "value"]), 
                lora_dropout=lora_cfg.get("dropout", 0.1),
                bias="none",
                modules_to_save=[],
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()

        self.embed_dim = self.backbone.config.hidden_size

        if use_mlp:
            self.classifier = nn.Sequential(
                nn.Linear(self.embed_dim, mlp_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(mlp_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(self.embed_dim, num_classes)
    
    def forward(self, images, features=None):
        outputs = self.backbone(images)
        cls_token = outputs.last_hidden_state[:, 0, :]

        return self.classifier(cls_token)

    def get_attention_map(self, images):
        backbone = self.backbone
        if hasattr(backbone, "base_model"):
            backbone = backbone.base_model.model

        outputs = backbone(images, output_attentions=True)
        return outputs.attentions[-1]

    def get_embedding(self, images, features=None):
        outputs = self.backbone(images)
        return outputs.last_hidden_state[:, 0, :]


@register_model("DINOv2FusionClassifier")
class DINOv2FusionClassifier(nn.Module):
    def __init__(
            self,
            tabular_dim: int,
            backbone_name: str = "facebook/dinov2-small",
            num_classes: int = 2,
            tabular_hidden_dim: int = 32,
            mlp_dim: int = 256,
            use_lora: bool = False,
            lora_cfg: dict = None,
            **kwargs,
    ):
        super().__init__()
        print(f"Loading DINOv2 Fusion (HF): {backbone_name} ...")

        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            attn_implementation="eager",
        )

        if use_lora and lora_cfg:
            print(f"Applying LoRA to Backbone: {lora_cfg}")
            peft_config = LoraConfig(
                r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("alpha", 16),
                target_modules=lora_cfg.get("target_modules", ["query", "value"]),
                lora_dropout=lora_cfg.get("dropout", 0.1),
                bias="none",
                modules_to_save=[],
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
        
        self.embed_dim = self.backbone.config.hidden_size # 384 usually

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
            nn.Linear(fusion_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_dim, num_classes),
        )

    def forward(self, images, features):
        if features is None or features.nelement() == 0:
            raise ValueError("Tabular features are missing. This model requires multimodal input.")
        
        outputs = self.backbone(images)
        cls_token = outputs.last_hidden_state[:, 0, :]
        tab_embeds = self.tabular_encoder(features)

        fused_features = torch.cat([cls_token, tab_embeds], dim=1)
        return self.classifier(fused_features)

    def get_embedding(self, images, features):
        if features is None or features.nelement() == 0:
            raise ValueError("Tabular features are missing. This model requires multimodal input.")

        outputs = self.backbone(images)
        cls_token = outputs.last_hidden_state[:, 0, :]
        tab_embeds = self.tabular_encoder(features)

        return torch.cat([cls_token, tab_embeds], dim=1)