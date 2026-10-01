import torch
import torch.nn as nn
from transformers import ViTModel
from peft import get_peft_model, LoraConfig

from .registry import register_model

@register_model("ViT")
class ViT(nn.Module):
    def __init__(
            self,

            # structure params
            backbone_name: str, 
            num_classes: int,
            classifier_hidden_dim: int = 256,
            dropout_rate: float = 0.5,
            backbone_dropout: float = 0.0,
            att_dropout: float = 0.0,
            
            # LoRA params
            use_lora: bool = False,
            lora_cfg: dict = None,
            
            **kwargs,
        ):
        super(ViT, self).__init__()
        
        self.backbone = ViTModel.from_pretrained(
            backbone_name,
            attn_implementation="eager",
            output_attentions=True,
            hidden_dropout_prob=backbone_dropout,
            attention_probs_dropout_prob=att_dropout,
        )
        
        hidden_size = self.backbone.config.hidden_size

        if use_lora and lora_cfg:
            peft_config = LoraConfig(
                r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("alpha", 16),
                target_modules=lora_cfg.get("target_modules", ["query", "value"]),
                lora_dropout=lora_cfg.get("dropout", 0.1),
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, peft_config)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(self, image, metadata=None):
        outputs = self.backbone(pixel_values=image, output_attentions=True)
        cls_embedding = outputs.pooler_output

        return self.classifier(cls_embedding)

    def get_embedding(self, images, features=None):
        outputs = self.backbone(pixel_values=images, output_attentions=False)
        return outputs.pooler_output


@register_model("ViTFusionClassifier")
class ViTFusionClassifier(nn.Module):
    def __init__(
            self,
            backbone_name: str, 
            num_classes: int,
            tabular_dim: int,
            tabular_hidden_dim: int = 32,
            classifier_hidden_dim: int = 256,
            dropout_rate: float = 0.5,
            backbone_dropout: float = 0.0,
            att_dropout: float = 0.0,
            use_lora: bool = False,
            lora_cfg: dict = None,
            **kwargs,
    ):
        super().__init__()
        print(f"Loading ViT Fusion: {backbone_name} ...")
        
        self.backbone = ViTModel.from_pretrained(
            backbone_name,
            attn_implementation="eager",
            output_attentions=True,
            hidden_dropout_prob=backbone_dropout,
            attention_probs_dropout_prob=att_dropout,
        )
        
        self.embed_dim = self.backbone.config.hidden_size

        if use_lora and lora_cfg:
            peft_config = LoraConfig(
                r=lora_cfg.get("r", 8),
                lora_alpha=lora_cfg.get("alpha", 16),
                target_modules=lora_cfg.get("target_modules", ["query", "value"]),
                lora_dropout=lora_cfg.get("dropout", 0.1),
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, peft_config)

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
            
        outputs = self.backbone(pixel_values=images, output_attentions=True)
        cls_token = outputs.pooler_output 
        tab_embeds = self.tabular_encoder(features)
        
        fused_features = torch.cat([cls_token, tab_embeds], dim=1)
        return self.classifier(fused_features)

    def get_embedding(self, images, features):
        if features is None or features.nelement() == 0:
            raise ValueError("Tabular features are missing. This model requires multimodal input.")

        outputs = self.backbone(pixel_values=images, output_attentions=False)
        cls_token = outputs.pooler_output
        tab_embeds = self.tabular_encoder(features)

        return torch.cat([cls_token, tab_embeds], dim=1)