"""
Vision backbones for histopathology classification.

All models expose:
  - forward(x) -> logits of shape (B, num_classes)
  - get_embedding(x) -> pooled feature embedding of shape (B, embed_dim),
    used by src/multimodal/fusion.py
"""
import timm
import torch
import torch.nn as nn


class TimmClassifier(nn.Module):
    """Thin wrapper around any timm backbone (EfficientNet, ResNet, ViT, ...)."""

    def __init__(self, backbone: str = "efficientnet_b0", num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        # num_classes=0 -> timm returns pooled features instead of logits,
        # which lets us reuse the same feature extractor for embeddings.
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.num_features
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.get_embedding(x)
        feats = self.dropout(feats)
        return self.classifier(feats)


def build_vision_model(backbone: str = "efficientnet_b0", num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Factory used by train.py / inference.py. `backbone` maps directly onto a
    timm model name, e.g.:
      - "efficientnet_b0"
      - "resnet50"
      - "vit_small_patch16_224"
    """
    return TimmClassifier(backbone=backbone, num_classes=num_classes, pretrained=pretrained)
