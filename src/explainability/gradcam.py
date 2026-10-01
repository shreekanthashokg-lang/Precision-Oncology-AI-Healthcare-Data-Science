"""
Grad-CAM explainability for the histopathology vision models.

Wraps pytorch-grad-cam so callers just need a trained model + one image
tensor, and get back a heatmap overlay ready for the Streamlit app or a
matplotlib figure in a notebook.
"""
from typing import Optional

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def _find_target_layer(model: torch.nn.Module):
    """
    Best-effort automatic target-layer selection: grabs the last
    convolutional layer in the backbone. Works for EfficientNet/ResNet-style
    timm backbones; for ViT backbones, Grad-CAM needs a different reshape
    transform (see pytorch-grad-cam's ViT examples) — not covered here since
    the default config uses a CNN backbone.
    """
    conv_layers = [m for m in model.backbone.modules() if isinstance(m, torch.nn.Conv2d)]
    if not conv_layers:
        raise ValueError(
            "No Conv2d layers found in model.backbone — Grad-CAM as configured "
            "here targets CNN backbones. For ViT backbones, use an "
            "attention-rollout visualization instead."
        )
    return [conv_layers[-1]]


def generate_gradcam_overlay(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_image: np.ndarray,
    target_class: Optional[int] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Args:
        model: trained TimmClassifier (see src/vision/models.py) in eval mode.
        input_tensor: normalized (1, 3, H, W) tensor, same preprocessing used
            at training time.
        rgb_image: (H, W, 3) float array in [0, 1], the *unnormalized* image
            to overlay the heatmap on.
        target_class: which class's Grad-CAM to compute; None = predicted class.

    Returns:
        (H, W, 3) uint8 heatmap-overlaid image.
    """
    model.eval()
    target_layers = _find_target_layer(model)

    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor.to(device), targets=targets)[0]

    overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return overlay
