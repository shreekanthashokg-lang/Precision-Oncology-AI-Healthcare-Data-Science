"""
Unified inference helpers used by the Streamlit app (app/streamlit_app.py)
and available for standalone scripting.

Loads whichever checkpoints exist under models/ and exposes simple
predict_* functions that take raw inputs (a PIL image, a raw DNA sequence
string) and return predictions + explainability artifacts.
"""
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from src.explainability.attention_vis import get_attention_scores
from src.explainability.gradcam import generate_gradcam_overlay
from src.genomics.encoding import one_hot_encode
from src.genomics.models import build_genomics_model
from src.multimodal.fusion import LateFusionClassifier
from src.utils.helpers import load_checkpoint
from src.vision.models import build_vision_model
from src.vision.transforms import get_eval_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES_VISION = {0: "Benign / Normal Tissue", 1: "Malignant / Tumor Tissue"}
CLASS_NAMES_GENOMICS = {0: "Benign", 1: "Pathogenic"}


def load_vision_model(checkpoint_path: str = "models/vision_best.pt"):
    if not Path(checkpoint_path).exists():
        return None
    ckpt = load_checkpoint(checkpoint_path, map_location=str(DEVICE))
    model = build_vision_model(ckpt["backbone"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model


def load_genomics_model(checkpoint_path: Optional[str] = None):
    if checkpoint_path is None:
        # Prefer the transformer checkpoint (has attention viz); fall back to CNN.
        for candidate in ["models/genomics_transformer_best.pt", "models/genomics_cnn_best.pt"]:
            if Path(candidate).exists():
                checkpoint_path = candidate
                break
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        return None

    ckpt = load_checkpoint(checkpoint_path, map_location=str(DEVICE))
    model = build_genomics_model(
        ckpt["model_type"], ckpt["seq_length"], ckpt["embedding_dim"], ckpt["num_classes"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model


def predict_histopathology(model, pil_image: Image.Image, image_size: int = 96, with_gradcam: bool = True) -> Dict:
    rgb_image = np.array(pil_image.convert("RGB"))
    transform = get_eval_transforms(image_size)
    tensor = transform(image=rgb_image)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = int(np.argmax(probs))

    result = {
        "pred_class": pred_class,
        "pred_label": CLASS_NAMES_VISION[pred_class],
        "probabilities": {CLASS_NAMES_VISION[i]: float(p) for i, p in enumerate(probs)},
    }

    if with_gradcam:
        # Grad-CAM overlay needs the image resized to model input size, [0,1] float.
        import cv2

        resized = cv2.resize(rgb_image, (image_size, image_size)).astype(np.float32) / 255.0
        try:
            overlay = generate_gradcam_overlay(model, tensor, resized, target_class=pred_class, device=str(DEVICE))
            result["gradcam_overlay"] = overlay
        except Exception as e:  # pragma: no cover - defensive, backbone-dependent
            result["gradcam_error"] = str(e)

    return result


def predict_variant(model, sequence: str, seq_length: int = 200, with_attention: bool = True) -> Dict:
    encoded = one_hot_encode(sequence, seq_length)
    tensor = torch.from_numpy(encoded).permute(1, 0).unsqueeze(0).to(DEVICE)  # (1, 4, L)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = int(np.argmax(probs))

    result = {
        "pred_class": pred_class,
        "pred_label": CLASS_NAMES_GENOMICS[pred_class],
        "probabilities": {CLASS_NAMES_GENOMICS[i]: float(p) for i, p in enumerate(probs)},
    }

    if with_attention and hasattr(model, "get_attention"):
        try:
            attn = get_attention_scores(model, tensor, device=str(DEVICE))
            result["attention_scores"] = attn
        except Exception as e:  # pragma: no cover
            result["attention_error"] = str(e)

    return result


def predict_multimodal(
    vision_model,
    genomics_model,
    fusion_model: LateFusionClassifier,
    pil_image: Image.Image,
    sequence: str,
    image_size: int = 96,
    seq_length: int = 200,
) -> Dict:
    """Fuses a vision embedding and a genomics embedding through a trained LateFusionClassifier."""
    rgb_image = np.array(pil_image.convert("RGB"))
    vtransform = get_eval_transforms(image_size)
    vtensor = vtransform(image=rgb_image)["image"].unsqueeze(0).to(DEVICE)

    encoded = one_hot_encode(sequence, seq_length)
    gtensor = torch.from_numpy(encoded).permute(1, 0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        v_emb = vision_model.get_embedding(vtensor)
        g_emb = genomics_model.get_embedding(gtensor)
        logits = fusion_model(v_emb, g_emb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    return {
        "pred_class": pred_class,
        "probabilities": {f"class_{i}": float(p) for i, p in enumerate(probs)},
    }
