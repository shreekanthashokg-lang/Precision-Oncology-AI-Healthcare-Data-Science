import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.theme import inject_custom_css, render_hero  # noqa: E402
from src.inference import load_genomics_model, load_vision_model  # noqa: E402
from src.multimodal.fusion import LateFusionClassifier  # noqa: E402
from src.utils.helpers import load_checkpoint, load_config  # noqa: E402

st.set_page_config(page_title="Multimodal Demo", page_icon="🔗", layout="wide")
inject_custom_css()

render_hero(
    "🔗",
    "Multimodal Demo",
    "Combines the vision and genomics branches through a late-fusion classifier — "
    "illustrating how imaging and genomic signal can be jointly modeled for a patient.",
)

cfg = load_config()


@st.cache_resource
def _load_models():
    vision_model = load_vision_model()
    genomics_model = load_genomics_model()
    fusion_ckpt_path = Path(cfg["paths"]["models_dir"]) / "fusion_best.pt"
    fusion_model = None
    if fusion_ckpt_path.exists() and vision_model is not None and genomics_model is not None:
        ckpt = load_checkpoint(str(fusion_ckpt_path))
        fusion_model = LateFusionClassifier(
            vision_dim=vision_model.embed_dim,
            genomics_dim=genomics_model.embed_dim,
            hidden_dim=cfg["multimodal"]["hidden_dim"],
        )
        fusion_model.load_state_dict(ckpt["model_state"])
        fusion_model.eval()
    return vision_model, genomics_model, fusion_model


vision_model, genomics_model, fusion_model = _load_models()

if vision_model is None or genomics_model is None:
    st.warning(
        "This demo needs both a trained vision model and a trained genomics model. "
        "Train both first (see the Histopathology Analyzer and Genomic Variant Analyzer "
        "pages for the exact commands), then come back here."
    )
    st.stop()

if fusion_model is None:
    st.info(
        "No trained fusion head found at `models/fusion_best.pt` yet — this page will "
        "show individual-branch predictions plus an embedding correlation analysis "
        "instead of a jointly fused prediction. See "
        "`notebooks/05_multimodal_fusion.ipynb` to train the fusion head."
    )

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Histopathology image", type=["png", "jpg", "jpeg", "tif", "tiff"])
with col2:
    sequence = st.text_area("DNA sequence", value="ACGT" * (cfg["genomics"]["seq_length"] // 4), height=120)

if uploaded_file is not None and st.button("Run multimodal analysis", type="primary"):
    image = Image.open(uploaded_file)

    if fusion_model is not None:
        from src.inference import predict_multimodal

        result = predict_multimodal(
            vision_model,
            genomics_model,
            fusion_model,
            image,
            sequence,
            image_size=cfg["vision"]["image_size"],
            seq_length=cfg["genomics"]["seq_length"],
        )
        st.subheader("Fused prediction")
        st.json(result["probabilities"])
    else:
        from src.inference import predict_histopathology, predict_variant

        vres = predict_histopathology(vision_model, image, image_size=cfg["vision"]["image_size"], with_gradcam=False)
        gres = predict_variant(genomics_model, sequence, seq_length=cfg["genomics"]["seq_length"], with_attention=False)

        c1, c2 = st.columns(2)
        c1.metric("Vision branch prediction", vres["pred_label"])
        c2.metric("Genomics branch prediction", gres["pred_label"])
        st.caption(
            "These are independent, per-branch predictions since no fusion head is "
            "trained yet — train `LateFusionClassifier` to get a single joint score."
        )
