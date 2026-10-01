import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.theme import inject_custom_css, render_hero, render_result_banner  # noqa: E402
from src.inference import load_vision_model, predict_histopathology  # noqa: E402
from src.utils.helpers import load_config  # noqa: E402

st.set_page_config(page_title="Histopathology Analyzer", page_icon="🔬", layout="wide")
inject_custom_css()

render_hero(
    "🔬",
    "Histopathology Analyzer",
    "Upload a histopathology image patch to get a malignant/benign prediction with Grad-CAM explainability.",
)

cfg = load_config()
image_size = cfg["vision"]["image_size"]


@st.cache_resource
def _load_model():
    return load_vision_model()


model = _load_model()

if model is None:
    st.warning(
        "No trained vision model found at `models/vision_best.pt`.\n\n"
        "Train one first — the real Kaggle PatchCamelyon data is recommended:\n"
        "```bash\n"
        "python data/scripts/download_histopathology.py --dataset pcam\n"
        "python data/scripts/preprocess.py --task vision\n"
        "python -m src.vision.train\n"
        "```"
    )
    st.stop()

uploaded_file = st.file_uploader("Upload a histopathology image", type=["png", "jpg", "jpeg", "tif", "tiff"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col1:
        st.markdown('<div class="glass-card"><b>Input image</b></div>', unsafe_allow_html=True)
        st.image(image, width="stretch")

    with st.spinner("Running inference..."):
        result = predict_histopathology(model, image, image_size=image_size, with_gradcam=True)

    with col1:
        st.write("")
        pred_label = result["pred_label"]
        confidence = max(result["probabilities"].values())
        if "Malignant" in pred_label or "Tumor" in pred_label:
            render_result_banner("warn", f"<b>{pred_label}</b> (confidence: {confidence:.1%})")
        else:
            render_result_banner("good", f"<b>{pred_label}</b> (confidence: {confidence:.1%})")

        st.write("")
        for label, prob in result["probabilities"].items():
            st.progress(prob, text=f"{label}: {prob:.1%}")

    with col2:
        st.markdown('<div class="glass-card"><b>Grad-CAM explanation</b></div>', unsafe_allow_html=True)
        if "gradcam_overlay" in result:
            st.image(result["gradcam_overlay"], width="stretch", caption="Warmer regions influenced the prediction more")
        else:
            st.info(result.get("gradcam_error", "Grad-CAM unavailable for this backbone."))
else:
    st.info("👆 Upload an image to get started, or try a sample from `data/raw/histopathology/` after downloading the dataset.")
