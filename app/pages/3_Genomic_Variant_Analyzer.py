import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.theme import inject_custom_css, render_hero, render_result_banner  # noqa: E402
from src.explainability.attention_vis import plot_attention_strip  # noqa: E402
from src.genomics.encoding import BASES  # noqa: E402
from src.inference import load_genomics_model, predict_variant  # noqa: E402
from src.utils.helpers import load_config  # noqa: E402

st.set_page_config(page_title="Genomic Variant Analyzer", page_icon="🧬", layout="wide")
inject_custom_css()

render_hero(
    "🧬",
    "Genomic Variant Analyzer",
    "Paste a raw DNA sequence to get a pathogenicity score, with an attention heatmap when using the Transformer model.",
)

cfg = load_config()
seq_length = cfg["genomics"]["seq_length"]


@st.cache_resource
def _load_model():
    return load_genomics_model()


model = _load_model()

if model is None:
    st.warning(
        "No trained genomics model found under `models/`.\n\n"
        "Train one first — real, literature-cited variants ship in the repo:\n"
        "```bash\n"
        "python data/scripts/download_genomics.py --mode real_curated\n"
        "python data/scripts/preprocess.py --task genomics\n"
        "python -m src.genomics.train\n"
        "```"
    )
    st.stop()

default_seq = "ACGT" * (seq_length // 4)

sequence = st.text_area(
    f"DNA sequence (A/C/G/T, will be padded/truncated to {seq_length} bp)",
    value=default_seq,
    height=120,
)

invalid_chars = set(sequence.upper()) - set(BASES) - {"N"}
if invalid_chars:
    st.error(f"Sequence contains unsupported characters: {', '.join(sorted(invalid_chars))}")
    st.stop()

BASE_COLORS = {"A": "#14b8a6", "C": "#8b5cf6", "G": "#f472b6", "T": "#60a5fa", "N": "#475569"}
preview = sequence.upper()[:120]
chips = "".join(
    f'<span style="color:{BASE_COLORS.get(b, "#94a3b8")}; font-weight:700;">{b}</span>' for b in preview
)
st.markdown(f'<div class="glass-card" style="font-family: monospace; letter-spacing: 2px; word-break: break-all;">{chips}{"…" if len(sequence) > 120 else ""}</div>', unsafe_allow_html=True)
st.write("")

if st.button("Analyze variant", type="primary"):
    with st.spinner("Running inference..."):
        result = predict_variant(model, sequence, seq_length=seq_length, with_attention=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Prediction")
        pred_label = result["pred_label"]
        confidence = max(result["probabilities"].values())
        if pred_label == "Pathogenic":
            render_result_banner("warn", f"<b>{pred_label}</b> (confidence: {confidence:.1%})")
        else:
            render_result_banner("good", f"<b>{pred_label}</b> (confidence: {confidence:.1%})")

        st.write("")
        for label, prob in result["probabilities"].items():
            st.progress(prob, text=f"{label}: {prob:.1%}")

    with col2:
        st.markdown("#### Attention over sequence")
        if "attention_scores" in result:
            fig = plot_attention_strip(sequence, result["attention_scores"])
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(
                result.get(
                    "attention_error",
                    "Attention visualization requires the Transformer genomics model "
                    "(set genomics.model_type: transformer in config.yaml and retrain).",
                )
            )
