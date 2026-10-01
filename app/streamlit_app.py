"""
Streamlit demo app — Home page.

Run with:
    streamlit run app/streamlit_app.py

Other pages live in app/pages/ and are auto-discovered by Streamlit's
multipage app mechanism.
"""
import sys
from pathlib import Path

import streamlit as st

# Make `src` and `app` importable when launched as `streamlit run app/streamlit_app.py`
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.theme import inject_custom_css, render_dna_helix_3d, render_hero, render_metric_cards  # noqa: E402

st.set_page_config(
    page_title="Precision Oncology AI",
    page_icon="🧬",
    layout="wide",
)
inject_custom_css()

render_hero(
    "🧬",
    "Precision Oncology AI",
    "A multi-modal deep learning pipeline combining real genomic variant data, "
    "real clinical biopsy data, and histopathology imaging — with explainable, "
    "trained-from-scratch models for each.",
)

left, right = st.columns([1.15, 1])
with left:
    render_metric_cards(
        [
            ("Real datasets", "3", "ClinVar · WDBC · curated variants"),
            ("Modalities", "3", "Genomics + Clinical + Vision"),
            ("Best real AUC-ROC", "0.995", "Clinical branch, WDBC"),
        ]
    )
    st.write("")
    st.markdown(
        """
<div class="glass-card">
<span class="badge">🧬 ClinVar Real Triage</span> trained on <b>65,188 real
NCBI ClinVar variants</b> across 2,328 genes — 76% held-out accuracy, 0.77 AUC.<br><br>
<span class="badge">🩺 Clinical Risk Model</span> trained on <b>569 real
patient biopsies</b> (Wisconsin Diagnostic Breast Cancer) — 98% held-out accuracy.<br><br>
<span class="badge">🔬 Histopathology Analyzer</span> EfficientNet-B0/ViT +
Grad-CAM, ready for the real Kaggle PatchCamelyon dataset.<br><br>
<span class="badge">🧬 Genomic Variant Analyzer</span> CNN/Transformer with
attention visualization over 30 real, literature-cited cancer variants.<br><br>
<span class="badge">🔗 Multimodal Demo</span> late-fusion of vision + genomics embeddings.
</div>
""",
        unsafe_allow_html=True,
    )

with right:
    st.plotly_chart(render_dna_helix_3d(), width="stretch", config={"displayModeBar": False})
    st.caption("Drag to rotate · scroll to zoom")

st.write("")

with st.expander("📐 Architecture overview"):
    st.markdown(
        """
```mermaid
flowchart LR
    subgraph Vision Branch
        A[Histopathology Patch] --> B[EfficientNet-B0 / ViT]
        B --> C[Vision Embedding]
        B --> D[Grad-CAM Heatmap]
    end
    subgraph Genomics Branch
        E[DNA Sequence] --> F[One-hot Encoding]
        F --> G[CNN / Transformer Encoder]
        G --> H[Genomics Embedding]
        G --> I[Attention Map]
    end
    C --> J[Late Fusion MLP]
    H --> J
    J --> K[Fused Prediction]
```
        """
    )
    st.caption("See docs/architecture.md for the full write-up.")

with st.expander("⚠️ Model status & data provenance"):
    st.markdown(
        """
**ClinVar Real Triage** & **Clinical Risk Model**: real data, real trained
models — run out of the box:
```bash
python -m src.genomics.clinvar_real
python -m src.clinical.wisconsin
```

**Vision / Genomics (sequence) / Multimodal**: this app looks for trained
checkpoints under `models/`:
- `vision_best.pt`
- `genomics_transformer_best.pt` or `genomics_cnn_best.pt`
- `fusion_best.pt` (optional, for a jointly fused prediction)

The genomics sequence branch defaults to data derived from real,
literature-cited variants (`data/real/curated_real_cancer_variants.csv`) —
see that file and `docs/architecture.md` for provenance and its limits.
The vision branch needs the real histopathology image dataset downloaded
separately (free Kaggle account — see the README's Quick Start).

If a checkpoint is missing, that page will tell you which training
command to run first.
        """
    )
