import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.theme import inject_custom_css, render_hero  # noqa: E402

st.set_page_config(page_title="About / Skills Demonstrated", page_icon="📋", layout="wide")
inject_custom_css()

render_hero("📋", "About / Skills Demonstrated", "How this project maps onto real AI/ML engineer, deep learning intern, and data scientist job descriptions.")

st.markdown(
    """
### Why this project

Real precision-oncology teams (PathAI, Tempus, Foundation Medicine, Illumina,
Google Health, Recursion, Roche) increasingly need engineers who are
comfortable moving between **imaging deep learning** and **genomics/sequence
modeling** — not just one or the other. This project is a compact,
end-to-end demonstration of exactly that range, built to be run and audited
by anyone cloning the repo.

### Skills demonstrated, mapped to typical job requirements

| Job description ask | Where it's demonstrated |
|---|---|
| Real-world data wrangling at scale | `src/genomics/clinvar_real.py` — 65,188 real ClinVar variants, VEP feature engineering |
| PyTorch model development | `src/vision/models.py`, `src/genomics/models.py` |
| Transfer learning / CNNs (EfficientNet, ResNet) | `src/vision/models.py` via `timm` |
| Transformers / attention mechanisms | `src/genomics/models.py::DNATransformerEncoder` |
| Computer vision pipelines & augmentation | `src/vision/transforms.py`, `src/vision/dataset.py` |
| Genomic sequence encoding (one-hot, k-mer) | `src/genomics/encoding.py` |
| Classical ML on structured/tabular data | `src/clinical/wisconsin.py`, `src/genomics/clinvar_real.py` (scikit-learn pipelines) |
| Multi-modal fusion | `src/multimodal/fusion.py` |
| Explainable AI (Grad-CAM, attention viz) | `src/explainability/` |
| Reproducible experiment config | `config/config.yaml`, `src/utils/seed.py` |
| Training infra: AMP, early stopping, schedulers | `src/vision/train.py`, `src/genomics/train.py` |
| Evaluation & metrics (AUC-ROC, F1, confusion matrix) | `src/utils/metrics.py` |
| Deployment / demo-ready app | `app/streamlit_app.py` and pages, `docs/deployment.md` |
| Clean software engineering practices | modular `src/` package, `.gitignore`, docs |

### About the author

Built by a fresher-level AI/ML engineer as a portfolio project to
demonstrate readiness for AIML Engineer / ML Engineer / Deep Learning
Intern / Data Scientist roles in healthcare, biotech, and genomics.

See the main `README.md` for setup instructions, and `docs/architecture.md`
for a deeper technical write-up.
"""
)
