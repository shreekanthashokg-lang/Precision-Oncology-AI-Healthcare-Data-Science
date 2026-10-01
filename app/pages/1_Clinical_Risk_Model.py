import sys
from pathlib import Path

import joblib
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.theme import inject_custom_css, render_hero, render_metric_cards, render_result_banner  # noqa: E402
from src.clinical.wisconsin import load_wdbc_dataframe  # noqa: E402

st.set_page_config(page_title="Clinical Risk Model (Real Data)", page_icon="🩺", layout="wide")
inject_custom_css()

render_hero(
    "🩺",
    "Clinical Risk Model — Real Patient Data",
    "Unlike the other branches (which can use synthetic or literature-curated data by default), this "
    "page runs on a genuinely real, patient-derived dataset: the Wisconsin Diagnostic Breast Cancer "
    "(WDBC) dataset — 569 real fine-needle-aspirate biopsies with pathologist-assigned labels.",
)

MODEL_PATH = Path("models/clinical_logreg.joblib")

if not MODEL_PATH.exists():
    st.warning(
        "No trained clinical model found at `models/clinical_logreg.joblib`.\n\n"
        "Train it (takes under a second, no download needed):\n"
        "```bash\n"
        "python -m src.clinical.wisconsin\n"
        "```"
    )
    st.stop()

artifact = joblib.load(MODEL_PATH)
scaler, model, feature_names = artifact["scaler"], artifact["model"], artifact["feature_names"]

df = load_wdbc_dataframe()

render_metric_cards(
    [
        ("Real patients", f"{len(df):,}", "fine-needle-aspirate biopsies"),
        ("Real features", f"{len(feature_names)}", "nuclear morphology measurements"),
        ("Test accuracy", "98.2%", "AUC-ROC 0.995"),
    ]
)
st.write("")

tab_lookup, tab_3d = st.tabs(["🔎 Sample lookup", "🌐 3D tumor landscape"])

with tab_lookup:
    st.markdown("#### Try a real biopsy sample")
    idx = st.slider("Pick a sample from the real dataset (index)", 0, len(df) - 1, 0)
    sample = df.iloc[idx]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="glass-card"><b>Real features for this sample</b></div>', unsafe_allow_html=True)
        st.dataframe(sample[feature_names].to_frame(name="value"), height=300)

    with col2:
        X = scaler.transform([sample[feature_names].values])
        prob = model.predict_proba(X)[0]
        pred_idx = int(model.predict(X)[0])
        pred_name = "benign" if pred_idx == 1 else "malignant"
        actual_name = sample["label_name"]

        if pred_name == "malignant":
            render_result_banner("warn", f"<b>Predicted: {pred_name}</b> (confidence: {max(prob):.1%})")
        else:
            render_result_banner("good", f"<b>Predicted: {pred_name}</b> (confidence: {max(prob):.1%})")

        st.write("")
        if pred_name == actual_name:
            render_result_banner("good", f"✅ Matches the real pathologist label: <b>{actual_name}</b>")
        else:
            render_result_banner("warn", f"⚠️ Real pathologist label was: <b>{actual_name}</b> — model disagreed on this sample.")

with tab_3d:
    st.markdown("#### All 569 real patients in 3D feature space, colored by real pathologist diagnosis")
    fig = px.scatter_3d(
        df,
        x="mean radius", y="mean texture", z="mean concavity",
        color="label_name",
        color_discrete_map={"malignant": "#f472b6", "benign": "#14b8a6"},
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#cbd5e1")),
        scene=dict(
            xaxis=dict(title="mean radius", color="#94a3b8", backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="mean texture", color="#94a3b8", backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)"),
            zaxis=dict(title="mean concavity", color="#94a3b8", backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)"),
        ),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Drag to rotate · scroll to zoom — notice how cleanly malignant/benign separate on just 3 real features")

st.markdown(
    """
### About this dataset

- **Source**: Street, Wolberg & Mangasarian (1993), *Nuclear feature extraction
  for breast tumor diagnosis* — 30 real morphology features (radius, texture,
  perimeter, smoothness, concavity, etc.) computed from digitized images of
  fine-needle-aspirate breast masses.
- **Size**: 569 real patient samples, each with a real pathologist-assigned
  diagnosis (malignant/benign).
- **No download required**: ships inside `scikit-learn`
  (`sklearn.datasets.load_breast_cancer`), so this page works fully offline.
- **Held-out test performance** (see `results/metrics/wdbc_real_results.json`
  for the full numbers): Logistic Regression reaches ~98% accuracy and
  ~0.995 AUC-ROC on a stratified 20% held-out test split.
"""
)
