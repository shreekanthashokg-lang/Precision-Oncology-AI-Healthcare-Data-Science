import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.theme import inject_custom_css, render_hero, render_metric_cards, render_result_banner  # noqa: E402
from src.genomics.clinvar_real import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_real_clinvar,
)

st.set_page_config(page_title="ClinVar Variant Triage (Real, 65K)", page_icon="🧬", layout="wide")
inject_custom_css()

render_hero(
    "🧬",
    "ClinVar Variant Triage — Real, Industry-Scale Data",
    "Trained on the real NCBI ClinVar dataset: 65,188 real human variants across 2,328 real genes, "
    "VEP-annotated with real population allele frequencies and in-silico pathogenicity scores "
    "(SIFT, PolyPhen, CADD). Task: flag variants likely to have conflicting clinical classifications "
    "across submitting labs — i.e. the ones that most need expert re-review.",
)

MODEL_PATH = Path("models/clinvar_real_rf.joblib")

if not MODEL_PATH.exists():
    st.warning(
        "No trained model found at `models/clinvar_real_rf.joblib`.\n\n"
        "Train it (a few seconds to ~1 minute on CPU):\n"
        "```bash\n"
        "python -m src.genomics.clinvar_real\n"
        "```"
    )
    st.stop()

artifact = joblib.load(MODEL_PATH)
preprocessor, model = artifact["preprocessor"], artifact["model"]

df = load_real_clinvar()

render_metric_cards(
    [
        ("Real variants", f"{len(df):,}", "NCBI ClinVar"),
        ("Real genes", f"{df['SYMBOL'].nunique():,}", "distinct HGNC symbols"),
        ("Test AUC-ROC", "0.77", "held-out, seed=42"),
    ]
)
st.write("")

tab_lookup, tab_3d = st.tabs(["🔎 Variant lookup", "🌐 3D variant landscape"])

with tab_lookup:
    st.markdown("#### Look up a real variant from the dataset")
    gene_options = sorted(df["SYMBOL"].dropna().unique().tolist())
    default_idx = gene_options.index("BRCA1") if "BRCA1" in gene_options else 0
    gene = st.selectbox("Gene symbol", gene_options, index=default_idx)

    gene_variants = df[df["SYMBOL"] == gene].reset_index(drop=True)
    row_idx = st.slider(f"Variant index for {gene} ({len(gene_variants)} real variants)", 0, max(len(gene_variants) - 1, 0), 0)
    variant = gene_variants.iloc[row_idx]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            f"""
<div class="glass-card">
<b>Real variant details</b><br><br>
<b>Position</b>: chr{variant['CHROM']}:{variant['POS']} ({variant['REF']}&gt;{variant['ALT']})<br>
<b>Consequence</b>: {variant['Consequence']}<br>
<b>Impact</b>: {variant['IMPACT']}<br>
<b>Condition(s)</b>: {variant['CLNDN']}<br>
<b>SIFT</b>: {variant['SIFT']} &nbsp;|&nbsp; <b>PolyPhen</b>: {variant['PolyPhen']}<br>
<b>CADD Phred</b>: {variant['CADD_PHRED']}
</div>
""",
            unsafe_allow_html=True,
        )

    with col2:
        X = pd.DataFrame([variant[NUMERIC_FEATURES + CATEGORICAL_FEATURES]])
        X_proc = preprocessor.transform(X)
        prob = model.predict_proba(X_proc)[0]
        pred = int(model.predict(X_proc)[0])
        actual = int(variant["CLASS"])

        if pred == 1:
            render_result_banner("warn", f"<b>Predicted: likely to have conflicting classifications</b> (confidence: {prob[1]:.1%})")
        else:
            render_result_banner("good", f"<b>Predicted: consensus classification</b> (confidence: {prob[0]:.1%})")

        st.write("")
        actual_label = "Conflicting" if actual == 1 else "Consensus"
        if pred == actual:
            render_result_banner("good", f"✅ Matches the real ClinVar record: <b>{actual_label}</b>")
        else:
            render_result_banner("warn", f"ℹ️ Real ClinVar record shows: <b>{actual_label}</b> — model disagreed on this one.")

with tab_3d:
    st.markdown(
        "#### Real variants plotted in 3D feature space "
        "(CADD Phred × BLOSUM62 × log allele frequency), colored by real ClinVar class"
    )
    sample_n = st.slider("Number of real variants to plot", 500, min(8000, len(df)), 2500, step=500)
    plot_df = df.dropna(subset=["CADD_PHRED", "BLOSUM62", "AF_ESP"]).sample(
        min(sample_n, len(df)), random_state=42
    ).copy()
    plot_df["log_AF_ESP"] = plot_df["AF_ESP"].clip(lower=1e-6).apply(lambda v: -1 * (v ** 0.15))
    plot_df["Class"] = plot_df["CLASS"].map({0: "Consensus", 1: "Conflicting"})

    fig = px.scatter_3d(
        plot_df,
        x="CADD_PHRED", y="BLOSUM62", z="log_AF_ESP",
        color="Class",
        color_discrete_map={"Consensus": "#14b8a6", "Conflicting": "#f472b6"},
        opacity=0.65,
        hover_data=["SYMBOL", "Consequence"],
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#cbd5e1")),
        scene=dict(
            xaxis=dict(title="CADD Phred", color="#94a3b8", backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(title="BLOSUM62", color="#94a3b8", backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)"),
            zaxis=dict(title="transformed allele freq.", color="#94a3b8", backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.08)"),
        ),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Drag to rotate · scroll to zoom · hover for real gene/consequence detail")

st.markdown(
    """
### About this model
- **Data**: real NCBI ClinVar variant records (65,188 rows, 2,328 genes),
  VEP-annotated — the same underlying database clinical genetics labs use.
- **Held-out test performance** (20% stratified split, seed=42 — see
  `results/metrics/clinvar_real_results.json`): Gradient Boosting reaches
  ~76% accuracy / 0.77 AUC-ROC; Random Forest reaches ~69% accuracy / 0.77
  AUC-ROC with higher recall on the minority (conflicting) class.
- This is a genuinely hard, realistic task — conflicting-classification
  prediction is an active research problem, and these numbers are reported
  as measured, not cherry-picked.
"""
)
