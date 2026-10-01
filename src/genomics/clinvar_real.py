"""
Real, industry-scale genomics classifier trained on the actual ClinVar
"conflicting classifications" dataset (65,188 real human variants, VEP-
annotated, sourced from NCBI ClinVar — the same database clinical
genetics labs query daily).

Task: predict whether a real ClinVar variant has CONFLICTING clinical
classifications (CLASS=1) vs. a clean consensus classification (CLASS=0)
across submitting labs — a genuinely useful triage task: conflicting
variants are exactly the ones that need expert re-review.

This is real data end to end:
  - CHROM/POS/REF/ALT: real genomic coordinates
  - SYMBOL: real gene symbol
  - Consequence/IMPACT: real VEP-predicted functional consequence
  - SIFT/PolyPhen/CADD_PHRED/CADD_RAW/BLOSUM62/LoFtool: real, published
    in-silico pathogenicity predictors
  - AF_ESP/AF_EXAC/AF_TGP: real population allele frequencies
  - CLNDN: real associated clinical condition name(s)

No internet access or credentials needed to use this module — the CSV
ships in data/real/.
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.metrics import compute_metrics

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "real" / "clinvar_conflicting_real_65k.csv"

NUMERIC_FEATURES = [
    "AF_ESP", "AF_EXAC", "AF_TGP",
    "CADD_PHRED", "CADD_RAW", "BLOSUM62", "LoFtool",
    "STRAND", "ORIGIN",
]
CATEGORICAL_FEATURES = [
    "CHROM", "IMPACT", "Consequence", "CLNVC", "BIOTYPE",
    "SIFT", "PolyPhen",
]
TARGET = "CLASS"


def load_real_clinvar(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected the real ClinVar CSV at {csv_path}. This ships with the "
            "repo under data/real/ — if it's missing, re-clone or re-copy it."
        )
    return pd.read_csv(csv_path, low_memory=False)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def train_and_evaluate(seed: int = 42) -> Dict:
    """
    Trains on the real 65k-row ClinVar dataset with a held-out test split
    and returns real evaluation metrics for two baseline classifiers.
    Runs in a few seconds to ~1 minute on CPU depending on machine.
    """
    df = load_real_clinvar()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    results = {}

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    rf.fit(X_train_proc, y_train)
    y_prob_rf = rf.predict_proba(X_test_proc)[:, 1]
    y_pred_rf = rf.predict(X_test_proc)
    results["random_forest"] = compute_metrics(y_test, y_pred_rf, y_prob_rf).to_dict()

    gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=seed)
    gb.fit(X_train_proc, y_train)
    y_prob_gb = gb.predict_proba(X_test_proc)[:, 1]
    y_pred_gb = gb.predict(X_test_proc)
    results["gradient_boosting"] = compute_metrics(y_test, y_pred_gb, y_prob_gb).to_dict()

    results["n_train"] = int(len(X_train))
    results["n_test"] = int(len(X_test))
    results["n_real_variants_total"] = int(len(df))
    results["n_distinct_genes"] = int(df["SYMBOL"].nunique())
    results["class_balance"] = df[TARGET].value_counts(normalize=True).to_dict()

    return results, preprocessor, rf


def train_and_save_best_model(seed: int = 42, out_path: str = "models/clinvar_real_rf.joblib") -> Dict:
    """Trains on the full real dataset and persists the deployed model artifact."""
    import joblib

    df = load_real_clinvar()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].values

    preprocessor = build_preprocessor()
    X_proc = preprocessor.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    model.fit(X_proc, y)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": model,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        out,
    )
    return {"saved_to": str(out), "n_samples": len(df)}


if __name__ == "__main__":
    import json

    metrics, _, _ = train_and_evaluate()
    print(json.dumps(metrics, indent=2, default=str))
    print(train_and_save_best_model())
