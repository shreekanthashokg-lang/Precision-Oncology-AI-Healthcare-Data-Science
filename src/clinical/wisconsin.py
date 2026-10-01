"""
Real, offline, no-download tabular oncology dataset: the Wisconsin Diagnostic
Breast Cancer (WDBC) dataset — 569 real fine-needle-aspirate biopsy samples
with 30 real computed nuclear-morphology features per sample (radius,
texture, perimeter, area, smoothness, etc.), each labeled malignant or
benign by an actual pathologist. This ships inside scikit-learn itself
(`sklearn.datasets.load_breast_cancer`), so it requires no internet access,
no API keys, and no large downloads — a genuinely real clinical dataset
that runs anywhere `scikit-learn` is installed.

Reference: Street, W.N., Wolberg, W.H., Mangasarian, O.L. (1993). Nuclear
feature extraction for breast tumor diagnosis. IS&T/SPIE International
Symposium on Electronic Imaging.

This is intentionally a *different modality* (tabular clinical features)
from the imaging/genomics branches — it's included so the project ships
with at least one branch trained end-to-end on real patient-derived data
out of the box, with real, reproducible performance numbers, rather than
only synthetic or placeholder ones.
"""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.metrics import compute_metrics


def load_wdbc_dataframe() -> pd.DataFrame:
    """Returns the real WDBC dataset as a single labeled DataFrame."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    # sklearn encodes malignant=0, benign=1; keep that convention but add a readable column.
    df["label"] = data.target
    df["label_name"] = [data.target_names[i] for i in data.target]
    return df


def train_and_evaluate(seed: int = 42) -> Dict:
    """
    Trains two baseline classifiers (Logistic Regression, Random Forest) on
    the real WDBC data with a held-out test split, and returns real
    evaluation metrics for both. Fast enough (< 1 second) to run on any
    machine, including this repo's CI or a fresh clone with no GPU.
    """
    df = load_wdbc_dataframe()
    X = df.drop(columns=["label", "label_name"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    logreg = LogisticRegression(max_iter=5000, random_state=seed)
    logreg.fit(X_train_scaled, y_train)
    y_prob = logreg.predict_proba(X_test_scaled)[:, 1]
    y_pred = logreg.predict(X_test_scaled)
    # sklearn's label 1 = benign here; flip probability convention isn't needed
    # since compute_metrics just needs a consistent "positive class" probability.
    results["logistic_regression"] = compute_metrics(y_test, y_pred, y_prob).to_dict()

    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = rf.predict(X_test)
    results["random_forest"] = compute_metrics(y_test, y_pred_rf, y_prob_rf).to_dict()

    results["n_train"] = int(len(X_train))
    results["n_test"] = int(len(X_test))
    results["n_features"] = int(X.shape[1])
    results["feature_importances_top5_rf"] = sorted(
        zip(df.drop(columns=["label", "label_name"]).columns, rf.feature_importances_),
        key=lambda t: -t[1],
    )[:5]

    return results


def save_real_wdbc_csv(out_path: str = "data/real/wdbc_breast_cancer.csv") -> Path:
    """Materializes the real dataset as a CSV in the repo for inspection/notebooks."""
    df = load_wdbc_dataframe()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def train_and_save_best_model(seed: int = 42, out_path: str = "models/clinical_logreg.joblib") -> Dict:
    """
    Trains on the full real dataset (no held-out split — this is the
    deployed-model artifact, not the evaluation run) and persists a
    scaler + logistic regression pipeline for the Streamlit app to load.
    Evaluation numbers should come from `train_and_evaluate()`'s held-out
    split, not this fitted-on-everything artifact.
    """
    import joblib

    df = load_wdbc_dataframe()
    X = df.drop(columns=["label", "label_name"]).values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=5000, random_state=seed)
    model.fit(X_scaled, y)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model, "feature_names": list(df.columns[:-2])}, out)
    return {"saved_to": str(out), "n_samples": len(df)}


if __name__ == "__main__":
    save_real_wdbc_csv()
    metrics = train_and_evaluate()
    import json

    print(json.dumps(metrics, indent=2, default=str))
    print(train_and_save_best_model())
