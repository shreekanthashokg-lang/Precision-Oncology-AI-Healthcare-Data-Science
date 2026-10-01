"""Classification metrics shared by the vision and genomics training loops."""
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    auc_roc: float
    f1: float
    precision: float
    recall: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> ClassificationMetrics:
    """
    y_true: (N,) int labels
    y_pred: (N,) int predicted labels (argmax)
    y_prob: (N,) float probability of the positive class
    """
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        # Happens if a batch/epoch only contains one class.
        auc = float("nan")

    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        auc_roc=auc,
        f1=f1_score(y_true, y_pred, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
    )


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)
