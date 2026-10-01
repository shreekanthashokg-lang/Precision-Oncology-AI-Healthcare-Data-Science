"""
Multimodal fusion of vision and genomics embeddings.

Two ideas are implemented, matching typical "day 1 multimodal" approaches
seen in precision-oncology literature:

1. LateFusionClassifier: concatenates the pooled vision embedding and the
   pooled genomics embedding, then runs a small MLP head — a standard late
   ("decision-level") fusion baseline.

2. embedding_correlation(): a lightweight, training-free analysis that
   measures how correlated the two modalities' predictions/embeddings are
   for the same patient — useful as an exploratory notebook cell rather
   than a trained model.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr


class LateFusionClassifier(nn.Module):
    """
    Takes pre-computed vision and genomics embeddings (e.g. produced by
    `model.get_embedding(x)` for each branch) and predicts a fused label.
    Kept modality-agnostic: any two embedding dimensions can be fused.
    """

    def __init__(self, vision_dim: int, genomics_dim: int, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim + genomics_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, vision_embedding: torch.Tensor, genomics_embedding: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([vision_embedding, genomics_embedding], dim=-1)
        return self.mlp(fused)


def embedding_correlation(vision_embeddings: np.ndarray, genomics_embeddings: np.ndarray) -> Tuple[float, float]:
    """
    Exploratory correlation analysis between modalities for matched samples:
    flattens each modality's embedding to a single score per sample (mean
    activation) and computes Pearson correlation. This is intentionally
    simple — meant as a notebook diagnostic, not a trained model — see
    notebooks/05_multimodal_fusion.ipynb.

    Returns (correlation_coefficient, p_value).
    """
    vision_score = vision_embeddings.mean(axis=1)
    genomics_score = genomics_embeddings.mean(axis=1)
    corr, p_value = pearsonr(vision_score, genomics_score)
    return float(corr), float(p_value)
