"""
Genomics models for variant pathogenicity classification.

Both models take a one-hot encoded DNA sequence of shape (B, 4, L) and
expose:
  - forward(x) -> logits (B, num_classes)
  - get_embedding(x) -> pooled feature embedding (B, embed_dim), used by
    src/multimodal/fusion.py
  - get_attention(x) (Transformer only) -> attention weights for
    src/explainability/attention_vis.py
"""
import math

import torch
import torch.nn as nn


class DeepSEAStyleCNN(nn.Module):
    """A compact 1D-CNN in the spirit of DeepSEA / DeepVariant sequence encoders."""

    def __init__(self, seq_length: int = 200, embedding_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=8, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=8, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=4, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embed_dim = embedding_dim
        self.embedding_head = nn.Linear(128, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes),
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L)
        feats = self.conv_block(x).squeeze(-1)  # (B, 128)
        return torch.relu(self.embedding_head(feats))  # (B, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding(x)
        return self.classifier(emb)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DNATransformerEncoder(nn.Module):
    """
    A small DNABERT-style Transformer encoder operating directly on one-hot
    nucleotide inputs (rather than a k-mer tokenizer + pretraining, to keep
    this trainable from scratch on a laptop). Exposes attention weights from
    its final layer for explainability.
    """

    def __init__(
        self,
        seq_length: int = 200,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        embedding_dim: int = 64,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(4, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_length + 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Separate single-head attention module purely for visualization —
        # cheap, and decouples explainability from the internal encoder stack.
        self.attn_probe = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)

        self.embed_dim = embedding_dim
        self.embedding_head = nn.Linear(d_model, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, num_classes),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L) -> (B, L, 4)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)  # (B, L, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoding(x)
        return self.encoder(x)  # (B, L+1, d_model)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encode(x)
        cls_out = encoded[:, 0, :]  # (B, d_model)
        return torch.relu(self.embedding_head(cls_out))

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, L+1) attention weights of the [CLS] token over the sequence."""
        encoded = self._encode(x)
        cls_query = encoded[:, 0:1, :]
        _, attn_weights = self.attn_probe(cls_query, encoded, encoded)
        return attn_weights.squeeze(1)  # (B, L+1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding(x)
        return self.classifier(emb)


def build_genomics_model(
    model_type: str = "cnn",
    seq_length: int = 200,
    embedding_dim: int = 64,
    num_classes: int = 2,
) -> nn.Module:
    if model_type == "cnn":
        return DeepSEAStyleCNN(seq_length=seq_length, embedding_dim=embedding_dim, num_classes=num_classes)
    elif model_type == "transformer":
        return DNATransformerEncoder(seq_length=seq_length, embedding_dim=embedding_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown genomics model_type: {model_type}")
