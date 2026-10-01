"""PyTorch Dataset for DNA variant sequences."""
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.genomics.encoding import one_hot_encode


class VariantSequenceDataset(Dataset):
    """
    Reads a CSV with columns [sequence, label] (produced by
    data/scripts/preprocess.py) and serves one-hot encoded sequences.
    The CNN and Transformer models both consume the same (L, 4) one-hot
    tensor as input.
    """

    def __init__(self, csv_path: str, seq_length: int = 200):
        self.df = pd.read_csv(csv_path)
        self.seq_length = seq_length

        if "sequence" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(
                f"Expected columns 'sequence' and 'label' in {csv_path}. "
                "Did you run data/scripts/preprocess.py --task genomics?"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        encoded = one_hot_encode(row["sequence"], self.seq_length)  # (L, 4)
        # Models expect channels-first: (4, L)
        encoded = torch.from_numpy(encoded).permute(1, 0)
        label = int(row["label"])
        return encoded, label
