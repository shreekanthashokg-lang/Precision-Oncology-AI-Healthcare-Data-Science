"""PyTorch Dataset for histopathology image patches."""
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class HistopathologyDataset(Dataset):
    """
    Reads a CSV with columns [path, label, label_idx] (produced by
    data/scripts/preprocess.py) and serves (image, label) pairs.
    """

    def __init__(self, csv_path: str, transform: Optional[Callable] = None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        if "path" not in self.df.columns or "label_idx" not in self.df.columns:
            raise ValueError(
                "Expected columns 'path' and 'label_idx' in "
                f"{csv_path}. Did you run data/scripts/preprocess.py?"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["path"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = int(row["label_idx"])

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


class InMemoryImageDataset(Dataset):
    """
    Lightweight dataset for a single image (or a small list of images) held
    in memory — used by the Streamlit app / inference script where images
    come from an upload widget rather than disk.
    """

    def __init__(self, images: np.ndarray, transform: Optional[Callable] = None):
        # images: list of HxWxC uint8 RGB arrays
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image
