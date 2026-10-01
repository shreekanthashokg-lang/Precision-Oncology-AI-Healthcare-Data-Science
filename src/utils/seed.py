"""Reproducibility helpers."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix all relevant RNG seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Trade a little speed for determinism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
