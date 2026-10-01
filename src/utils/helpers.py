"""Miscellaneous shared helpers: config loading, logging, checkpointing."""
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


def load_config(path: str = "config/config.yaml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_logger(name: str, log_dir: str = "results/logs", level: str = "INFO") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers on repeated calls
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


class EarlyStopping:
    """Stops training when a monitored metric stops improving."""

    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True  # first score is always "best"

        improved = (
            score < self.best_score - self.min_delta
            if self.mode == "min"
            else score > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
