"""
Training loop for the histopathology vision model.

Usage
-----
    python -m src.vision.train --config config/config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.helpers import EarlyStopping, get_logger, load_config, save_checkpoint
from src.utils.metrics import compute_metrics
from src.utils.seed import set_seed
from src.vision.dataset import HistopathologyDataset
from src.vision.models import build_vision_model
from src.vision.transforms import get_eval_transforms, get_train_transforms


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc="train" if train else "eval", leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = logits.argmax(dim=1).detach().cpu().numpy()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)
            all_probs.append(probs)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vcfg = cfg["vision"]
    set_seed(cfg["seed"])
    logger = get_logger("vision_train", cfg["logging"]["log_dir"], cfg["logging"]["level"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    processed_dir = Path(cfg["paths"]["processed_dir"]) / "histopathology"
    train_csv, val_csv = processed_dir / "train.csv", processed_dir / "val.csv"
    if not train_csv.exists():
        logger.error(
            f"{train_csv} not found. Run data/scripts/download_histopathology.py "
            "then data/scripts/preprocess.py --task vision first."
        )
        return

    train_ds = HistopathologyDataset(str(train_csv), transform=get_train_transforms(vcfg["image_size"]))
    val_ds = HistopathologyDataset(str(val_csv), transform=get_eval_transforms(vcfg["image_size"]))

    train_loader = DataLoader(train_ds, batch_size=vcfg["batch_size"], shuffle=True, num_workers=vcfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=vcfg["batch_size"], shuffle=False, num_workers=vcfg["num_workers"])

    model = build_vision_model(vcfg["backbone"], vcfg["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=vcfg["lr"], weight_decay=vcfg["weight_decay"])

    if vcfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=vcfg["epochs"])
    elif vcfg["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, vcfg["epochs"] // 3), gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    scaler = GradScaler(device.type) if (vcfg["mixed_precision"] and device.type == "cuda") else None
    early_stopper = EarlyStopping(patience=vcfg["early_stopping_patience"], mode="min")

    best_path = Path(cfg["paths"]["models_dir"]) / "vision_best.pt"

    for epoch in range(1, vcfg["epochs"] + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer, scaler, device, train=False)

        if vcfg["scheduler"] == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        logger.info(
            f"Epoch {epoch}/{vcfg['epochs']} | "
            f"train_loss={train_loss:.4f} acc={train_metrics.accuracy:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_metrics.accuracy:.4f} "
            f"auc={val_metrics.auc_roc:.4f} f1={val_metrics.f1:.4f}"
        )

        if early_stopper.step(val_loss):
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "backbone": vcfg["backbone"],
                    "num_classes": vcfg["num_classes"],
                    "epoch": epoch,
                    "val_metrics": val_metrics.to_dict(),
                },
                str(best_path),
            )
            logger.info(f"New best model saved to {best_path}")

        if early_stopper.should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    logger.info("Vision training complete.")


if __name__ == "__main__":
    main()
