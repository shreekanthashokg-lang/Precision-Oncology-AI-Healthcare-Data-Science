"""
Training loop for the genomics variant-impact classifier.

Usage
-----
    python -m src.genomics.train --config config/config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.genomics.dataset import VariantSequenceDataset
from src.genomics.models import build_genomics_model
from src.utils.helpers import EarlyStopping, get_logger, load_config, save_checkpoint
from src.utils.metrics import compute_metrics
from src.utils.seed import set_seed


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for seqs, labels in tqdm(loader, desc="train" if train else "eval", leave=False):
            seqs, labels = seqs.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(seqs)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * seqs.size(0)
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
    gcfg = cfg["genomics"]
    set_seed(cfg["seed"])
    logger = get_logger("genomics_train", cfg["logging"]["log_dir"], cfg["logging"]["level"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    processed_dir = Path(cfg["paths"]["processed_dir"]) / "genomics"
    train_csv, val_csv = processed_dir / "train.csv", processed_dir / "val.csv"
    if not train_csv.exists():
        logger.error(
            f"{train_csv} not found. Run data/scripts/download_genomics.py "
            "then data/scripts/preprocess.py --task genomics first."
        )
        return

    train_ds = VariantSequenceDataset(str(train_csv), seq_length=gcfg["seq_length"])
    val_ds = VariantSequenceDataset(str(val_csv), seq_length=gcfg["seq_length"])

    train_loader = DataLoader(train_ds, batch_size=gcfg["batch_size"], shuffle=True, num_workers=gcfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=gcfg["batch_size"], shuffle=False, num_workers=gcfg["num_workers"])

    model = build_genomics_model(
        gcfg["model_type"], gcfg["seq_length"], gcfg["embedding_dim"], gcfg["num_classes"]
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=gcfg["lr"], weight_decay=gcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=gcfg["epochs"])

    early_stopper = EarlyStopping(patience=gcfg["early_stopping_patience"], mode="min")
    best_path = Path(cfg["paths"]["models_dir"]) / f"genomics_{gcfg['model_type']}_best.pt"

    for epoch in range(1, gcfg["epochs"] + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{gcfg['epochs']} | "
            f"train_loss={train_loss:.4f} acc={train_metrics.accuracy:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_metrics.accuracy:.4f} "
            f"auc={val_metrics.auc_roc:.4f} f1={val_metrics.f1:.4f}"
        )

        if early_stopper.step(val_loss):
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "model_type": gcfg["model_type"],
                    "seq_length": gcfg["seq_length"],
                    "embedding_dim": gcfg["embedding_dim"],
                    "num_classes": gcfg["num_classes"],
                    "epoch": epoch,
                    "val_metrics": val_metrics.to_dict(),
                },
                str(best_path),
            )
            logger.info(f"New best model saved to {best_path}")

        if early_stopper.should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    logger.info("Genomics training complete.")


if __name__ == "__main__":
    main()
