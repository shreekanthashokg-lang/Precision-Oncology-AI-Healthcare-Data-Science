"""
Shared preprocessing utilities: turns raw downloaded data into the
train/val/test splits consumed by src/vision and src/genomics.

Usage
-----
    python data/scripts/preprocess.py --task vision
    python data/scripts/preprocess.py --task genomics
    python data/scripts/preprocess.py --task all
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def preprocess_vision(seed: int = 42) -> None:
    """
    Builds an image-path/label CSV for whichever histopathology dataset was
    downloaded into data/raw/histopathology/.

    Supports two real layouts, auto-detected:
    1. PCam / "Histopathologic Cancer Detection" Kaggle competition format:
       data/raw/histopathology/pcam/train_labels.csv (columns: id,label)
       + data/raw/histopathology/pcam/train/<id>.tif
       This is the exact real format Kaggle ships for that competition —
       327,680 real 96x96 histopathology patches.
    2. Folder-per-class layout (BreakHis, Kather colorectal, etc.):
       data/raw/histopathology/<class_name>/<image files>
    """
    src_dir = RAW_DIR / "histopathology"
    out_dir = PROCESSED_DIR / "histopathology"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        print(
            f"No raw histopathology data found at {src_dir}. "
            "Run data/scripts/download_histopathology.py first."
        )
        return

    # --- Try the real PCam Kaggle competition format first ---
    pcam_labels_csv = src_dir / "pcam" / "train_labels.csv"
    pcam_train_dir = src_dir / "pcam" / "train"
    if pcam_labels_csv.exists() and pcam_train_dir.exists():
        print(f"Detected real PCam-format data at {src_dir / 'pcam'}")
        labels_df = pd.read_csv(pcam_labels_csv)
        labels_df["path"] = labels_df["id"].apply(lambda x: str(pcam_train_dir / f"{x}.tif"))
        labels_df = labels_df[labels_df["path"].apply(lambda p: Path(p).exists())]
        labels_df = labels_df.rename(columns={"label": "label_idx"})
        labels_df["label"] = labels_df["label_idx"].map({0: "no_tumor", 1: "tumor"})

        if len(labels_df) == 0:
            print(
                f"{pcam_labels_csv} exists but no matching .tif files were found under "
                f"{pcam_train_dir}. Did the Kaggle download finish?"
            )
            return

        train_df, temp_df = train_test_split(
            labels_df, test_size=0.3, stratify=labels_df["label_idx"], random_state=seed
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df["label_idx"], random_state=seed
        )

        cols = ["path", "label", "label_idx"]
        train_df[cols].to_csv(out_dir / "train.csv", index=False)
        val_df[cols].to_csv(out_dir / "val.csv", index=False)
        test_df[cols].to_csv(out_dir / "test.csv", index=False)
        pd.Series({"no_tumor": 0, "tumor": 1}).to_json(out_dir / "class_to_idx.json")

        print(
            f"Vision splits written to {out_dir} from real PCam data: "
            f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )
        return

    # --- Fall back to folder-per-class layout ---
    records = []
    class_dirs = [d for d in src_dir.rglob("*") if d.is_dir()]
    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    for class_dir in class_dirs:
        label = class_dir.name
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in image_exts:
                records.append({"path": str(img_path), "label": label})

    if not records:
        print(f"No images found under {src_dir}. Nothing to preprocess yet.")
        return

    df = pd.DataFrame(records)
    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label_idx"] = df["label"].map(class_to_idx)

    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["label_idx"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label_idx"], random_state=seed
    )

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    pd.Series(class_to_idx).to_json(out_dir / "class_to_idx.json")

    print(
        f"Vision splits written to {out_dir}: "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )


def preprocess_genomics(seed: int = 42) -> None:
    genomics_raw_dir = RAW_DIR / "genomics"
    # Prefer the real-curated-variant-derived dataset; fall back to the
    # purely synthetic generator if that's what was downloaded instead.
    candidates = [
        genomics_raw_dir / "real_curated_variants.csv",
        genomics_raw_dir / "synthetic_variants.csv",
    ]
    src_path = next((p for p in candidates if p.exists()), None)
    out_dir = PROCESSED_DIR / "genomics"
    out_dir.mkdir(parents=True, exist_ok=True)

    if src_path is None:
        print(
            f"No raw genomics data found under {genomics_raw_dir}. "
            "Run data/scripts/download_genomics.py first "
            "(try --mode real_curated for real, cited variant labels)."
        )
        return

    print(f"Using {src_path.name} as the genomics data source.")

    df = pd.read_csv(src_path)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed
    )

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print(
        f"Genomics splits written to {out_dir}: "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess raw data into splits")
    parser.add_argument("--task", choices=["vision", "genomics", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.task in ("vision", "all"):
        preprocess_vision(args.seed)
    if args.task in ("genomics", "all"):
        preprocess_genomics(args.seed)


if __name__ == "__main__":
    main()
