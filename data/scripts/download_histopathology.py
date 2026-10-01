"""
Download / prepare a histopathology dataset for the vision branch of the pipeline.

Supported datasets
-------------------
1. PatchCamelyon (PCam) via the real Kaggle competition
   "Histopathologic Cancer Detection" — https://www.kaggle.com/c/histopathologic-cancer-detection
   327,680 real 96x96 H&E-stained lymph-node patches, labeled tumor/no_tumor.
   Requires the Kaggle API (`pip install kaggle`, and a ~/.kaggle/kaggle.json
   token — free, from your Kaggle account settings).

2. BreakHis               — https://www.kaggle.com/datasets/ambarish/breakhis
   Requires the Kaggle API as well.

3. Colorectal histology    — a small, permissively-licensed dataset (Kather et al.)
   hosted on Zenodo; downloaded directly with `requests`, no auth needed. Good
   for a quick local smoke test since it needs no credentials, but PCam
   (option 1) is the recommended real dataset for this project — it's what
   `data/scripts/preprocess.py` auto-detects and `config.yaml`'s
   `vision.image_size: 96` is already set to match.

Usage
-----
    python data/scripts/download_histopathology.py --dataset pcam
    python data/scripts/download_histopathology.py --dataset colorectal
    python data/scripts/download_histopathology.py --dataset breakhis

Notes
-----
This script intentionally keeps network calls simple and prints clear
instructions when a dataset requires manual steps (Kaggle credentials, license
acceptance, etc.) rather than trying to silently work around them.
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "histopathology"

ZENODO_COLORECTAL_URL = (
    "https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip"
)


def _download_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_colorectal() -> None:
    dest_zip = RAW_DIR / "colorectal_tiles.zip"
    print(f"Downloading Kather colorectal histology tiles to {dest_zip} ...")
    _download_with_progress(ZENODO_COLORECTAL_URL, dest_zip)
    print("Extracting ...")
    with zipfile.ZipFile(dest_zip, "r") as zf:
        zf.extractall(RAW_DIR / "colorectal")
    print(f"Done. Images extracted to {RAW_DIR / 'colorectal'}")


def download_pcam() -> None:
    dest = RAW_DIR / "pcam"
    print(
        "PatchCamelyon ships via the real Kaggle competition "
        "'Histopathologic Cancer Detection' and requires a free Kaggle account + API token.\n\n"
        "1. pip install kaggle\n"
        "2. Get a token: Kaggle account settings -> 'Create New API Token' -> "
        "downloads kaggle.json -> place it at ~/.kaggle/kaggle.json (chmod 600)\n"
        "3. Accept the competition rules once at: "
        "https://www.kaggle.com/c/histopathologic-cancer-detection/rules\n"
        "4. Run:\n"
        f"   kaggle competitions download -c histopathologic-cancer-detection -p {dest}\n"
        f"   unzip {dest / 'histopathologic-cancer-detection.zip'} -d {dest}\n\n"
        f"This produces exactly the layout data/scripts/preprocess.py auto-detects:\n"
        f"  {dest / 'train_labels.csv'}   (columns: id, label)\n"
        f"  {dest / 'train'}/<id>.tif     (327,680 real 96x96 patches)\n"
        f"  {dest / 'test'}/<id>.tif      (unlabeled real test patches, for a Kaggle submission)\n\n"
        "Once downloaded, just run:\n"
        "   python data/scripts/preprocess.py --task vision\n"
        "   python -m src.vision.train\n"
    )


def download_breakhis() -> None:
    print(
        "BreakHis requires the Kaggle API.\n"
        "  1. pip install kaggle\n"
        "  2. Place your kaggle.json token in ~/.kaggle/kaggle.json\n"
        "  3. Run: kaggle datasets download -d ambarish/breakhis "
        f"-p {RAW_DIR / 'breakhis'} --unzip\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a histopathology dataset")
    parser.add_argument(
        "--dataset",
        choices=["pcam", "colorectal", "breakhis"],
        default="pcam",
        help="Which dataset to fetch. 'pcam' is the real Kaggle competition dataset "
        "this project's vision branch is tuned for (image_size=96 in config.yaml "
        "matches it exactly). 'colorectal' works with no credentials for a quick "
        "smoke test.",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset == "colorectal":
        download_colorectal()
    elif args.dataset == "pcam":
        download_pcam()
    elif args.dataset == "breakhis":
        download_breakhis()
    else:
        print(f"Unknown dataset: {args.dataset}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
