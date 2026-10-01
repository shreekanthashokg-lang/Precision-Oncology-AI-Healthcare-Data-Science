#!/usr/bin/env bash
# End-to-end demo runner: prepares synthetic/offline data, trains both
# branches for a few epochs, and launches the Streamlit app.
#
# Usage: ./run_demo.sh
set -euo pipefail

echo "== 1/6 Installing dependencies =="
pip install -r requirements.txt

echo "== 2/6 Training real models with zero downloads (Clinical WDBC + ClinVar 65K) =="
python -m src.clinical.wisconsin
python -m src.genomics.clinvar_real

echo "== 3/6 Preparing genomics sequence data (real-curated cancer variants, offline) =="
python data/scripts/download_genomics.py --mode real_curated
python data/scripts/preprocess.py --task genomics

echo "== 4/6 Preparing histopathology data (Kather colorectal, small download) =="
python data/scripts/download_histopathology.py --dataset colorectal
python data/scripts/preprocess.py --task vision

echo "== 5/6 Training vision + genomics sequence branches =="
python -m src.genomics.train
python -m src.vision.train

echo "== 6/6 Launching Streamlit app =="
streamlit run app/streamlit_app.py
