# Deployment Guide

This app is set up to deploy directly to **Streamlit Community Cloud**
(free, easiest option) with zero code changes. Other targets (Hugging Face
Spaces, Render, a Docker container) are also covered below.

## Option A — Streamlit Community Cloud (recommended)

1. Push this repo to GitHub (public or private).
2. Go to <https://share.streamlit.io>, sign in with GitHub, click
   **New app**.
3. Select this repo, branch `main`, and set **Main file path** to:
   ```
   app/streamlit_app.py
   ```
4. Streamlit Cloud automatically picks up:
   - `requirements.txt` — Python dependencies
   - `packages.txt` — system/apt dependencies (`libgl1`, `libglib2.0-0`,
     needed by OpenCV)
   - `runtime.txt` — Python version pin (3.10)
   - `.streamlit/config.toml` — theme + server settings
5. Click **Deploy**. First build takes a few minutes (installing PyTorch +
   timm).

**Before deploying**, at minimum train and commit the clinical + ClinVar
models so the app has something real to show immediately:
```bash
python -m src.clinical.wisconsin
python -m src.genomics.clinvar_real
```
This writes `models/clinical_logreg.joblib` (a few KB, safe to commit
directly) and `models/clinvar_real_rf.joblib` (~30MB — large for a normal
git repo; use Git LFS, or just regenerate it on first deploy since
training takes under a minute and the source data is already in the
repo). For the vision/sequence-genomics checkpoints, either:
- Train them locally and `git add -f models/*.pt` (checkpoints are
  gitignored by default — see `.gitignore` — since they're normally
  regenerated, not committed), or
- Accept that the Histopathology/Genomics/Multimodal pages will show their
  "train this first" message until a checkpoint exists in the deployed
  environment.

## Option B — Hugging Face Spaces

1. Create a new Space, SDK = **Streamlit**.
2. Push this repo's contents to the Space's git remote.
3. Hugging Face Spaces reads the same `requirements.txt` and
   `packages.txt` automatically. Set the Space's "App file" to
   `app/streamlit_app.py` in the Space settings if it isn't auto-detected.

## Option C — Docker (self-hosted / Render / Fly.io / any container host)

Minimal `Dockerfile` (not included by default, to keep the repo lean —
create it as needed):

```dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t precision-oncology-ai .
docker run -p 8501:8501 precision-oncology-ai
```

## What ships "ready" vs. what you train first

| Branch | Deploys working out of the box? | Why |
|---|---|---|
| 🧬 ClinVar Real Triage (65K) | ✅ Yes, immediately | Real data ships in `data/real/`; train (~1 min) and commit/regenerate the `.joblib` |
| 🩺 Clinical Risk Model | ✅ Yes, immediately | Real data ships inside `scikit-learn`; train (<1s) and commit the tiny `.joblib` |
| 🧬 Genomic Variant Analyzer (sequence) | ⚠️ After one training run | Real-curated variant data ships in the repo; train the CNN/Transformer (minutes on CPU) and commit/regenerate the checkpoint |
| 🔬 Histopathology Analyzer | ⚠️ After download + training | Needs the real PCam data (`data/scripts/download_histopathology.py --dataset pcam` — free Kaggle account), then training (GPU recommended) |
| 🔗 Multimodal Demo | ⚠️ After both above + fusion training | Needs both branch checkpoints, then the fusion head (`notebooks/05_multimodal_fusion.ipynb`) |

This is intentional: real medical imaging datasets are large and
credential-gated by their hosts, so they can't be embedded in a portfolio
repo. Once you've trained locally, commit the resulting `models/*.pt` /
`*.joblib` files (or host them externally and adjust `src/inference.py`'s
loaders to fetch them) so the deployed app serves real predictions rather
than the "train this first" placeholder message.
