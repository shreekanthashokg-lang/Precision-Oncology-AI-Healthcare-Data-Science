# Future Work

Ideas for extending this project beyond its current 4–6 week scope:

## Vision branch
- Swap in a true whole-slide-image (WSI) pipeline with tiling and
  attention-based multiple-instance learning (MIL), rather than
  pre-cropped patches.
- Add stain normalization (Macenko/Reinhard) to reduce inter-lab color
  variance, a common real-world histopathology issue.
- Benchmark additional backbones (ConvNeXt, Swin Transformer) and report
  compute/accuracy tradeoffs.

## Genomics branch
- Replace the synthetic dataset with real ClinVar variants + reference
  genome flanking sequences (see `docs/architecture.md` extension notes).
- Pretrain the Transformer encoder with masked-language-modeling on a large
  unlabeled genomic corpus before fine-tuning (true DNABERT-style
  pretraining), rather than training from scratch on the labeled task
  alone.
- Add structural variant / copy-number features alongside point variants.

## Multimodal
- Move beyond late fusion to cross-attention fusion (image patches
  attending to genomic tokens and vice versa).
- Use real patient-matched imaging + genomic data (e.g. TCGA, which has
  both histopathology slides and genomic profiles for the same patients)
  instead of a simulated pairing.
- Add uncertainty quantification (e.g. deep ensembles or MC dropout) so the
  fused prediction reports a calibrated confidence interval.

## MLOps / productionization
- Containerize with Docker and add a CI pipeline (lint + unit tests + a
  smoke-test training run on a tiny data subset).
- Track experiments with Weights & Biases or MLflow instead of flat log
  files.
- Add model versioning and a lightweight model registry for the Streamlit
  app to pull from.
- Add unit tests for `src/genomics/encoding.py` and `src/utils/metrics.py`
  (currently untested pure functions, good first targets).

## Clinical validity
- Everything here is a portfolio/demonstration project using public or
  synthetic data with placeholder performance numbers — it is **not**
  validated for clinical use. Real deployment would require IRB-approved
  data, rigorous external validation, and regulatory review (e.g. FDA
  SaMD pathway).
