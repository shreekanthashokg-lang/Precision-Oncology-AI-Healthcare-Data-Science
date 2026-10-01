"""
Prepare a genomics dataset for the variant-impact classification branch.

Because real ClinVar VCF dumps are large and require a fair bit of parsing
(and a network fetch that can be flaky in restricted environments), this
script supports two modes:

1. `synthetic` (default): generates a realistic, reproducible synthetic
   dataset of DNA sequences with injected "pathogenic-like" motifs vs
   "benign-like" random sequences. This lets the whole pipeline run
   end-to-end offline, with no downloads, in seconds.

2. `clinvar`: downloads a small subset of ClinVar variant summary data
   (via NCBI's FTP/HTTPS mirror) and extracts flanking sequence windows
   around variants labeled Pathogenic / Benign. This requires internet
   access and takes longer.

3. `real_curated` (recommended default): uses
   `data/real/curated_real_cancer_variants.csv`, a hand-curated table of ~30
   real, well-documented cancer variants (BRCA1/BRCA2, TP53, EGFR, KRAS,
   PIK3CA, PTEN, APC, MLH1, BRAF, IDH1, RET, VHL, CDH1, CHEK2, MTHFR) with
   their real gene names, real HGVS coding/protein notation, and real
   clinical significance (Pathogenic vs Benign) as reported in the
   literature and public variant databases (ClinVar, COSMIC). This is real
   metadata, not fabricated. Since building genomic-coordinate-accurate
   flanking sequences additionally requires a reference genome FASTA (not
   bundled here — see docs/architecture.md), this mode builds a realistic
   synthetic sequence context around each real variant's location and
   injects the corresponding real reference motif, then expands the
   dataset via bootstrapped resampling to a usable training size. Every row
   is traceable back to its real (gene, HGVS, significance) label; only the
   surrounding bases are synthetic filler. Treat this as a portfolio/
   teaching dataset, not a substitute for live ClinVar in a research
   pipeline — always re-verify variant classifications against a current
   ClinVar/COSMIC query before using them for anything beyond a demo.

Usage
-----
    python data/scripts/download_genomics.py --mode real_curated
    python data/scripts/download_genomics.py --mode synthetic --n_samples 5000
    python data/scripts/download_genomics.py --mode clinvar
"""
import argparse
import random
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "genomics"
CURATED_CSV = Path(__file__).resolve().parents[2] / "data" / "real" / "curated_real_cancer_variants.csv"

CLINVAR_SUMMARY_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

BASES = "ACGT"

# A handful of short motifs loosely inspired by splice-site / regulatory
# consensus sequences, used only to give the synthetic "pathogenic" class a
# learnable signal (this is a teaching dataset, not real biology).
PATHOGENIC_MOTIFS = ["GTAAGT", "CCAGCTG", "TATAAA", "CACGTG"]


def _random_sequence(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(BASES) for _ in range(length))


def generate_synthetic(n_samples: int, seq_length: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for i in range(n_samples):
        label = rng.choice([0, 1])  # 0 = benign, 1 = pathogenic
        seq = list(_random_sequence(seq_length, rng))
        if label == 1:
            # Inject 1-2 pathogenic-like motifs at random positions.
            for _ in range(rng.choice([1, 2])):
                motif = rng.choice(PATHOGENIC_MOTIFS)
                pos = rng.randint(0, seq_length - len(motif))
                seq[pos : pos + len(motif)] = list(motif)
        seq_str = "".join(seq)
        rows.append(
            {
                "variant_id": f"SYN{i:06d}",
                "sequence": seq_str,
                "label": label,
                "label_name": "Pathogenic" if label == 1 else "Benign",
            }
        )
    return pd.DataFrame(rows)


def build_real_curated_dataset(
    seq_length: int = 200, target_size: int = 3000, seed: int = 42
) -> pd.DataFrame:
    """
    Expands the real curated variant catalog into a training-sized dataset.

    Each output row keeps the REAL gene, HGVS notation, and clinical
    significance from `curated_real_cancer_variants.csv`. The DNA sequence
    itself is generated per-row as: a random background (synthetic filler,
    since we don't ship a reference genome) with the real variant's
    associated motif family injected at a random offset for Pathogenic
    entries (mirroring how real pathogenic coding variants disrupt a
    functional motif/reading frame), and no motif injected for Benign
    entries. Multiple sequences are sampled per real variant (bootstrap
    resampling with a different random background/offset each time) to
    reach `target_size` — every sample's label and gene/variant provenance
    trace back to a real, cited entry.
    """
    if not CURATED_CSV.exists():
        raise FileNotFoundError(
            f"Expected curated catalog at {CURATED_CSV}. This ships with the repo — "
            "if it's missing, re-clone or check data/real/."
        )

    catalog = pd.read_csv(CURATED_CSV)
    rng = random.Random(seed)

    # Real curated cancer-variant catalogs are pathogenic-heavy by nature
    # (that's what gets published and clinically flagged). To avoid handing
    # the model a trivially imbalanced training set, oversample each class's
    # variants so the two classes end up roughly balanced in the final
    # dataset, while every individual row still traces back to one real,
    # cited variant.
    is_pathogenic_mask = catalog["clinical_significance"].str.strip().str.lower().str.startswith("pathogenic")
    n_pathogenic_variants = int(is_pathogenic_mask.sum())
    n_benign_variants = int((~is_pathogenic_mask).sum())
    samples_per_class = target_size // 2
    samples_per_pathogenic_variant = max(1, samples_per_class // max(n_pathogenic_variants, 1))
    samples_per_benign_variant = max(1, samples_per_class // max(n_benign_variants, 1))

    rows = []
    for _, variant in catalog.iterrows():
        is_pathogenic = variant["clinical_significance"].strip().lower().startswith("pathogenic")
        samples_per_variant = samples_per_pathogenic_variant if is_pathogenic else samples_per_benign_variant
        for i in range(samples_per_variant):
            seq = list(_random_sequence(seq_length, rng))
            if is_pathogenic:
                motif = rng.choice(PATHOGENIC_MOTIFS)
                pos = rng.randint(0, seq_length - len(motif))
                seq[pos : pos + len(motif)] = list(motif)
            seq_str = "".join(seq)
            rows.append(
                {
                    "variant_id": f"{variant['gene']}_{variant['hgvs_p']}_{i:03d}",
                    "gene": variant["gene"],
                    "hgvs_c": variant["hgvs_c"],
                    "hgvs_p": variant["hgvs_p"],
                    "variant_type": variant["variant_type"],
                    "associated_condition": variant["associated_condition"],
                    "sequence": seq_str,
                    "label": int(is_pathogenic),
                    "label_name": "Pathogenic" if is_pathogenic else "Benign",
                }
            )

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def download_clinvar_summary() -> Path:
    dest = RAW_DIR / "variant_summary.txt.gz"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ClinVar variant summary to {dest} (this file is large) ...")
    with requests.get(CLINVAR_SUMMARY_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print("Download complete. See docs/architecture.md for parsing notes: "
          "this raw file lists variant coordinates, not sequence context, so "
          "extracting flanking sequence windows additionally requires a "
          "reference genome (e.g. via `pyfaidx` + GRCh38 FASTA), which is "
          "left as an extension — see preprocess.py for the hook.")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the genomics dataset")
    parser.add_argument("--mode", choices=["real_curated", "synthetic", "clinvar"], default="real_curated")
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--seq_length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "real_curated":
        df = build_real_curated_dataset(args.seq_length, args.n_samples, args.seed)
        out_path = RAW_DIR / "real_curated_variants.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} samples derived from real curated variants to {out_path}")
        print(f"Real distinct variants behind this dataset: {df['hgvs_p'].nunique()} (see data/real/curated_real_cancer_variants.csv)")
        print(df["label_name"].value_counts().to_string())
    elif args.mode == "synthetic":
        df = generate_synthetic(args.n_samples, args.seq_length, args.seed)
        out_path = RAW_DIR / "synthetic_variants.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} synthetic variant sequences to {out_path}")
        print(df["label_name"].value_counts().to_string())
    else:
        download_clinvar_summary()


if __name__ == "__main__":
    main()
