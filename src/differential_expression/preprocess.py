"""Quality control and preprocessing for the heat-stress proteomics data.

Runs on the two pipeline artifacts produced earlier:
  - data/Physiological_Data_Cleaned.csv   (tidy physiology)
  - data/Physiological_NPX_Merged.csv      (physiology + Olink NPX)

It performs:
  1. QC   -- duplicated samples, inconsistent identifiers, missing values.
  2. EDA  -- distribution summary + potential outliers for physiology variables
             (flagged, NOT removed: n is small and the extremes are biological).
  3. Protein filtering -- remove proteins missing (QC-excluded) in more than
     20/40 samples; below-LOD values are kept, not imputed, and no variance
     threshold is applied.

Outputs:
  - data/Physiological_NPX_Preprocessed.csv   (filtered, downstream-ready)
  - data/preprocess_dropped_proteins.csv      (which proteins were dropped & why)
  - data/preprocess_physiology_outliers.csv   (flagged physiology outliers)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# shared ingestion modules live in src/data_pipeline/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
from read_physiological_data import DATA_DIR

# ---- Protein filtering policy ---------------------------------------------
# Remove a protein if it is missing (QC-excluded) in MORE than 20 of the 40
# samples (> 50%). Below-LOD values are NOT counted as missing -- they are
# retained and NOT imputed, so proteins mostly below the limit of detection
# stay in the analysis. No variance threshold is applied.
MAX_MISSING_FRAC = 0.5   # drop a protein missing in > 20/40 samples
# ---------------------------------------------------------------------------

PHYS_PATH = DATA_DIR / "Physiological_Data_Cleaned.csv"
MERGED_PATH = DATA_DIR / "Physiological_NPX_Merged.csv"
# Differential-expression outputs live in their own subfolder.
OUT_DIR = DATA_DIR / "differential_expression"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "Physiological_NPX_Preprocessed.csv"
DROPPED_PATH = OUT_DIR / "preprocess_dropped_proteins.csv"
OUTLIERS_PATH = OUT_DIR / "preprocess_physiology_outliers.csv"

KEYS = ["Participant", "Exposure"]
PHYS_META = ["Participant", "Acclimation", "Thermal_Stage", "Exposure"]


def _rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def qc_duplicates(name, df):
    print(f"[{name}] duplicated full rows        : {df.duplicated().sum()}")
    print(f"[{name}] duplicated [Participant,Exp]: {df.duplicated(KEYS).sum()}")


def qc_identifiers(phys, merged):
    phys_ids = set(phys["Participant"].unique())
    npx_ids = set(merged["Participant"].unique())
    print("physiology Participant IDs:", sorted(phys_ids))
    print("NPX/merged Participant IDs:", sorted(npx_ids))
    # Compare after stripping non-alphanumerics to expose formatting-only diffs.
    norm = lambda s: {i.replace("-", "") for i in s}
    only_phys = phys_ids - npx_ids
    if only_phys and norm(phys_ids) == norm(npx_ids):
        print(
            f"INCONSISTENT FORMAT (resolved by merge via hyphen removal): "
            f"{sorted(only_phys)} -> {sorted(i.replace('-', '') for i in only_phys)}"
        )
    elif norm(phys_ids) != norm(npx_ids):
        print("WARNING: identifier sets differ even after normalization!")
    else:
        print("Identifiers consistent.")


def qc_missing_physiology(phys):
    mv = phys.isna().sum()
    mv = mv[mv > 0]
    print("missing values:", "none" if mv.empty else f"\n{mv.to_string()}")


def physiology_outliers(phys):
    """Flag potential outliers per physiological variable (IQR and z-score)."""
    phys_vars = [c for c in phys.columns if c not in PHYS_META]
    rows = []
    for v in phys_vars:
        s = phys[v].astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        z = (s - s.mean()) / s.std(ddof=0)
        mask = (s < lo) | (s > hi) | (z.abs() > 3)
        for idx in phys.index[mask]:
            rows.append(
                {
                    "Variable": v,
                    "Participant": phys.at[idx, "Participant"],
                    "Exposure": phys.at[idx, "Exposure"],
                    "Value": round(float(s[idx]), 3),
                    "IQR_outlier": bool((s[idx] < lo) or (s[idx] > hi)),
                    "z_gt_3": bool(abs(z[idx]) > 3),
                }
            )
    report = pd.DataFrame(rows)
    print(f"flagged {len(report)} potential outlier value(s) "
          f"across {report['Variable'].nunique() if len(report) else 0} variable(s).")
    if len(report):
        print(report.to_string(index=False))
    print("\nNote: flagged values are physiologically plausible extremes "
          "(e.g. SSNA %baseline, heat vasodilation) and are NOT removed.")
    return report


def filter_proteins(merged, phys_cols):
    protein_cols = [c for c in merged.columns if c not in phys_cols]
    proteins = merged[protein_cols]
    n = len(merged)

    miss_count = proteins.isna().sum()
    miss_frac = miss_count / n
    n_sample_qc = int(proteins.isna().values.sum())
    thr = int(round(MAX_MISSING_FRAC * n))  # 20 of 40

    print(f"proteins in       : {len(protein_cols)}")
    print(f"sample-assay QC exclusions (NaN) : {n_sample_qc} "
          f"({100 * n_sample_qc / proteins.size:.1f}% of {proteins.size})")
    print(f"\nmissingness (n of {n} samples) profile:")
    for k in [0, 2, 4, 8, 20]:
        print(f"   > {k:2d}/{n} missing : {int((miss_count > k).sum())}")

    # Remove a protein missing (QC-excluded) in more than 20/40 samples.
    # Below-LOD values are NOT missing (kept, not imputed); no variance filter.
    drop_missing = miss_frac[miss_frac > MAX_MISSING_FRAC].index
    keep_proteins = [c for c in protein_cols if c not in set(drop_missing)]

    dropped = pd.DataFrame(
        [{"Protein": p, "Reason": f"missing in > {thr}/{n} samples",
          "MissingFrac": round(float(miss_frac[p]), 3)} for p in drop_missing]
    )

    print(f"\nDROPPED (missing in > {thr}/{n} samples): "
          f"{len(drop_missing)}  {list(drop_missing)}")
    print(f"proteins kept     : {len(keep_proteins)}")
    return keep_proteins, dropped


def main():
    phys = pd.read_csv(PHYS_PATH)
    merged = pd.read_csv(MERGED_PATH)
    phys_cols = [c for c in merged.columns if c in phys.columns]

    _rule("1. QUALITY CONTROL")
    print("-- Duplicated samples --")
    qc_duplicates("physiology", phys)
    qc_duplicates("merged", merged)
    print("\n-- Inconsistent identifiers --")
    qc_identifiers(phys, merged)
    print("\n-- Missing values (physiology) --")
    qc_missing_physiology(phys)

    _rule("2. PHYSIOLOGY DISTRIBUTIONS & OUTLIERS")
    phys_vars = [c for c in phys.columns if c not in PHYS_META]
    print(phys[phys_vars].describe().T[["count", "mean", "std", "min", "50%", "max"]]
          .round(2).to_string())
    print("\n-- Potential outliers --")
    outliers = physiology_outliers(phys)

    _rule("3. PROTEIN FILTERING")
    keep_proteins, dropped = filter_proteins(merged, phys_cols)

    preprocessed = merged[phys_cols + keep_proteins]

    preprocessed.to_csv(OUT_PATH, index=False)
    dropped.to_csv(DROPPED_PATH, index=False)
    outliers.to_csv(OUTLIERS_PATH, index=False)

    _rule("OUTPUTS")
    print(f"preprocessed dataset : {OUT_PATH}  {preprocessed.shape}")
    print(f"dropped proteins log : {DROPPED_PATH}  ({len(dropped)} rows)")
    print(f"physiology outliers  : {OUTLIERS_PATH}  ({len(outliers)} rows)")


if __name__ == "__main__":
    main()
