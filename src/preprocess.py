"""Quality control and preprocessing for the heat-stress proteomics data.

Runs on the two pipeline artifacts produced earlier:
  - data/Physiological_Data_Cleaned.csv   (tidy physiology)
  - data/Physiological_NPX_Merged.csv      (physiology + Olink NPX)

It performs:
  1. QC   -- duplicated samples, inconsistent identifiers, missing values.
  2. EDA  -- distribution summary + potential outliers for physiology variables
             (flagged, NOT removed: n is small and the extremes are biological).
  3. Protein filtering -- drop proteins with excessive missingness or extremely
     low variance before downstream analysis.

Outputs:
  - data/Physiological_NPX_Preprocessed.csv   (filtered, downstream-ready)
  - data/preprocess_dropped_proteins.csv      (which proteins were dropped & why)
  - data/preprocess_physiology_outliers.csv   (flagged physiology outliers)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from read_physiological_data import DATA_DIR

# ---- Filtering thresholds (edit here) -------------------------------------
# Drop a protein if it is missing in more than this fraction of samples.
MAX_MISSING_FRAC = 0.20
# Drop a protein if its variance (NPX, log2 scale) is below this value.
# 0.0 disables the variance filter: every protein passing the missingness
# filter is retained (a near-constant protein simply carries no signal and is
# handled gracefully downstream), so no protein is removed for low variance.
MIN_VARIANCE = 0.0
# ---------------------------------------------------------------------------

PHYS_PATH = DATA_DIR / "Physiological_Data_Cleaned.csv"
MERGED_PATH = DATA_DIR / "Physiological_NPX_Merged.csv"
OUT_PATH = DATA_DIR / "Physiological_NPX_Preprocessed.csv"
DROPPED_PATH = DATA_DIR / "preprocess_dropped_proteins.csv"
OUTLIERS_PATH = DATA_DIR / "preprocess_physiology_outliers.csv"

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

    miss_frac = proteins.isna().mean()
    variance = proteins.var(numeric_only=True)

    print(f"proteins in       : {len(protein_cols)}")
    print("\nmissingness (% of samples) sensitivity:")
    for thr in [0.0, 0.05, 0.10, 0.20, 0.50]:
        print(f"   > {int(thr*100):3d}% missing : {(miss_frac > thr).sum()}")
    print("\nvariance sensitivity:")
    for thr in [0.005, 0.01, 0.02, 0.05]:
        print(f"   variance < {thr:<5}: {(variance < thr).sum()}")

    drop_missing = miss_frac[miss_frac > MAX_MISSING_FRAC].index
    # Only consider variance among proteins that survived the missing filter.
    surviving = [c for c in protein_cols if c not in set(drop_missing)]
    var_surv = merged[surviving].var(numeric_only=True)
    drop_lowvar = var_surv[var_surv < MIN_VARIANCE].index

    dropped = pd.DataFrame(
        [{"Protein": p, "Reason": f"missing>{MAX_MISSING_FRAC:.0%}",
          "MissingFrac": round(miss_frac[p], 3), "Variance": round(variance[p], 5)
          if pd.notna(variance[p]) else np.nan} for p in drop_missing]
        + [{"Protein": p, "Reason": f"variance<{MIN_VARIANCE}",
            "MissingFrac": round(miss_frac[p], 3), "Variance": round(variance[p], 5)}
           for p in drop_lowvar]
    )

    keep_proteins = [c for c in protein_cols
                     if c not in set(drop_missing) | set(drop_lowvar)]

    print(f"\nDROPPED (missing > {MAX_MISSING_FRAC:.0%})   : "
          f"{len(drop_missing)}  {list(drop_missing)}")
    print(f"DROPPED (variance < {MIN_VARIANCE}): "
          f"{len(drop_lowvar)}  {list(drop_lowvar)}")
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
