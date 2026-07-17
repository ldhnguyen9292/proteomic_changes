"""Complete-case WGCNA preprocessing: drop every protein with ANY missing value.

Alternative to preprocess_wgcna.py (which KNN-imputes). Here a value is MISSING
if it was QC-excluded (NaN) OR below the limit of detection (NPX < LOD) -- both
are unreliable. Any protein with at least one missing value across the 40
samples is dropped, so the matrix is complete WITHOUT imputation (only
fully-detected, QC-pass proteins survive).

Everything else matches the impute workflow (remove assay-QC failures, a light
variance floor, keep all 40 samples). Outputs go to a case-specific folder so
the two workflows never overwrite each other.

Outputs (data/wgcna_complete/):
  wgcna_expression.csv  -- 40 samples (rows) x fully-observed proteins (cols)
  wgcna_traits.csv      -- 40 samples (rows) x physiological traits
Run after merge_physiological_npx.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# shared ingestion modules live in src/data_pipeline/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
from read_physiological_data import DATA_DIR
from preprocess_wgcna import _load_raw_long, below_lod_matrix, TRAIT_COLS

OUT_DIR = DATA_DIR / "wgcna" / "complete"
MERGED_PATH = DATA_DIR / "Physiological_NPX_Merged.csv"

MIN_VARIANCE = 0.01   # light variance floor (same as the impute workflow)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = pd.read_csv(MERGED_PATH)
    phys = pd.read_csv(DATA_DIR / "Physiological_Data_Cleaned.csv")
    phys_cols = [c for c in merged.columns if c in phys.columns]
    protein_cols = [c for c in merged.columns if c not in phys_cols]

    sample_id = merged["Participant"].astype(str) + "-" + merged["Exposure"].astype(str)
    expr = merged[protein_cols].copy()
    expr.index = sample_id

    print("=" * 60)
    print("COMPLETE-CASE WGCNA PREPROCESSING (drop any missing)")
    print("=" * 60)
    print(f"proteins in merged                         : {len(protein_cols)}")

    # 1. remove assay-QC failures (entirely missing).
    expr = expr[expr.columns[expr.isna().mean() < 1.0]]
    print(f"after removing assay-QC failures (all-NaN) : {expr.shape[1]}")

    # 2. unified missing mask: QC-excluded (NaN) OR below LOD.
    raw_long = _load_raw_long()
    blod = below_lod_matrix(raw_long, expr.index, expr.columns)
    missing = expr.isna() | blod
    expr = expr.mask(missing)

    # 3. complete-case: drop any protein with at least one missing value.
    complete = expr.columns[expr.isna().sum() == 0]
    print(f"after dropping proteins with ANY missing    : {len(complete)}  "
          f"(-{expr.shape[1] - len(complete)})")
    expr = expr[complete]

    # 4. light variance floor (no imputation needed -- matrix is complete).
    var = expr.var()
    keep = var[var >= MIN_VARIANCE].index.tolist()
    print(f"after variance filter (>= {MIN_VARIANCE})            : {len(keep)}  "
          f"(-{len(complete) - len(keep)})")
    expr = expr[keep]
    assert int(expr.isna().values.sum()) == 0, "expression still has NaN"

    # 5. sample-outlier check (report only).
    dmat = squareform(pdist(expr.values))
    md = dmat.mean(axis=1)
    outliers = list(expr.index[md > md.mean() + 3 * md.std()])
    print(f"sample outliers flagged                    : {outliers or 'none'}")

    # 6. traits (all 40 samples).
    traits = merged[TRAIT_COLS].copy()
    traits.index = sample_id
    traits.insert(0, "Hyperthermic",
                  (merged["Thermal_Stage"] == "Hyperthermic").astype(int).values)
    traits.insert(1, "Post_acclimation",
                  (merged["Acclimation"] == "Post").astype(int).values)

    expr.to_csv(OUT_DIR / "wgcna_expression.csv")
    traits.to_csv(OUT_DIR / "wgcna_traits.csv")

    print("\nOUTPUTS")
    print(f"  {OUT_DIR / 'wgcna_expression.csv'}  {expr.shape}  (samples x proteins)")
    print(f"  {OUT_DIR / 'wgcna_traits.csv'}  {traits.shape}  (samples x traits)")
    print("run visualize_wgcna_complete.py for the results figure")


if __name__ == "__main__":
    main()
