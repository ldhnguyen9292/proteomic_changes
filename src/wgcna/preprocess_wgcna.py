"""Build a WGCNA-ready expression matrix + trait table from the merged data.

Separate path from preprocess.py (which reproduces the source study's DE
numbers). WGCNA builds a co-expression network, so poorly-detected /
near-constant proteins add spurious correlations and the matrix must be
complete.

A value is treated as MISSING if it was QC-excluded (NaN) OR below the limit
of detection (NPX < LOD) -- both are unreliable measurements. LOD lives only
in olink_raw, so it is read from there and aligned onto the merged columns.

Steps (agreed recipe):
  1. Remove the 5 assay-QC failures (entirely missing)            -> 2,938
  2. Filter: drop a protein missing in > 10% of samples OR with
     variance < 0.01                                              -> ~1,707
  3. Impute the remaining missing cells (KNN, k=10; or LOD/sqrt2)
  4. Sample-outlier check: hierarchical clustering (flagged, not dropped)
  5. Keep all 40 samples (repeated measures; non-independence noted later)

Outputs (data/):
  wgcna_expression.csv  -- 40 samples (rows) x filtered/imputed proteins (cols)
  wgcna_traits.csv      -- 40 samples (rows) x physiological traits
Figure (results/):
  wgcna_preprocessing.png -- filtering waterfall + sample dendrogram

Run after merge_physiological_npx.py.
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.impute import KNNImputer

# shared ingestion modules live in src/data_pipeline/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
from read_physiological_data import DATA_DIR

MERGED_PATH = DATA_DIR / "Physiological_NPX_Merged.csv"
OLINK_RAW_DIR = DATA_DIR / "raw_data" / "olink_raw"
OUT_DIR = DATA_DIR / "wgcna" / "impute"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXPR_OUT = OUT_DIR / "wgcna_expression.csv"
TRAITS_OUT = OUT_DIR / "wgcna_traits.csv"

# ---- WGCNA filtering parameters (edit here) -------------------------------
MAX_MISSING_FRAC = 0.10   # drop a protein missing (NaN or < LOD) in > 10% of samples
MIN_VARIANCE = 0.01       # drop a protein with observed variance < 0.01
KNN_K = 10                # neighbours for KNN imputation
IMPUTE_METHOD = "knn"     # "knn"  or  "lod"  (LOD / sqrt(2))
# ---------------------------------------------------------------------------

TRAIT_COLS = [
    "Sweat rate (L/h)",                            # whole-body (primary outcome)
    "LSR (mg/min/cm2)",                            # local sweat rate (per condition)
    "Tcore (°C)", "Mean Tskin (°C)", "Mean Tbody (°C)",
    "LDF (units)", "CVC (units/mmHg)",
    "HR (bpm)", "SBP (mmHg)", "DBP (mmHg)", "MAP (mmHg)", "SSNA (%baseline)",
    "Change (post-pre) in nude body weight (kg)", "Heat exposure duration (s)",
]


def _rule(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def _load_raw_long():
    raw = pd.concat([pd.read_csv(f, sep=";") for f in glob.glob(f"{OLINK_RAW_DIR}/*.csv")],
                    ignore_index=True)
    tok = raw["SampleID"].str.split("-", expand=True)
    raw["Participant"], raw["Exposure"] = tok[1], tok[2]
    real = raw[(tok[0] == "SSNA") & raw["Exposure"].isin(["PR1", "PT1", "PR2", "PT2"])].copy()
    n_panels = real.groupby("Assay")["Panel"].nunique()
    bridge = set(n_panels[n_panels > 1].index)
    real["col"] = [f"{a}__{p}" if a in bridge else a
                   for a, p in zip(real["Assay"], real["Panel"])]
    real["sample"] = real["Participant"].astype(str) + "-" + real["Exposure"].astype(str)
    return real


def below_lod_matrix(raw_long, index, columns):
    """Boolean (samples x proteins): True where NPX < LOD, aligned to expr."""
    raw_long = raw_long.copy()
    raw_long["below"] = raw_long["NPX"] < raw_long["LOD"]
    blod = raw_long.pivot_table(index="sample", columns="col", values="below",
                                aggfunc="first")
    return blod.reindex(index=index, columns=columns).fillna(False).astype(bool)


def lod_floor(raw_long, columns):
    """Per-protein LOD/sqrt(2) floor for LOD-based imputation."""
    lod = raw_long.groupby("col")["LOD"].median().reindex(columns)
    return lod / np.sqrt(2)


def main():
    merged = pd.read_csv(MERGED_PATH)
    phys = pd.read_csv(DATA_DIR / "Physiological_Data_Cleaned.csv")
    phys_cols = [c for c in merged.columns if c in phys.columns]
    protein_cols = [c for c in merged.columns if c not in phys_cols]

    sample_id = (merged["Participant"].astype(str) + "-"
                 + merged["Exposure"].astype(str))
    expr = merged[protein_cols].copy()
    expr.index = sample_id

    _rule("WGCNA PROTEIN FILTERING  (missing = NaN or < LOD)")
    n0 = len(protein_cols)
    print(f"proteins in merged                         : {n0}")

    # 1. assay-QC failures = entirely missing (all-NaN).
    n1_keep = expr.columns[expr.isna().mean() < 1.0]
    expr = expr[n1_keep]
    print(f"after removing assay-QC failures (all-NaN) : {expr.shape[1]}  "
          f"(-{n0 - expr.shape[1]})")

    # Build the unified missing mask: QC-excluded (NaN) OR below LOD.
    raw_long = _load_raw_long()
    blod = below_lod_matrix(raw_long, expr.index, expr.columns)
    missing = expr.isna() | blod
    expr = expr.mask(missing)                       # set all missing cells to NaN

    # 2. filter: missing > 10%  OR  variance < 0.01.
    miss_frac = missing.mean(axis=0)
    var = expr.var(numeric_only=True)               # variance on observed values
    drop_missing = set(miss_frac[miss_frac > MAX_MISSING_FRAC].index)
    drop_var = set(var[var < MIN_VARIANCE].index)
    keep = [c for c in expr.columns if c not in (drop_missing | drop_var)]
    print(f"drop missing > {MAX_MISSING_FRAC:.0%}                          : "
          f"{len(drop_missing)}")
    print(f"drop variance < {MIN_VARIANCE}                        : {len(drop_var)}")
    print(f"after filter (union removed {len(drop_missing | drop_var)})            : "
          f"{len(keep)}")
    expr = expr[keep]

    # 3. impute remaining missing cells.
    n_missing_cells = int(expr.isna().values.sum())
    if IMPUTE_METHOD == "knn":
        expr_imp = pd.DataFrame(
            KNNImputer(n_neighbors=KNN_K).fit_transform(expr.values),
            index=expr.index, columns=expr.columns)
    elif IMPUTE_METHOD == "lod":
        floor = lod_floor(raw_long, expr.columns)
        expr_imp = expr.fillna(floor)
    else:
        raise ValueError(IMPUTE_METHOD)
    print(f"\nimputed missing cells ({IMPUTE_METHOD}, k={KNN_K})            : "
          f"{n_missing_cells}")
    print(f"NaNs remaining after imputation            : "
          f"{int(expr_imp.isna().values.sum())}")

    # 4. sample-outlier check (report only; dendrogram drawn by visualize_wgcna).
    _rule("SAMPLE OUTLIER CHECK (mean pairwise distance)")
    dmat = squareform(pdist(expr_imp.values))
    md = dmat.mean(axis=1)
    outliers = list(expr_imp.index[md > md.mean() + 3 * md.std()])
    print(f"samples: {len(expr_imp)}   flagged outliers: {outliers or 'none'}")
    print("(flagged only -- inspect the dendrogram before deciding)")

    # 5. traits (all 40 samples).
    traits = merged[TRAIT_COLS].copy()
    traits.index = sample_id
    traits.insert(0, "Hyperthermic",
                  (merged["Thermal_Stage"] == "Hyperthermic").astype(int).values)
    traits.insert(1, "Post_acclimation",
                  (merged["Acclimation"] == "Post").astype(int).values)

    expr_imp.to_csv(EXPR_OUT)
    traits.to_csv(TRAITS_OUT)

    _rule("OUTPUTS")
    print(f"expression : {EXPR_OUT}  {expr_imp.shape}  (samples x proteins)")
    print(f"traits     : {TRAITS_OUT}  {traits.shape}  (samples x traits)")
    print("run visualize_wgcna.py to render the results figure")


if __name__ == "__main__":
    main()
