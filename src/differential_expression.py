"""Differential protein abundance across the heat-acclimation contrasts.

Recreates the differential-expression results described in the source study's
abstract, using paired within-participant tests on the preprocessed Olink NPX
data. NPX is on a log2 scale, so a paired mean difference IS the log2
fold-change (log2FC).

Contrasts (each paired across the 10 participants):
  pre_heat   PT1 vs PR1  -- heat stress, before acclimation
  acclim     PR2 vs PR1  -- acclimation effect, measured normothermic (at rest)
  post_heat  PT2 vs PR2  -- heat stress, after acclimation

A protein is "changed" when BH-FDR < 0.05; a stricter FDR < 0.01 tier is also
reported (this tier reproduces the abstract's "9 proteins reached the p value
threshold" for pre_heat). Proteins "unique to the post-acclimation heat
response" are those changed in post_heat but not in pre_heat.

Both a paired t-test and a Wilcoxon signed-rank test are computed for every
protein so the two can be compared; the t-test is treated as primary.

Outputs (data/):
  de_<contrast>.csv   -- full per-protein table (log2FC, p, FDR for both tests)
  de_summary.csv      -- counts per contrast, next to the abstract's figures

Run after preprocess.py (it reads Physiological_NPX_Preprocessed.csv).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from read_physiological_data import DATA_DIR
from preprocess import PHYS_META

PREP_PATH = DATA_DIR / "Physiological_NPX_Preprocessed.csv"

# ---- Significance thresholds (edit here) ----------------------------------
FDR_CHANGED = 0.05   # "changed in concentration"
FDR_STRICT = 0.01    # stricter tier ("reached the p value threshold")
# ---------------------------------------------------------------------------

# contrast key -> (exposure_a, exposure_b, description); log2FC = mean(a - b).
CONTRASTS = {
    "pre_heat":  ("PT1", "PR1", "Heat stress, before acclimation"),
    "acclim":    ("PR2", "PR1", "Acclimation effect, normothermic (rest)"),
    "post_heat": ("PT2", "PR2", "Heat stress, after acclimation"),
}

# Marker genes named in the abstract, per result, for verification.
MARKERS = {
    "pre_heat":  ["FGFBP3", "TIMP4", "OBP2B", "CXCL17"],
    "acclim":    ["MMP7", "CA6", "KLK14", "DNER"],
    "post_heat": ["MOG", "SMOC1", "PTPRN2", "TNC", "C1QTNF1", "MATN2", "NTF3"],
}

# What the abstract reports, for side-by-side comparison (not used to fit).
ABSTRACT = {
    "pre_heat":  "~1% (n=29) changed; 9 reached the p value threshold",
    "acclim":    "54 changed (normothermic)",
    "post_heat": "175 up (~6%), 19 down (~0.6%); 180 unique to post-acclimation",
}


def _rule(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def benjamini_hochberg(pvals):
    """BH-FDR adjusted p-values; NaNs in -> NaNs out (excluded from the count)."""
    p = np.asarray(pvals, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    q[ok] = adj
    return q


def paired_contrast(df, proteins, a, b):
    """Paired t-test and Wilcoxon per protein for exposure a vs b.

    Uses only participants with a non-missing value in BOTH exposures.
    """
    A = df[df["Exposure"] == a].set_index("Participant")[proteins]
    B = df[df["Exposure"] == b].set_index("Participant")[proteins]
    common = A.index.intersection(B.index)
    A, B = A.loc[common], B.loc[common]

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # small-n Wilcoxon approximation notes
        for p in proteins:
            x, y = A[p].values, B[p].values
            mask = ~(np.isnan(x) | np.isnan(y))
            xd, yd = x[mask], y[mask]
            if len(xd) < 3 or np.allclose(xd, yd):
                rows.append((p, np.nan, np.nan, np.nan, len(xd)))
                continue
            t_p = stats.ttest_rel(xd, yd).pvalue
            try:
                w_p = stats.wilcoxon(xd, yd).pvalue
            except ValueError:
                w_p = np.nan
            rows.append((p, (xd - yd).mean(), t_p, w_p, len(xd)))

    r = pd.DataFrame(rows, columns=["Protein", "log2FC", "t_p", "w_p", "n_pairs"])
    r["t_fdr"] = benjamini_hochberg(r["t_p"].values)
    r["w_fdr"] = benjamini_hochberg(r["w_p"].values)
    r["changed"] = r["t_fdr"] < FDR_CHANGED
    r["strict"] = r["t_fdr"] < FDR_STRICT
    r["direction"] = np.where(r["log2FC"] > 0, "up", "down")
    return r.sort_values("t_p").set_index("Protein")


def main():
    df = pd.read_csv(PREP_PATH)
    proteins = [c for c in df.columns if c not in PHYS_META
                and c not in _phys_var_cols(df)]
    n_tested = len(proteins)

    _rule("DIFFERENTIAL EXPRESSION (paired; 'changed' = BH-FDR < 0.05)")
    print(f"proteins tested: {n_tested}   samples: {len(df)}")

    results = {}
    for key, (a, b, desc) in CONTRASTS.items():
        r = paired_contrast(df, proteins, a, b)
        results[key] = r
        r.to_csv(DATA_DIR / f"de_{key}.csv")

    # "Unique to post-acclimation": changed in post_heat but not in pre_heat.
    post_sig = set(results["post_heat"].index[results["post_heat"]["changed"]])
    pre_sig = set(results["pre_heat"].index[results["pre_heat"]["changed"]])
    unique_post = post_sig - pre_sig

    # ---- summary table --------------------------------------------------
    summary = []
    for key, (a, b, desc) in CONTRASTS.items():
        r = results[key]
        chg = r[r["changed"]]
        summary.append({
            "contrast": key,
            "comparison": f"{a} vs {b}",
            "description": desc,
            "n_tested": n_tested,
            "changed_fdr05": int(len(chg)),
            "up": int((chg["direction"] == "up").sum()),
            "down": int((chg["direction"] == "down").sum()),
            "strict_fdr01": int(r["strict"].sum()),
            "wilcoxon_fdr05": int((r["w_fdr"] < FDR_CHANGED).sum()),
            "abstract": ABSTRACT[key],
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(DATA_DIR / "de_summary.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(summary_df[["comparison", "description", "changed_fdr05", "up",
                          "down", "strict_fdr01", "wilcoxon_fdr05"]].to_string(index=False))
    print(f"\nunique to post-acclimation heat response (post \\ pre): "
          f"{len(unique_post)}   [abstract: 180]")

    _rule("ABSTRACT COMPARISON")
    for row in summary:
        print(f"[{row['comparison']}] recreated: {row['changed_fdr05']} changed "
              f"({row['up']} up / {row['down']} down), {row['strict_fdr01']} at FDR<0.01")
        print(f"            abstract : {row['abstract']}")

    _rule("MARKER VERIFICATION (genes named in the abstract)")
    for key, genes in MARKERS.items():
        r = results[key]
        print(f"\n{key} ({CONTRASTS[key][0]} vs {CONTRASTS[key][1]}):")
        for g in genes:
            if g not in r.index:
                print(f"  {g:8s} NOT IN DATA")
                continue
            row = r.loc[g]
            uniq = " unique-to-post" if (key == "post_heat" and g in unique_post) else ""
            print(f"  {g:8s} log2FC={row.log2FC:+.3f}  t_FDR={row.t_fdr:.3g}  "
                  f"w_FDR={row.w_fdr:.3g}  changed={bool(row.changed)}"
                  f"  dir={row.direction}{uniq}")

    _rule("OUTPUTS")
    for key in CONTRASTS:
        print(f"de_{key}.csv")
    print("de_summary.csv")


def _phys_var_cols(df):
    """Non-protein physiological measurement columns (kept out of the tests)."""
    phys = pd.read_csv(DATA_DIR / "Physiological_Data_Cleaned.csv")
    return [c for c in df.columns if c in phys.columns]


if __name__ == "__main__":
    main()
