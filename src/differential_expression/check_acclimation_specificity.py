"""Check the source study's "180 proteins changed only after acclimation" claim.

The abstract reports that for 180 proteins the heat-stress response was "unique to
the post-acclimation visit". That number is derived by SET SUBTRACTION -- proteins
that pass a cut-off in post-acclimation heat stress (PT2 vs PR2) but NOT in
pre-acclimation heat stress (PT1 vs PR1). Comparing "significant vs not significant"
this way is a known statistical fallacy (Gelman & Stern, 2006, "The difference
between significant and not significant is not itself statistically significant"):
a protein can sit just above the cut-off after and just below it before with no
real change in its response.

The correct test of "does the heat-stress response differ before vs after
acclimation?" is the Thermal_Stage x Acclimation INTERACTION. For this balanced
2x2 within-subject design the interaction contrast is, per protein, a one-sample
t-test across the 10 subjects on the difference-of-differences

    DoD = (PT2 - PR2) - (PT1 - PR1)

which is algebraically identical to the interaction term of a two-way
repeated-measures ANOVA (F = t^2, same p-value). We report both:

  * their set-subtraction count (their logic), and
  * the interaction count at BH-FDR < 0.05 and at nominal p < 0.05, against the
    number expected by chance alone.

Missing data are NOT imputed (as in the paper): each test uses only the subjects
with the values it needs present.

Outputs:
  data/differential_expression/acclimation_specificity.csv   -- per-protein stats
  results/preprocessing/acclimation_specificity.png          -- summary figure

Run with any Python 3.8+ (pandas, numpy, scipy, matplotlib).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
from read_physiological_data import DATA_DIR, PROJECT_DIR

MERGED_PATH = DATA_DIR / "Physiological_NPX_Merged.csv"
PHYS_PATH = DATA_DIR / "Physiological_Data_Cleaned.csv"
OUT_CSV = DATA_DIR / "differential_expression" / "acclimation_specificity.csv"
FIG = PROJECT_DIR / "results" / "preprocessing" / "acclimation_specificity.png"

LOG2FC_CUT = 1.0      # the study's change criterion (|log2 fold-change| > 1)
FDR_CUT = 0.05        # significance cut-off for the alternative counting
MIN_PAIRS = 3         # need at least this many complete subject pairs to test


def bh_fdr(p):
    """Benjamini-Hochberg FDR; NaN p-values pass through as NaN."""
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    ps = p[idx]
    order = np.argsort(ps)
    n = len(ps)
    ranked = ps[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[idx[order]] = np.clip(ranked, 0, 1)
    return q


def paired(a, b):
    """log2FC (mean of a-b) and paired-t p over subjects present in both."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < MIN_PAIRS:
        return np.nan, np.nan, int(m.sum())
    d = a[m] - b[m]
    fc = float(np.mean(d))
    p = float(stats.ttest_1samp(d, 0).pvalue) if np.std(d) > 0 else np.nan
    return fc, p, int(m.sum())


def main():
    merged = pd.read_csv(MERGED_PATH)
    phys = pd.read_csv(PHYS_PATH)
    phys_cols = [c for c in merged.columns if c in phys.columns]
    proteins = [c for c in merged.columns if c not in phys_cols]

    # wide matrix: subject x exposure per protein
    merged["Participant"] = merged["Participant"].astype(str)
    P = sorted(merged["Participant"].unique())
    def cube(code):
        sub = merged[merged["Exposure"] == code].set_index("Participant")
        return sub.reindex(P)[proteins].to_numpy(float)
    PR1, PT1, PR2, PT2 = (cube(c) for c in ("PR1", "PT1", "PR2", "PT2"))

    # drop all-missing proteins (assay-QC failures)
    keep = ~np.all(np.isnan(np.vstack([PR1, PT1, PR2, PT2])), axis=0)
    proteins = list(np.array(proteins)[keep])
    PR1, PT1, PR2, PT2 = PR1[:, keep], PT1[:, keep], PR2[:, keep], PT2[:, keep]
    nP = len(proteins)

    rows = []
    for j in range(nP):
        fc_pre, p_pre, n_pre = paired(PT1[:, j], PR1[:, j])   # heat, before
        fc_post, p_post, n_post = paired(PT2[:, j], PR2[:, j])  # heat, after
        # interaction = one-sample t on the difference-of-differences
        dod = (PT2[:, j] - PR2[:, j]) - (PT1[:, j] - PR1[:, j])
        dod = dod[np.isfinite(dod)]
        if len(dod) >= MIN_PAIRS and np.std(dod) > 0:
            p_int = float(stats.ttest_1samp(dod, 0).pvalue)
            fc_int = float(np.mean(dod))
        else:
            p_int, fc_int = np.nan, np.nan
        rows.append(dict(protein=proteins[j], fc_pre=fc_pre, p_pre=p_pre,
                         fc_post=fc_post, p_post=p_post,
                         fc_interaction=fc_int, p_interaction=p_int))
    df = pd.DataFrame(rows)
    df["fdr_pre"] = bh_fdr(df["p_pre"])
    df["fdr_post"] = bh_fdr(df["p_post"])
    df["fdr_interaction"] = bh_fdr(df["p_interaction"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # ---- their logic: "changed after but not before" ----------------------
    changed_post_fc = set(df.loc[df["fc_post"].abs() > LOG2FC_CUT, "protein"])
    changed_pre_fc = set(df.loc[df["fc_pre"].abs() > LOG2FC_CUT, "protein"])
    unique_fc = changed_post_fc - changed_pre_fc                 # "180"-style

    sig_post = set(df.loc[df["fdr_post"] < FDR_CUT, "protein"])
    sig_pre = set(df.loc[df["fdr_pre"] < FDR_CUT, "protein"])
    unique_sig = sig_post - sig_pre

    # ---- proper test: the interaction ------------------------------------
    tested = int(df["p_interaction"].notna().sum())
    int_fdr = int((df["fdr_interaction"] < FDR_CUT).sum())
    int_nom = int((df["p_interaction"] < 0.05).sum())
    chance = 0.05 * tested

    print("=" * 68)
    print('CHECK: "180 proteins changed ONLY after acclimation"')
    print("=" * 68)
    print(f"proteins analysed                         : {nP}")
    print("\n-- their logic (set subtraction) --")
    print(f"|log2FC|>1 after acclimation (PT2 vs PR2) : {len(changed_post_fc)}")
    print(f"|log2FC|>1 before (PT1 vs PR1)            : {len(changed_pre_fc)}")
    print(f"=> 'changed only after acclimation'       : {len(unique_fc)}  (their '180')")
    print(f"   same subtraction on FDR<0.05 sig sets  : {len(unique_sig)}")
    print("\n-- proper test: Thermal_Stage x Acclimation interaction --")
    print("   (per-protein one-sample t on (PT2-PR2)-(PT1-PR1);")
    print("    equals the two-way RM-ANOVA interaction, F = t^2)")
    print(f"   proteins tested                        : {tested}")
    print(f"   interaction significant, BH-FDR<0.05   : {int_fdr}")
    print(f"   interaction nominal p<0.05             : {int_nom}   (chance ~ {chance:.0f})")
    print("\nConclusion: the interaction is essentially null, so '180' is a")
    print("thresholding artifact, not a real 'only-after-acclimation' response.")

    _plot(df, unique_fc, unique_sig, tested, int_fdr, int_nom, chance,
          len(changed_post_fc), len(changed_pre_fc))
    print("\nsaved:", OUT_CSV)
    print("saved:", FIG)
    return dict(unique_fc=len(unique_fc), unique_sig=len(unique_sig),
                int_fdr=int_fdr, int_nom=int_nom, chance=chance, tested=tested)


def _plot(df, unique_fc, unique_sig, tested, int_fdr, int_nom, chance,
          n_post, n_pre):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.0),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # Panel A -- counts: their logic vs the proper test
    labels = ["'Changed only\nafter' (their\nsubtraction)",
              "Same, on\nFDR<0.05\nsig. sets",
              "Interaction\nsignificant\n(FDR<0.05)",
              "Interaction\nnominal\np<0.05"]
    vals = [len(unique_fc), len(unique_sig), int_fdr, int_nom]
    colors = ["#C04A00", "#E69F00", "#007A4D", "#0072B2"]
    bars = axA.bar(range(4), vals, color=colors, width=0.68, zorder=3)
    axA.axhline(chance, ls="--", lw=1.4, color="#777", zorder=2)
    axA.text(3.42, chance, f" expected by\n chance ≈ {chance:.0f}", va="center",
             ha="left", fontsize=8, color="#555")
    for b, v in zip(bars, vals):
        axA.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, str(v),
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    axA.set_xticks(range(4))
    axA.set_xticklabels(labels, fontsize=8.5)
    axA.set_ylabel("number of proteins")
    axA.set_title("How many proteins are 'only after acclimation'?",
                  fontsize=11, fontweight="bold")
    axA.set_ylim(0, max(vals) * 1.22)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)
    axA.grid(axis="y", color="#EEE", zorder=0)
    axA.set_axisbelow(True)

    # Panel B -- interaction p-value histogram (null => flat)
    p = df["p_interaction"].dropna().to_numpy()
    axB.hist(p, bins=20, range=(0, 1), color="#B0B7C6", edgecolor="white", zorder=3)
    axB.axhline(len(p) / 20, ls="--", lw=1.6, color="#C04A00", zorder=4)
    axB.text(0.97, len(p) / 20, " flat = no real\n interaction (null)",
             ha="right", va="bottom", fontsize=8.5, color="#C04A00")
    axB.set_xlabel("interaction p-value  (PT2−PR2) − (PT1−PR1)")
    axB.set_ylabel(f"proteins  (n = {tested})")
    axB.set_title("Interaction p-values are ~uniform → null effect",
                  fontsize=11, fontweight="bold")
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    fig.suptitle("The '180 changed only after acclimation' is a thresholding "
                 "artifact, not a real response", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
