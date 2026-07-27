"""What the soft-threshold power β actually does to the modules.

β cannot change which correlation is bigger than which -- x^β is monotone, so the
ranking is identical at every β. It changes the RATIOS, and that matters because
topological overlap SUMS over shared neighbours:

    TOM_ij  ∝  Σ_u  a_iu · a_uj        with  a = max(r, 0)^β

At β = 1 a protein has ~1,700 partners each carrying moderate weight, so the sum
is dominated by the sheer number of weak links: everything overlaps with
everything and no group can be separated. Raising β makes weak links contribute
almost nothing, so the sum is carried by a handful of genuinely strong partners
and distinct groups appear.

Three panels, all from the real PT1/PT2 network:
  A  TOM at four values of β, proteins in module order -- blocks appear
  B  separation ratio (mean within-module TOM / mean between-module TOM) vs β
  C  scale-free fit R² vs β, and why β = 13 was chosen

Output: results/wgcna/beta_effect.png
Run with Python 3.8+ (uses the cached PT1/PT2 fit if present).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
from read_physiological_data import DATA_DIR, PROJECT_DIR

CASE_DIR = DATA_DIR / "wgcna" / "impute"
OUT_DIR = PROJECT_DIR / "results" / "wgcna"
CODES = ["PT1", "PT2"]
CHOSEN = 13
SHOW_BETAS = [1, 3, 6, 13]
SWEEP = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30]
MOD_COLOR = {"dimgrey": "#D55E00", "lightgrey": "#0072B2",
             "black": "#009E73", "darkgrey": "#E69F00"}
N_SUB = {"lightgrey": 26, "black": 24, "darkgrey": 18, "dimgrey": 16}


def tom_from_adjacency(a):
    a = a.copy()
    np.fill_diagonal(a, 0.0)
    k = a.sum(axis=1)
    num = a @ a + a
    den = np.minimum.outer(k, k) + 1.0 - a
    t = num / den
    np.fill_diagonal(t, 1.0)
    return t


def main():
    expr = pd.read_csv(CASE_DIR / "wgcna_expression.csv", index_col=0)
    tr = pd.read_csv(CASE_DIR / "wgcna_traits.csv", index_col=0)
    common = expr.index.intersection(tr.index)
    expr, tr = expr.loc[common], tr.loc[common]
    expr = expr.loc[tr.index.str.split("-").str[-1].isin(CODES)]

    import pickle
    obj = pickle.load(open("/tmp/pt1pt2_obj.pkl", "rb"))
    mods = obj.datExpr.var["moduleColors"]
    sft = obj.sft

    X = expr[mods.index]
    labels_full = mods.to_numpy()
    corr_full = np.corrcoef(X.to_numpy(), rowvar=False)
    same = labels_full[:, None] == labels_full[None, :]
    off = ~np.eye(len(labels_full), dtype=bool)

    # ---- separation ratio across the sweep (full 1,707-protein network) ------
    sep, kbar, top10 = [], [], []
    for b in SWEEP:
        a = np.maximum(corr_full, 0) ** b
        t = tom_from_adjacency(a)
        win = t[same & off].mean()
        bet = t[(~same) & off].mean()
        sep.append(win / bet)
        np.fill_diagonal(a, 0.0)
        kbar.append(a.sum(axis=1).mean())
        srt = np.sort(a, axis=1)[:, ::-1]
        top10.append((srt[:, :10].sum(axis=1) / srt.sum(axis=1)).mean())
        print(f"β={b:>2}  within/between TOM = {win/bet:6.2f}   "
              f"mean k = {kbar[-1]:8.2f}   top-10 share = {top10[-1]:.3f}")

    # ---- legible subset, ordered by module, for the heatmaps -----------------
    picks = []
    for m, n in N_SUB.items():
        g = mods.index[mods == m]
        sub = X[g]
        me = np.linalg.svd(((sub - sub.mean()) / sub.std(ddof=0).replace(0, 1)
                            ).to_numpy(), full_matrices=False)[0][:, 0]
        kme = sub.apply(lambda c: abs(np.corrcoef(c.values, me)[0, 1]))
        picks += kme.sort_values(ascending=False).head(n).index.to_list()
    lab_sub = mods[picks].to_numpy()
    corr_sub = np.corrcoef(X[picks].to_numpy(), rowvar=False)

    fig = plt.figure(figsize=(14.5, 9.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.92], hspace=0.55, wspace=0.42,
                          top=0.79, bottom=0.075, left=0.065, right=0.945)

    # ---- A: TOM at four betas ------------------------------------------------
    for j, b in enumerate(SHOW_BETAS):
        ax = fig.add_subplot(gs[0, j])
        t = tom_from_adjacency(np.maximum(corr_sub, 0) ** b)
        ax.imshow(t, cmap="Reds", vmin=0, vmax=np.percentile(t, 99.5),
                  aspect="auto", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for i, m in enumerate(lab_sub):
            ax.add_patch(plt.Rectangle((i - 0.5, len(lab_sub) - 0.5), 1, 3.0,
                                       color=MOD_COLOR[m], clip_on=False))
        w = t[(lab_sub[:, None] == lab_sub[None, :]) & ~np.eye(len(lab_sub), dtype=bool)].mean()
        bt = t[(lab_sub[:, None] != lab_sub[None, :])].mean()
        ax.set_title(f"β = {b}" + ("   ← chosen" if b == CHOSEN else "")
                     + f"\nwithin / between = {w / bt:.1f}×",
                     fontsize=11, fontweight="bold",
                     color="#2E7D32" if b == CHOSEN else "#333")
        if j == 0:
            ax.set_ylabel("proteins, in module order", fontsize=9)

    fig.text(0.5, 0.885, "A.  Topological overlap at four values of β  —  "
             "same proteins, same order, only β changes",
             ha="center", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.851, "At β = 1 every protein overlaps with every other and no "
             "group can be separated. Raising β lets the real blocks emerge.",
             ha="center", fontsize=9.5, color="#555")

    # ---- B: separation ratio -------------------------------------------------
    axB = fig.add_subplot(gs[1, :2])
    axB.plot(SWEEP, sep, "o-", color="#0072B2", lw=2, ms=5)
    axB.set_yscale("log")
    axB.axvline(CHOSEN, color="#2E7D32", ls=":", lw=1.8)
    axB.axhline(1.0, color="#999", ls="--", lw=1)
    axB.text(1.4, 1.12, "1.0 = no separation at all", fontsize=8.5, color="#777")
    i13 = SWEEP.index(CHOSEN)
    axB.annotate(f"β = 13 → modules overlap\n{sep[i13]:.0f}× more within than between",
                 xy=(CHOSEN, sep[i13]), xytext=(CHOSEN - 10.5, sep[i13] * 0.62),
                 fontsize=9, fontweight="bold", color="#2E7D32",
                 arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.2))
    axB.set_xlabel("soft-threshold power  β")
    axB.set_ylabel("within-module TOM ÷\nbetween-module TOM", fontsize=9.5)
    axB.set_title("B.  How separable the modules are, as β increases",
                  fontsize=11.5, fontweight="bold", loc="left")
    axB.grid(color="#EEEEEE"); axB.set_axisbelow(True)
    for s in ("top", "right"):
        axB.spines[s].set_visible(False)

    axB2 = axB.twinx()
    axB2.plot(SWEEP, top10, "s--", color="#D55E00", lw=1.3, ms=4, alpha=0.85)
    axB2.set_ylabel("top-10 partners' share", fontsize=8.5, color="#D55E00")
    axB2.tick_params(axis="y", labelcolor="#D55E00")
    axB2.set_ylim(0, 1.05)
    axB2.spines["top"].set_visible(False)

    # ---- C: scale-free fit ---------------------------------------------------
    axC = fig.add_subplot(gs[1, 2:])
    axC.plot(sft["Power"], sft["SFT.R.sq"], "o-", color="#0072B2", lw=2, ms=5)
    axC.axhline(0.8, color="#C0392B", ls="--", lw=1.4)
    axC.axvline(CHOSEN, color="#2E7D32", ls=":", lw=1.8)
    axC.text(1.2, 0.82, "R² > 0.8 required", color="#C0392B", fontsize=8.5)
    axC.annotate("smallest β that clears the line",
                 xy=(CHOSEN, 0.83), xytext=(CHOSEN - 11, 0.45), fontsize=9,
                 fontweight="bold", color="#2E7D32",
                 arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.2))
    axC.set_ylim(0, 1.02)
    axC.set_xlabel("soft-threshold power  β")
    axC.set_ylabel("scale-free fit  R²")
    axC.set_title("C.  Why β = 13 and not higher: the stopping rule",
                  fontsize=11.5, fontweight="bold", loc="left")
    axC.grid(color="#EEEEEE"); axC.set_axisbelow(True)
    for s in ("top", "right"):
        axC.spines[s].set_visible(False)

    fig.suptitle("The soft-threshold power β: what it changes, and why 13",
                 fontsize=14.5, fontweight="bold", y=0.955)
    out = OUT_DIR / "beta_effect.png"
    fig.savefig(out, bbox_inches="tight", dpi=165)
    plt.close(fig)
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
