"""Short animated walkthrough of how WGCNA separates proteins into modules.

Eight scenes, built from the real PT1/PT2 data:

  1. the input matrix           20 samples x proteins
  2. correlate every pair       a messy, unordered correlation matrix
  3. raise to the power beta    animated 1 -> 13; weak correlations collapse
  4. topological overlap (TOM)  "correlated AND sharing neighbours"
  5. cluster on 1 - TOM         the tree builds up merge by merge
  6. Dynamic Tree Cut           branches become modules
  7. reorder by module          block structure on the diagonal
  8. module eigengene vs LSR    what the modules are then tested against

For legibility the matrices are drawn on a subset of proteins (the strongest
members of each module); the module assignment shown is the real one from the
full 1,707-protein network, and scene 8 uses the full module. This is stated on
the frames themselves so the clip cannot be mistaken for the full computation.

Outputs (results/wgcna/):
  wgcna_module_detection.mp4   -- for slides
  wgcna_module_detection.gif   -- for anything that will not take video

Run with Python 3.8+ (needs scipy, sklearn, matplotlib; ffmpeg for the mp4).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_pipeline"))
from read_physiological_data import DATA_DIR, PROJECT_DIR

CASE_DIR = DATA_DIR / "wgcna" / "impute"
OUT_DIR = PROJECT_DIR / "results" / "wgcna"
CODES = ["PT1", "PT2"]
BETA = 13
TRAIT = "LSR (mg/min/cm2)"
FPS = 12

# real module colours -> display colour + report label
MOD_STYLE = {
    "dimgrey":   ("#D55E00", "M1"),
    "lightgrey": ("#0072B2", "M3"),
    "black":     ("#009E73", "—"),
    "darkgrey":  ("#E69F00", "—"),
}
N_PER_MODULE = {"lightgrey": 26, "black": 24, "darkgrey": 18, "dimgrey": 16}

SCENES = [
    ("1. The input", "20 samples (PT1 + PT2) × 1,707 proteins.\n"
                     "Each column is one protein's profile across the samples.", 26),
    ("2. Correlate every protein pair", "Pearson correlation between all protein "
     "pairs.\nIn an arbitrary protein order there is no visible structure.", 26),
    ("3. Raise to the power β", "adjacency = max(r, 0)^β  (signed hybrid).\n"
     "A high β keeps strong correlations and collapses weak ones toward zero.", 40),
    ("4. Topological overlap (TOM)", "Two proteins are close if they are "
     "correlated AND\nthey share the same neighbours — this sharpens real links.", 30),
    ("5. Cluster on 1 − TOM", "Average-linkage hierarchical clustering.\n"
     "The tree is built one merge at a time.", 34),
    ("6. Dynamic Tree Cut", "Each branch of at least 30 proteins becomes a module.\n"
     "Modules whose eigengenes correlate > 0.8 are then merged.", 28),
    ("7. Reorder by module", "The same TOM, proteins sorted by module: each module "
     "is a\nblock of proteins that overlap with each other, not with the rest.", 28),
    ("8. Module eigengene vs sweat rate", "Each module is summarised by its first "
     "principal component\n(the eigengene), which is then correlated with LSR.", 34),
]


def tom_from_adjacency(a):
    """Standard WGCNA topological overlap matrix."""
    a = a.copy()
    np.fill_diagonal(a, 0.0)
    k = a.sum(axis=1)
    num = a @ a + a
    den = np.minimum.outer(k, k) + 1.0 - a
    tom = num / den
    np.fill_diagonal(tom, 1.0)
    return tom


def load():
    expr = pd.read_csv(CASE_DIR / "wgcna_expression.csv", index_col=0)
    tr = pd.read_csv(CASE_DIR / "wgcna_traits.csv", index_col=0)
    common = expr.index.intersection(tr.index)
    expr, tr = expr.loc[common], tr.loc[common]
    keep = tr.index.str.split("-").str[-1].isin(CODES)
    return expr.loc[keep], tr.loc[keep]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expr, tr = load()
    sel = pd.read_csv(OUT_DIR / "impute" / "enrichment" / "module_selection.csv")
    del sel  # (only used to confirm the file exists / stays in step with the screen)

    # module assignment from the real full network, cached by module_detection_figure
    import pickle
    pkl = Path("/tmp/pt1pt2_obj.pkl")
    if pkl.exists():
        obj = pickle.load(open(pkl, "rb"))
        mods = obj.datExpr.var["moduleColors"]
    else:                                   # rebuild if the cache is gone
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import model_development as md
        ex = tr.index.str.split("-").str[-1]
        obj = md.build(expr, tr, pd.Series(ex.isin(CODES), index=tr.index), "PT1_PT2")
        mods = obj.datExpr.var["moduleColors"]

    # ---- pick a legible subset: strongest members of each module -------------
    X = expr[mods.index]
    picks = []
    for m, n in N_PER_MODULE.items():
        genes = mods.index[mods == m]
        sub = X[genes]
        me = PCA(n_components=1).fit_transform(
            (sub - sub.mean()) / sub.std(ddof=0).replace(0, 1))[:, 0]
        kme = sub.apply(lambda c: np.corrcoef(c.values, me)[0, 1]).abs()
        picks.append(kme.sort_values(ascending=False).head(n).index.to_list())
    genes = [g for p in picks for g in p]
    labels = mods[genes].to_numpy()
    Z = X[genes]
    Zs = ((Z - Z.mean()) / Z.std(ddof=0).replace(0, 1)).to_numpy()

    # Display order for scenes 1-4 must be arbitrary: the subset was picked
    # module by module, so leaving it in that order would already show blocks
    # and contradict the caption. Maths is unaffected -- display only.
    shuf = np.random.default_rng(7).permutation(len(genes))

    corr = np.corrcoef(Zs, rowvar=False)
    adj_full = np.maximum(corr, 0) ** BETA
    tom = tom_from_adjacency(adj_full)
    corr_s = corr[np.ix_(shuf, shuf)]
    tom_s = tom[np.ix_(shuf, shuf)]
    Zs_s = Zs[:, shuf]
    link = linkage(squareform(np.clip(1 - tom, 0, None), checks=False), "average")
    dn = dendrogram(link, no_plot=True)
    leaf = dn["leaves"]
    tom_leaf = tom[np.ix_(leaf, leaf)]
    lab_leaf = labels[leaf]
    blk = np.argsort([list(N_PER_MODULE).index(m) for m in labels], kind="stable")
    tom_blk = tom[np.ix_(blk, blk)]
    lab_blk = labels[blk]

    # scene 8: real eigengene of M1 (dimgrey) vs LSR, full module
    m1 = mods.index[mods == "dimgrey"]
    sub = X[m1]
    me = PCA(n_components=1).fit_transform(
        (sub - sub.mean()) / sub.std(ddof=0).replace(0, 1))[:, 0]
    y = tr.loc[X.index, TRAIT].astype(float).to_numpy()
    if np.corrcoef(me, y)[0, 1] < 0:
        me = -me
    r_me = np.corrcoef(me, y)[0, 1]

    starts = np.cumsum([0] + [s[2] for s in SCENES])
    total = int(starts[-1])

    fig = plt.figure(figsize=(11.6, 5.9))
    fig.patch.set_facecolor("white")
    ax_cap = fig.add_axes([0.03, 0.845, 0.94, 0.13]); ax_cap.axis("off")
    ax_main = fig.add_axes([0.055, 0.135, 0.44, 0.665])
    ax_side = fig.add_axes([0.565, 0.135, 0.40, 0.665])
    ax_bar = fig.add_axes([0.055, 0.028, 0.91, 0.018]); ax_bar.axis("off")

    def scene_of(f):
        i = int(np.searchsorted(starts, f, side="right") - 1)
        i = min(i, len(SCENES) - 1)
        return i, (f - starts[i]) / max(1, SCENES[i][2] - 1)

    def draw_caption(i):
        ax_cap.clear(); ax_cap.axis("off")
        title, sub_t, _ = SCENES[i]
        ax_cap.text(0, 0.95, title, fontsize=16, fontweight="bold", va="top",
                    color="#1F4E79", transform=ax_cap.transAxes)
        ax_cap.text(0, 0.36, sub_t, fontsize=10.5, va="top", color="#333",
                    transform=ax_cap.transAxes)
        ax_cap.text(1.0, 0.95, f"step {i + 1} of {len(SCENES)}", fontsize=9.5,
                    va="top", ha="right", color="#888", transform=ax_cap.transAxes)

    def draw_progress(f):
        ax_bar.clear(); ax_bar.axis("off")
        ax_bar.add_patch(plt.Rectangle((0, 0), 1, 1, color="#E8E8E8",
                                       transform=ax_bar.transAxes))
        ax_bar.add_patch(plt.Rectangle((0, 0), (f + 1) / total, 1, color="#1F4E79",
                                       transform=ax_bar.transAxes))

    def heat(ax, M, title, cmap, vmin, vmax, bars=None):
        ax.clear()
        ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")
        if bars is not None:
            for i, m in enumerate(bars):
                ax.add_patch(plt.Rectangle((i - 0.5, len(bars) - 0.5), 1, 3.2,
                                           color=MOD_STYLE[m][0], clip_on=False))

    def side_text(lines):
        ax_side.clear(); ax_side.axis("off")
        for k, (t, sz, bold, col) in enumerate(lines):
            ax_side.text(0.02, 0.94 - k * 0.088, t, fontsize=sz, fontweight=
                         "bold" if bold else "normal", color=col, va="top",
                         transform=ax_side.transAxes)

    def update(f):
        i, t = scene_of(f)
        draw_caption(i); draw_progress(f)
        ax_main.set_facecolor("white")

        if i == 0:
            heat(ax_main, Zs_s, f"expression (z-scored) — {len(genes)} proteins shown",
                 "RdBu_r", -2.5, 2.5)
            ax_main.set_xlabel("proteins →", fontsize=9)
            ax_main.set_ylabel("20 samples", fontsize=9)
            side_text([("Input", 13, True, "#1F4E79"),
                       ("• 20 samples: 10 participants × PT1, PT2", 10.5, False, "#333"),
                       ("• 1,707 proteins after detection filtering", 10.5, False, "#333"),
                       ("• NPX values, log2 scale", 10.5, False, "#333"),
                       ("", 10, False, "#333"),
                       ("Matrices in this clip are drawn on", 9.5, False, "#777"),
                       (f"{len(genes)} representative proteins so the", 9.5, False, "#777"),
                       ("structure is visible; module labels are", 9.5, False, "#777"),
                       ("the real ones from all 1,707.", 9.5, False, "#777")])
        elif i == 1:
            heat(ax_main, corr_s, "correlation matrix (unordered)", "RdBu_r", -1, 1)
            side_text([("Every protein pair", 13, True, "#1F4E79"),
                       (f"• {len(genes)}×{len(genes)} correlations here;", 10.5, False, "#333"),
                       ("  1,707×1,707 ≈ 1.5 million in the real run", 10.5, False, "#333"),
                       ("• red = positive, blue = negative", 10.5, False, "#333"),
                       ("", 10, False, "#333"),
                       ("Nothing is grouped yet — the order is", 10, False, "#777"),
                       ("arbitrary, so no structure is visible.", 10, False, "#777")])
        elif i == 2:
            b = 1 + t * (BETA - 1)
            heat(ax_main, np.maximum(corr_s, 0) ** b,
                 f"adjacency = max(r, 0)^β    β = {b:.1f}", "Reds", 0, 1)
            side_text([("Soft thresholding", 13, True, "#1F4E79"),
                       (f"β = {b:.1f}", 15, True, "#D55E00"),
                       ("• r = 0.9 → " + f"{0.9 ** b:.2f}", 10.5, False, "#333"),
                       ("• r = 0.5 → " + f"{0.5 ** b:.3f}", 10.5, False, "#333"),
                       ("• r = 0.3 → " + f"{0.3 ** b:.4f}", 10.5, False, "#333"),
                       ("", 10, False, "#333"),
                       ("Weak correlations are pushed to ~0", 10, False, "#777"),
                       ("without a hard cut-off. β is chosen as", 10, False, "#777"),
                       ("the smallest value giving scale-free R² > 0.8.", 9.5, False, "#777")])
        elif i == 3:
            heat(ax_main, tom_s, "topological overlap (TOM)", "Reds", 0, tom.max())
            side_text([("Sharing neighbours", 13, True, "#1F4E79"),
                       ("Two proteins overlap if they are", 10.5, False, "#333"),
                       ("connected AND connected to the", 10.5, False, "#333"),
                       ("same other proteins.", 10.5, False, "#333"),
                       ("", 10, False, "#333"),
                       ("This is what makes WGCNA robust:", 10, False, "#777"),
                       ("a single noisy correlation cannot", 10, False, "#777"),
                       ("create a module on its own.", 10, False, "#777")])
        elif i == 4:
            heat(ax_main, tom_leaf, "TOM, proteins in tree order", "Reds", 0, tom.max())
            ax_side.clear()
            hmax = link[:, 2].max()
            order_h = np.argsort([max(d) for d in dn["dcoord"]])
            n_show = int(np.ceil(t * len(order_h)))
            for j in order_h[:n_show]:
                ax_side.plot(dn["icoord"][j], dn["dcoord"][j], color="#4A4A4A", lw=0.9)
            ax_side.set_xticks([])
            ax_side.set_xlim(0, 10 * len(genes))
            ax_side.set_ylim(link[:, 2].min() * 0.98, hmax * 1.02)
            ax_side.set_ylabel("1 − TOM", fontsize=9)
            ax_side.set_title("hierarchical tree, building up", fontsize=10,
                              fontweight="bold")
            for s in ("top", "right"):
                ax_side.spines[s].set_visible(False)
        elif i == 5:
            heat(ax_main, tom_leaf, "TOM, proteins in tree order", "Reds", 0,
                 tom.max(), bars=lab_leaf)
            ax_side.clear()
            for xs, ys in zip(dn["icoord"], dn["dcoord"]):
                ax_side.plot(xs, ys, color="#4A4A4A", lw=0.9)
            for k, m in enumerate(lab_leaf):
                ax_side.add_patch(plt.Rectangle(
                    (5 + 10 * k - 5, link[:, 2].min() * 0.98), 10,
                    (link[:, 2].max() - link[:, 2].min()) * 0.06,
                    color=MOD_STYLE[m][0], alpha=min(1.0, t * 2)))
            ax_side.set_xticks([])
            ax_side.set_xlim(0, 10 * len(genes))
            ax_side.set_ylim(link[:, 2].min() * 0.98, link[:, 2].max() * 1.02)
            ax_side.set_ylabel("1 − TOM", fontsize=9)
            ax_side.set_title("branches → modules", fontsize=10, fontweight="bold")
            for s in ("top", "right"):
                ax_side.spines[s].set_visible(False)
        elif i == 6:
            heat(ax_main, tom_blk, "TOM, proteins sorted by module", "Reds", 0,
                 tom.max(), bars=lab_blk)
            pos = 0
            side_rows = [("Four modules found", 13, True, "#1F4E79")]
            for m, n in N_PER_MODULE.items():
                c, lb = MOD_STYLE[m]
                real_n = int((mods == m).sum())
                side_rows.append((f"■  {m}{'  (' + lb + ')' if lb != '—' else ''}"
                                  f"   {real_n} proteins", 11, True, c))
                pos += n
            side_rows += [("", 10, False, "#333"),
                          ("Bright blocks on the diagonal are the", 10, False, "#777"),
                          ("modules: high overlap within, low", 10, False, "#777"),
                          ("overlap with everything else. Here no", 10, False, "#777"),
                          ("two eigengenes correlated above 0.8,", 10, False, "#777"),
                          ("so nothing was merged.", 10, False, "#777")]
            side_text(side_rows)
        else:
            ax_main.clear()
            k = max(2, int(2 + t * 18))
            ax_main.plot(range(1, len(me) + 1), me, "-o", color=MOD_STYLE["dimgrey"][0],
                         ms=4, lw=1.4)
            ax_main.set_title("M1 eigengene across the 20 samples", fontsize=10,
                              fontweight="bold")
            ax_main.set_xlabel("sample", fontsize=9)
            ax_main.set_ylabel("eigengene (PC1)", fontsize=9)
            ax_main.grid(color="#EEEEEE"); ax_main.set_axisbelow(True)
            for s in ("top", "right"):
                ax_main.spines[s].set_visible(False)
            ax_side.clear()
            n_show = int(np.ceil(t * len(me))) if t < 1 else len(me)
            ax_side.scatter(me[:n_show], y[:n_show], s=46,
                            color=MOD_STYLE["dimgrey"][0], edgecolor="white", zorder=3)
            if n_show > 2:
                b1, b0 = np.polyfit(me[:n_show], y[:n_show], 1)
                xx = np.linspace(me.min(), me.max(), 10)
                ax_side.plot(xx, b1 * xx + b0, color="#333", lw=1.3, ls="--")
            ax_side.set_xlabel("M1 eigengene", fontsize=9)
            ax_side.set_ylabel("LSR (mg/min/cm²)", fontsize=9)
            ax_side.set_title(f"module–trait  r = {r_me:+.2f}   (FDR = 0.028)",
                              fontsize=10, fontweight="bold")
            ax_side.grid(color="#EEEEEE"); ax_side.set_axisbelow(True)
            for s in ("top", "right"):
                ax_side.spines[s].set_visible(False)
        return []

    anim = animation.FuncAnimation(fig, update, frames=total, interval=1000 / FPS,
                                   blit=False)
    mp4 = OUT_DIR / "wgcna_module_detection.mp4"
    anim.save(str(mp4), writer=animation.FFMpegWriter(fps=FPS, bitrate=2600), dpi=125)
    print("saved:", mp4, f"({total} frames, {total / FPS:.0f}s)")

    gif = OUT_DIR / "wgcna_module_detection.gif"
    anim.save(str(gif), writer=animation.PillowWriter(fps=FPS), dpi=70)
    print("saved:", gif)
    plt.close(fig)


if __name__ == "__main__":
    main()
