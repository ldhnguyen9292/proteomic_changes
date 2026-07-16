"""Visualize the data-preprocessing / QC results into the results/ folder.

Figures produced:
  results/physiology_distributions.png  -- per-variable boxplots split by
      exposure stage (thermal x acclimation), PR1 --> PT1 --> PR2 --> PT2, with
      individual samples overlaid so outliers are visible.
  results/session_distributions.png     -- session-level variables (measured once
      per acclimation session, identical across thermal stages) split by
      acclimation only, Pre --> Post.
  results/protein_missingness.png       -- protein missingness profile with the
      drop threshold, and the low-variance profile with its threshold.
  results/protein_filtering_summary.png -- proteins kept vs dropped (by reason).
  results/preprocessing_summary.png     -- table of QC/preprocessing problems
      found and the action taken for each (numbers derived from the data).

Run after preprocess.py (it reads the same artifacts).
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from read_physiological_data import DATA_DIR, PROJECT_DIR
from preprocess import MAX_MISSING_FRAC, MIN_VARIANCE, PHYS_META, KEYS, OUTLIERS_PATH

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Okabe-Ito colorblind-safe palette (validated, widely used for CVD safety).
C_NORMO = "#0072B2"   # blue  -> Normothermic (cool)
C_HYPER = "#D55E00"   # vermillion -> Hyperthermic (hot)

# Four exposure stages (thermal x acclimation), ordered PR1 -> PT1 -> PR2 -> PT2.
# Hue encodes thermal stage (cool blues vs warm oranges); lightness encodes
# acclimation (Pre = lighter, Post = darker) -- all Okabe-Ito, CVD-safe.
EXPOSURE_ORDER = ["PR1", "PT1", "PR2", "PT2"]
EXPOSURE_COLORS = {
    "PR1": "#56B4E9",   # sky blue    -> Pre  / Normothermic
    "PT1": "#E69F00",   # orange      -> Pre  / Hyperthermic
    "PR2": "#0072B2",   # blue        -> Post / Normothermic
    "PT2": "#D55E00",   # vermillion  -> Post / Hyperthermic
}
EXPOSURE_LABELS = {
    "PR1": "PR1 · Pre / Normothermic",
    "PT1": "PT1 · Pre / Hyperthermic",
    "PR2": "PR2 · Post / Normothermic",
    "PT2": "PT2 · Post / Hyperthermic",
}

# These are measured once per acclimation session (identical across the
# Normothermic/Hyperthermic stages), so splitting them into 4 exposure stages
# only duplicates values -- show them by acclimation (Pre -> Post) instead.
SESSION_VARS = {
    "Change (post-pre) in nude body weight (kg)",
    "Heat exposure duration (s)",
    "Sweat rate (L/h)",
}
ACCLIM_ORDER = ["Pre", "Post"]
ACCLIM_COLORS = {"Pre": "#BBBBBB", "Post": "#666666"}  # neutral grey: not a thermal contrast
ACCLIM_LABELS = {"Pre": "Pre-acclimation", "Post": "Post-acclimation"}

C_KEEP = "#0072B2"
C_DROP_MISS = "#D55E00"
C_DROP_VAR = "#E69F00"  # orange
C_THRESH = "#CC3311"    # threshold reference lines
INK = "#222222"
GRID = "#DDDDDD"

# Preprocessing-summary table: colour per processing stage.
STAGE_COLORS = {"QC": "#0072B2", "EDA": "#E69F00", "Filter": "#009E73"}

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.8,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def _recessive(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _grouped_box(ax, src, col, var, groups, gcolors):
    """Boxplot of `var` split by `col` into `groups`, with jittered samples."""
    _recessive(ax)
    positions = list(range(len(groups)))
    data = [src.loc[src[col] == g, var].values for g in groups]
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color=INK, linewidth=1.4))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(gcolors[g])
        patch.set_alpha(0.30)
        patch.set_edgecolor(gcolors[g])
    # Overlay individual samples (jittered) so every point / outlier shows.
    for i, g in enumerate(groups):
        y = src.loc[src[col] == g, var].values
        jitter = np.linspace(-0.16, 0.16, len(y))
        ax.scatter(np.full(len(y), i) + jitter, y, s=22,
                   color=gcolors[g], edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_title(var, fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels(groups, fontsize=9)


def plot_physiology_distributions(phys):
    """Variables that vary by thermal stage, split by exposure PR1->PT1->PR2->PT2."""
    phys_vars = [c for c in phys.columns
                 if c not in PHYS_META and c not in SESSION_VARS]
    stages = [s for s in EXPOSURE_ORDER if s in set(phys["Exposure"])]

    ncol = 4
    nrow = int(np.ceil(len(phys_vars) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.1 * nrow))
    axes = axes.ravel()

    for ax, var in zip(axes, phys_vars):
        _grouped_box(ax, phys, "Exposure", var, stages, EXPOSURE_COLORS)

    for ax in axes[len(phys_vars):]:
        ax.set_visible(False)

    # Single shared legend (identity is not color-alone: axis labels also name it).
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=EXPOSURE_COLORS[s],
                          markeredgecolor="white", label=EXPOSURE_LABELS[s])
               for s in stages]
    fig.legend(handles=handles, loc="upper center", ncol=len(stages), frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("Physiological variable distributions by exposure stage "
                 "(PR1 → PT1 → PR2 → PT2; points = individual samples)",
                 y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = RESULTS_DIR / "physiology_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_session_distributions(phys):
    """Session-level variables (one value per acclimation session), split Pre->Post.

    These are measured once per session and are identical across the
    Normothermic/Hyperthermic stages, so splitting them by thermal stage would
    only duplicate values -- they are shown by acclimation only.
    """
    session_vars = [c for c in phys.columns if c in SESSION_VARS]
    acclim = [a for a in ACCLIM_ORDER if a in set(phys["Acclimation"])]
    # One row per (participant, acclimation) so samples aren't double-counted.
    session_phys = phys.drop_duplicates(["Participant", "Acclimation"])

    fig, axes = plt.subplots(1, len(session_vars),
                             figsize=(4.4 * len(session_vars), 4.0))
    axes = np.atleast_1d(axes).ravel()

    for ax, var in zip(axes, session_vars):
        _grouped_box(ax, session_phys, "Acclimation", var, acclim, ACCLIM_COLORS)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=ACCLIM_COLORS[a],
                          markeredgecolor="white", label=ACCLIM_LABELS[a])
               for a in acclim]
    fig.legend(handles=handles, loc="upper center", ncol=len(acclim), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Session-level variable distributions by acclimation "
                 "(Pre → Post; identical across thermal stages)",
                 y=1.10, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = RESULTS_DIR / "session_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_protein_missingness(miss_pct, variance):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # -- Missingness: discrete levels (0, 2.5, 5, 7.5, 10, 100 %) --
    _recessive(ax1)
    counts = miss_pct.round(1).value_counts().sort_index()
    bars = ax1.bar(range(len(counts)), counts.values, color=C_NORMO, alpha=0.85,
                   width=0.7, zorder=2)
    ax1.set_yscale("log")
    ax1.set_xticks(range(len(counts)))
    ax1.set_xticklabels([f"{v:g}" for v in counts.index])
    ax1.set_xlabel("Missing values per protein (% of 40 samples)")
    ax1.set_ylabel("Number of proteins (log scale)")
    ax1.set_title("Protein missingness")
    for b, v in zip(bars, counts.values):
        ax1.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center",
                 va="bottom", fontsize=8)
    thr = MAX_MISSING_FRAC * 100
    ax1.axvline(np.interp(thr, counts.index, range(len(counts))),
                color=C_THRESH, linestyle="--", linewidth=1.4)
    ax1.text(0.97, 0.95, f"drop > {thr:g}% missing\n(removes {(miss_pct > thr).sum()})",
             transform=ax1.transAxes, ha="right", va="top", color=C_THRESH, fontsize=9)

    # -- Variance: histogram on log10 axis --
    _recessive(ax2)
    logv = np.log10(variance.dropna())
    ax2.hist(logv, bins=45, color=C_NORMO, alpha=0.85, zorder=2)
    ax2.axvline(np.log10(MIN_VARIANCE), color=C_THRESH,
                linestyle="--", linewidth=1.4)
    ax2.set_xlabel("log10(variance)  [NPX, log2 scale]")
    ax2.set_ylabel("Number of proteins")
    ax2.set_title("Protein variance")
    ax2.text(0.03, 0.95,
             f"drop variance < {MIN_VARIANCE}\n(removes {(variance < MIN_VARIANCE).sum()})",
             transform=ax2.transAxes, ha="left", va="top", color=C_THRESH, fontsize=9)

    fig.suptitle("Protein filters: missingness & variance",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = RESULTS_DIR / "protein_missingness.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_filtering_summary(n_total, dropped):
    n_miss = int((dropped["Reason"].str.startswith("missing")).sum())
    n_var = int((dropped["Reason"].str.startswith("variance")).sum())
    n_keep = n_total - n_miss - n_var

    labels = ["Kept", f"Dropped: missing > {MAX_MISSING_FRAC:.0%}",
              f"Dropped: variance < {MIN_VARIANCE}"]
    values = [n_keep, n_miss, n_var]
    colors = [C_KEEP, C_DROP_MISS, C_DROP_VAR]

    fig, ax = plt.subplots(figsize=(9, 3.2))
    _recessive(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    bars = ax.barh(labels, values, color=colors,
                   alpha=0.9, height=0.6, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("Number of proteins (log scale)")
    ax.invert_yaxis()
    for b, v in zip(bars, values):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v}", va="center",
                ha="left", fontsize=10, fontweight="bold")
    ax.set_title(f"Protein filtering result  ({n_total} in → {n_keep} kept)",
                 fontsize=12)
    fig.tight_layout()
    out = RESULTS_DIR / "protein_filtering_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_preprocessing_summary(phys, merged, dropped):
    """Table of QC/preprocessing problems and the action taken for each.

    Numbers are derived from the data so the table cannot drift out of sync
    with the pipeline or the filtering thresholds.
    """
    phys_cols = [c for c in merged.columns if c in phys.columns]
    protein_cols = [c for c in merged.columns if c not in phys_cols]

    n_dup = int(phys.duplicated().sum() + phys.duplicated(KEYS).sum()
                + merged.duplicated().sum() + merged.duplicated(KEYS).sum())
    n_missing = int(phys.isna().sum().sum())
    id_only_phys = sorted(set(phys["Participant"]) - set(merged["Participant"]))

    miss_list = dropped.loc[dropped["Reason"].str.startswith("missing"), "Protein"].tolist()
    var_list = dropped.loc[dropped["Reason"].str.startswith("variance"), "Protein"].tolist()
    n_total, n_kept = len(protein_cols), len(protein_cols) - len(dropped)

    outliers = pd.read_csv(OUTLIERS_PATH)
    short = lambda v: ("body-weight change" if v.startswith("Change")
                       else v.split(" (")[0])
    out_vars = ", ".join(sorted({short(v) for v in outliers["Variable"]}))

    # (Stage, Problem, How detected, Action taken & outcome)
    rows = [
        ("QC", "Inconsistent participant ID format (e.g. '001-B' vs '001B')",
         "Compared ID sets after stripping non-alphanumeric characters",
         f"Reconciled by hyphen removal during the merge "
         f"({len(id_only_phys)} ID relabelled: {', '.join(id_only_phys)} → "
         f"{', '.join(i.replace('-', '') for i in id_only_phys)})"),
        ("QC", "Duplicate samples",
         "Full-row and [Participant, Exposure] duplication checks",
         f"None found ({n_dup}); no rows removed"),
        ("QC", "Missing physiological values",
         "Per-column null counts",
         f"None found ({n_missing}); no imputation required"),
        ("EDA", f"Extreme physiological values "
                f"({len(outliers)} values across {outliers['Variable'].nunique()} "
                f"variables: {out_vars})",
         "IQR (1.5×) rule and |z-score| > 3 per variable",
         "Flagged only, NOT removed — physiologically plausible extremes "
         "(e.g. heat vasodilation, SSNA %baseline)"),
        ("Filter", "Proteins with excessive missingness",
         f"Missing in > {MAX_MISSING_FRAC:.0%} of {len(merged)} samples",
         f"Dropped {len(miss_list)}: {', '.join(miss_list)}"),
        ("Filter", "Near-constant (low-variance) proteins",
         f"NPX variance < {MIN_VARIANCE} (among proteins surviving the "
         f"missingness filter)",
         f"Dropped {len(var_list)}: {', '.join(var_list)}"),
    ]

    col_labels = ["Stage", "Problem", "How it was detected",
                  "Action taken & outcome"]
    col_widths = [0.08, 0.26, 0.26, 0.40]
    wrap_chars = [10, 34, 34, 52]

    cell_text = [[textwrap.fill(c, w) for c, w in zip(row, wrap_chars)]
                 for row in rows]
    # Row heights track the tallest wrapped cell so nothing is clipped.
    row_lines = [1] + [max(cell.count("\n") + 1 for cell in r) for r in cell_text]
    unit = 0.92 / sum(row_lines)

    fig, ax = plt.subplots(figsize=(15, 0.42 * sum(row_lines) + 1.0))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   colWidths=col_widths, cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#FFFFFF")
        cell.set_linewidth(1.2)
        cell.set_height(row_lines[r] * unit)
        cell.set_text_props(va="center")
        if r == 0:  # header
            cell.set_facecolor(INK)
            cell.set_text_props(color="white", fontweight="bold", va="center")
        else:
            stage = rows[r - 1][0]
            cell.set_facecolor("#F5F5F5" if r % 2 else "#FFFFFF")
            if c == 0:  # colour-code the stage cell
                cell.set_text_props(color=STAGE_COLORS.get(stage, INK),
                                    fontweight="bold", va="center")

    ax.set_title(
        f"Data preprocessing: problems found & actions taken\n"
        f"{len(phys)} samples retained · {n_total} proteins in → {n_kept} kept",
        fontsize=13, fontweight="bold", pad=18)
    fig.tight_layout()
    out = RESULTS_DIR / "preprocessing_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    phys = pd.read_csv(DATA_DIR / "Physiological_Data_Cleaned.csv")
    merged = pd.read_csv(DATA_DIR / "Physiological_NPX_Merged.csv")
    dropped = pd.read_csv(DATA_DIR / "preprocess_dropped_proteins.csv")

    phys_cols = [c for c in merged.columns if c in phys.columns]
    protein_cols = [c for c in merged.columns if c not in phys_cols]
    miss_pct = merged[protein_cols].isna().mean() * 100
    variance = merged[protein_cols].var(numeric_only=True)

    outs = [
        plot_physiology_distributions(phys),
        plot_session_distributions(phys),
        plot_preprocessing_summary(phys, merged, dropped),
        plot_protein_missingness(miss_pct, variance),
        plot_filtering_summary(len(protein_cols), dropped),
    ]
    print("Saved figures:")
    for o in outs:
        print(" -", o)


if __name__ == "__main__":
    main()
