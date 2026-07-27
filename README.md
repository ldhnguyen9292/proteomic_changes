# proteomic_changes

Analysis pipeline for **"Proteomic changes during human heat stress and heat
acclimation."** Olink NPX proteomics + physiology from 10 participants × 4
exposures. A shared ingestion step feeds two analysis tracks:

1. **differential_expression** — QC/preprocessing + paired differential-expression
   (reproduction of the source study's contrasts).
2. **wgcna** — weighted co-expression networks relating protein modules to
   physiological traits (primary outcome: sweat rate), with two missing-data
   workflows (impute vs complete-case) plus a comparison.

## Repository layout

```
proteomic_changes/
├── src/
│   ├── data_pipeline/                     # shared ingestion (imported by all)
│   │   ├── read_physiological_data.py     #   raw xlsx -> Physiological_Data_Cleaned.csv
│   │   └── merge_physiological_npx.py     #   + npx/   -> Physiological_NPX_Merged.csv
│   ├── differential_expression/
│   │   ├── preprocess.py                  #   QC + filtering -> data/differential_expression/
│   │   ├── visualize_preprocessing.py     #   QC/EDA figures -> results/preprocessing/
│   │   ├── check_acclimation_specificity.py  # tests the "180 only after acclimation" claim
│   │   └── pair_t-test.ipynb              #   paired differential-expression analysis
│   └── wgcna/
│       ├── preprocess_wgcna.py            #   impute workflow   -> data/wgcna/impute/
│       ├── preprocess_wgcna_complete.py   #   complete-case     -> data/wgcna/complete/
│       ├── visualize_wgcna.py             #   -> results/wgcna/impute/
│       ├── visualize_wgcna_complete.py    #   -> results/wgcna/complete/
│       ├── compare_wgcna.py               #   -> results/wgcna/impute_vs_complete.png
│       ├── model_development.ipynb            # WGCNA on the imputed set (1707 proteins)
│       └── model_development_complete.ipynb   # WGCNA on the complete set (868 proteins)
├── data/
│   ├── raw_data/                          # RAW DATA (not tracked — download separately)
│   ├── Physiological_Data_Cleaned.csv     # shared (read_physiological_data.py)
│   ├── Physiological_NPX_Merged.csv       # shared (merge_physiological_npx.py)
│   ├── differential_expression/           # Preprocessed + dropped + outlier logs
│   └── wgcna/
│       ├── impute/    wgcna_expression.csv, wgcna_traits.csv   # KNN-imputed
│       └── complete/  wgcna_expression.csv, wgcna_traits.csv   # drop-any-missing
└── results/
    ├── preprocessing/                     # QC / preprocessing figures
    └── wgcna/
        ├── impute/    wgcna_preprocessing.png
        ├── complete/  wgcna_preprocessing.png
        └── impute_vs_complete.png         # side-by-side sweat comparison
```

## 1. Download the raw data

Not stored in this repo. Download from Figshare and unzip under `data/raw_data/`:

**https://figshare.com/projects/Proteomic_changes_during_human_heat_stress_and_heat_acclimation/163291**

```
data/raw_data/
├── Physiological_data.xlsx
├── npx/            # 8 per-panel NPX files (used by merge_physiological_npx.py)
├── olink_raw/      # 2 long-format Olink exports (used for LOD / QC flags)
├── delta_npx/      # per-subject paired differences
└── fold_changes/   # group-level log2 fold changes
```

## 2. Install dependencies

Base scripts need Python 3.8+ with `pandas openpyxl numpy scipy scikit-learn matplotlib`.

The **WGCNA notebooks** and `compare_wgcna.py` additionally require **PyWGCNA**,
which here is installed under **Python 3.9** — run those with the Python 3.9
kernel/interpreter.

## 3. Run order

```bash
# 1) shared ingestion
python src/data_pipeline/read_physiological_data.py     # -> data/Physiological_Data_Cleaned.csv
python src/data_pipeline/merge_physiological_npx.py     # -> data/Physiological_NPX_Merged.csv

# 2) differential-expression track
python src/differential_expression/preprocess.py            # -> data/differential_expression/
python src/differential_expression/visualize_preprocessing.py  # -> results/preprocessing/
python src/differential_expression/check_acclimation_specificity.py  # "180" reproducibility check
#    then run src/differential_expression/pair_t-test.ipynb

# 3) WGCNA track (impute + complete-case + comparison)
python src/wgcna/preprocess_wgcna.py            # -> data/wgcna/impute/   (1707 proteins)
python src/wgcna/visualize_wgcna.py             # -> results/wgcna/impute/
python src/wgcna/preprocess_wgcna_complete.py   # -> data/wgcna/complete/ (868 proteins)
python src/wgcna/visualize_wgcna_complete.py    # -> results/wgcna/complete/
python src/wgcna/compare_wgcna.py               # -> results/wgcna/impute_vs_complete.png
#    then run src/wgcna/model_development{,_complete}.ipynb  (Python 3.9 / PyWGCNA)

# 4) hub prioritization + GO/KEGG enrichment (Python 3.9; enrichment needs internet)
python src/wgcna/enrichment.py impute           # -> results/wgcna/impute/enrichment/
python src/wgcna/enrichment.py complete         # -> results/wgcna/complete/enrichment/
python src/wgcna/enrichment.py summary          # -> results/wgcna/enrichment_theme_summary.{csv,png}
python src/wgcna/hub_prioritization.py          # -> results/wgcna/impute/hub_prioritization.{csv,png}
#   (hub_prioritization reads enrichment's module_selection.csv — run enrichment first)
python src/wgcna/mmgs_threshold.py              # -> results/wgcna/mmgs_threshold_impute.png
#   (reads enrichment's module_selection.csv; no WGCNA rebuild, runs on Python 3.8)
```

`src/` is split into subfolders; each script prepends `src/data_pipeline/` to
`sys.path` so the shared `read_physiological_data` module stays importable, and
the notebooks do the same in their first cell.

## Preprocessing & QC

**Protein filtering** removes only assays that failed QC (missing in > 20/40
samples); below-LOD values are kept, not imputed, and no variance filter is
applied. QC-excluded measurements (`QC_Warning=WARN`, ~3.0% of sample-assays)
are set to NaN and handled by complete-pairs downstream.

**Reproducibility check — "180 changed only after acclimation".** The study derives
"180" by *set subtraction* (proteins passing the cut-off in post-acclimation heat
stress but not pre-acclimation). `check_acclimation_specificity.py` shows this is a
thresholding artifact: the correct test — the `Thermal_Stage × Acclimation`
interaction (a per-protein one-sample t-test on `(PT2−PR2)−(PT1−PR1)`, identical to
the two-way RM-ANOVA interaction) — finds **0** proteins at BH-FDR < 0.05, and its
nominal count (153) sits at the chance level (~147). The interaction p-values are
uniform (`results/preprocessing/acclimation_specificity.png`). This is the classic
"difference between significant and non-significant is not itself significant"
fallacy (Gelman & Stern, 2006).

**WGCNA "missing"** = QC-excluded (`QC_Warning=WARN`) **or** below the limit of
detection. Three missing-value scenarios were tested and compared on the sweat
modules (`results/wgcna/preprocessing_comparison.png`); **impute · keep ≥ 36/40
was chosen** — it is the only scenario where both sweat networks pass all three
evaluation checks:

| scenario | proteins | rule |
|----------|----------|------|
| **impute · keep ≥ 36/40** (`wgcna/impute`, chosen) | **1707** | keep proteins detected in ≥ 36 of 40 samples (≤ 10% missing); KNN-impute the rest |
| impute · keep ≥ 32/40 (sensitivity) | 1835 | keep proteins detected in ≥ 32 of 40 samples (≤ 20% missing); KNN-impute the rest |
| complete case (`wgcna/complete`) | 868 | drop any protein with a missing value (no imputation) |

`preprocess_wgcna.py` builds the chosen (10%) input; `preprocess_wgcna_complete.py`
builds the complete-case input; the 20% scenario is the relaxed-threshold check.

## Module selection for enrichment

**Scope: the chosen impute 10% workflow, with local sweat rate (LSR) as the sweat
outcome.** Whole-body `Sweat rate (L/h)` is not screened; the complete-case run is a
sensitivity check only.

A module is selected only if it passes **all three proposal criteria**:

| # | criterion | cut-off |
|---|-----------|---------|
| 1 | scale-free topology fit | R² > 0.8 at the selected soft-power |
| 2 | module eigengene vs LSR | \|r\| ≥ 0.5 **and** BH-FDR < 0.05 |
| 3 | MM vs GS within the module | \|r\| > 0.6 **and** p < 0.05 |

This gives **three modules**, both in the heat contrasts:

| network | module | n | R² | r | FDR | MM-GS |
|---------|--------|---|----|---|-----|-------|
| PT1/PT2 | dimgrey | 127 | 0.830 | +0.554 | 0.028 | 0.837 |
| PR2/PT2 | black | 946 | 0.820 | +0.556 | 0.044 | 0.762 |
| PT1/PT2 | lightgrey | 695 | 0.830 | +0.513 | 0.028 | 0.666 |

Criterion 3 uses the proposal's original **0.6**, not the relaxed 0.5: all three have
MM-GS ≥ 0.666, so the two cut-offs give an identical result and **no deviation from the
proposal needs defending**.

### Excluded modules

| module | clears | fails |
|--------|--------|-------|
| `PT1/PT2 darkgrey` | 1, 2 (r = +0.528, FDR = 0.028) | **3** — MM-GS = **0.162** |
| `full black` | 1, FDR = 0.004 | **2** — r = **0.485** < 0.5 |

`PT1/PT2 darkgrey` is the important one. A low MM-GS means module membership and sweat
association are essentially unrelated inside that module — how central a protein is
tells you nothing about how sweat-associated it is. Its enrichment is real, but no
protein-level or hub claim from it would be supported, which is why it is dropped
rather than reported with a caveat.

`full black` also fails on design grounds independent of the threshold: the full
network pools all 40 samples across all four exposures, so its LSR correlation is
confounded by thermal stage (hyperthermic exposures raise LSR and the whole
heat-response programme together). Only the contrast networks are interpretable.

Excluded modules keep their per-criterion flags (`passes_topology`, `passes_fdr`,
`passes_mt_r`, `passes_mmgs`) in `module_selection.csv`, and their enrichment CSVs are
retained under `out_of_scope/` rather than deleted.

### Complete-case sensitivity: zero modules

Under all three criteria the complete-case workflow selects **nothing**:

- `PR2/PT2 black` — R² = **0.79998**, just under the 0.8 topology cut. Its power sweep
  never reaches 0.8 (max 0.79998 at power 19, the top of the range).
- `PT1/PT2 black` — MM-GS = **0.437**, fails criterion 3.

Marginal as the first one is, this is a further argument for the impute 10% choice
already made on the three-check comparison: the complete-case network cannot support
the analysis under the proposal's own standards.

### Where the MM-GS cut-off should sit

`mmgs_threshold.py` → `results/wgcna/mmgs_threshold_impute.png`.

**The proposal's 0.6 is used.** Among the four modules clearing criteria 1–2, MM-GS is
bimodal — 0.837, 0.762, 0.666, then a gap down to 0.162 — so **any cut-off in
0.162–0.666 gives the same three modules**. 0.5 and 0.6 are equivalent here, and 0.6
avoids justifying a deviation from the proposal. Do not go above 0.666: at 0.7,
lightgrey would drop out too.

**Judge MM-GS on effect size, not its p-value.** MM-GS is tested over n = proteins in
the module (127–946 here), and those proteins are co-expressed by construction, so the
test is anti-conservative twice over: `PT1/PT2 darkgrey` reaches **p = 0.011 at
r = 0.162**. The `p < 0.05` half of criterion 3 is satisfied by trivially small
correlations and should not be relied on.

MM-GS is also now shown as a bold final column (after a blank spacer) in
`module_trait_heatmaps.png`. It is **not** the same quantity as the trait columns: those
correlate the module *eigengene* with a trait across samples, while MM-GS correlates
Module Membership with Gene Significance across the proteins *inside* the module.

`out_of_scope/` subfolders hold enrichment CSVs generated under a wider screen
(whole-body sweat rate, complete-case full network); they are outside the chosen
scenario and are excluded from the theme summary.

## Data notes

- **Exposure codes**: letters = thermal stage (`PR` = Normothermic, `PT` =
  Hyperthermic), digit = acclimation (`1` = Pre, `2` = Post).

  | Code | Thermal stage | Acclimation |
  |------|---------------|-------------|
  | PR1  | Normothermic  | Pre         |
  | PT1  | Hyperthermic  | Pre         |
  | PR2  | Normothermic  | Post        |
  | PT2  | Hyperthermic  | Post        |

- NPX values are **log2**; a paired difference is a log2 fold-change.
- Participant IDs are normalized on merge: physiology `001-B` vs NPX `001B`.
- Bridge assays on multiple panels (`CXCL8`, `IL6`, `TNF`, `IDO1`, `LMOD1`,
  `SCRIB`) are suffixed with the panel name (e.g. `IL6__Inflammation`) so no
  measurement is overwritten.
