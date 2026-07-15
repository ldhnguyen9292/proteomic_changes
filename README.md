# proteomic_changes

Processing pipeline for the study **"Proteomic changes during human heat stress
and heat acclimation."** It reshapes the raw physiological measurements into a
tidy table and merges them with the Olink NPX proteomics panels.

## Repository layout

```
proteomic_changes/
├── src/                              # Python scripts
│   ├── read_physiological_data.py    # raw Excel -> tidy long-format physiology
│   ├── merge_physiological_npx.py    # join physiology + NPX panels by sample
│   ├── preprocess.py                 # QC + protein filtering -> preprocessed data
│   └── visualize_preprocessing.py    # QC/EDA figures -> results/
├── data/
│   ├── raw_data/                     # RAW DATA (not tracked — download separately)
│   ├── Physiological_Data_Cleaned.csv    # from read_physiological_data.py
│   ├── Physiological_NPX_Merged.csv      # from merge_physiological_npx.py
│   ├── Physiological_NPX_Preprocessed.csv # from preprocess.py (downstream-ready)
│   ├── preprocess_dropped_proteins.csv   # provenance: dropped proteins & reason
│   └── preprocess_physiology_outliers.csv # flagged physiology outliers
└── results/                          # QC/EDA figures from visualize_preprocessing.py
```

## 1. Download the raw data

The raw dataset is **not** stored in this repository. Download it from Figshare:

**https://figshare.com/projects/Proteomic_changes_during_human_heat_stress_and_heat_acclimation/163291**

Unzip the download and arrange the files under `data/raw_data/` exactly as below
(the scripts read `Physiological_data.xlsx` and the `npx/` panels):

```
data/raw_data/
├── Physiological_data.xlsx
├── npx/
│   ├── cardiometabolic_npx.csv
│   ├── cardiometabolic_ii_npx.csv
│   ├── inflammation_npx.csv
│   ├── inflammation_ii_npx.csv
│   ├── neurology_npx.csv
│   ├── neurology_ii_npx.csv
│   ├── oncology_npx.csv
│   └── oncology_ii_npx.csv
├── delta_npx/        # optional (not used by the current scripts)
├── fold_changes/     # optional
└── olink_raw/        # optional
```

## 2. Install dependencies

Requires Python 3.8+.

```bash
pip install pandas openpyxl matplotlib seaborn
```

## 3. Run

From the repository root, run in order:

```bash
# 1) Reshape the raw physiological workbook into a tidy long table
python src/read_physiological_data.py
#    -> data/Physiological_Data_Cleaned.csv

# 2) Merge physiology with the Olink NPX panels (by Participant + Exposure)
python src/merge_physiological_npx.py
#    -> data/Physiological_NPX_Merged.csv

# 3) Quality control + protein filtering (prints a QC report)
python src/preprocess.py
#    -> data/Physiological_NPX_Preprocessed.csv  (+ dropped/outlier logs)

# 4) Visualize the QC / preprocessing results
python src/visualize_preprocessing.py
#    -> results/*.png
```

## Preprocessing & QC

`preprocess.py` checks for duplicated samples, inconsistent identifiers, and
missing values; summarizes physiological distributions and flags potential
outliers (kept, not removed — n is small and the extremes are biological); and
removes proteins with excessive missingness (> 20% of samples) or extremely low
variance (< 0.01). Both thresholds are constants at the top of the script, with a
printed sensitivity table. `visualize_preprocessing.py` renders these into
`results/`:

- `physiology_distributions.png` — per-variable boxplots by thermal stage.
- `protein_missingness.png` — missingness and variance profiles with thresholds.
- `protein_filtering_summary.png` — proteins kept vs dropped, by reason.

## Data notes

- **Exposure codes** encode the two experimental factors: the letters give the
  thermal stage (`PR` = Normothermic, `PT` = Hyperthermic) and the digit gives
  the acclimation state (`1` = before / Pre, `2` = after / Post).

  | Code | Thermal stage | Acclimation |
  |------|---------------|-------------|
  | PR1  | Normothermic  | Pre         |
  | PT1  | Hyperthermic  | Pre         |
  | PR2  | Normothermic  | Post        |
  | PT2  | Hyperthermic  | Post        |

- Participant IDs are normalized when merging: physiology uses `001-B` while the
  NPX files use `001B`.
- A few Olink bridge assays appear on multiple panels (`CXCL8`, `IL6`, `TNF`,
  `IDO1`, `LMOD1`, `SCRIB`). In the merged file these are suffixed with the panel
  name (e.g. `IL6__Inflammation`) so no measurement is overwritten.
