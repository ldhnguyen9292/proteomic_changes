"""Read the raw Physiological_data.xlsx and reshape it into a tidy long frame.

The raw workbook is a human-formatted, two-block report (Pre-acclimation and
Post-acclimation). Each thermal measure appears twice per block: an
``… initial (…)`` column (Normothermic baseline) and a ``… final (…)`` column
(Hyperthermic / heat). Three trailing columns (body-weight change, heat-exposure
duration, sweat rate) are per-participant and not split by thermal stage.

This module rebuilds the tidy long frame that downstream analysis expects
(one row per Participant x Acclimation x Thermal_Stage), so we no longer depend
on a pre-cleaned ``Physiological_Data_Cleaned.xlsx``.
"""

from pathlib import Path

import pandas as pd

# This script lives in src/data_pipeline/; the project root is two levels up.
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
# Raw workbook (git-ignored) inside data/raw_data/.
DATA_PATH = DATA_DIR / "raw_data" / "Physiological_data.xlsx"
# Cleaned long-format output written into data/.
CLEANED_PATH = DATA_DIR / "Physiological_Data_Cleaned.csv"

# Row indices in the raw sheet (0-based, header=None).
HEADER_ROW = 2
PRE_ROWS = slice(3, 13)    # 10 Pre-acclimation participants (rows 3..12)
POST_ROWS = slice(18, 28)  # 10 Post-acclimation participants (rows 18..27)

# Olink NPX "Exposure" code semantics (npx/ folder):
#   PR = Normal (Normothermic),  PT = Hot (Hyperthermic)
#   ...1 = before thermal adaptation (Pre),  ...2 = after thermal adaptation (Post)
# Keyed by (Thermal_Stage, Acclimation) so physiology rows carry the same code
# as the NPX files and can be joined directly.
EXPOSURE_FROM_STAGE = {
    ("Normothermic", "Pre"): "PR1",
    ("Hyperthermic", "Pre"): "PT1",
    ("Normothermic", "Post"): "PR2",
    ("Hyperthermic", "Post"): "PT2",
}

# Categorical key columns that should lead the tidy frame.
KEY_COLS = ["Participant", "Acclimation", "Thermal_Stage", "Exposure"]


def process_thermal_stages(sub_df: pd.DataFrame, acclimation_label: str) -> pd.DataFrame:
    """Melt one acclimation block (Pre or Post) into long form by thermal stage.

    ``initial`` columns become Normothermic rows and ``final`` columns become
    Hyperthermic rows; stripping those words aligns the two halves so they can be
    stacked. The three trailing per-participant columns are broadcast onto both
    rows via a merge on Participant.
    """
    participants = sub_df[["Participant"]]

    # Normothermic = the "… initial …" columns.
    normo_cols = [c for c in sub_df.columns if "initial" in str(c)]
    normo_df = participants.join(sub_df[normo_cols])
    normo_df.columns = [str(c).replace(" initial", "") for c in normo_df.columns]
    normo_df["Thermal_Stage"] = "Normothermic"

    # Hyperthermic = the "… final …" columns.
    hyper_cols = [c for c in sub_df.columns if "final" in str(c)]
    hyper_df = participants.join(sub_df[hyper_cols])
    hyper_df.columns = [str(c).replace(" final", "") for c in hyper_df.columns]
    hyper_df["Thermal_Stage"] = "Hyperthermic"

    # Per-participant common vars = the last 3 columns of the block.
    common_cols = ["Participant"] + list(sub_df.columns[-3:])
    common_df = sub_df[common_cols]

    merged_thermal = pd.concat([normo_df, hyper_df], axis=0, ignore_index=True)
    final_sub_df = pd.merge(merged_thermal, common_df, on="Participant")
    final_sub_df["Acclimation"] = acclimation_label
    return final_sub_df


def read_physiological_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw workbook and return the tidy long-format physiology frame."""
    df = pd.read_excel(path, header=None)

    headers = df.iloc[HEADER_ROW].tolist()
    headers[0] = "Participant"

    pre_df = df.iloc[PRE_ROWS].copy()
    pre_df.columns = headers

    post_df = df.iloc[POST_ROWS].copy()
    post_df.columns = headers

    pre_processed = process_thermal_stages(pre_df, "Pre")
    post_processed = process_thermal_stages(post_df, "Post")

    final_dataset = pd.concat([pre_processed, post_processed], axis=0, ignore_index=True)

    # Encode the NPX Exposure code (PR1/PT1/PR2/PT2) from thermal stage + acclimation.
    final_dataset["Exposure"] = [
        EXPOSURE_FROM_STAGE[(stage, accl)]
        for stage, accl in zip(final_dataset["Thermal_Stage"], final_dataset["Acclimation"])
    ]

    # Lead with the categorical keys, keep the rest in their original order.
    remaining_cols = [c for c in final_dataset.columns if c not in KEY_COLS]
    final_dataset = final_dataset[KEY_COLS + remaining_cols]

    # Tidy the column labels the way the analysis cells expect.
    final_dataset.columns = final_dataset.columns.str.strip()
    return final_dataset


if __name__ == "__main__":
    dataset = read_physiological_data()
    dataset.to_csv(CLEANED_PATH, index=False)
    print("Shape:", dataset.shape)
    print("\nColumns:")
    print(list(dataset.columns))
    print("\nFirst 5 rows:")
    print(dataset.head())
    print(f"\nSaved cleaned data -> {CLEANED_PATH}")
