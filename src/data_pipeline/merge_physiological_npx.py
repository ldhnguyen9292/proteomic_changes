"""Merge the tidy physiological data with the Olink NPX panels.

Join key: Participant + Exposure code (PR1/PT1/PR2/PT2).

The 8 NPX panels are combined side-by-side into one wide protein table
(one row per sample), then joined to the physiology frame. A few Olink bridge
assays appear on more than one panel (e.g. IL6, TNF, IDO1); those collided
columns are suffixed with the panel name so no measurement is overwritten.

Output: Physiological_NPX_Merged.csv (one row per Participant x Exposure).
"""

from collections import Counter
from pathlib import Path

import pandas as pd

from read_physiological_data import DATA_DIR, read_physiological_data

NPX_DIR = DATA_DIR / "raw_data" / "npx"
MERGED_PATH = DATA_DIR / "Physiological_NPX_Merged.csv"

KEYS = ["Participant", "Exposure"]

# Panel display name -> filename (matches the notebook's npx_files naming).
PANEL_FILES = {
    "Cardiometabolic": "cardiometabolic_npx.csv",
    "Cardiometabolic_II": "cardiometabolic_ii_npx.csv",
    "Inflammation": "inflammation_npx.csv",
    "Inflammation_II": "inflammation_ii_npx.csv",
    "Neurology": "neurology_npx.csv",
    "Neurology_II": "neurology_ii_npx.csv",
    "Oncology": "oncology_npx.csv",
    "Oncology_II": "oncology_ii_npx.csv",
}


def load_npx_panel(path: Path) -> pd.DataFrame:
    """Load one NPX panel keyed by Participant+Exposure (controls dropped)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Drop CONTROL_SAMPLE_* rows, which have no Participant/Exposure.
    df = df.dropna(subset=KEYS)
    df = df.drop(columns=["SampleID"], errors="ignore")
    return df


def load_combined_npx() -> pd.DataFrame:
    """Combine all panels side-by-side into one wide protein table."""
    panels = {name: load_npx_panel(NPX_DIR / fn) for name, fn in PANEL_FILES.items()}

    # Find protein names that appear on more than one panel.
    counts = Counter(
        col for df in panels.values() for col in df.columns if col not in KEYS
    )
    collided = {p for p, n in counts.items() if n > 1}
    if collided:
        print(f"Bridge proteins on multiple panels (suffixed by panel): {sorted(collided)}")

    combined = None
    for name, df in panels.items():
        # Disambiguate only the collided proteins; leave unique ones bare.
        df = df.rename(columns={c: f"{c}__{name}" for c in df.columns if c in collided})
        df = df.set_index(KEYS)
        combined = df if combined is None else combined.join(df, how="outer")

    return combined.reset_index()


def merge_physiology_npx() -> pd.DataFrame:
    """Join physiology and NPX on Participant + Exposure."""
    physio = read_physiological_data()
    # Normalize IDs: physiology uses "001-B", NPX uses "001B".
    physio["Participant"] = physio["Participant"].str.replace("-", "", regex=False)

    npx = load_combined_npx()

    merged = physio.merge(npx, on=KEYS, how="inner", validate="one_to_one")

    # Report any physiology samples that failed to match an NPX sample.
    unmatched = len(physio) - len(merged)
    if unmatched:
        got = set(map(tuple, merged[KEYS].values))
        missing = [k for k in map(tuple, physio[KEYS].values) if k not in got]
        print(f"WARNING: {unmatched} physiology sample(s) had no NPX match: {missing}")

    return merged


if __name__ == "__main__":
    merged = merge_physiology_npx()
    merged.to_csv(MERGED_PATH, index=False)
    n_protein = merged.shape[1] - 18  # 18 physiology columns (3+1 keys + 14 vars)
    print(f"\nMerged shape: {merged.shape}  (~{n_protein} protein columns)")
    print("Samples per exposure:")
    print(merged["Exposure"].value_counts().sort_index().to_string())
    print(f"\nSaved merged data -> {MERGED_PATH}")
