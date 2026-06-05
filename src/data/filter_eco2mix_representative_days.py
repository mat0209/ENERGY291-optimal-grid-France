"""
Filter eco2mix-regional-cons-def.csv to keep only rows matching the representative dates
defined in new_representative_days.csv, retaining only MW columns (no TCO/TCH percentages).
Output is written to data/final/.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
FINAL = ROOT / "data" / "final"

KEEP_COLS = [
    "Code INSEE région",
    "Région",
    "Date",
    "Heure",
    "Consommation (MW)",
    "Thermique (MW)",
    "Nucléaire (MW)",
    "Eolien (MW)",
    "Solaire (MW)",
    "Hydraulique (MW)",
    "Pompage (MW)",
    "Bioénergies (MW)",
    "Ech. physiques (MW)",
    "Stockage batterie",
    "Déstockage batterie",
    "Eolien terrestre",
    "Eolien offshore",
]


def main():
    rep_days = pd.read_csv(FINAL / "new_representative_days.csv")
    representative_dates = set(rep_days["representative_date"].astype(str))
    print(f"Filtering for {len(representative_dates)} representative dates: {sorted(representative_dates)}")

    eco = pd.read_csv(
        RAW / "eco2mix-regional-cons-def.csv",
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
    )

    mask = eco["Date"].isin(representative_dates)
    filtered = eco[mask].reset_index(drop=True)
    print(f"Kept {len(filtered):,} rows out of {len(eco):,}")

    cols = [c for c in KEEP_COLS if c in filtered.columns]
    filtered = filtered[cols]

    gen_cols = [
        "Thermique (MW)", "Nucléaire (MW)", "Eolien (MW)", "Solaire (MW)",
        "Hydraulique (MW)", "Pompage (MW)", "Bioénergies (MW)",
        "Stockage batterie", "Déstockage batterie",
    ]
    existing_gen_cols = [c for c in gen_cols if c in filtered.columns]
    filtered["Total gen (MW)"] = (
        filtered[existing_gen_cols]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1)
    )

    out_path = FINAL / "eco2mix_regional_new_representative_days.csv"
    filtered.to_csv(out_path, index=False, sep=";")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
