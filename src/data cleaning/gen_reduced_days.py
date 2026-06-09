"""
Compute reduced generation per region by subtracting selected nuclear output from total generation.
Merges eco2mix regional data with nuclear unit production; outputs gen_reduced_days.csv to data/final/.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "data" / "final"

UNIT_TO_REGION = {
    "BUGEY-2":     "Auvergne-Rhône-Alpes",
    "BUGEY-3":     "Auvergne-Rhône-Alpes",
    "BUGEY-4":     "Auvergne-Rhône-Alpes",
    "BUGEY-5":     "Auvergne-Rhône-Alpes",
    "TRICASTIN-1": "Auvergne-Rhône-Alpes",
    "TRICASTIN-2": "Auvergne-Rhône-Alpes",
    "DAMPIERRE-1": "Centre-Val de Loire",
    "DAMPIERRE-2": "Centre-Val de Loire",
    "GRAVELINES-1": "Hauts-de-France",
    "GRAVELINES-2": "Hauts-de-France",
    "GRAVELINES-3": "Hauts-de-France",
}

# --- Load nuclear production (hourly, local Paris time) ---
nuc = pd.read_csv(FINAL / "production_nucleaire.csv")
nuc["start_date"] = pd.to_datetime(nuc["start_date"], utc=True).dt.tz_convert("Europe/Paris")
nuc["Date"] = nuc["start_date"].dt.strftime("%Y-%m-%d")
nuc["Heure"] = nuc["start_date"].dt.strftime("%H:%M")
nuc["Région"] = nuc["unit"].map(UNIT_TO_REGION)

nuclear_per_region = (
    nuc.groupby(["Région", "Date", "Heure"])["value_MW"]
    .sum()
    .reset_index()
    .rename(columns={"value_MW": "nuclear_selected_MW"})
)

# --- Load eco2mix (half-hourly), keep row with max consumption per hour ---
eco = pd.read_csv(FINAL / "eco2mix_regional_new_representative_days.csv", sep=";")

num_cols = [c for c in eco.columns if c not in ["Code INSEE région", "Région", "Date", "Heure"]]
eco[num_cols] = eco[num_cols].apply(pd.to_numeric, errors="coerce")
eco["datetime"] = pd.to_datetime(eco["Date"] + " " + eco["Heure"])
eco["hour"] = eco["datetime"].dt.floor("h")

# Within each (region, hour) group, keep the row with highest consumption
idx = eco.groupby(["Code INSEE région", "Région", "Date", "hour"])["Consommation (MW)"].idxmax()
eco_hourly = eco.loc[idx].copy()
eco_hourly["Heure"] = eco_hourly["hour"].dt.strftime("%H:%M")
eco_hourly = eco_hourly.drop(columns=["datetime", "hour"])

# --- Merge and compute gen_reduced ---
result = eco_hourly.merge(nuclear_per_region, on=["Région", "Date", "Heure"], how="left")
result["nuclear_selected_MW"] = result["nuclear_selected_MW"].fillna(0)
result["gen_reduced_MW"] = result["Total gen (MW)"] - result["nuclear_selected_MW"]
result["nuclear_gen_reduced_MW"] = (
    pd.to_numeric(result["Nucléaire (MW)"], errors="coerce") - result["nuclear_selected_MW"]
)

result.to_csv(FINAL / "gen_reduced_days.csv", index=False, sep=";")
print(f"Saved {len(result)} rows to gen_reduced_days.csv")

print("\nNuclear selected by region (sum over all days):")
print(
    result.groupby("Région")[["nuclear_selected_MW", "Total gen (MW)", "gen_reduced_MW"]]
    .sum()
    .round(0)
    .to_string()
)
