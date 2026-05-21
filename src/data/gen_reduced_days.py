import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

# Mapping: nuclear unit -> region name in eco2mix
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

# --- Load selected nuclear production ---
nuc = pd.read_csv(PROCESSED / "production_nucleaire.csv")
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

# --- Load eco2mix regional (representative days, hourly) ---
eco = pd.read_csv(PROCESSED / "eco2mix_regional_representative_days_2024.csv", sep=";")

# Total real generation = sum of all production sources
gen_cols = ["Thermique (MW)", "Nucléaire (MW)", "Eolien terrestre", "Eolien offshore",
            "Solaire (MW)", "Hydraulique (MW)", "Pompage (MW)", "Bioénergies (MW)"]
eco["real_gen_MW"] = eco[gen_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

# --- Merge and compute gen_reduced ---
result = eco.merge(nuclear_per_region, on=["Région", "Date", "Heure"], how="left")
result["nuclear_selected_MW"] = result["nuclear_selected_MW"].fillna(0)
result["gen_reduced_MW"] = result["real_gen_MW"] - result["nuclear_selected_MW"]
result["nuclear_gen_reduced_MW"] = pd.to_numeric(result["Nucléaire (MW)"], errors="coerce") - result["nuclear_selected_MW"]

result.to_csv(PROCESSED / "gen_reduced_days_2024.csv", index=False, sep=";")
print(f"Saved {len(result)} rows to gen_reduced_days_2024.csv")

# Quick check
print("\nNuclear selected by region (sum over all days):")
print(
    result.groupby("Région")[["nuclear_selected_MW", "real_gen_MW", "gen_reduced_MW"]]
    .sum()
    .round(0)
    .to_string()
)
