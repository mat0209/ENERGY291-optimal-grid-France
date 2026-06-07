"""
Filter Renewables.ninja PV/wind data to representative dates, aggregate NUTS2 zones
to 13 French administrative regions, and process regional eco2mix consumption data.
Outputs to data/processed/.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Load the 12 representative dates
rep_days = pd.read_csv(PROCESSED / "representative_days_weighted_2024.csv")
rep_dates = pd.to_datetime(rep_days["representative_date"]).dt.date

# --- Solar PV ---
pv_raw = pd.read_csv(RAW / "ninja-pv-country-FR-nuts2-merra2.csv", skiprows=3)
pv_raw["time"] = pd.to_datetime(pv_raw["time"], utc=True)

mask_pv = pv_raw["time"].dt.date.isin(rep_dates)
pv_filtered = pv_raw[mask_pv].reset_index(drop=True)

pv_filtered.to_csv(PROCESSED / "ninja_pv_representative_days_2024.csv", index=False)
print(f"PV: {len(pv_filtered)} rows saved ({len(pv_filtered) // 24} days x 24h)")

# --- Wind ---
wind_raw = pd.read_csv(RAW / "ninja-wind-country-FR-current_onshore-merra2.csv", skiprows=3)
wind_raw["time"] = pd.to_datetime(wind_raw["time"], utc=True)

mask_wind = wind_raw["time"].dt.date.isin(rep_dates)
wind_filtered = wind_raw[mask_wind].reset_index(drop=True)

wind_filtered.to_csv(PROCESSED / "ninja_wind_representative_days_2024.csv", index=False)
print(f"Wind: {len(wind_filtered)} rows saved ({len(wind_filtered) // 24} days x 24h)")

# --- Harmonise PV and Wind to 13 administrative regions ---
# Mapping: old NUTS2 (PV) → new admin region name
PV_TO_REGION = {
    "FR10": "Île-de-France",
    "FR24": "Centre-Val de Loire",
    "FR26": "Bourgogne-Franche-Comté",  "FR43": "Bourgogne-Franche-Comté",
    "FR23": "Normandie",                 "FR25": "Normandie",
    "FR22": "Hauts-de-France",           "FR30": "Hauts-de-France",
    "FR21": "Grand Est",  "FR41": "Grand Est",  "FR42": "Grand Est",
    "FR51": "Pays de la Loire",
    "FR52": "Bretagne",
    "FR53": "Nouvelle-Aquitaine",  "FR61": "Nouvelle-Aquitaine",  "FR63": "Nouvelle-Aquitaine",
    "FR62": "Occitanie",           "FR81": "Occitanie",
    "FR71": "Auvergne-Rhône-Alpes",  "FR72": "Auvergne-Rhône-Alpes",
    "FR82": "Provence-Alpes-Côte d'Azur",
    # FR83 (Corse) excluded — not in eco2mix
}

# Mapping: new NUTS2 (Wind) → new admin region name
WIND_TO_REGION = {
    "FR10": "Île-de-France",
    "FRB0": "Centre-Val de Loire",
    "FRC1": "Bourgogne-Franche-Comté",  "FRC2": "Bourgogne-Franche-Comté",
    "FRD1": "Normandie",                 "FRD2": "Normandie",
    "FRE1": "Hauts-de-France",           "FRE2": "Hauts-de-France",
    "FRF1": "Grand Est",  "FRF2": "Grand Est",  "FRF3": "Grand Est",
    "FRG0": "Pays de la Loire",
    "FRH0": "Bretagne",
    "FRI1": "Nouvelle-Aquitaine",  "FRI2": "Nouvelle-Aquitaine",  "FRI3": "Nouvelle-Aquitaine",
    "FRJ1": "Occitanie",           "FRJ2": "Occitanie",
    "FRK1": "Auvergne-Rhône-Alpes",  "FRK2": "Auvergne-Rhône-Alpes",
    "FRL0": "Provence-Alpes-Côte d'Azur",
    # FRM0 (Corse) excluded
}

def aggregate_to_regions(df, zone_to_region):
    time_col = df["time"]
    cf_cols = {col: region for col, region in zone_to_region.items() if col in df.columns}
    records = {"time": time_col}
    for region in sorted(set(cf_cols.values())):
        zones = [col for col, reg in cf_cols.items() if reg == region]
        # Equal-weight mean across sub-zones (no sub-region capacity data available)
        records[region] = df[zones].mean(axis=1)
    return pd.DataFrame(records)

pv_regions = aggregate_to_regions(pv_filtered, PV_TO_REGION)
pv_regions.to_csv(PROCESSED / "ninja_pv_representative_days_2024_regions.csv", index=False)
print(f"PV (13 regions): {len(pv_regions)} rows, {len(pv_regions.columns)-1} regions")

wind_regions = aggregate_to_regions(wind_filtered, WIND_TO_REGION)
wind_regions.to_csv(PROCESSED / "ninja_wind_representative_days_2024_regions.csv", index=False)
print(f"Wind (13 regions): {len(wind_regions)} rows, {len(wind_regions.columns)-1} regions")

# --- Regional Eco2mix ---
eco_raw = pd.read_csv(
    RAW / "eco2mix-regional-cons-def.csv",
    sep=";",
    encoding="utf-8-sig",
    low_memory=False,
)

# Construct datetime and filter on representative days
eco_raw["datetime"] = pd.to_datetime(eco_raw["Date"] + " " + eco_raw["Heure"], dayfirst=False)
mask_eco = eco_raw["datetime"].dt.date.isin(rep_dates)
eco_filtered = eco_raw[mask_eco].copy()

# Resample to hourly by averaging the two half-hours (XX:00 and XX:30)
drop_cols = ["Heure", "Date - Heure", "datetime", "Date"]
num_cols = [c for c in eco_filtered.columns if c not in ["Code INSEE région", "Région", "Nature"] + drop_cols]

eco_filtered["hour"] = eco_filtered["datetime"].dt.floor("h")

eco_hourly = (
    eco_filtered
    .groupby(["Code INSEE région", "Région", "Nature", "hour"])
    [num_cols]
    .mean()
    .reset_index()
)

eco_hourly["Date"] = eco_hourly["hour"].dt.strftime("%Y-%m-%d")
eco_hourly["Heure"] = eco_hourly["hour"].dt.strftime("%H:%M")
eco_hourly = eco_hourly.drop(columns=["hour"])

eco_hourly.to_csv(PROCESSED / "eco2mix_regional_representative_days_2024.csv", index=False, sep=";")
n_regions = eco_hourly["Région"].nunique()
print(f"Eco2mix: {len(eco_hourly)} rows saved (12 days x 24h x {n_regions} regions)")
