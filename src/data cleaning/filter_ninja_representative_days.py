"""
Filter full-year Renewables.ninja PV and wind files to keep only rows matching
the new representative dates, aggregate NUTS2 zones to 13 administrative regions,
and write results to data/final/.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
FINAL = ROOT / "data" / "final"

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
    # FR83 (Corse) excluded
}

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
    cf_cols = {col: region for col, region in zone_to_region.items() if col in df.columns}
    records = {"time": df["time"]}
    for region in sorted(set(cf_cols.values())):
        zones = [col for col, reg in cf_cols.items() if reg == region]
        records[region] = df[zones].mean(axis=1)
    return pd.DataFrame(records)


def main():
    rep_days = pd.read_csv(FINAL / "new_representative_days.csv")
    rep_dates = set(pd.to_datetime(rep_days["representative_date"]).dt.date)
    print(f"Filtering for {len(rep_dates)} dates: {sorted(str(d) for d in rep_dates)}")

    # --- Solar PV ---
    pv_raw = pd.read_csv(RAW / "ninja-pv-country-FR-nuts2-merra2.csv", skiprows=3)
    pv_raw["time"] = pd.to_datetime(pv_raw["time"], utc=True)
    pv_filtered = pv_raw[pv_raw["time"].dt.date.isin(rep_dates)].reset_index(drop=True)
    pv_regions = aggregate_to_regions(pv_filtered, PV_TO_REGION)
    out_pv = FINAL / "ninja_pv_new_representative_days_regions.csv"
    pv_regions.to_csv(out_pv, index=False)
    print(f"PV: {len(pv_regions)} rows ({len(pv_regions) // 24} days x 24h) -> {out_pv}")

    # --- Wind ---
    wind_raw = pd.read_csv(RAW / "ninja-wind-country-FR-current_onshore-merra2.csv", skiprows=3)
    wind_raw["time"] = pd.to_datetime(wind_raw["time"], utc=True)
    wind_filtered = wind_raw[wind_raw["time"].dt.date.isin(rep_dates)].reset_index(drop=True)
    wind_regions = aggregate_to_regions(wind_filtered, WIND_TO_REGION)
    out_wind = FINAL / "ninja_wind_new_representative_days_regions.csv"
    wind_regions.to_csv(out_wind, index=False)
    print(f"Wind: {len(wind_regions)} rows ({len(wind_regions) // 24} days x 24h) -> {out_wind}")


if __name__ == "__main__":
    main()
