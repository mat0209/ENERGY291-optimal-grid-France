import pandas as pd
import numpy as np

TRANSMISSION_IN = "data/final/lignes_transmission.csv"
COMMUNES_PATH   = "data/raw/communes-france-2025.csv"
OUTPUT_PATH     = "data/final/capacites_interregionales.csv"

# ── Region centroids ───────────────────────────────────────────────────────────
MAINLAND_CODES = {11, 24, 27, 28, 32, 44, 52, 53, 75, 76, 84, 93}
communes = pd.read_csv(COMMUNES_PATH, index_col=0)
centroids = (
    communes[communes["reg_code"].isin(MAINLAND_CODES)]
    .groupby("reg_nom")[["latitude_centre", "longitude_centre"]]
    .mean()
)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

df = pd.read_csv(TRANSMISSION_IN)

# Keep only tronçons where both endpoints are assigned to a mainland French region
EXCLUDED = {"Étranger", "Corse"}
both_assigned = (
    df["Reg_Depart"].notna()  & df["Reg_Arrivee"].notna() &
    (~df["Reg_Depart"].isin(EXCLUDED)) &
    (~df["Reg_Arrivee"].isin(EXCLUDED)) &
    (df["Reg_Depart_Code"] != df["Reg_Arrivee_Code"])
)
cross = df[both_assigned].copy()

# Canonicalize direction: sort region pair alphabetically so A-B == B-A
def _ordered(row):
    a, ca = row["Reg_Depart"],  int(row["Reg_Depart_Code"])
    b, cb = row["Reg_Arrivee"], int(row["Reg_Arrivee_Code"])
    if a <= b:
        return pd.Series({"Region_A": a, "Code_A": ca, "Region_B": b, "Code_B": cb})
    return pd.Series({"Region_A": b, "Code_A": cb, "Region_B": a, "Code_B": ca})

cross[["Region_A", "Code_A", "Region_B", "Code_B"]] = cross.apply(_ordered, axis=1)

# Aggregate
agg = (
    cross
    .groupby(["Region_A", "Code_A", "Region_B", "Code_B"])
    .agg(
        Capacite_MW_total=("Capacite_MW", "sum"),
        Nb_troncons=("Capacite_MW", "count"),
        Capacite_400kV=("Capacite_MW", lambda s: s[cross.loc[s.index, "Tension"] == "400kV"].sum()),
        Capacite_225kV=("Capacite_MW", lambda s: s[cross.loc[s.index, "Tension"] == "225kV"].sum()),
    )
    .reset_index()
    .sort_values("Capacite_MW_total", ascending=False)
)

# ── Filter: keep adjacent pairs + non-adjacent backbone lines (≥6 GW, 400kV) ──
ADJACENT = {
    (11,24),(11,27),(11,28),(11,32),(11,44),
    (24,27),(24,28),(24,52),(24,75),(24,84),
    (27,44),(27,84),
    (28,32),(28,52),(28,53),
    (32,44),
    (52,53),(52,75),
    (75,76),(75,84),
    (76,84),(76,93),
    (84,27),(84,93),
}
ADJACENT = {(a, b) for a, b in ADJACENT} | {(b, a) for a, b in ADJACENT}

def _keep(row):
    pair = (int(row["Code_A"]), int(row["Code_B"]))
    if pair in ADJACENT:
        return True
    # Non-adjacent backbone: large capacity with significant 400 kV share
    return row["Capacite_MW_total"] >= 6000 and row["Capacite_400kV"] > 0

agg = agg[agg.apply(_keep, axis=1)].reset_index(drop=True)

# ── Add centroid-to-centroid distance ─────────────────────────────────────────
def _distance(row):
    try:
        a = centroids.loc[row["Region_A"]]
        b = centroids.loc[row["Region_B"]]
        return round(haversine_km(a["latitude_centre"], a["longitude_centre"],
                                  b["latitude_centre"], b["longitude_centre"]), 1)
    except KeyError:
        return None

TORTUOSITY = 1.25   # detour factor from Hagspiel et al. (2014) / Fürsch et al. (2013)

agg["Distance_km"]          = agg.apply(_distance, axis=1)
agg["Distance_km_adjusted"] = (agg["Distance_km"] * TORTUOSITY).round(1)

agg.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(agg)} corridors interrégionaux -> {OUTPUT_PATH}")
print(f"Capacité totale de transit : {agg['Capacite_MW_total'].sum():,.0f} MW")
print()
print("Top 20 corridors :")
print(agg[["Region_A", "Region_B", "Capacite_MW_total", "Nb_troncons"]].head(20).to_string(index=False))
