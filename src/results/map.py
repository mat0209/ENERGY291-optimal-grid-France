"""
Map optimal investment results and nuclear decommissioning sites over metropolitan France.
Reads capacity results from data/results/; outputs HTML/PNG to figures/results/.
"""

import json
import os
import pandas as pd
import plotly.graph_objects as go
import requests

COMMUNES_PATH  = "data/raw/communes-france-2025.csv"
CORRIDORS_PATH = "data/final/capacites_interregionales.csv"
CAPACITY_PATH  = "data/results/capacity_results_demand_growth_8pct.csv"
GEOJSON_CACHE  = "data/raw/regions-france.geojson"
GEOJSON_URL    = (
    "https://raw.githubusercontent.com/gregoiredavid/"
    "france-geojson/master/regions-version-simplifiee.geojson"
)
OUTPUT_HTML    = "figures/results/results_map.html"
OUTPUT_PNG     = "figures/results/results_map.png"

SCENARIO_LABEL = "Scenario: +8% demand growth, constrained transmission"
MAINLAND_CODES = {11, 24, 27, 28, 32, 44, 52, 53, 75, 76, 84, 93}

# ── 1. Region centroids ───────────────────────────────────────────────────────
communes = pd.read_csv(COMMUNES_PATH, index_col=0)
centroids = (
    communes[communes["reg_code"].isin(MAINLAND_CODES)]
    .groupby(["reg_code", "reg_nom"])[["latitude_centre", "longitude_centre"]]
    .mean()
    .reset_index()
    .set_index("reg_nom")
)

ABBREV = {
    "Auvergne-Rhone-Alpes":       "ARA",
    "Auvergne-Rhône-Alpes":       "ARA",
    "Bourgogne-Franche-Comte":    "BFC",
    "Bourgogne-Franche-Comté":    "BFC",
    "Bretagne":                   "BRE",
    "Centre-Val de Loire":        "CVL",
    "Grand Est":                  "GES",
    "Hauts-de-France":            "HDF",
    "Ile-de-France":              "IDF",
    "Île-de-France":              "IDF",
    "Normandie":                  "NOR",
    "Nouvelle-Aquitaine":         "NAQ",
    "Occitanie":                  "OCC",
    "Pays de la Loire":           "PDL",
    "Provence-Alpes-Cote d'Azur": "PAC",
    "Provence-Alpes-Côte d'Azur": "PAC",
}

# ── 2. Nuclear reactor sites decommissioned (>= 50-year lifetime) ─────────────
# Capacities: net MW per unit × number of units (IAEA PRIS approximations)
REACTOR_SITES = [
    {
        "name": "Gravelines",
        "units": "GRAVELINES 1–3",
        "n_units": 3,
        "lat": 51.008, "lon": 2.128,
        "region": "Hauts-de-France",
        "cap_GW": 2.7,   # 3 × 910 MW
    },
    {
        "name": "Dampierre",
        "units": "DAMPIERRE 1–2",
        "n_units": 2,
        "lat": 47.728, "lon": 2.519,
        "region": "Centre-Val de Loire",
        "cap_GW": 1.8,   # 2 × 890 MW
    },
    {
        "name": "Bugey",
        "units": "BUGEY 2–5",
        "n_units": 4,
        "lat": 45.796, "lon": 5.271,
        "region": "Auvergne-Rhone-Alpes",
        "cap_GW": 3.7,   # 4 × 920 MW
    },
    {
        "name": "Tricastin",
        "units": "TRICASTIN 1–2",
        "n_units": 2,
        "lat": 44.331, "lon": 4.732,
        "region": "Auvergne-Rhone-Alpes",
        "cap_GW": 1.8,   # 2 × 915 MW
    },
]

# ── 3. Capacity results ───────────────────────────────────────────────────────
cap = pd.read_csv(CAPACITY_PATH, sep=";").set_index("Région")
cap["Battery_MWh"] = cap["Battery_MWh"].clip(lower=0)

MARKER_SIZE = 25   # fixed size for all wind / battery markers

# ── 4. Load regions GeoJSON (download once, cache locally) ────────────────────
def _load_geojson():
    if os.path.exists(GEOJSON_CACHE):
        with open(GEOJSON_CACHE, encoding="utf-8") as f:
            return json.load(f)
    try:
        resp = requests.get(GEOJSON_URL, timeout=15)
        resp.raise_for_status()
        geo = resp.json()
        with open(GEOJSON_CACHE, "w", encoding="utf-8") as f:
            json.dump(geo, f)
        print(f"GeoJSON downloaded and cached -> {GEOJSON_CACHE}")
        return geo
    except Exception as exc:
        print(f"Warning: could not load regions GeoJSON ({exc}). Borders skipped.")
        return None

regions_geo = _load_geojson()

# ── 5. Build figure ───────────────────────────────────────────────────────────
fig = go.Figure()

# ── Region borders (dashed, light gray) from GeoJSON ─────────────────────────
if regions_geo:
    border_lons, border_lats = [], []
    for feature in regions_geo["features"]:
        try:
            code = int(feature["properties"]["code"])
        except (KeyError, ValueError):
            continue
        if code not in MAINLAND_CODES:
            continue
        geom = feature["geometry"]
        polys = (
            [geom["coordinates"]] if geom["type"] == "Polygon"
            else geom["coordinates"]
        )
        for poly in polys:
            for ring in poly:
                border_lons += [pt[0] for pt in ring] + [ring[0][0], None]
                border_lats += [pt[1] for pt in ring] + [ring[0][1], None]

    fig.add_trace(go.Scattergeo(
        lon=border_lons, lat=border_lats,
        mode="lines",
        line=dict(color="#bbbbbb", width=0.8, dash="dot"),
        hoverinfo="skip",
        showlegend=False,
    ))

# ── Transmission corridors (faint gray background) ────────────────────────────
corridors = pd.read_csv(CORRIDORS_PATH)
for _, row in corridors.iterrows():
    ra, rb = row["Region_A"], row["Region_B"]
    if ra not in centroids.index or rb not in centroids.index:
        continue
    fig.add_trace(go.Scattergeo(
        lon=[centroids.loc[ra, "longitude_centre"],
             centroids.loc[rb, "longitude_centre"], None],
        lat=[centroids.loc[ra, "latitude_centre"],
             centroids.loc[rb, "latitude_centre"], None],
        mode="lines",
        line=dict(width=1.0, color="#d4d4d4"),
        hoverinfo="skip",
        showlegend=False,
    ))

# ── Region nodes ──────────────────────────────────────────────────────────────
fig.add_trace(go.Scattergeo(
    lon=centroids["longitude_centre"].values,
    lat=centroids["latitude_centre"].values,
    mode="markers+text",
    marker=dict(size=7, color="white", line=dict(color="#aaaaaa", width=1.2)),
    text=[ABBREV.get(r, r) for r in centroids.index],
    textposition="top center",
    textfont=dict(size=9, color="#999999", family="Arial"),
    hoverinfo="skip",
    showlegend=False,
))

# ── Wind capacity  (y-up = turbine viewed from above) ────────────────────────
wind_regions = cap[cap["Wind_MW"] > 0]
if not wind_regions.empty:
    w_lons, w_lats, w_sizes, w_hover = [], [], [], []
    for region, row in wind_regions.iterrows():
        if region not in centroids.index:
            continue
        w_lons.append(centroids.loc[region, "longitude_centre"] + 0.65)
        w_lats.append(centroids.loc[region, "latitude_centre"]  - 0.30)
        w_sizes.append(MARKER_SIZE)
        w_hover.append(
            f"<b>Wind — {ABBREV.get(region, region)}</b><br>"
            f"New capacity: {row['Wind_MW']:,.0f} MW"
        )
    fig.add_trace(go.Scattergeo(
        lon=w_lons, lat=w_lats,
        mode="markers",
        marker=dict(
            symbol="y-up",
            size=w_sizes,
            color="#1976b0",
            opacity=0.90,
            line=dict(color="#0d4870", width=2),
        ),
        hoverinfo="text",
        hovertext=w_hover,
        name="Wind (new capacity)",
        showlegend=True,
    ))

# ── Battery capacity  (filled square) ────────────────────────────────────────
bat_regions = cap[cap["Battery_MWh"] > 0]
if not bat_regions.empty:
    b_lons, b_lats, b_sizes, b_hover = [], [], [], []
    for region, row in bat_regions.iterrows():
        if region not in centroids.index:
            continue
        b_lons.append(centroids.loc[region, "longitude_centre"] - 0.65)
        b_lats.append(centroids.loc[region, "latitude_centre"]  - 0.30)
        b_sizes.append(MARKER_SIZE)
        b_hover.append(
            f"<b>Battery — {ABBREV.get(region, region)}</b><br>"
            f"New capacity: {row['Battery_MWh']:,.0f} MWh"
        )
    fig.add_trace(go.Scattergeo(
        lon=b_lons, lat=b_lats,
        mode="markers",
        marker=dict(
            symbol="square",
            size=b_sizes,
            color="#e07b27",
            opacity=0.90,
            line=dict(color="#9c5510", width=2),
        ),
        hoverinfo="text",
        hovertext=b_hover,
        name="Battery (new capacity)",
        showlegend=True,
    ))

# ── Capacity labels below each marker ────────────────────────────────────────
for region, row in cap.iterrows():
    if region not in centroids.index:
        continue
    lat0 = centroids.loc[region, "latitude_centre"]
    lon0 = centroids.loc[region, "longitude_centre"]
    pairs = []
    if row["Wind_MW"] > 0:
        pairs.append((lon0 + 0.65, lat0 - 0.65, f"<b>{row['Wind_MW']/1000:.1f} GW</b>"))
    if row["Battery_MWh"] > 0:
        pairs.append((lon0 - 0.65, lat0 - 0.65, f"<b>{row['Battery_MWh']/1000:.0f} GWh</b>"))
    for alon, alat, label in pairs:
        fig.add_trace(go.Scattergeo(
            lon=[alon], lat=[alat],
            mode="text",
            text=[label],
            textfont=dict(size=9, color="#2c3e50", family="Arial"),
            hoverinfo="skip",
            showlegend=False,
        ))

# ── Decommissioned reactor sites (red X) ─────────────────────────────────────
fig.add_trace(go.Scattergeo(
    lon=[r["lon"] for r in REACTOR_SITES],
    lat=[r["lat"] for r in REACTOR_SITES],
    mode="markers+text",
    marker=dict(
        symbol="x",
        size=18,
        color="#c0392b",
        opacity=0.93,
        line=dict(color="#7b241c", width=3),
    ),
    text=[f"<b>{r['name']}</b><br><b>({r['cap_GW']:.1f} GW)</b>" for r in REACTOR_SITES],
    textposition="bottom center",
    textfont=dict(size=10, color="#c0392b", family="Arial Bold"),
    hoverinfo="text",
    hovertext=[
        f"<b>Decommissioned: {r['name']}</b><br>"
        f"Units: {r['units']}<br>"
        f"Region: {r['region']}<br>"
        f"Total capacity: {r['cap_GW']:.1f} GW"
        for r in REACTOR_SITES
    ],
    name="Decommissioned reactors",
    showlegend=True,
))

# ── Regions annotation (bottom-left) ─────────────────────────────────────────
unique_abbrev = {}
for name, abbr in ABBREV.items():
    if abbr not in unique_abbrev:
        unique_abbrev[abbr] = name
region_lines = "<br>".join(
    f"<b>{abbr}</b>  {name}"
    for abbr, name in sorted(unique_abbrev.items())
)

# ── Layout ────────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=(
            "<b>Optimal Investments and Nuclear Decommissioning"
            " — Metropolitan France</b><br>"
            f"<sup>{SCENARIO_LABEL} &nbsp;|&nbsp;"
            f" Total decommissioned: ~10.0 GW</sup>"
        ),
        x=0.5,
        font=dict(size=17),
    ),
    showlegend=True,
    legend=dict(
        x=0.815, y=0.98,
        xanchor="right", yanchor="top",
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#cccccc",
        borderwidth=1,
        font=dict(size=11),
        itemsizing="constant",
        grouptitlefont=dict(size=12, color="#2c3e50"),
    ),
    geo=dict(
        scope="europe",
        resolution=50,
        showland=True,       landcolor="#f8f8f2",
        showocean=True,      oceancolor="#e8f4f8",
        showcoastlines=True, coastlinecolor="#aaaaaa",
        showcountries=True,  countrycolor="#cccccc",
        lonaxis=dict(range=[-5.2, 8.3]),
        lataxis=dict(range=[42.2, 51.8]),
        bgcolor="white",
    ),
    paper_bgcolor="white",
    margin=dict(l=0, r=0, t=70, b=0),
    width=960,
    height=720,
    annotations=[
        dict(
            x=0.355, y=0.3,
            xanchor="right", yanchor="top",
            xref="paper", yref="paper",
            align="left",
            text="<b>Regions</b><br>" + region_lines,
            showarrow=False,
            font=dict(size=10, family="Arial"),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=7,
        )
    ],
)

os.makedirs("figures/results", exist_ok=True)
fig.write_html(OUTPUT_HTML)
print(f"Saved -> {OUTPUT_HTML}")

try:
    fig.write_image(OUTPUT_PNG, scale=2)
    print(f"Saved -> {OUTPUT_PNG}")
except Exception:
    print("PNG skipped -- install kaleido: pip install kaleido")
