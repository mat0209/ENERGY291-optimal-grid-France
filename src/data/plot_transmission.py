import pandas as pd
import plotly.graph_objects as go
import os

CORRIDORS_PATH = "data/final/capacites_interregionales.csv"
COMMUNES_PATH  = "data/raw/communes-france-2025.csv"
OUTPUT_HTML    = "figures/transmission/transmission_network.html"
OUTPUT_PNG     = "figures/transmission/transmission_network.png"

# ── 1. Region centroids (computed from communes) ──────────────────────────────
communes = pd.read_csv(COMMUNES_PATH, index_col=0)
MAINLAND_CODES = {11, 24, 27, 28, 32, 44, 52, 53, 75, 76, 84, 93}
centroids = (
    communes[communes["reg_code"].isin(MAINLAND_CODES)]
    .groupby(["reg_code", "reg_nom"])[["latitude_centre", "longitude_centre"]]
    .mean()
    .reset_index()
    .set_index("reg_nom")
)

ABBREV = {
    "Auvergne-Rhône-Alpes":          "ARA",
    "Bourgogne-Franche-Comté":        "BFC",
    "Bretagne":                       "BRE",
    "Centre-Val de Loire":            "CVL",
    "Grand Est":                      "GES",
    "Hauts-de-France":                "HDF",
    "Île-de-France":                  "IDF",
    "Normandie":                      "NOR",
    "Nouvelle-Aquitaine":             "NAQ",
    "Occitanie":                      "OCC",
    "Pays de la Loire":               "PDL",
    "Provence-Alpes-Côte d'Azur":     "PAC",
}

# ── 2. Load corridors ─────────────────────────────────────────────────────────
df = pd.read_csv(CORRIDORS_PATH)
cap_max = df["Capacite_MW_total"].max()

def _edge_width(cap):
    return 1 + 9 * (cap / cap_max)

CAP_BINS = [
    ("< 15 GW",     0,      15_000),
    ("15 – 35 GW",  15_000, 35_000),
    ("> 35 GW",     35_000, float("inf")),
]

# ── 3. Build figure ───────────────────────────────────────────────────────────
fig = go.Figure()

# ── Legend traces: capacity bins (native Plotly legend → correct line widths) ─
for label, lo, hi in CAP_BINS:
    mid = (lo + min(hi, cap_max)) / 2
    fig.add_trace(go.Scattergeo(
        lon=[None], lat=[None],
        mode="lines",
        line=dict(width=_edge_width(mid), color="#2980b9"),
        opacity=0.75,
        name=label,
        showlegend=True,
        legendgroup="capacity",
        legendgrouptitle=dict(text="Transfer capacity") if lo == 0 else None,
    ))

fig.add_trace(go.Scattergeo(
    lon=[None], lat=[None],
    mode="markers",
    marker=dict(size=12, color="white", line=dict(color="#2980b9", width=2.5)),
    name="Region node",
    showlegend=True,
    legendgroup="capacity",
))

# ── Edges ─────────────────────────────────────────────────────────────────────
for _, row in df.iterrows():
    ra, rb = row["Region_A"], row["Region_B"]
    if ra not in centroids.index or rb not in centroids.index:
        continue
    lat_a = centroids.loc[ra, "latitude_centre"]
    lon_a = centroids.loc[ra, "longitude_centre"]
    lat_b = centroids.loc[rb, "latitude_centre"]
    lon_b = centroids.loc[rb, "longitude_centre"]

    cap = row["Capacite_MW_total"]

    fig.add_trace(go.Scattergeo(
        lon=[lon_a, lon_b, None],
        lat=[lat_a, lat_b, None],
        mode="lines",
        line=dict(width=_edge_width(cap), color="#2980b9"),
        opacity=0.65,
        hoverinfo="text",
        hovertext=(
            f"<b>{ABBREV.get(ra, ra)} ↔ {ABBREV.get(rb, rb)}</b><br>"
            f"Total capacity: {cap:,.0f} MW<br>"
            f"400 kV: {row['Capacite_400kV']:,.0f} MW &nbsp;|&nbsp; "
            f"225 kV: {row['Capacite_225kV']:,.0f} MW<br>"
            f"Segments: {int(row['Nb_troncons'])}"
        ),
        showlegend=False,
    ))

# ── Nodes ─────────────────────────────────────────────────────────────────────
fig.add_trace(go.Scattergeo(
    lon=centroids["longitude_centre"].values,
    lat=centroids["latitude_centre"].values,
    mode="markers+text",
    marker=dict(size=14, color="white", line=dict(color="#2980b9", width=2.5)),
    text=[ABBREV.get(r, r) for r in centroids.index],
    textposition="top center",
    textfont=dict(size=11, color="#2c3e50", family="Arial Black"),
    hoverinfo="text",
    hovertext=list(centroids.index),
    showlegend=False,
))

# ── Region annotation (right side, below legend) ─────────────────────────────
region_lines = "<br>".join(
    f"<b>{abbr}</b>  {name}"
    for name, abbr in sorted(ABBREV.items(), key=lambda x: x[1])
)

fig.update_layout(
    title=dict(
        text=(
            "<b>Inter-regional transmission network — Metropolitan France</b><br>"
            "<sup>Aggregated thermal capacities (RTE data) — "
            "edge width proportional to total transfer capacity</sup>"
        ),
        x=0.5,
        font=dict(size=19),
    ),
    showlegend=True,
    legend=dict(
        x=0.845, y=0.98,
        xanchor="right", yanchor="top",
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#cccccc",
        borderwidth=1,
        font=dict(size=11),
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
        lataxis=dict(range=[42.2, 51.3]),
        bgcolor="white",
    ),
    paper_bgcolor="white",
    margin=dict(l=0, r=0, t=60, b=0),
    width=920,
    height=700,
    annotations=[
        dict(
            x=0.355, y=0.33,
            xanchor="right", yanchor="top",
            xref="paper", yref="paper",
            align="left",
            text="<b>Regions</b><br>" + region_lines,
            showarrow=False,
            font=dict(size=11, family="Arial"),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=7,
        )
    ],
)

os.makedirs("figures/transmission", exist_ok=True)
fig.write_html(OUTPUT_HTML)
print(f"Saved -> {OUTPUT_HTML}")

try:
    fig.write_image(OUTPUT_PNG, scale=2)
    print(f"Saved -> {OUTPUT_PNG}")
except Exception:
    print("PNG skipped — install kaleido: pip install kaleido")
