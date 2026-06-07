"""
Four-scenario comparison: ±Gas × ±Flamanville 3
Positive bars  = new capacity to build (solar / wind / 4h battery)
Negative bars  = supply removed from baseline (decommissioned nuclear / gas)  [hatched]
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT         = Path(__file__).resolve().parents[2]
DATA_RESULTS = ROOT / "data" / "results"
DATA_FINAL   = ROOT / "data" / "final"
FIG_DIR      = ROOT / "figures" / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ABBREV = {
    "Auvergne-Rhône-Alpes":       "ARA",
    "Bourgogne-Franche-Comté":    "BFC",
    "Bretagne":                   "BRE",
    "Centre-Val de Loire":        "CVL",
    "Grand Est":                  "GES",
    "Hauts-de-France":            "HDF",
    "Île-de-France":              "IDF",
    "Normandie":                  "NOR",
    "Nouvelle-Aquitaine":         "NAQ",
    "Occitanie":                  "OCC",
    "Pays de la Loire":           "PDL",
    "Provence-Alpes-Côte d'Azur": "PAC",
}

UNIT_TO_REGION = {
    "BUGEY-2":      "Auvergne-Rhône-Alpes",
    "BUGEY-3":      "Auvergne-Rhône-Alpes",
    "BUGEY-4":      "Auvergne-Rhône-Alpes",
    "BUGEY-5":      "Auvergne-Rhône-Alpes",
    "TRICASTIN-1":  "Auvergne-Rhône-Alpes",
    "TRICASTIN-2":  "Auvergne-Rhône-Alpes",
    "DAMPIERRE-1":  "Centre-Val de Loire",
    "DAMPIERRE-2":  "Centre-Val de Loire",
    "GRAVELINES-1": "Hauts-de-France",
    "GRAVELINES-2": "Hauts-de-France",
    "GRAVELINES-3": "Hauts-de-France",
}

C_SOLAR   = "#e8892b"
C_WIND    = "#3e9a5f"
C_BAT     = "#6a5aad"
C_FLA3_F  = "#ff8080"   # solid red for FLA3
C_FLA3_E  = "#cc0000"
C_NUC     = "#c0392b"   # decommissioned nuclear (hatched)
C_GAS     = "#7f8c8d"   # gas removed (hatched)
THRESHOLD_MW = 50

# =============================================================================
# Base data
# =============================================================================
gen_df = pd.read_csv(DATA_FINAL / "gen_reduced_days.csv", sep=";", low_memory=False)
gen_df["Thermique (MW)"] = pd.to_numeric(gen_df["Thermique (MW)"], errors="coerce")
regions = sorted(gen_df["Région"].dropna().unique())
labels  = [ABBREV.get(r, r) for r in regions]
x       = np.arange(len(regions))
NOR_IDX = labels.index("NOR")
BAR_W   = 0.72

# =============================================================================
# Removed supply per region (GW)
# =============================================================================
# Nuclear: nameplate capacity per unit (RTE/EDF official, in MW)
NUC_NAMEPLATE = {
    "BUGEY-2":      945,  "BUGEY-3":      945,
    "BUGEY-4":      945,  "BUGEY-5":      945,
    "TRICASTIN-1":  915,  "TRICASTIN-2":  915,
    "DAMPIERRE-1":  890,  "DAMPIERRE-2":  890,
    "GRAVELINES-1": 910,  "GRAVELINES-2": 910,  "GRAVELINES-3": 910,
}
nuc_per_region: dict[str, float] = {}
for unit, region in UNIT_TO_REGION.items():
    nuc_per_region[region] = nuc_per_region.get(region, 0.0) + NUC_NAMEPLATE[unit]

nuc_neg = np.array([-nuc_per_region.get(r, 0.0) / 1e3 for r in regions])

# Gas: peak observed dispatch per region (proxy for nameplate; from 10 representative days)
gas_by_region = gen_df.groupby("Région")["Thermique (MW)"].max() / 1e3
gas_neg = np.array([-gas_by_region.get(r, 0.0) for r in regions])

FLA3_GW   = round(1_630 / 1e3, 2)
decom_gw  = round(-nuc_neg.sum(), 1)
gas_tot   = round(-gas_neg.sum(), 1)

# =============================================================================
# Helpers
# =============================================================================
def read_cost(f):
    p = DATA_RESULTS / f
    return round(float(p.read_text().strip()) / 1000, 1) if p.exists() else None

def load_scenario(cap_file, cost_file):
    df = (pd.read_csv(DATA_RESULTS / cap_file, sep=";")
            .set_index("Région").reindex(regions).fillna(0))
    def clean(v): return 0.0 if v < THRESHOLD_MW else v
    sol = np.array([clean(v) for v in df["Solar_MW"]])         / 1e3
    win = np.array([clean(v) for v in df["Wind_MW"]])          / 1e3
    bat = np.array([clean(v / 4) for v in df["Battery_MWh"]])  / 1e3
    return sol, win, bat, read_cost(cost_file)

# =============================================================================
# Scenarios
# =============================================================================
scenarios = [
    dict(label="With gas  |  No FLA3",  row=0, col=0, gas=True,  fla=False,
         cap="capacity_results_demand_growth_8pct.csv", cost="cost_with_gas.txt"),
    dict(label="No gas  |  No FLA3",    row=0, col=1, gas=False, fla=False,
         cap="capacity_results_no_fossil.csv",          cost="cost_no_fossil.txt"),
    dict(label="With gas  |  FLA3",     row=1, col=0, gas=True,  fla=True,
         cap="capacity_results_fla_with_gas.csv",       cost="cost_fla_with_gas.txt"),
    dict(label="No gas  |  FLA3",       row=1, col=1, gas=False, fla=True,
         cap="capacity_results_fla_without_gas.csv",    cost="cost_fla_without_gas.txt"),
]
for s in scenarios:
    s["solar"], s["wind"], s["bat"], s["cost"] = load_scenario(s["cap"], s["cost"])

# =============================================================================
# Y-axis — round to nearest 5 GW, no artificial headroom
# =============================================================================
pos_max = max((s["solar"] + s["wind"] + s["bat"]).max() for s in scenarios)
neg_min = (nuc_neg + gas_neg).min()
y_max   = np.ceil(pos_max  / 5) * 5
y_min   = np.floor(neg_min / 2) * 2

# =============================================================================
# Figure
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True, sharex=False)
fig.patch.set_facecolor("white")

for s in scenarios:
    ax = axes[s["row"]][s["col"]]

    # Positive bars — solid
    ax.bar(x, s["solar"],                             width=BAR_W, color=C_SOLAR, zorder=3)
    ax.bar(x, s["wind"],  bottom=s["solar"],           width=BAR_W, color=C_WIND,  zorder=3)
    ax.bar(x, s["bat"],   bottom=s["solar"]+s["wind"], width=BAR_W, color=C_BAT,   zorder=3)

    # FLA3 — solid, no hatch
    if s["fla"]:
        ax.bar([NOR_IDX], [FLA3_GW], width=BAR_W,
               color=C_FLA3_F, edgecolor=C_FLA3_E, linewidth=1.3, zorder=4)

    # Negative bars — hatched
    ax.bar(x, nuc_neg, width=BAR_W,
           facecolor=C_NUC, edgecolor="white", linewidth=0.5, hatch="///", zorder=3)
    if not s["gas"]:
        ax.bar(x, gas_neg, bottom=nuc_neg, width=BAR_W,
               facecolor=C_GAS, edgecolor="white", linewidth=0.5, hatch="\\\\", zorder=3)

    # Styling
    cost_str = f"{s['cost']:.1f} B€/yr" if s["cost"] is not None else "—"
    ax.set_title(f"{s['label']}\nTotal cost: {cost_str}",
                 fontsize=11, fontweight="bold", pad=8, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(y_min, y_max)
    ax.axhline(0, color="black", linewidth=0.9, zorder=5)
    ax.yaxis.grid(True, color="lightgray", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    if s["col"] == 0:
        ax.set_ylabel("Capacity (GW)", fontsize=10, labelpad=10)
    else:
        ax.tick_params(axis="y", left=False)

# =============================================================================
# Legend
# =============================================================================
legend_handles = [
    mpatches.Patch(facecolor=C_SOLAR, edgecolor="none", label="Solar (GW)"),
    mpatches.Patch(facecolor=C_WIND,  edgecolor="none", label="Wind (GW)"),
    mpatches.Patch(facecolor=C_BAT,   edgecolor="none", label="4h Battery (GW)"),
    mpatches.Patch(facecolor=C_FLA3_F, edgecolor=C_FLA3_E, linewidth=1.3,
                   label=f"FLA3 nameplate (GW)"),
    mpatches.Patch(facecolor=C_NUC, edgecolor="white", hatch="///",
                   label="Nuclear decommissioned — nameplate (GW)"),
    mpatches.Patch(facecolor=C_GAS, edgecolor="white", hatch="\\\\",
                   label="Gas removed — peak dispatch (GW)"),
]
fig.legend(handles=legend_handles, loc="upper right",
           bbox_to_anchor=(0.985, 0.930), fontsize=9.5,
           framealpha=0.92, edgecolor="#cccccc")

# =============================================================================
# Title + subtitle
# =============================================================================
fig.text(0.5, 0.977,
    "Optimal Capacity Expansion  —  ±Gas  ×  ±Flamanville 3   (+8% demand growth)",
    ha="center", fontsize=14, fontweight="bold")
fig.text(0.5, 0.954,
    f"Decommissioned nuclear: {decom_gw} GW nameplate (11 units)   |   "
    f"Gas removed: ~{gas_tot} GW peak dispatch (proxy for nameplate)   |   FLA3: +{FLA3_GW} GW nameplate (NOR, modelled at 75% CF)",
    ha="center", fontsize=9, color="#555555")

# =============================================================================
# Layout and save — explicit subplots_adjust, no tight_layout
# =============================================================================
fig.subplots_adjust(top=0.880, bottom=0.055, left=0.070, right=0.970,
                    hspace=0.2, wspace=0.05)
fig.savefig(FIG_DIR / "comparison_4scenarios.png", dpi=150)
print(f"Saved: {FIG_DIR / 'comparison_4scenarios.png'}")
