"""
Visualize national generation mix and consumption for each representative day.
Reads from data/final/; writes figures to figures/.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "data" / "final"
FIG = ROOT / "figures" / "gen representative days"
FIG.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
gen = pd.read_csv(FINAL / "gen_reduced_days.csv", sep=";")
pv  = pd.read_csv(FINAL / "ninja_pv_new_representative_days_regions.csv")
wnd = pd.read_csv(FINAL / "ninja_wind_new_representative_days_regions.csv")

gen["hour"] = pd.to_datetime(gen["Heure"], format="%H:%M").dt.hour
pv["hour"]  = pd.to_datetime(pv["time"]).dt.hour
wnd["hour"] = pd.to_datetime(wnd["time"]).dt.hour
pv["date"]  = pd.to_datetime(pv["time"]).dt.date.astype(str)
wnd["date"] = pd.to_datetime(wnd["time"]).dt.date.astype(str)

GEN_SOURCES = {
    "Nucléaire (MW)":    "Nuclear",
    "Hydraulique (MW)":  "Hydro",
    "Thermique (MW)":    "Thermal",
    "Eolien terrestre":  "Wind onshore",
    "Eolien offshore":   "Wind offshore",
    "Solaire (MW)":      "Solar",
    "Bioénergies (MW)":  "Bioenergy",
    "Pompage (MW)":      "Pumping",
}
SOURCE_COLORS = {
    "Nuclear":       "#4e79a7",
    "Hydro":         "#76b7b2",
    "Thermal":       "#e15759",
    "Wind onshore":  "#59a14f",
    "Wind offshore": "#8cd17d",
    "Solar":         "#f28e2b",
    "Bioenergy":     "#b07aa1",
    "Pumping":       "#9c755f",
}

regions = sorted(gen["Région"].unique())
dates   = sorted(gen["Date"].unique())

MONTH_LABEL = {
    "2024-01-18": "January",
    "2024-02-04": "February",
    "2024-03-12": "March",
    "2024-06-12": "June (1)",
    "2024-06-22": "June (2)",
    "2024-08-13": "August",
    "2024-09-17": "September",
    "2024-10-04": "October",
    "2024-11-04": "November",
    "2024-12-02": "December",
}

# National aggregation
nat_gen_cols = [*GEN_SOURCES.keys(), "Consommation (MW)", "Total gen (MW)", "gen_reduced_MW"]
nat = (
    gen.groupby(["Date", "hour"])[[c for c in nat_gen_cols if c in gen.columns]]
    .sum()
    .reset_index()
)

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — National generation mix (stacked bar, one subplot per day)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
axes = axes.flatten()
fig.suptitle("National generation mix (all regions summed)", fontsize=14, fontweight="bold")

for ax, date in zip(axes, dates):
    ddf = nat[nat["Date"] == date].sort_values("hour")
    hours = ddf["hour"].values
    bottom = None
    for col, label in GEN_SOURCES.items():
        if col not in ddf.columns:
            continue
        vals = pd.to_numeric(ddf[col], errors="coerce").fillna(0).values
        if bottom is None:
            ax.bar(hours, vals, label=label, color=SOURCE_COLORS[label], width=0.85)
            bottom = vals.copy()
        else:
            ax.bar(hours, vals, bottom=bottom, label=label,
                   color=SOURCE_COLORS[label], width=0.85)
            bottom += vals
    ax.set_title(f"{MONTH_LABEL.get(date, date)}\n{date}", fontsize=9)
    ax.set_xlabel("Hour", fontsize=7)
    ax.set_ylabel("MW", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xticks(range(0, 24, 4))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(GEN_SOURCES),
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(FIG / "national_gen_mix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved national_gen_mix.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — National consumption, total gen and gen reduced (one subplot per day)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
axes = axes.flatten()
fig.suptitle("National — Consumption, Total generation & Gen reduced",
             fontsize=14, fontweight="bold")

for ax, date in zip(axes, dates):
    ddf = nat[nat["Date"] == date].sort_values("hour")
    hours = ddf["hour"].values

    ax.fill_between(hours,
                    pd.to_numeric(ddf["gen_reduced_MW"], errors="coerce").fillna(0),
                    alpha=0.35, color="#4e79a7", label="Gen reduced")
    ax.plot(hours,
            pd.to_numeric(ddf["gen_reduced_MW"], errors="coerce").fillna(0),
            color="#4e79a7", linewidth=1.2)
    ax.plot(hours,
            pd.to_numeric(ddf["Total gen (MW)"], errors="coerce").fillna(0),
            color="#59a14f", linewidth=1.5, linestyle="--", label="Total gen")
    ax.plot(hours,
            pd.to_numeric(ddf["Consommation (MW)"], errors="coerce").fillna(0),
            color="black", linewidth=1.5, label="Consumption")
    ax.set_title(f"{MONTH_LABEL.get(date, date)}\n{date}", fontsize=9)
    ax.set_xlabel("Hour", fontsize=7)
    ax.set_ylabel("MW", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xticks(range(0, 24, 4))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(FIG / "national_consumption_gen_reduced.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved national_consumption_gen_reduced.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — National generation mix with decommissioned nuclear + consumption line
# ══════════════════════════════════════════════════════════════════════════════
nat_v2_cols = [
    "nuclear_gen_reduced_MW", "Hydraulique (MW)", "Thermique (MW)",
    "Eolien terrestre", "Eolien offshore", "Solaire (MW)", "Bioénergies (MW)",
    "Pompage (MW)", "nuclear_selected_MW", "Consommation (MW)",
]
nat_v2 = (
    gen.groupby(["Date", "hour"])[[c for c in nat_v2_cols if c in gen.columns]]
    .sum()
    .reset_index()
)

GEN_SOURCES_V2 = {
    "nuclear_gen_reduced_MW":  "Nuclear",
    "Hydraulique (MW)":        "Hydro",
    "Thermique (MW)":          "Thermal",
    "Eolien terrestre":        "Wind onshore",
    "Eolien offshore":         "Wind offshore",
    "Solaire (MW)":            "Solar",
    "Bioénergies (MW)":        "Bioenergy",
    "Pompage (MW)":            "Pumping",
    "nuclear_selected_MW":     "Nuclear (decommissioned)",
}

SOURCE_COLORS_V2 = {
    "Nuclear":                  "#4e79a7",
    "Hydro":                    "#76b7b2",
    "Thermal":                  "#e15759",
    "Wind onshore":             "#59a14f",
    "Wind offshore":            "#8cd17d",
    "Solar":                    "#f28e2b",
    "Bioenergy":                "#b07aa1",
    "Pumping":                  "#9c755f",
    "Nuclear (decommissioned)": "#4e79a7",  # same blue, distinguished by hatching
}

fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=False)
axes = axes.flatten()
fig.suptitle("National generation mix — decommissioned nuclear units highlighted",
             fontsize=14, fontweight="bold")

for ax, date in zip(axes, dates):
    ddf = nat_v2[nat_v2["Date"] == date].sort_values("hour")
    hours = ddf["hour"].values
    bottom = None

    for col, label in GEN_SOURCES_V2.items():
        if col not in ddf.columns:
            continue
        vals = pd.to_numeric(ddf[col], errors="coerce").fillna(0).clip(lower=0).values
        hatch = "///" if label == "Nuclear (decommissioned)" else None
        edgecolor = "white" if hatch is None else "#2c4a6e"
        if bottom is None:
            ax.bar(hours, vals, label=label, color=SOURCE_COLORS_V2[label],
                   hatch=hatch, edgecolor=edgecolor, linewidth=0.4, width=0.85)
            bottom = vals.copy()
        else:
            ax.bar(hours, vals, bottom=bottom, label=label, color=SOURCE_COLORS_V2[label],
                   hatch=hatch, edgecolor=edgecolor, linewidth=0.4, width=0.85)
            bottom += vals

    # Consumption lines
    cons = pd.to_numeric(ddf["Consommation (MW)"], errors="coerce").fillna(0)
    ax.plot(hours, cons, color="black", linewidth=1.8, linestyle="-",
            label="Consumption", zorder=5)
    ax.plot(hours, cons * 1.08, color="black", linewidth=1.2, linestyle="--",
            label="Consumption +8%", zorder=5)

    ax.set_title(f"{MONTH_LABEL.get(date, date)}\n{date}", fontsize=9)
    ax.set_xlabel("Hour", fontsize=7)
    ax.set_ylabel("MW", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xticks(range(0, 24, 4))

handles, labels_leg = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc="lower center", ncol=len(GEN_SOURCES_V2) + 2,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(FIG / "national_gen_mix_v2.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved national_gen_mix_v2.png")

print(f"\nAll figures saved to {FIG}")
