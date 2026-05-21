import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
gen = pd.read_csv(PROCESSED / "gen_reduced_days_2024.csv", sep=";")
pv  = pd.read_csv(PROCESSED / "ninja_pv_representative_days_2024_regions.csv")
wnd = pd.read_csv(PROCESSED / "ninja_wind_representative_days_2024_regions.csv")

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

# month labels for titles
MONTH_LABEL = {
    "2024-01-29": "January",   "2024-02-14": "February",
    "2024-03-14": "March",     "2024-04-17": "April",
    "2024-05-13": "May",       "2024-06-11": "June",
    "2024-07-10": "July",      "2024-08-20": "August",
    "2024-09-12": "September", "2024-10-02": "October",
    "2024-11-08": "November",  "2024-12-20": "December",
}

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Generation mix per energy source (stacked area)
#          one figure per region, 12 subplots (one per representative day)
# ══════════════════════════════════════════════════════════════════════════════
for region in regions:
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharey=False)
    axes = axes.flatten()
    fig.suptitle(f"Generation mix — {region}", fontsize=14, fontweight="bold")

    rdf = gen[gen["Région"] == region].copy()

    for ax, date in zip(axes, dates):
        ddf = rdf[rdf["Date"] == date].sort_values("hour")
        hours = ddf["hour"].values

        bottom = None
        for col, label in GEN_SOURCES.items():
            vals = pd.to_numeric(ddf[col], errors="coerce").fillna(0).values
            if bottom is None:
                ax.bar(hours, vals, label=label, color=SOURCE_COLORS[label], width=0.85)
                bottom = vals.copy()
            else:
                ax.bar(hours, vals, bottom=bottom, label=label,
                       color=SOURCE_COLORS[label], width=0.85)
                bottom += vals

        ax.set_title(MONTH_LABEL[date], fontsize=9)
        ax.set_xlabel("Hour (UTC)", fontsize=7)
        ax.set_ylabel("MW", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xticks(range(0, 24, 4))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(GEN_SOURCES),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fname = region.replace(" ", "_").replace("'", "").replace("-", "_")
    fig.savefig(FIG / f"gen_mix_{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved gen_mix_{fname}.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Gen reduced vs demand (all regions, one figure per day)
# ══════════════════════════════════════════════════════════════════════════════
for date in dates:
    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharey=False)
    axes = axes.flatten()
    fig.suptitle(f"Gen reduced vs Demand — {MONTH_LABEL[date]} ({date})",
                 fontsize=14, fontweight="bold")

    for ax, region in zip(axes, regions):
        sub = gen[(gen["Date"] == date) & (gen["Région"] == region)].sort_values("hour")
        hours = sub["hour"].values
        ax.fill_between(hours,
                        pd.to_numeric(sub["gen_reduced_MW"], errors="coerce").fillna(0),
                        alpha=0.5, color="#4e79a7", label="Gen reduced")
        ax.plot(hours,
                pd.to_numeric(sub["Consommation (MW)"], errors="coerce").fillna(0),
                color="black", linewidth=1.5, label="Demand")
        ax.set_title(region, fontsize=8)
        ax.set_xlabel("Hour (UTC)", fontsize=7)
        ax.set_ylabel("MW", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xticks(range(0, 24, 4))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(FIG / f"gen_reduced_vs_demand_{date}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved gen_reduced_vs_demand_{date}.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Solar & Wind capacity factors per region
#          one figure per representative day, CF[0,1] per region
# ══════════════════════════════════════════════════════════════════════════════
for date in dates:
    pv_day  = pv[pv["date"]  == date].sort_values("hour")
    wnd_day = wnd[wnd["date"] == date].sort_values("hour")

    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharey=True)
    axes = axes.flatten()
    fig.suptitle(f"Solar & Wind capacity factors — {MONTH_LABEL[date]} ({date})",
                 fontsize=14, fontweight="bold")

    for ax, region in zip(axes, regions):
        hours = pv_day["hour"].values
        cf_pv  = pv_day[region].values  if region in pv_day.columns  else [0]*24
        cf_wnd = wnd_day[region].values if region in wnd_day.columns else [0]*24

        ax.fill_between(hours, cf_pv,  alpha=0.6, color="#f28e2b", label="Solar CF")
        ax.fill_between(hours, cf_wnd, alpha=0.6, color="#59a14f", label="Wind CF")
        ax.set_ylim(0, 1)
        ax.set_title(region, fontsize=8)
        ax.set_xlabel("Hour (UTC)", fontsize=7)
        ax.set_ylabel("CF", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xticks(range(0, 24, 4))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(FIG / f"cf_solar_wind_{date}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved cf_solar_wind_{date}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — National generation mix (summed across all regions)
# ══════════════════════════════════════════════════════════════════════════════
nat = gen.groupby(["Date", "hour"])[[*GEN_SOURCES.keys(), "Consommation (MW)",
                                      "gen_reduced_MW", "real_gen_MW"]].sum().reset_index()

fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharey=False)
axes = axes.flatten()
fig.suptitle("National generation mix (all regions summed)", fontsize=14, fontweight="bold")

for ax, date in zip(axes, dates):
    ddf = nat[nat["Date"] == date].sort_values("hour")
    hours = ddf["hour"].values
    bottom = None
    for col, label in GEN_SOURCES.items():
        vals = pd.to_numeric(ddf[col], errors="coerce").fillna(0).values
        if bottom is None:
            ax.bar(hours, vals, label=label, color=SOURCE_COLORS[label], width=0.85)
            bottom = vals.copy()
        else:
            ax.bar(hours, vals, bottom=bottom, label=label,
                   color=SOURCE_COLORS[label], width=0.85)
            bottom += vals
    ax.set_title(MONTH_LABEL[date], fontsize=9)
    ax.set_xlabel("Hour (UTC)", fontsize=7)
    ax.set_ylabel("MW", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xticks(range(0, 24, 4))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(GEN_SOURCES),
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(FIG / "national_gen_mix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved national_gen_mix.png")

# ══════════════════════════════════════════════════════════════════════════════
# Plot 5 — National gen reduced vs demand (all 12 days, one subplot each)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharey=False)
axes = axes.flatten()
fig.suptitle("National gen reduced vs Demand (all regions summed)",
             fontsize=14, fontweight="bold")

for ax, date in zip(axes, dates):
    ddf = nat[nat["Date"] == date].sort_values("hour")
    hours = ddf["hour"].values
    ax.fill_between(hours,
                    pd.to_numeric(ddf["gen_reduced_MW"], errors="coerce").fillna(0),
                    alpha=0.4, color="#4e79a7", label="Gen reduced")
    ax.plot(hours,
            pd.to_numeric(ddf["real_gen_MW"], errors="coerce").fillna(0),
            color="#4e79a7", linewidth=1.5, linestyle="--", label="Real gen")
    ax.plot(hours,
            pd.to_numeric(ddf["Consommation (MW)"], errors="coerce").fillna(0),
            color="black", linewidth=1.5, label="Demand")
    ax.set_title(MONTH_LABEL[date], fontsize=9)
    ax.set_xlabel("Hour (UTC)", fontsize=7)
    ax.set_ylabel("MW", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xticks(range(0, 24, 4))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(FIG / "national_gen_reduced_vs_demand.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved national_gen_reduced_vs_demand.png")

print(f"\nAll figures saved to {FIG}")
