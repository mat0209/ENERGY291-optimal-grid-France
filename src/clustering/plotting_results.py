"""
Figures for the K-medoids analysis — 10 representative days 2024 (no monthly constraint)
Expected inputs:
  - data/final/new_representative_days.csv
  - data/final/new_representative_day_profiles.csv
  - data/final/day_cluster_assignment.csv
  - data/processed/mix_rte_2024_clean.csv
"""

import os
import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
final_dir    = os.path.join(project_root, 'data', 'final')
processed    = os.path.join(project_root, 'data', 'processed')
out_dir      = os.path.join(project_root, 'figures', 'clustering')
os.makedirs(out_dir, exist_ok=True)

summary    = pd.read_csv(os.path.join(final_dir,  'new_representative_days.csv'))
profiles   = pd.read_csv(os.path.join(final_dir,  'new_representative_day_profiles.csv'))
assignment = pd.read_csv(os.path.join(final_dir,  'day_cluster_assignment.csv'))
raw        = pd.read_csv(os.path.join(processed,  'mix_rte_2024_clean.csv'))

raw["datetime"]  = pd.to_datetime(raw["date"] + " " + raw["heures"], dayfirst=True)
raw["date_only"] = raw["datetime"].dt.date.astype(str)
raw["month"]     = raw["datetime"].dt.month

profiles["datetime"]  = pd.to_datetime(profiles["date"] + " " + profiles["heures"], dayfirst=True)
profiles["date_only"] = profiles["datetime"].dt.date.astype(str)

K = len(summary)   # 10

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

PROD_COLS   = ["nucleaire","hydraulique","gaz","eolien_total",
               "solaire","bioenergies","fioul","charbon","pompage"]
PROD_LABELS = ["Nuclear","Hydro","Gas","Wind",
               "Solar","Bioenergy","Oil","Coal","Pumping"]
PROD_COLORS = ["#4e8fc7","#5bbfa8","#e8944a","#a8d36e",
               "#f7d155","#8bc67e","#c0645e","#8c7565","#9b7ec8"]

# 10 distinct cluster colors
CLUSTER_CMAP = plt.cm.get_cmap("tab10", K)
CLUSTER_COLORS = [CLUSTER_CMAP(i) for i in range(K)]

STYLE = dict(figure_facecolor="#fafaf8", axes_facecolor="#fafaf8")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Feature weights (unchanged — same weighting scheme)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_weights():
    W_DEMAND = 0.50
    prod_means     = raw[PROD_COLS].mean()
    prod_means_pos = prod_means.abs().clip(lower=0)
    prod_weights   = prod_means_pos / prod_means_pos.sum() * (1 - W_DEMAND)

    labels  = ["Consumption"] + PROD_LABELS
    weights = np.concatenate(([W_DEMAND], prod_weights.values))
    colors  = ["#d45f5f"] + PROD_COLORS

    order     = np.argsort(weights)
    labels_s  = [labels[i]  for i in order]
    weights_s = [weights[i] for i in order]
    colors_s  = [colors[i]  for i in order]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=STYLE["figure_facecolor"])
    ax.set_facecolor(STYLE["axes_facecolor"])
    bars = ax.barh(labels_s, weights_s, color=colors_s, height=0.6,
                   edgecolor="white", linewidth=0.5)
    for bar, w in zip(bars, weights_s):
        ax.text(w + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{w*100:.1f}%", va="center", fontsize=8.5, color="#444")
    ax.set_xlabel("Weight in the weighted distance", fontsize=10)
    ax.set_title("Feature weights — K-medoids 2024\n"
                 "50% consumption · 50% production mix (proportional to mean absolute contribution)",
                 fontsize=10, pad=12)
    ax.set_xlim(0, max(weights_s) * 1.18)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_feature_weights.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ok  {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Full-year calendar coloured by cluster assignment
# ═══════════════════════════════════════════════════════════════════════════════
def fig_calendar():
    """3×4 monthly calendars; each day is coloured by its cluster. Medoid days
    are outlined in black and labelled with the cluster number."""
    assign_map = dict(zip(assignment["actual_date"], assignment["cluster_id"]))
    rep_dates  = set(summary["representative_date"])

    fig, axes = plt.subplots(3, 4, figsize=(14, 9), facecolor=STYLE["figure_facecolor"])
    fig.suptitle("Cluster assignment — 10 representative days 2024 (k-medoids, no monthly constraint)",
                 fontsize=11, y=1.01)

    for m_idx, ax in enumerate(axes.flat):
        month = m_idx + 1
        _, n_days = calendar.monthrange(2024, month)
        first_wd  = calendar.monthrange(2024, month)[0]

        ax.set_facecolor(STYLE["axes_facecolor"])
        ax.set_xlim(0, 7); ax.set_ylim(0, 6)
        ax.set_xticks(np.arange(0.5, 7.5, 1))
        ax.set_xticklabels(["Mo","Tu","We","Th","Fr","Sa","Su"], fontsize=8)
        ax.set_yticks([])
        ax.set_title(MONTH_NAMES[m_idx], fontsize=10, pad=4)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for day in range(1, n_days + 1):
            date_str = f"2024-{month:02d}-{day:02d}"
            cid      = assign_map.get(date_str, None)
            is_rep   = date_str in rep_dates

            pos = day - 1 + first_wd
            col = pos % 7
            row = 5 - pos // 7
            x, y = col + 0.5, row + 0.5

            face_color = CLUSTER_COLORS[cid - 1] if cid else "#e0ddd8"
            circle = plt.Circle((x, y), 0.38, color=face_color,
                                 zorder=2, alpha=0.85)
            ax.add_patch(circle)

            if is_rep:
                ring = plt.Circle((x, y), 0.38, fill=False,
                                   edgecolor="black", linewidth=1.8, zorder=4)
                ax.add_patch(ring)

            ax.text(x, y, str(day), ha="center", va="center",
                    fontsize=7 if not is_rep else 8,
                    color="white",
                    fontweight="bold" if is_rep else "normal",
                    zorder=5)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=CLUSTER_COLORS[i], edgecolor="black" if True else "none",
                       label=f"Cluster {i+1}  ({summary.iloc[i]['representative_date'][5:]},"
                             f" {summary.iloc[i]['cluster_size']} days)")
        for i in range(K)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_calendar_representative.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ok  {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Consumption profiles (one panel per cluster)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_consumption_profiles():
    """2×5 grid; each panel shows the P5–P95 envelope of all days in the cluster
    and the consumption profile of the representative day."""
    # Build mapping: cluster_id → list of actual dates
    cluster_days = assignment.groupby("cluster_id")["actual_date"].apply(list).to_dict()

    fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharey=False,
                             facecolor=STYLE["figure_facecolor"])
    fig.suptitle("Consumption profiles — representative day vs cluster envelope 2024",
                 fontsize=12, y=1.01)

    steps = np.arange(48) * 0.5   # 0 → 23.5 h

    for idx, ax in enumerate(axes.flat):
        if idx >= K:
            ax.set_visible(False)
            continue

        row_s  = summary.iloc[idx]
        cid    = int(row_s["cluster_id"])
        color  = CLUSTER_COLORS[idx]
        ax.set_facecolor(STYLE["axes_facecolor"])

        # Envelope of all days in cluster
        dates_in_cluster = cluster_days.get(cid, [])
        cluster_data = raw[raw["date_only"].isin(dates_in_cluster)].copy()
        day_groups   = cluster_data.groupby("date_only")["consommation"].apply(list)
        full_days    = [v for v in day_groups if len(v) == 48]
        if full_days:
            mat = np.array(full_days)
            ax.fill_between(steps,
                            mat.min(axis=0),
                            mat.max(axis=0),
                            alpha=0.2, color=color)

        # Representative day profile
        rep_prof = profiles[profiles["date_only"] == row_s["representative_date"]].sort_values("datetime")
        if len(rep_prof) == 48:
            ax.plot(steps, rep_prof["consommation"].values,
                    color=color, lw=2, label="Rep. day")

        ax.set_xlim(0, 23.5)
        ax.set_xticks([0, 6, 12, 18])
        ax.tick_params(labelsize=7)
        ax.set_ylabel("MW" if idx % 5 == 0 else "", fontsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.set_title(f"Cluster {cid} — {row_s['representative_date'][5:]}\n"
                     f"{row_s['cluster_size']} days  |  weight {row_s['weight']*100:.1f}%",
                     fontsize=8, pad=3)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_consumption_profiles.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ok  {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Stacked production mix
# ═══════════════════════════════════════════════════════════════════════════════
def fig_production_mix():
    means = []
    for _, row in summary.iterrows():
        day_prof = profiles[profiles["date_only"] == row["representative_date"]]
        means.append(day_prof[PROD_COLS].mean().values if len(day_prof) > 0
                     else np.zeros(len(PROD_COLS)))

    means  = np.array(means)   # (10, 9)
    x      = np.arange(K)
    bottom = np.zeros(K)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=STYLE["figure_facecolor"])
    ax.set_facecolor(STYLE["axes_facecolor"])

    for col, label, color, vals in zip(PROD_COLS, PROD_LABELS, PROD_COLORS, means.T):
        ax.bar(x, vals.clip(min=0), bottom=bottom, color=color, label=label,
               width=0.7, edgecolor="white", linewidth=0.4)
        bottom += vals.clip(min=0)

    ax.plot(x, summary["avg_consumption_MW"].values, "o--", color="#333",
            lw=1.5, ms=5, label="Avg consumption", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"C{int(summary.iloc[i]['cluster_id'])}\n{summary.iloc[i]['representative_date'][5:]}\n"
         f"({summary.iloc[i]['cluster_size']}d)"
         for i in range(K)],
        fontsize=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.set_ylabel("Average power (MW)", fontsize=10)
    ax.set_title("Production mix and consumption — 10 representative days 2024",
                 fontsize=11, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=5, fontsize=8, frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_production_mix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ok  {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Heatmap (10 clusters × 48 half-hours) + avg/min/max bars
# ═══════════════════════════════════════════════════════════════════════════════
def fig_heatmap():
    mat = np.zeros((K, 48))
    for i, row in summary.iterrows():
        day = profiles[profiles["date_only"] == row["representative_date"]].sort_values("datetime")
        if len(day) == 48:
            mat[i] = day["consommation"].values

    mat_norm = (mat - mat.min(axis=1, keepdims=True)) / \
               (mat.max(axis=1, keepdims=True) - mat.min(axis=1, keepdims=True) + 1e-9)

    ylabels = [f"C{int(summary.iloc[i]['cluster_id'])} — {summary.iloc[i]['representative_date'][5:]}"
               for i in range(K)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   facecolor=STYLE["figure_facecolor"],
                                   gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("Consumption profile heatmap — 10 representative days 2024", fontsize=11)

    im = ax1.imshow(mat_norm, aspect="auto", cmap="RdYlBu_r",
                    extent=[0, 23.5, K + 0.5, 0.5], vmin=0, vmax=1)
    ax1.set_yticks(range(1, K + 1))
    ax1.set_yticklabels(ylabels, fontsize=8.5)
    ax1.set_xlabel("Hour of day", fontsize=10)
    ax1.set_xticks([0, 6, 12, 18, 23])
    ax1.set_title("Normalised profile (0 = daily min, 1 = daily max)", fontsize=9, pad=6)
    plt.colorbar(im, ax=ax1, fraction=0.025, pad=0.02, label="Relative intensity")

    ax2.set_facecolor(STYLE["axes_facecolor"])
    y = np.arange(K)
    ax2.barh(y, summary["avg_consumption_MW"] / 1000, color="#c07a7a", height=0.6)
    ax2.errorbar(
        summary["avg_consumption_MW"] / 1000, y,
        xerr=[
            (summary["avg_consumption_MW"] - summary["min_consumption_MW"]) / 1000,
            (summary["max_consumption_MW"] - summary["avg_consumption_MW"]) / 1000,
        ],
        fmt="none", ecolor="#555", capsize=3, linewidth=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels(ylabels, fontsize=8.5)
    ax2.set_xlabel("Consumption (GW)", fontsize=9)
    ax2.set_title("Avg +/- [min, max]", fontsize=9, pad=6)
    ax2.invert_yaxis()
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ok  {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...\n")
    fig_weights()
    fig_calendar()
    fig_consumption_profiles()
    fig_production_mix()
    fig_heatmap()
    print(f"\nAll figures saved to: {out_dir}")
