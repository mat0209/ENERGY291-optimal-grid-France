"""
Figures for the K-medoids analysis — representative days 2024
Expected inputs (in data/processed/):
  - representative_days_weighted_2024.csv
  - representative_day_profiles_weighted_2024.csv
  - mix_rte_2024_clean.csv  (for monthly context)

Usage:
  python plot_representative_days.py
"""

import os
import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
processed    = os.path.join(project_root, 'data', 'processed')
out_dir      = os.path.join(project_root, 'figures')
os.makedirs(out_dir, exist_ok=True)

summary_path  = os.path.join(processed, 'representative_days_weighted_2024.csv')
profiles_path = os.path.join(processed, 'representative_day_profiles_weighted_2024.csv')
raw_path      = os.path.join(processed, 'mix_rte_2024_clean.csv')

# ── Data ──────────────────────────────────────────────────────────────────────
summary  = pd.read_csv(summary_path)
profiles = pd.read_csv(profiles_path)
raw      = pd.read_csv(raw_path, sep=",", decimal=".")

raw["datetime"]  = pd.to_datetime(raw["date"] + " " + raw["heures"], dayfirst=True)
raw["date_only"] = raw["datetime"].dt.date.astype(str)
raw["month"]     = raw["datetime"].dt.month

profiles["datetime"]  = pd.to_datetime(profiles["date"] + " " + profiles["heures"], dayfirst=True)
profiles["date_only"] = profiles["datetime"].dt.date.astype(str)
profiles["month"]     = profiles["datetime"].dt.month

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

PROD_COLS   = ["nucleaire","hydraulique","gaz","eolien_total",
               "solaire","bioenergies","fioul","charbon","pompage"]
PROD_LABELS = ["Nuclear","Hydro","Gas","Wind",
               "Solar","Bioenergy","Oil","Coal","Pumping"]

PROD_COLORS = ["#4e8fc7","#5bbfa8","#e8944a","#a8d36e",
               "#f7d155","#8bc67e","#c0645e","#8c7565","#9b7ec8"]

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
# Figure 1 — Feature weights
# ═══════════════════════════════════════════════════════════════════════════════
def fig_weights():
    """Horizontal bar chart of each feature's weight in the weighted distance."""

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
    print(f"  ✓  {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Calendar of representative days
# ═══════════════════════════════════════════════════════════════════════════════
def fig_calendar():
    """Calendar view: one panel per month with the representative day highlighted."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), facecolor=STYLE["figure_facecolor"])
    fig.suptitle("Representative days 2024 — K-medoids (50% consumption / 50% production)",
                 fontsize=12, y=1.01)

    rep_map = dict(zip(summary["month"], summary["representative_date"]))

    for m_idx, ax in enumerate(axes.flat):
        month    = m_idx + 1
        rep_date = rep_map[month]
        rep_day  = int(rep_date.split("-")[2])

        _, n_days = calendar.monthrange(2024, month)
        first_wd  = calendar.monthrange(2024, month)[0]  # 0 = Monday

        ax.set_facecolor(STYLE["axes_facecolor"])
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 6)
        ax.set_xticks(np.arange(0.5, 7.5, 1))
        ax.set_xticklabels(["Mo","Tu","We","Th","Fr","Sa","Su"], fontsize=8)
        ax.set_yticks([])
        ax.set_title(MONTH_NAMES[m_idx], fontsize=10, pad=4)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for day in range(1, n_days + 1):
            pos    = day - 1 + first_wd
            col    = pos % 7
            row    = 5 - pos // 7   # top = week 1
            x, y   = col + 0.5, row + 0.5
            is_rep = (day == rep_day)

            circle = plt.Circle((x, y), 0.38,
                                 color="#d45f5f" if is_rep else "#e0ddd8",
                                 zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, str(day), ha="center", va="center",
                    fontsize=7.5 if not is_rep else 8.5,
                    color="white" if is_rep else "#666",
                    fontweight="bold" if is_rep else "normal",
                    zorder=3)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_calendar_representative.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ✓  {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Half-hourly consumption profiles
# ═══════════════════════════════════════════════════════════════════════════════
def fig_consumption_profiles():
    """Consumption curve over 48 half-hour steps for each representative day,
    with the monthly P5–P95 envelope in the background."""
    cmap = plt.cm.get_cmap("tab20", 12)

    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharey=False,
                             facecolor=STYLE["figure_facecolor"])
    fig.suptitle("Consumption profiles — representative days vs monthly envelope 2024",
                 fontsize=12, y=1.01)

    steps = np.arange(48) * 0.5  # hours 0 → 23.5

    for m_idx, ax in enumerate(axes.flat):
        month = m_idx + 1
        ax.set_facecolor(STYLE["axes_facecolor"])

        # All complete days in the month
        month_data = raw[raw["month"] == month].copy()
        day_groups = month_data.groupby("date_only")["consommation"].apply(list)
        full_days  = [v for v in day_groups if len(v) == 48]

        if full_days:
            mat = np.array(full_days)
            ax.fill_between(steps,
                            np.percentile(mat, 5,  axis=0),
                            np.percentile(mat, 95, axis=0),
                            alpha=0.15, color=cmap(m_idx), label="P5–P95")

        # Representative day
        rep_row = profiles[profiles["month"] == month].sort_values("datetime")
        if len(rep_row) == 48:
            ax.plot(steps, rep_row["consommation"].values,
                    color=cmap(m_idx), lw=1.8, label="Rep. day")
            rep_date = summary.loc[summary.month == month, "representative_date"].values[0]
            ax.set_title(f"{MONTH_NAMES[m_idx]} — {rep_date[5:]}", fontsize=9, pad=3)

        ax.set_xlim(0, 23.5)
        ax.set_xticks([0, 6, 12, 18])
        ax.tick_params(labelsize=7)
        ax.set_ylabel("MW" if m_idx % 4 == 0 else "", fontsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_consumption_profiles.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ✓  {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Stacked production mix
# ═══════════════════════════════════════════════════════════════════════════════
def fig_production_mix():
    """Stacked bar chart of average production by source for each representative day."""
    means = []
    for _, row in summary.iterrows():
        day_prof = profiles[profiles["date_only"] == row["representative_date"]]
        means.append(day_prof[PROD_COLS].mean().values if len(day_prof) > 0
                     else np.zeros(len(PROD_COLS)))

    means  = np.array(means)   # (12, 9)
    x      = np.arange(12)
    bottom = np.zeros(12)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=STYLE["figure_facecolor"])
    ax.set_facecolor(STYLE["axes_facecolor"])

    for col, label, color, vals in zip(PROD_COLS, PROD_LABELS, PROD_COLORS, means.T):
        ax.bar(x, vals.clip(min=0), bottom=bottom, color=color, label=label,
               width=0.7, edgecolor="white", linewidth=0.4)
        bottom += vals.clip(min=0)

    # Average consumption line
    ax.plot(x, summary["avg_consumption_MW"].values, "o--", color="#333",
            lw=1.5, ms=5, label="Avg consumption", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{MONTH_NAMES[i]}\n{summary.iloc[i]['representative_date'][5:]}"
         for i in range(12)],
        fontsize=8.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.set_ylabel("Average power (MW)", fontsize=10)
    ax.set_title("Production mix and consumption — representative days 2024",
                 fontsize=11, pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=5, fontsize=8, frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_production_mix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ✓  {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Consumption heatmap + min/avg/max bar chart
# ═══════════════════════════════════════════════════════════════════════════════
def fig_heatmap():
    """12×48 heatmap of row-normalised consumption profiles + avg ± [min, max] bars."""
    mat = np.zeros((12, 48))
    for _, row in summary.iterrows():
        m   = row["month"] - 1
        day = profiles[profiles["date_only"] == row["representative_date"]].sort_values("datetime")
        if len(day) == 48:
            mat[m] = day["consommation"].values

    # Row-normalise so daily shapes are comparable across months
    mat_norm = (mat - mat.min(axis=1, keepdims=True)) / \
               (mat.max(axis=1, keepdims=True) - mat.min(axis=1, keepdims=True) + 1e-9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   facecolor=STYLE["figure_facecolor"],
                                   gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("Consumption profile heatmap — representative days 2024", fontsize=11)

    # Heatmap
    im = ax1.imshow(mat_norm, aspect="auto", cmap="RdYlBu_r",
                    extent=[0, 23.5, 12.5, 0.5], vmin=0, vmax=1)
    ax1.set_yticks(range(1, 13))
    ax1.set_yticklabels(MONTH_NAMES, fontsize=9)
    ax1.set_xlabel("Hour of day", fontsize=10)
    ax1.set_xticks([0, 6, 12, 18, 23])
    ax1.set_title("Normalised profile (0 = daily min, 1 = daily max)", fontsize=9, pad=6)
    plt.colorbar(im, ax=ax1, fraction=0.025, pad=0.02, label="Relative intensity")

    # Avg ± [min, max] bars
    ax2.set_facecolor(STYLE["axes_facecolor"])
    y = np.arange(12)
    ax2.barh(y, summary["avg_consumption_MW"] / 1000,
             color="#c07a7a", height=0.6)
    ax2.errorbar(
        summary["avg_consumption_MW"] / 1000, y,
        xerr=[
            (summary["avg_consumption_MW"] - summary["min_consumption_MW"]) / 1000,
            (summary["max_consumption_MW"] - summary["avg_consumption_MW"]) / 1000,
        ],
        fmt="none", ecolor="#555", capsize=3, linewidth=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels(MONTH_NAMES, fontsize=9)
    ax2.set_xlabel("Consumption (GW)", fontsize=9)
    ax2.set_title("Avg ± [min, max]", fontsize=9, pad=6)
    ax2.invert_yaxis()
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  ✓  {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...\n")
    fig_weights()
    fig_calendar()
    fig_consumption_profiles()
    fig_production_mix()
    fig_heatmap()
    print(f"\nAll figures saved to: {out_dir}")