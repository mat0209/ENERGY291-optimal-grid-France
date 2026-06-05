"""
K-Medoids: Find one representative day per month (12 days for 2024).
Weighting: 50% on consumption, 50% split equally across the production mix columns.
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── 1. Data Loading ────────────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
infile = os.path.join(project_root, 'data', 'processed', 'mix_rte_2024_clean.csv')
df = pd.read_csv(infile, sep=",", decimal=".")

df["datetime"] = pd.to_datetime(df["date"] + " " + df["heures"], dayfirst=True)
df = df.sort_values("datetime").reset_index(drop=True)
df["date_only"] = df["datetime"].dt.date
df["month"]     = df["datetime"].dt.month

# ── 2. Column Groups & Weights ────────────────────────────────────────────────
DEMAND_COL   = "consommation"
PRODUCTION_COLS = ["fioul", "charbon", "gaz", "nucleaire",
                   "eolien_total", "solaire", "hydraulique", "pompage", "bioenergies"]

# Compute weights: 50% on demand, 50% distributed across production columns
# proportionally to their average contribution (columns that produce more get larger weight).
W_DEMAND = 0.50

# compute production means (use non-null values)
prod_means = df[PRODUCTION_COLS].mean(axis=0)
if prod_means.sum() == 0:
# fallback to equal weights if all zeros
    prod_weights = np.repeat(1.0 / len(PRODUCTION_COLS), len(PRODUCTION_COLS))
else:
    # use absolute/positive contributions so resources with negative net (e.g. pumping)
    # don't produce negative weights
    prod_means_pos = prod_means.abs().clip(lower=0.0)
    prod_weights = prod_means_pos / prod_means_pos.sum()

# scale to sum to remaining 50%
prod_weights = prod_weights * (1.0 - W_DEMAND)

ALL_COLS = [DEMAND_COL] + PRODUCTION_COLS
WEIGHTS = np.concatenate(([W_DEMAND], prod_weights.values))

print("Column weights (50% demand; production proportional to mean share):")
print(f"  {DEMAND_COL:<20} {W_DEMAND:.4f}  (50%)")
for col, w in zip(PRODUCTION_COLS, prod_weights):
    print(f"  {col:<20} {w:.4f}  ({w*100:.2f}%)")

# ── 3. Build Daily Profiles ────────────────────────────────────────────────────
# Shape per day: (48 steps × 10 cols) — kept as 2D for weighted normalisation
def build_profiles(data, cols):
    rows = []
    for (date, month), grp in data.groupby(["date_only", "month"]):
        grp_sorted = grp.sort_values("datetime")
        if len(grp_sorted) != 48:
            continue
        mat = grp_sorted[cols].values.astype(float)   # (48, 10)
        rows.append({"date_only": date, "month": month, "profile": mat})
    return pd.DataFrame(rows).reset_index(drop=True)

daily_profiles = build_profiles(df, ALL_COLS)

# Stack into 3D array: (n_days, 48, 10)
profile_3d = np.stack(daily_profiles["profile"].values)   # (n_days, 48, 10)
n_days, n_steps, n_cols = profile_3d.shape

print(f"\nComplete days found : {n_days}")
print(f"Array shape         : {profile_3d.shape}  (days × steps × features)")

# ── 4. Z-score Normalisation Per Column ───────────────────────────────────────
# Normalise across all days & steps so units are comparable
flat = profile_3d.reshape(-1, n_cols)           # (n_days*48, 10)
col_mean = flat.mean(axis=0)
col_std  = flat.std(axis=0)
col_std[col_std == 0] = 1

profile_norm = (profile_3d - col_mean) / col_std   # (n_days, 48, 10)

# ── 5. Apply Column Weights & Flatten ─────────────────────────────────────────
# Multiply each column by sqrt(weight) so that the squared Euclidean distance
# equals the weighted sum of squared distances per feature.
sqrt_w = np.sqrt(WEIGHTS)                          # (10,)
profile_weighted = profile_norm * sqrt_w           # (n_days, 48, 10) broadcast

# Flatten to (n_days, 48*10) for distance computation
X = profile_weighted.reshape(n_days, -1)

# ── 6. K-Medoids (PAM) ────────────────────────────────────────────────────────
def k_medoids_euclidean(X, k, max_iter=300, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(X)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    D = np.sqrt((diff ** 2).sum(axis=-1))
    medoid_idx = rng.choice(n, k, replace=False).tolist()

    for iteration in range(max_iter):
        assigned = D[:, medoid_idx].argmin(axis=1)
        new_medoids = []
        for c in range(k):
            members = np.where(assigned == c)[0]
            if len(members) == 0:
                new_medoids.append(medoid_idx[c])
                continue
            costs = D[np.ix_(members, members)].sum(axis=1)
            new_medoids.append(members[costs.argmin()])
        if new_medoids == medoid_idx:
            print(f"  Converged at iteration {iteration + 1}")
            break
        medoid_idx = new_medoids

    return medoid_idx, assigned

# ── 7. One K-Medoids per Month ────────────────────────────────────────────────
results = []
MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

for month in range(1, 13):
    mask = daily_profiles["month"] == month
    idx  = daily_profiles.index[mask].tolist()
    X_m  = X[idx]

    print(f"\nMonth {month:02d} ({MONTH_NAMES[month-1]}) — {len(idx)} days")

    medoid_local, _ = k_medoids_euclidean(X_m, k=1)
    medoid_global   = idx[medoid_local[0]]

    row      = daily_profiles.iloc[medoid_global]
    raw_mat  = profile_3d[medoid_global]           # (48, 10)
    conso    = raw_mat[:, 0]                        # consumption column

    results.append({
        "month":               month,
        "month_name":          MONTH_NAMES[month - 1],
        "representative_date": str(row["date_only"]),
        "avg_consumption_MW":  round(float(conso.mean()), 1),
        "min_consumption_MW":  round(float(conso.min()),  1),
        "max_consumption_MW":  round(float(conso.max()),  1),
    })

# ── 8. Print Results ──────────────────────────────────────────────────────────
print("\n" + "═" * 72)
print("  REPRESENTATIVE DAYS 2024  (50% demand / 50% production mix)")
print("═" * 72)
print(f"  {'Month':<12} {'Date':<14} {'Avg (MW)':>12} {'Min (MW)':>10} {'Max (MW)':>10}")
print("─" * 72)
for r in results:
    print(f"  {r['month_name']:<12} {r['representative_date']:<14} "
          f"{r['avg_consumption_MW']:>12,.0f} "
          f"{r['min_consumption_MW']:>10,.0f} "
          f"{r['max_consumption_MW']:>10,.0f}")
print("═" * 72)

# ── 9. Export CSVs ────────────────────────────────────────────────────────────
out_dir = os.path.join(project_root, 'data', 'processed')
os.makedirs(out_dir, exist_ok=True)
summary_path = os.path.join(out_dir, 'representative_days_weighted_2024.csv')
pd.DataFrame(results).to_csv(summary_path, index=False)
print(f"\nSummary exported to: {summary_path}")

rep_dates = [r["representative_date"] for r in results]
df_rep = df[df["date_only"].astype(str).isin(rep_dates)].copy()
profiles_path = os.path.join(out_dir, 'representative_day_profiles_weighted_2024.csv')
df_rep.to_csv(profiles_path, index=False)
print(f"Detailed profiles exported to: {profiles_path}")

# ── 10. Comparison vs consumption-only ────────────────────────────────────────
try:
    prev     = pd.read_csv("representative_days_2024.csv")
    prev_map = dict(zip(prev["month"], prev["representative_date"]))
    print("\n" + "─" * 60)
    print("  Comparison: consumption-only vs weighted (50/50)")
    print("─" * 60)
    print(f"  {'Month':<12} {'Conso-only':<14} {'Weighted':<14} {'Same?'}")
    print("─" * 60)
    for r in results:
        m      = r["month"]
        d_prev = prev_map.get(m, "N/A")
        d_new  = r["representative_date"]
        same   = "✓" if d_prev == d_new else "✗ differs"
        print(f"  {r['month_name']:<12} {d_prev:<14} {d_new:<14} {same}")
    print("─" * 60)
except FileNotFoundError:
    print("\n(No consumption-only results found for comparison.)")