"""
K-Medoids: Find 10 representative days across the full year 2024 (no monthly constraint).
Weighting: 50% on consumption, 50% split proportionally across production mix columns.
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
DEMAND_COL      = "consommation"
PRODUCTION_COLS = ["fioul", "charbon", "gaz", "nucleaire",
                   "eolien_total", "solaire", "hydraulique", "pompage", "bioenergies"]
W_DEMAND = 0.50

prod_means     = df[PRODUCTION_COLS].mean(axis=0)
prod_means_pos = prod_means.abs().clip(lower=0.0)
prod_weights   = (prod_means_pos / prod_means_pos.sum()) * (1.0 - W_DEMAND)

ALL_COLS = [DEMAND_COL] + PRODUCTION_COLS
WEIGHTS  = np.concatenate(([W_DEMAND], prod_weights.values))

print("Column weights (50% demand; production proportional to mean share):")
print(f"  {DEMAND_COL:<20} {W_DEMAND:.4f}  (50%)")
for col, w in zip(PRODUCTION_COLS, prod_weights):
    print(f"  {col:<20} {w:.4f}  ({w*100:.2f}%)")

# ── 3. Build Daily Profiles ────────────────────────────────────────────────────
def build_profiles(data, cols):
    rows = []
    for (date, month), grp in data.groupby(["date_only", "month"]):
        grp_sorted = grp.sort_values("datetime")
        if len(grp_sorted) != 48:
            continue
        mat = grp_sorted[cols].values.astype(float)   # (48, n_cols)
        rows.append({"date_only": date, "month": month, "profile": mat})
    return pd.DataFrame(rows).reset_index(drop=True)

daily_profiles = build_profiles(df, ALL_COLS)
profile_3d     = np.stack(daily_profiles["profile"].values)   # (n_days, 48, n_cols)
n_days, n_steps, n_cols = profile_3d.shape

print(f"\nComplete days found : {n_days}")
print(f"Array shape         : {profile_3d.shape}  (days × steps × features)")

# ── 4. Z-score Normalisation ───────────────────────────────────────────────────
flat     = profile_3d.reshape(-1, n_cols)
col_mean = flat.mean(axis=0)
col_std  = flat.std(axis=0)
col_std[col_std == 0] = 1
profile_norm = (profile_3d - col_mean) / col_std

# ── 5. Apply Column Weights & Flatten ─────────────────────────────────────────
sqrt_w           = np.sqrt(WEIGHTS)
profile_weighted = profile_norm * sqrt_w
X                = profile_weighted.reshape(n_days, -1)   # (n_days, 48*n_cols)

# ── 6. K-Medoids (PAM) on the full year ───────────────────────────────────────
K = 10

def k_medoids_euclidean(X, k, max_iter=300, random_state=42):
    rng = np.random.default_rng(random_state)
    n   = len(X)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    D    = np.sqrt((diff ** 2).sum(axis=-1))
    medoid_idx = rng.choice(n, k, replace=False).tolist()

    for iteration in range(max_iter):
        assigned    = D[:, medoid_idx].argmin(axis=1)
        new_medoids = []
        for c in range(k):
            members = np.where(assigned == c)[0]
            if len(members) == 0:
                new_medoids.append(medoid_idx[c])
                continue
            costs = D[np.ix_(members, members)].sum(axis=1)
            new_medoids.append(int(members[costs.argmin()]))
        if new_medoids == medoid_idx:
            print(f"  Converged at iteration {iteration + 1}")
            break
        medoid_idx = new_medoids

    return medoid_idx, assigned

print(f"\nRunning k-medoids with k={K} on the full year …")
medoid_indices, cluster_assignments = k_medoids_euclidean(X, k=K)

# ── 7. Build Results ──────────────────────────────────────────────────────────
MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

results = []
for orig_idx, med_idx in enumerate(medoid_indices):   # orig_idx: 0-based, matches cluster_assignments
    members      = np.where(cluster_assignments == orig_idx)[0]
    cluster_size = len(members)
    weight       = cluster_size / n_days

    row     = daily_profiles.iloc[med_idx]
    raw_mat = profile_3d[med_idx]
    conso   = raw_mat[:, 0]
    month   = row["month"]

    results.append({
        "orig_idx":            orig_idx,              # keep to rebuild lookup after sort
        "cluster_id":          orig_idx + 1,          # temporary, overwritten below
        "representative_date": str(row["date_only"]),
        "month":               month,
        "month_name":          MONTH_NAMES[int(month) - 1],
        "cluster_size":        cluster_size,
        "weight":              round(weight, 6),
        "avg_consumption_MW":  round(float(conso.mean()), 1),
        "min_consumption_MW":  round(float(conso.min()),  1),
        "max_consumption_MW":  round(float(conso.max()),  1),
    })

# Sort chronologically and renumber — orig_idx is kept to allow correct day assignment later
results.sort(key=lambda r: r["representative_date"])
for i, r in enumerate(results, start=1):
    r["cluster_id"] = i

# ── 8. Print Results ──────────────────────────────────────────────────────────
print("\n" + "=" * 88)
print("  10 REPRESENTATIVE DAYS 2024  (50% demand / 50% production mix, no monthly constraint)")
print("=" * 88)
print(f"  {'#':<4} {'Date':<14} {'Month':<12} {'Cluster size':>12} {'Weight':>8} "
      f"{'Avg (MW)':>12} {'Min (MW)':>10} {'Max (MW)':>10}")
print("-" * 88)
for r in results:
    print(f"  {r['cluster_id']:<4} {r['representative_date']:<14} {r['month_name']:<12} "
          f"{r['cluster_size']:>12} {r['weight']:>8.4f} "
          f"{r['avg_consumption_MW']:>12,.0f} "
          f"{r['min_consumption_MW']:>10,.0f} "
          f"{r['max_consumption_MW']:>10,.0f}")
print("-" * 88)
total_w = sum(r["weight"] for r in results)
print(f"  {'Total weight:':<62} {total_w:>8.4f}")
print("=" * 88)

# ── 9. Export CSVs ────────────────────────────────────────────────────────────
out_dir   = os.path.join(project_root, 'data', 'processed')
final_dir = os.path.join(project_root, 'data', 'final')
os.makedirs(out_dir,   exist_ok=True)
os.makedirs(final_dir, exist_ok=True)

summary_path = os.path.join(final_dir, 'new_representative_days.csv')
pd.DataFrame(results).to_csv(summary_path, index=False)
print(f"\nSummary exported to          : {summary_path}")

rep_dates     = [r["representative_date"] for r in results]
df_rep        = df[df["date_only"].astype(str).isin(rep_dates)].copy()
profiles_path = os.path.join(final_dir, 'new_representative_day_profiles.csv')
df_rep.to_csv(profiles_path, index=False)
print(f"Detailed profiles exported to: {profiles_path}")

# ── 10. Assign Every Day of the Year to Its Representative ────────────────────
# Lookup by ORIGINAL 0-based index (matches cluster_assignments values directly)
orig_lookup = {r["orig_idx"]: r for r in results}

# Output A — mapping CSV (366 rows)
mapping_rows = []
for i, row in daily_profiles.iterrows():
    orig_idx = int(cluster_assignments[i])   # 0-based, original k-medoids order
    rep      = orig_lookup[orig_idx]
    mapping_rows.append({
        "actual_date":          str(row["date_only"]),
        "cluster_id":           rep["cluster_id"],
        "representative_date":  rep["representative_date"],
        "weight":               rep["weight"],
    })

df_mapping = pd.DataFrame(mapping_rows).sort_values("actual_date").reset_index(drop=True)
mapping_path = os.path.join(final_dir, 'day_cluster_assignment.csv')
df_mapping.to_csv(mapping_path, index=False)
print(f"Day-cluster mapping exported to: {mapping_path}  ({len(df_mapping)} rows)")
