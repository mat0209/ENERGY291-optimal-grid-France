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
for cluster_id, med_idx in enumerate(medoid_indices, start=1):
    members      = np.where(cluster_assignments == (cluster_id - 1))[0]
    cluster_size = len(members)
    weight       = cluster_size / n_days   # fraction of the year

    row     = daily_profiles.iloc[med_idx]
    raw_mat = profile_3d[med_idx]          # (48, n_cols)
    conso   = raw_mat[:, 0]
    month   = row["month"]

    results.append({
        "cluster_id":          cluster_id,
        "representative_date": str(row["date_only"]),
        "month":               month,
        "month_name":          MONTH_NAMES[int(month) - 1],
        "cluster_size":        cluster_size,
        "weight":              round(weight, 6),
        "avg_consumption_MW":  round(float(conso.mean()), 1),
        "min_consumption_MW":  round(float(conso.min()),  1),
        "max_consumption_MW":  round(float(conso.max()),  1),
    })

# Sort chronologically by representative date
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
# Build a lookup: cluster_id (1-based) → dict with representative_date and weight
cluster_lookup = {r["cluster_id"]: r for r in results}

# Output A — mapping CSV (366 rows)
mapping_rows = []
for i, row in daily_profiles.iterrows():
    assignment   = int(cluster_assignments[i])   # 0-based cluster index
    cluster_id   = assignment + 1                # 1-based
    rep          = cluster_lookup[cluster_id]
    mapping_rows.append({
        "actual_date":          str(row["date_only"]),
        "cluster_id":           cluster_id,
        "representative_date":  rep["representative_date"],
        "weight":               rep["weight"],
    })

df_mapping = pd.DataFrame(mapping_rows).sort_values("actual_date").reset_index(drop=True)
mapping_path = os.path.join(final_dir, 'day_cluster_assignment.csv')
df_mapping.to_csv(mapping_path, index=False)
print(f"Day-cluster mapping exported to: {mapping_path}  ({len(df_mapping)} rows)")

# Output B — full-year profiles (366 × 48 = 17 568 rows)
# For each actual date, replicate the representative profile and substitute the date columns.
rep_profiles = df_rep.set_index("date_only")   # index on representative date_only (as date object)

expanded_chunks = []
for _, map_row in df_mapping.iterrows():
    actual_date = map_row["actual_date"]         # string "YYYY-MM-DD"
    rep_date    = map_row["representative_date"] # string "YYYY-MM-DD"

    # Extract 48 half-hours for this representative date
    chunk = df_rep[df_rep["date_only"].astype(str) == rep_date].copy()

    # Substitute date columns so the chunk appears as the actual date
    actual_dt = pd.to_datetime(actual_date)
    chunk["actual_date"] = actual_date
    chunk["date"]        = actual_dt.strftime("%d/%m/%Y")
    chunk["date_only"]   = actual_dt.date()
    chunk["datetime"]    = pd.to_datetime(
        actual_date + " " + chunk["heures"].astype(str)
    )
    chunk["month"]       = actual_dt.month

    expanded_chunks.append(chunk)

df_expanded = pd.concat(expanded_chunks, ignore_index=True)

# Aggregate to hourly: max of all columns per (actual_date, hour)
df_expanded["hour"] = df_expanded["heures"].str[:2].astype(int)

skip = {"hour", "heures", "datetime", "actual_date"}
numeric_cols = [c for c in df_expanded.columns
                if c not in skip and pd.api.types.is_numeric_dtype(df_expanded[c])]
categ_cols   = [c for c in df_expanded.columns
                if c not in skip and not pd.api.types.is_numeric_dtype(df_expanded[c])]

agg_dict = {c: "max"   for c in numeric_cols}
agg_dict.update({c: "first" for c in categ_cols})

df_hourly = (df_expanded
             .groupby(["actual_date", "hour"], sort=True)
             .agg(agg_dict)
             .reset_index())

df_hourly["heures"]   = df_hourly["hour"] + 1   # 1 = 00:00-01:00, 24 = 23:00-24:00
df_hourly["datetime"] = pd.to_datetime(df_hourly["actual_date"]) + pd.to_timedelta(df_hourly["hour"], unit="h")
df_hourly = df_hourly.drop(columns=["hour"])

# Reorder: actual_date first
cols = ["actual_date", "heures"] + [c for c in df_hourly.columns if c not in ("actual_date", "heures")]
df_hourly = df_hourly[cols]

expanded_path = os.path.join(final_dir, 'full_year_assigned_profiles.csv')
df_hourly.to_csv(expanded_path, index=False)
print(f"Full-year profiles exported to : {expanded_path}  ({len(df_hourly)} rows, hourly)")
