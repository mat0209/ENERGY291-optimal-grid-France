"""
K-Medoids: Find one representative day per month (12 days for 2024).
Each day is represented by ALL numeric columns (48 half-hourly steps × n_features),
flattened into a single vector for distance computation.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── 1. Data Loading ────────────────────────────────────────────────────────────
df = pd.read_csv("data\\processed\\mix_rte_2024_clean.csv", sep=",", decimal=".")

df["datetime"] = pd.to_datetime(df["date"] + " " + df["heures"], dayfirst=True)
df = df.sort_values("datetime").reset_index(drop=True)
df["date_only"] = df["datetime"].dt.date
df["month"]     = df["datetime"].dt.month

# ── 2. Select All Numeric Feature Columns ─────────────────────────────────────
EXCLUDE = {"date_only", "month", "datetime"}
numeric_cols = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in EXCLUDE
]
print(f"Features used ({len(numeric_cols)}): {numeric_cols}")

# ── 3. Build Daily Profiles (48 steps × all features → flat vector) ───────────
def build_profiles(data, cols):
    """
    For each day, stack the 48 half-hourly rows of `cols` into a flat vector.
    Returns a DataFrame with date_only, month, and a 'profile' list column.
    """
    rows = []
    for (date, month), grp in data.groupby(["date_only", "month"]):
        grp_sorted = grp.sort_values("datetime")
        if len(grp_sorted) != 48:
            continue
        # Shape: (48, n_features) → flatten to (48 * n_features,)
        vec = grp_sorted[cols].values.flatten().tolist()
        rows.append({"date_only": date, "month": month, "profile": vec})
    return pd.DataFrame(rows).reset_index(drop=True)

daily_profiles = build_profiles(df, numeric_cols)
profile_matrix = np.array(daily_profiles["profile"].tolist(), dtype=float)  # (n_days, 48*n_feat)

print(f"Complete days found : {len(daily_profiles)}")
print(f"Vector length per day: {profile_matrix.shape[1]}  (48 steps × {len(numeric_cols)} features)")

# ── 4. Normalise Each Feature Across All Days ─────────────────────────────────
# Z-score per feature-slot so no single large-magnitude column dominates
col_means = profile_matrix.mean(axis=0)
col_stds  = profile_matrix.std(axis=0)
col_stds[col_stds == 0] = 1  # avoid div-by-zero for constant columns
profile_matrix_norm = (profile_matrix - col_means) / col_stds

# ── 5. K-Medoids (PAM) ────────────────────────────────────────────────────────
def k_medoids_euclidean(X, k, max_iter=300, random_state=42):
    """
    PAM-style K-Medoids with Euclidean distance.
    Returns (medoid_indices, cluster_assignments).
    """
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

# ── 6. One K-Medoids per Month ────────────────────────────────────────────────
results = []
MONTH_NAMES = ["January","February","March","April","May","June",
               "July","August","September","October","November","December"]

for month in range(1, 13):
    mask = daily_profiles["month"] == month
    idx  = daily_profiles.index[mask].tolist()
    X_m  = profile_matrix_norm[idx]

    print(f"\nMonth {month:02d} ({MONTH_NAMES[month-1]}) — {len(idx)} days")

    medoid_local, _ = k_medoids_euclidean(X_m, k=1)
    medoid_global   = idx[medoid_local[0]]

    row = daily_profiles.iloc[medoid_global]

    # Recover raw consumption values for summary stats
    raw = profile_matrix[medoid_global]                          # flat raw vector
    conso_idx = [i * len(numeric_cols) + numeric_cols.index("consommation")
                 for i in range(48)]
    conso_vals = raw[conso_idx]

    results.append({
        "month":               month,
        "month_name":          MONTH_NAMES[month - 1],
        "representative_date": str(row["date_only"]),
        "avg_consumption_MW":  round(float(conso_vals.mean()), 1),
        "min_consumption_MW":  round(float(conso_vals.min()),  1),
        "max_consumption_MW":  round(float(conso_vals.max()),  1),
    })

# ── 7. Print Results ──────────────────────────────────────────────────────────
print("\n" + "═" * 72)
print("  REPRESENTATIVE DAYS 2024  (k-medoids — ALL features, z-score norm)")
print("═" * 72)
print(f"  {'Month':<12} {'Date':<14} {'Avg (MW)':>12} {'Min (MW)':>10} {'Max (MW)':>10}")
print("─" * 72)
for r in results:
    print(f"  {r['month_name']:<12} {r['representative_date']:<14} "
          f"{r['avg_consumption_MW']:>12,.0f} "
          f"{r['min_consumption_MW']:>10,.0f} "
          f"{r['max_consumption_MW']:>10,.0f}")
print("═" * 72)

# ── 8. Export Summary CSV ─────────────────────────────────────────────────────
out = pd.DataFrame(results)
out.to_csv("representative_days_all_features_2024.csv", index=False)
print("\nSummary exported to: representative_days_all_features_2024.csv")

# ── 9. Export Detailed Profiles of the 12 Representative Days ─────────────────
rep_dates = [r["representative_date"] for r in results]
df_rep = df[df["date_only"].astype(str).isin(rep_dates)].copy()
df_rep.to_csv("representative_day_profiles_all_features_2024.csv", index=False)
print("Detailed profiles exported to: representative_day_profiles_all_features_2024.csv")

# ── 10. Comparison table: consumption-only vs all-features ────────────────────
prev_dates = {
    1: "2024-01-08", 2: "2024-02-05", 3: "2024-03-11", 4: "2024-04-15",
    5: "2024-05-13", 6: "2024-06-10", 7: "2024-07-08", 8: "2024-08-12",
    9: "2024-09-09", 10: "2024-10-14", 11: "2024-11-11", 12: "2024-12-09",
}  # placeholder — will be overwritten if comparison CSV exists

try:
    prev = pd.read_csv("representative_days_2024.csv")
    prev_map = dict(zip(prev["month"], prev["representative_date"]))
    print("\n" + "─" * 60)
    print("  Comparison: consumption-only vs all-features")
    print("─" * 60)
    print(f"  {'Month':<12} {'Conso-only':<14} {'All-features':<14} {'Same?'}")
    print("─" * 60)
    for r in results:
        m = r["month"]
        d_prev = prev_map.get(m, "N/A")
        d_new  = r["representative_date"]
        same   = "✓" if d_prev == d_new else "✗ differs"
        print(f"  {r['month_name']:<12} {d_prev:<14} {d_new:<14} {same}")
    print("─" * 60)
except FileNotFoundError:
    print("\n(No previous consumption-only results found for comparison.)")