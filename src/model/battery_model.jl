# Capacity expansion model with seasonal battery storage (Kotzur et al. 2018) and transmission.
# Inputs: data/final/; Output: printed results.

using JuMP, CSV, DataFrames, Printf, Dates, Clp

# =============================================================================
# Battery model with seasonal storage — Kotzur et al. (2018)
#
# The battery state of charge is decomposed into two layers:
#   SOC(i, t) = e_inter[r, i] + e_intra[r, d=f(i), t]
#
#   e_intra[r, d, t] : intra-period SOC accumulated within typical day d
#                      starts at 0 each day (e_intra[r,d,0] = 0)
#                      can be negative (net discharge so far in the day)
#
#   e_inter[r, i]    : inter-period SOC at the start of actual day i (1:N)
#                      carries energy across days, enables seasonal storage
#
# Cyclic annual condition:
#   Σ_d cluster_size[d] * e_intra[r, d, T] = 0
#   (net annual change of the battery must be zero)
#
# Capacity constraints use auxiliary max/min per typical day:
#   e_inter[r,i] + e_intra_max[r, f(i)] ≤ x_bat[r]   ∀ i
#   e_inter[r,i] + e_intra_min[r, f(i)] ≥ 0           ∀ i
# This gives N constraints instead of N×T, reducing model size.
# =============================================================================

# =============================================================================
# Paths
# =============================================================================
const DATA_FINAL = joinpath(@__DIR__, "..", "..", "data", "final")

# =============================================================================
# Load data
# =============================================================================
gen_df      = CSV.read(joinpath(DATA_FINAL, "gen_reduced_days.csv"),    DataFrame; delim=';')
pv_df       = CSV.read(joinpath(DATA_FINAL, "ninja_pv_new_representative_days_regions.csv"),   DataFrame)
wind_df     = CSV.read(joinpath(DATA_FINAL, "ninja_wind_new_representative_days_regions.csv"), DataFrame)
cap_df      = CSV.read(joinpath(DATA_FINAL, "capacites_interregionales.csv"), DataFrame)
clusters_df = CSV.read(joinpath(DATA_FINAL, "day_cluster_assignment.csv"), DataFrame)

# =============================================================================
# Sets
# =============================================================================
regions = sort(unique(gen_df.Région))
dates   = sort(unique(gen_df.Date))

R = length(regions)        # 12 regions
D = length(dates)          # 10 typical days
T = 24                     # hours per day
N = nrow(clusters_df)      # 366 actual days (2024 is a leap year)

region_idx = Dict(r => i for (i, r) in enumerate(regions))
date_idx   = Dict(d => i for (i, d) in enumerate(dates))

# For each actual day i (1:N): index of its typical day in dates (1:D)
day_to_typday = [date_idx[string(row.representative_date)] for row in eachrow(clusters_df)]

# Number of actual days assigned to each typical day
cluster_size = zeros(Int, D)
for d in day_to_typday
    cluster_size[d] += 1
end


# Interregional capacities
cap = Dict(
    (region_idx[row.Region_A], region_idx[row.Region_B]) => Float64(row.Capacite_MW_total)
    for row in eachrow(cap_df)
    if haskey(region_idx, row.Region_A) && haskey(region_idx, row.Region_B)
)
pairs     = collect(keys(cap))
pairs_in  = [[(rA,rB) for (rA,rB) in pairs if rB == r] for r in 1:R]
pairs_out = [[(rA,rB) for (rA,rB) in pairs if rA == r] for r in 1:R]

# =============================================================================
# Parameters — arrays indexed [r, d, t]
# =============================================================================
demand      = zeros(R, D, T)
gen_reduced = zeros(R, D, T)
cf_solar    = zeros(R, D, T)
cf_wind     = zeros(R, D, T)

for row in eachrow(gen_df)
    r = region_idx[row.Région]
    d = date_idx[row.Date]
    t = Dates.hour(row.Heure) + 1
    demand[r, d, t]      = coalesce(row[Symbol("Consommation (MW)")], 0.0)
    gen_reduced[r, d, t] = coalesce(row.gen_reduced_MW, 0.0)
end

demand .*= 1.08  # +8% demand growth scenario

for i in 1:nrow(pv_df)
    time_str = string(pv_df.time[i])
    date_str = time_str[1:10]
    utc_hour = parse(Int, time_str[12:13])
    !haskey(date_idx, date_str) && continue
    d = date_idx[date_str]
    t = utc_hour + 1
    for (r_name, r) in region_idx
        cf_solar[r, d, t] = coalesce(pv_df[i,   Symbol(r_name)], 0.0)
        cf_wind[r, d, t]  = coalesce(wind_df[i, Symbol(r_name)], 0.0)
    end
end

# =============================================================================
# Costs
# =============================================================================
c_solar = 84207.18832    # €/MW/yr
c_wind  = 170873.2996    # €/MW/yr
c_bat   = 25529.34076    # €/MWh/yr

# =============================================================================
# Model
# =============================================================================
model = Model(Clp.Optimizer)
set_silent(model)

# --- Capacity decision variables ---
@variable(model, x_solar[1:R] >= 0)   # new solar capacity [MW]
@variable(model, x_wind[1:R]  >= 0)   # new wind capacity  [MW]
@variable(model, x_bat[1:R]   >= 0)   # new battery energy capacity [MWh]

# --- Dispatch variables (indexed by typical day d) ---
@variable(model, b_charge[1:R, 1:D, 1:T]    >= 0)   # charging power    [MW]
@variable(model, b_discharge[1:R, 1:D, 1:T] >= 0)   # discharging power [MW]
@variable(model, flow[pairs, 1:D, 1:T])              # interregional flow [MW]

# --- Kotzur battery SOC variables ---

# Intra-period SOC: accumulated within a typical day, starts at 0
# No lower bound: can be negative (net discharge in first hours of day)
@variable(model, e_intra[1:R, 1:D, 0:T])

# Auxiliary max/min intra-period SOC per typical day (for capacity constraints)
@variable(model, e_intra_max[1:R, 1:D])
@variable(model, e_intra_min[1:R, 1:D])

# Inter-period SOC: energy carried from one actual day to the next
@variable(model, e_inter[1:R, 1:N] >= 0)

# =============================================================================
# Objective: minimise total annualised investment cost
# =============================================================================
@objective(model, Min,
    sum(c_solar * x_solar[r] +
        c_wind  * x_wind[r]  +
        c_bat   * x_bat[r]
        for r in 1:R)
)

# =============================================================================
# Constraints
# =============================================================================

for r in 1:R, d in 1:D

    # Intra-period SOC starts at 0 at the beginning of each typical day
    @constraint(model, e_intra[r, d, 0] == 0)

    for t in 1:T

        # (1) Energy balance
        @constraint(model,
            gen_reduced[r, d, t]
            + cf_solar[r, d, t] * x_solar[r]
            + cf_wind[r, d, t]  * x_wind[r]
            + b_discharge[r, d, t]
            - b_charge[r, d, t]
            + sum(flow[p, d, t] for p in pairs_in[r];  init=AffExpr(0))
            - sum(flow[p, d, t] for p in pairs_out[r]; init=AffExpr(0))
            >= demand[r, d, t]
        )

        # (2) Intra-period SOC evolution
        @constraint(model,
            e_intra[r, d, t] == e_intra[r, d, t-1] + b_charge[r, d, t] - b_discharge[r, d, t]
        )

        # (3) Auxiliary max/min tracking for capacity constraints
        @constraint(model, e_intra[r, d, t] <= e_intra_max[r, d])
        @constraint(model, e_intra[r, d, t] >= e_intra_min[r, d])

        # (4) Power capacity — 4-hour battery (max power = capacity / 4)
        @constraint(model, b_charge[r, d, t]    <= x_bat[r] / 4.0)
        @constraint(model, b_discharge[r, d, t] <= x_bat[r] / 4.0)
    end
end

# (5) Cyclic annual condition: net SOC change over the year = 0
#     Σ_d cluster_size[d] × e_intra[r, d, T] = 0  ∀ r
for r in 1:R
    @constraint(model,
        sum(cluster_size[d] * e_intra[r, d, T] for d in 1:D) == 0
    )
end

# (6) Inter-period SOC evolution across actual days
#     e_inter[r, i+1] = e_inter[r, i] + e_intra[r, f(i), T]
for r in 1:R, i in 1:N-1
    @constraint(model,
        e_inter[r, i+1] == e_inter[r, i] + e_intra[r, day_to_typday[i], T]
    )
end

# (7) Capacity constraints via auxiliary min/max (N constraints per region,
#     instead of N×T — efficiency gain from Appendix B of Kotzur et al.)
for r in 1:R, i in 1:N
    d = day_to_typday[i]
    @constraint(model, e_inter[r, i] + e_intra_max[r, d] <= x_bat[r])
    @constraint(model, e_inter[r, i] + e_intra_min[r, d] >= 0)
end

# (8) Interregional transmission capacity
for (rA, rB) in pairs, d in 1:D, t in 1:T
    @constraint(model, -cap[(rA,rB)] <= flow[(rA,rB), d, t] <= cap[(rA,rB)])
end

# =============================================================================
# Solve
# =============================================================================
println("\nSolving...")
optimize!(model)

# =============================================================================
# Results
# =============================================================================
println("\n=== Optimal Capacity Expansion — Seasonal Battery (Kotzur) + 8% Demand ===")
println("Status    : ", termination_status(model))
println("Objective : ", round(objective_value(model) / 1e6, digits=1), " M€/yr")
println()
@printf("%-32s %10s %10s %14s\n", "Region", "Solar(MW)", "Wind(MW)", "Battery(MWh)")
println("-" ^ 70)
for r_name in regions
    r = region_idx[r_name]
    @printf("%-32s %10.1f %10.1f %14.1f\n",
        r_name,
        value(x_solar[r]),
        value(x_wind[r]),
        value(x_bat[r]))
end
println("-" ^ 70)
@printf("%-32s %10.1f %10.1f %14.1f\n", "TOTAL",
    sum(value(x_solar[r]) for r in 1:R),
    sum(value(x_wind[r])  for r in 1:R),
    sum(value(x_bat[r])   for r in 1:R))

# Inter-period SOC range per region (shows seasonal storage usage)
println("\nBattery seasonal swing (e_inter max - min) per region:")
for r_name in regions
    r = region_idx[r_name]
    inter_vals = [value(e_inter[r, i]) for i in 1:N]
    swing = maximum(inter_vals) - minimum(inter_vals)
    @printf("  %-32s  swing = %8.0f MWh  (min=%8.0f  max=%8.0f)\n",
        r_name, swing, minimum(inter_vals), maximum(inter_vals))
end
