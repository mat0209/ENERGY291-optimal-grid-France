# Capacity expansion model: no fossil (thermal) generation — all gas/oil removed from gen_reduced.
# Otherwise identical to final_model.jl (Kotzur seasonal battery, endogenous transmission, +8% demand).
# Inputs: data/final/; Output: data/results/ and figures/results/.

using JuMP, CSV, DataFrames, Printf, Dates, Plots, HiGHS

# =============================================================================
# Paths
# =============================================================================
const DATA = joinpath(@__DIR__, "..", "..", "data", "final")
const DATA_results = joinpath(@__DIR__, "..", "..", "data", "results")

# =============================================================================
# Load data
# =============================================================================
gen_df  = CSV.read(joinpath(DATA, "gen_reduced_days.csv"),
                   DataFrame; delim=';')
pv_df   = CSV.read(joinpath(DATA, "ninja_pv_new_representative_days_regions.csv"),
                   DataFrame)
wind_df = CSV.read(joinpath(DATA, "ninja_wind_new_representative_days_regions.csv"),
                   DataFrame)
cap_df  = CSV.read(joinpath(@__DIR__, "..", "..", "data", "final", "capacites_interregionales.csv"), DataFrame)

# =============================================================================
# Sets
# =============================================================================
regions = sort(unique(gen_df.Région))   # R = 12 administrative regions
dates   = sort(unique(gen_df.Date))     # D = 10 representative days

R = length(regions)   # 12
D = length(dates)     # 10
T = 24                # hours per day

region_idx = Dict(r => i for (i, r) in enumerate(regions))
date_idx   = Dict(string(d) => i for (i, d) in enumerate(dates))

# Interregional capacities: (rA, rB) => cap_MW (one direction per CSV pair)
cap = Dict(
    (region_idx[row.Region_A], region_idx[row.Region_B]) => Float64(row.Capacite_MW_total)
    for row in eachrow(cap_df)
    if haskey(region_idx, row.Region_A) && haskey(region_idx, row.Region_B)
)
dist = Dict(
    (region_idx[row.Region_A], region_idx[row.Region_B]) => Float64(row.Distance_km_adjusted)
    for row in eachrow(cap_df)
    if haskey(region_idx, row.Region_A) && haskey(region_idx, row.Region_B)
)
pairs = collect(keys(cap))

# For each region: pairs where it is the import (in) or export (out) endpoint
pairs_in  = [[(rA,rB) for (rA,rB) in pairs if rB == r] for r in 1:R]
pairs_out = [[(rA,rB) for (rA,rB) in pairs if rA == r] for r in 1:R]

# =============================================================================
# Parameters — arrays indexed [r, d, t], t ∈ 1:24
# =============================================================================
demand      = zeros(R, D, T)   # D_{r,d,t}  [MW]
gen_reduced = zeros(R, D, T)   # G̃_{r,d,t}  [MW]  — fossil excluded
cf_solar    = zeros(R, D, T)   # CF^solar_{r,d,t}  ∈ [0,1]
cf_wind     = zeros(R, D, T)   # CF^wind_{r,d,t}   ∈ [0,1]

# --- Demand and reduced generation (eco2mix, local French time) ---
# gen_reduced = gen_reduced_MW − Thermique: removes all fossil (gas/oil) production
for row in eachrow(gen_df)
    r = region_idx[row.Région]
    d = date_idx[string(row.Date)]
    t = Dates.hour(row.Heure) + 1   # Time(0,0)→1, Time(23,0)→24
    demand[r, d, t]      = coalesce(row[Symbol("Consommation (MW)")], 0.0)
    thermal_MW           = coalesce(row[Symbol("Thermique (MW)")], 0.0)
    gen_reduced[r, d, t] = coalesce(row.gen_reduced_MW, 0.0) - thermal_MW
end

demand .*= 1.08  # +8% demand growth scenario

# --- Capacity factors (Ninja timestamps are UTC, matched by date and UTC hour) ---
for i in 1:nrow(pv_df)
    time_str = string(pv_df.time[i])
    date_str = time_str[1:10]
    utc_hour = parse(Int, time_str[12:13])

    !haskey(date_idx, date_str) && continue
    d = date_idx[date_str]
    t = utc_hour + 1   # 1-indexed, t ∈ 1:24
    for (r_name, r) in region_idx
        cf_solar[r, d, t] = coalesce(pv_df[i,   Symbol(r_name)], 0.0)
        cf_wind[r, d, t]  = coalesce(wind_df[i, Symbol(r_name)], 0.0)
    end
end

# =============================================================================
# Cluster assignment for Kotzur seasonal battery model
# =============================================================================
cluster_df = CSV.read(joinpath(@__DIR__, "..", "..", "data", "final", "day_cluster_assignment.csv"), DataFrame)
sort!(cluster_df, :actual_date)

N = nrow(cluster_df)   # 366 actual days

# For each actual day i (1:N) in chronological order: index of its typical day (1:D)
day_to_typday = [date_idx[string(row.representative_date)] for row in eachrow(cluster_df)]

# Number of actual days assigned to each typical day
cluster_size = zeros(Int, D)
for d in day_to_typday
    cluster_size[d] += 1
end

# =============================================================================
# Costs
# =============================================================================
c_solar     = 84207.18832   # €/MW/yr     — onshore solar PV  (annualized)
c_wind      = 170873.2996   # €/MW/yr     — onshore wind       (annualized)
c_bat       = 25529.34076   # €/MWh/yr    — 4-hour Li-ion battery (energy capacity)
c_trans_400 = 61.09615194   # €/MW/km/yr  — 400 kV transmission lines (annualized)
c_trans_225 = 101.8269199   # €/MW/km/yr  — 225 kV transmission lines (annualized)

cap_trans_400 = 1500.0  # MW per 400 kV transmission line
cap_trans_225 = 400.0   # MW per 225 kV transmission line

# =============================================================================
# Model
# =============================================================================
model = Model(HiGHS.Optimizer)
set_silent(model)

# --- Decision variables ---
@variable(model, x_solar[1:R] >= 0)                     # new solar capacity    [MW]
@variable(model, x_wind[1:R]  >= 0)                     # new wind capacity     [MW]
@variable(model, x_bat[1:R]   >= 0)                     # new battery capacity  [MWh]
@variable(model, y_trans_400[pairs] >= 0, Int)           # new 400 kV transmission lines
@variable(model, y_trans_225[pairs] >= 0, Int)           # new 225 kV transmission lines

@variable(model, b_charge[1:R, 1:D, 1:T]    >= 0)  # battery charging         [MW]
@variable(model, b_discharge[1:R, 1:D, 1:T] >= 0)  # battery discharging      [MW]
@variable(model, flow[pairs, 1:D, 1:T])             # flow rA→rB, >0 = export  [MW]

# --- Kotzur seasonal battery SOC (Kotzur et al. 2018) ---
@variable(model, e_intra[1:R, 1:D, 0:T])           # intra-period SOC, starts at 0, can be negative
@variable(model, e_intra_max[1:R, 1:D])            # aux: max of e_intra over t per typical day
@variable(model, e_intra_min[1:R, 1:D])            # aux: min of e_intra over t per typical day
@variable(model, e_inter[1:R, 1:N] >= 0)           # inter-period SOC at start of each actual day

# --- Objective: minimise total annualised investment cost ---
@objective(model, Min,
    sum(c_solar * x_solar[r] + c_wind * x_wind[r] + c_bat * x_bat[r]
    for r in 1:R)
    + sum(y_trans_400[p] * c_trans_400 * dist[p] * cap_trans_400
    + y_trans_225[p] * c_trans_225 * dist[p] * cap_trans_225 for p in pairs)
)

# --- Constraints ---
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

        # (3) Auxiliary max/min tracking (used for capacity constraints)
        @constraint(model, e_intra[r, d, t] <= e_intra_max[r, d])
        @constraint(model, e_intra[r, d, t] >= e_intra_min[r, d])

        # (4) Power capacity — 4-hour battery
        @constraint(model, b_charge[r, d, t]    <= x_bat[r] / 4.0)
        @constraint(model, b_discharge[r, d, t] <= x_bat[r] / 4.0)
    end
end

# (5) Cyclic annual condition: Σ_d cluster_size[d] × e_intra[r,d,T] = 0
for r in 1:R
    @constraint(model, sum(cluster_size[d] * e_intra[r, d, T] for d in 1:D) == 0)
end

# (6) Inter-period SOC evolution — follows actual chronological order of 366 days
for r in 1:R, i in 1:N-1
    @constraint(model, e_inter[r, i+1] == e_inter[r, i] + e_intra[r, day_to_typday[i], T])
end

# (7) Capacity constraints via aux min/max — N constraints instead of N×T (Kotzur Appendix B)
for r in 1:R, i in 1:N
    d = day_to_typday[i]
    @constraint(model, e_inter[r, i] + e_intra_max[r, d] <= x_bat[r])
    @constraint(model, e_inter[r, i] + e_intra_min[r, d] >= 0)
end

# (8) Interregional transmission capacities
for (rA, rB) in pairs
    cap_new = cap[(rA,rB)] + y_trans_400[(rA,rB)] * cap_trans_400 + y_trans_225[(rA,rB)] * cap_trans_225
    for d in 1:D, t in 1:T
        @constraint(model, flow[(rA,rB), d, t] <=  cap_new)
        @constraint(model, flow[(rA,rB), d, t] >= -cap_new)
    end
end

# =============================================================================
# Solve
# =============================================================================
optimize!(model)

# =============================================================================
# Results
# =============================================================================
println("\n=== Optimal Capacity Expansion — No Fossil Generation + Limited Transmission + 8% Demand Growth ===")
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

println()
println("=== New transmission lines ===")
@printf("%-35s %-35s %10s %10s %12s %12s\n",
    "Region A", "Region B", "400kV(nb)", "225kV(nb)", "Cap400(MW)", "Cap225(MW)")
println("-" ^ 90)
region_name = Dict(v => k for (k, v) in region_idx)
trans_built = false
for p in sort(pairs)
    n400 = round(Int, value(y_trans_400[p]))
    n225 = round(Int, value(y_trans_225[p]))
    if n400 > 0 || n225 > 0
        @printf("%-35s %-35s %10d %10d %12.0f %12.0f\n",
            region_name[p[1]], region_name[p[2]],
            n400, n225,
            n400 * cap_trans_400, n225 * cap_trans_225)
        trans_built = true
    end
end
trans_built || println("  No new lines built.")
println("-" ^ 90)

# --- Plot capacity results by region (stacked bar chart) ---
fig_dir = joinpath(@__DIR__, "..", "..", "figures", "results")
mkpath(fig_dir)
regions_plot = collect(regions)
solar_vals = [value(x_solar[r])        for r in 1:R]
wind_vals  = [value(x_wind[r])         for r in 1:R]
bat_vals   = [value(x_bat[r]) / 4.0   for r in 1:R]   # MWh → MW (4h battery)

bar(regions_plot, solar_vals,
    label = "Solar (MW)",
    bar_width = 0.7,
    color = "#f28e2b",
    xlabel = "Region",
    ylabel = "Capacity Added (MW)",
    title = "Optimal Capacity Expansion by Region — No Fossil",
    legend = :topright,
    xrotation = 45)
bar!(regions_plot, wind_vals,
    label = "Wind (MW)",
    color = "#59a14f",
    bottom = solar_vals)
bar!(regions_plot, bat_vals,
    label = "Battery (MW)",
    color = "#7b5ea6",
    bottom = solar_vals .+ wind_vals)

output_path = joinpath(fig_dir, "capacity_results_no_fossil.png")
savefig(output_path)
println("Saved plot to ", output_path)

# --- Save capacity results to CSV ---
results = DataFrame(
    Région = regions,
    Solar_MW = [value(x_solar[r]) for r in 1:R],
    Wind_MW = [value(x_wind[r]) for r in 1:R],
    Battery_MWh = [value(x_bat[r]) for r in 1:R],
)
CSV.write(joinpath(DATA_results, "capacity_results_no_fossil.csv"), results; delim=';')
println("Saved results to ", joinpath(DATA_results, "capacity_results_no_fossil.csv"))

# --- Save transmission results to CSV ---
trans_results = DataFrame(
    Region_A   = [region_name[p[1]] for p in pairs],
    Region_B   = [region_name[p[2]] for p in pairs],
    Dist_km    = [dist[p]           for p in pairs],
    Cap_exist_MW = [cap[p]          for p in pairs],
    New_lines_400kV = [round(Int, value(y_trans_400[p])) for p in pairs],
    New_lines_225kV = [round(Int, value(y_trans_225[p])) for p in pairs],
    New_cap_400_MW  = [round(Int, value(y_trans_400[p])) * cap_trans_400 for p in pairs],
    New_cap_225_MW  = [round(Int, value(y_trans_225[p])) * cap_trans_225 for p in pairs],
    Total_cap_MW    = [cap[p] + round(Int, value(y_trans_400[p])) * cap_trans_400 + round(Int, value(y_trans_225[p])) * cap_trans_225 for p in pairs],
)
CSV.write(joinpath(DATA_results, "transmission_results_no_fossil.csv"), trans_results; delim=';')
println("Saved transmission results to ", joinpath(DATA_results, "transmission_results_no_fossil.csv"))

open(joinpath(DATA_results, "cost_no_fossil.txt"), "w") do f
    write(f, string(round(objective_value(model) / 1e6, digits=2)))
end
