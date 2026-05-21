using JuMP, Clp, CSV, DataFrames, Printf, Dates

# =============================================================================
# Paths
# =============================================================================
const DATA = joinpath(@__DIR__, "..", "..", "data", "processed")

# =============================================================================
# Load data
# =============================================================================
gen_df  = CSV.read(joinpath(DATA, "gen_reduced_days_2024.csv"),
                   DataFrame; delim=';')
pv_df   = CSV.read(joinpath(DATA, "ninja_pv_representative_days_2024_regions.csv"),
                   DataFrame)
wind_df = CSV.read(joinpath(DATA, "ninja_wind_representative_days_2024_regions.csv"),
                   DataFrame)

# =============================================================================
# Sets
# =============================================================================
regions = sort(unique(gen_df.Région))   # R = 12 administrative regions
dates   = sort(unique(gen_df.Date))     # D = 12 representative days

R = length(regions)   # 12
D = length(dates)     # 12
T = 24                # hours per day

region_idx = Dict(r => i for (i, r) in enumerate(regions))
date_idx   = Dict(d => i for (i, d) in enumerate(dates))

# =============================================================================
# Parameters — arrays indexed [r, d, t], t ∈ 1:24
# =============================================================================
demand      = zeros(R, D, T)   # D_{r,d,t}  [MW]
gen_reduced = zeros(R, D, T)   # G̃_{r,d,t}  [MW]
cf_solar    = zeros(R, D, T)   # CF^solar_{r,d,t}  ∈ [0,1]
cf_wind     = zeros(R, D, T)   # CF^wind_{r,d,t}   ∈ [0,1]

# --- Demand and reduced generation (eco2mix, local French time) ---
for row in eachrow(gen_df)
    r = region_idx[row.Région]
    d = date_idx[row.Date]
    t = Dates.hour(row.Heure) + 1   # Time(0,0)→1, Time(23,0)→24
    demand[r, d, t]      = coalesce(row[Symbol("Consommation (MW)")], 0.0)
    gen_reduced[r, d, t] = coalesce(row.gen_reduced_MW, 0.0)
end

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
# Costs
# =============================================================================
c_solar = 84207.18832   # €/MW/yr   — onshore solar PV  (annualised CAPEX)
c_wind  = 170873.2996   # €/MW/yr   — onshore wind       (annualised CAPEX)
c_bat   = 23522.78488   # €/MWh/yr  — 4-hour Li-ion battery (energy capacity)

# =============================================================================
# Model
# =============================================================================
model = Model(Clp.Optimizer)
set_silent(model)

# --- Decision variables ---
@variable(model, x_solar[1:R] >= 0)               # new solar capacity    [MW]
@variable(model, x_wind[1:R]  >= 0)               # new wind capacity     [MW]
@variable(model, x_bat[1:R]   >= 0)               # new battery capacity  [MWh]

@variable(model, b_charge[1:R, 1:D, 1:T]    >= 0) # battery charging      [MW]
@variable(model, b_discharge[1:R, 1:D, 1:T] >= 0) # battery discharging   [MW]
@variable(model, e[1:R, 1:D, 0:T]           >= 0) # state of charge       [MWh]
@variable(model, flow[1:R, 1:D, 1:T])             # net import per region [MW], free (no limit)

# --- Objective: minimise total annualised investment cost ---
@objective(model, Min,
    sum(c_solar * x_solar[r] +
        c_wind  * x_wind[r]  +
        c_bat   * x_bat[r]
        for r in 1:R)
)

# --- Constraints ---
for r in 1:R, d in 1:D

    # (4) Battery initialisation: each representative day starts at 50% SoC
    @constraint(model, e[r, d, 0] == 0.0 * x_bat[r])

    for t in 1:T

        # (1) Energy balance with net import flow (no transmission limit)
        @constraint(model,
            gen_reduced[r, d, t]
            + cf_solar[r, d, t] * x_solar[r]
            + cf_wind[r, d, t]  * x_wind[r]
            + b_discharge[r, d, t]
            - b_charge[r, d, t]
            + flow[r, d, t]
            >= demand[r, d, t]
        )

        # (3) Battery state of charge evolution
        @constraint(model,
            e[r, d, t] == e[r, d, t-1] + b_charge[r, d, t] - b_discharge[r, d, t]
        )

        # (5) Battery energy capacity
        @constraint(model, e[r, d, t] <= x_bat[r])

        # (6) Battery power capacity — 4-hour battery: max power = capacity / 4
        @constraint(model, b_charge[r, d, t]    <= x_bat[r] / 4.0)
        @constraint(model, b_discharge[r, d, t] <= x_bat[r] / 4.0)
    end
end

# Flow conservation: net imports sum to zero across all regions at each (d,t)
for d in 1:D, t in 1:T
    @constraint(model, sum(flow[r, d, t] for r in 1:R) == 0)
end


# =============================================================================
# Solve
# =============================================================================
optimize!(model)

# =============================================================================
# Results
# =============================================================================
println("\n=== Optimal Capacity Expansion — With Unlimited Transmission ===")
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
