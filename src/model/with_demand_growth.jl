using JuMP, Clp, CSV, DataFrames, Printf, Dates, Plots

# =============================================================================
# Paths
# =============================================================================
const DATA = joinpath(@__DIR__, "..", "..", "data", "processed")
const DATA_results = joinpath(@__DIR__, "..", "..", "data", "results")

# =============================================================================
# Load data
# =============================================================================
gen_df  = CSV.read(joinpath(DATA, "gen_reduced_days_2024.csv"),
                   DataFrame; delim=';')
pv_df   = CSV.read(joinpath(DATA, "ninja_pv_representative_days_2024_regions.csv"),
                   DataFrame)
wind_df = CSV.read(joinpath(DATA, "ninja_wind_representative_days_2024_regions.csv"),
                   DataFrame)
cap_df  = CSV.read(joinpath(DATA, "capacites_interregionales.csv"), DataFrame)

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

# Capacités interrégionales : (rA, rB) => cap_MW  (une seule direction par paire du CSV)
cap = Dict(
    (region_idx[row.Region_A], region_idx[row.Region_B]) => Float64(row.Capacite_MW_total)
    for row in eachrow(cap_df)
    if haskey(region_idx, row.Region_A) && haskey(region_idx, row.Region_B)
)
pairs = collect(keys(cap))

# Pour chaque région : paires dont elle est l'extrémité import (in) ou export (out)
pairs_in  = [[(rA,rB) for (rA,rB) in pairs if rB == r] for r in 1:R]
pairs_out = [[(rA,rB) for (rA,rB) in pairs if rA == r] for r in 1:R]

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

demand .*= 1.08   # scénario +8% de consommation

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
@variable(model, flow[pairs, 1:D, 1:T])            # flow (rA→rB) > 0 = export de rA vers rB [MW]

# --- Objective: minimise total annualised investment cost ---
@objective(model, Min,
    sum(c_solar * x_solar[r] +
        c_wind  * x_wind[r]  +
        c_bat   * x_bat[r]
        for r in 1:R)
)

# --- Constraints ---
for r in 1:R, d in 1:D

    # (4) Battery initialisation cyclique : début du jour d = fin du jour précédent (mod D)
    @constraint(model, e[r, d, 0] == e[r, d == 1 ? D : d-1, T])

    for t in 1:T

        # (1) Energy balance avec flux bilatéraux
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

# (2) Capacités de transmission interrégionales
for (rA, rB) in pairs, d in 1:D, t in 1:T
    @constraint(model, -cap[(rA,rB)] <= flow[(rA,rB), d, t] <= cap[(rA,rB)])
end


# =============================================================================
# Solve
# =============================================================================
optimize!(model)

# =============================================================================
# Results
# =============================================================================
println("\n=== Optimal Capacity Expansion — With Limited Transmission + 8% Demand Growth ===")
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

# --- Plot capacity results par région (stacked bar chart) ---
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
    title = "Optimal Capacity Expansion by Region",
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

output_path = joinpath(fig_dir, "capacity_results_demand_growth_8pct.png")
savefig(output_path)
println("Saved plot to ", output_path)

# --- Save results to CSV ---
results = DataFrame(
    Région = regions,
    Solar_MW = [value(x_solar[r]) for r in 1:R],
    Wind_MW = [value(x_wind[r]) for r in 1:R],
    Battery_MWh = [value(x_bat[r]) for r in 1:R],
)
CSV.write(joinpath(DATA_results, "capacity_results_demand_growth_8pct.csv"), results; delim=';')
println("Saved results to ", joinpath(DATA_results, "capacity_results_demand_growth_8pct.csv"))
