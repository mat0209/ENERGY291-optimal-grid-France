# Sensitivity analysis: sweep demand growth from 0% to 20% (steps of 2%).
# Runs the final model for each scenario and plots national solar, wind, battery and cost.
# Inputs: data/final/; Output: data/results/ and figures/results/.

using JuMP, CSV, DataFrames, Printf, Dates, Plots, HiGHS

# =============================================================================
# Paths
# =============================================================================
const DATA         = joinpath(@__DIR__, "..", "..", "data", "final")
const DATA_RESULTS = joinpath(@__DIR__, "..", "..", "data", "results")
const FIG_DIR      = joinpath(@__DIR__, "..", "..", "figures", "results")
mkpath(DATA_RESULTS)
mkpath(FIG_DIR)

# =============================================================================
# Load data once (outside the loop)
# =============================================================================
gen_df  = CSV.read(joinpath(DATA, "gen_reduced_days.csv"),                          DataFrame; delim=';')
pv_df   = CSV.read(joinpath(DATA, "ninja_pv_new_representative_days_regions.csv"),  DataFrame)
wind_df = CSV.read(joinpath(DATA, "ninja_wind_new_representative_days_regions.csv"), DataFrame)
cap_df  = CSV.read(joinpath(DATA, "capacites_interregionales.csv"),                  DataFrame)

# =============================================================================
# Sets
# =============================================================================
regions = sort(unique(gen_df.Région))
dates   = sort(unique(gen_df.Date))

R = length(regions)   # 12
D = length(dates)     # 10
T = 24

region_idx = Dict(r => i for (i, r) in enumerate(regions))
date_idx   = Dict(string(d) => i for (i, d) in enumerate(dates))

# Transmission
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
pairs     = collect(keys(cap))
pairs_in  = [[(rA, rB) for (rA, rB) in pairs if rB == r] for r in 1:R]
pairs_out = [[(rA, rB) for (rA, rB) in pairs if rA == r] for r in 1:R]

# =============================================================================
# Base demand + generation arrays (growth = 0%)
# =============================================================================
demand_base = zeros(R, D, T)
gen_reduced = zeros(R, D, T)
cf_solar    = zeros(R, D, T)
cf_wind     = zeros(R, D, T)

for row in eachrow(gen_df)
    r = region_idx[row.Région]
    d = date_idx[string(row.Date)]
    t = Dates.hour(row.Heure) + 1
    demand_base[r, d, t] = coalesce(row[Symbol("Consommation (MW)")], 0.0)
    gen_reduced[r, d, t] = coalesce(row.gen_reduced_MW, 0.0)
end

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
# Kotzur seasonal battery — cluster assignment
# =============================================================================
cluster_df = CSV.read(joinpath(DATA, "day_cluster_assignment.csv"), DataFrame)
sort!(cluster_df, :actual_date)
N = nrow(cluster_df)   # 366 actual days

day_to_typday = [date_idx[string(row.representative_date)] for row in eachrow(cluster_df)]

cluster_size = zeros(Int, D)
for d in day_to_typday
    cluster_size[d] += 1
end

# =============================================================================
# Costs
# =============================================================================
c_solar     = 84207.18832    # €/MW/yr
c_wind      = 170873.2996    # €/MW/yr
c_bat       = 25529.34076    # €/MWh/yr
c_trans_400 = 61.09615194    # €/MW/km/yr
c_trans_225 = 101.8269199    # €/MW/km/yr
cap_trans_400 = 1500.0       # MW per 400 kV line
cap_trans_225 = 400.0        # MW per 225 kV line

# =============================================================================
# Sensitivity analysis: demand_growth ∈ [0%, 20%] in steps of 2%
# =============================================================================
growth_rates = collect(0.0:0.02:0.20)   # 11 scenarios

sens_results = DataFrame(
    growth_pct  = Float64[],
    total_cost  = Float64[],   # M€/yr
    solar_MW    = Float64[],
    wind_MW     = Float64[],
    battery_MWh = Float64[],
    trans_new_MW = Float64[],  # total new transmission capacity added (MW)
)

for rate in growth_rates
    pct = round(Int, rate * 100)
    println("\n─────────────────────────────────────────────────")
    println("  Demand growth = $(pct)%  (scenario $(findfirst(==(rate), growth_rates))/$(length(growth_rates)))")
    println("─────────────────────────────────────────────────")

    demand = demand_base .* (1.0 + rate)

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Decision variables
    @variable(model, x_solar[1:R] >= 0)
    @variable(model, x_wind[1:R]  >= 0)
    @variable(model, x_bat[1:R]   >= 0)
    @variable(model, y_trans_400[pairs] >= 0, Int)
    @variable(model, y_trans_225[pairs] >= 0, Int)

    @variable(model, b_charge[1:R, 1:D, 1:T]    >= 0)
    @variable(model, b_discharge[1:R, 1:D, 1:T] >= 0)
    @variable(model, flow[pairs, 1:D, 1:T])

    # Kotzur seasonal battery
    @variable(model, e_intra[1:R, 1:D, 0:T])
    @variable(model, e_intra_max[1:R, 1:D])
    @variable(model, e_intra_min[1:R, 1:D])
    @variable(model, e_inter[1:R, 1:N] >= 0)

    # Objective
    @objective(model, Min,
        sum(c_solar * x_solar[r] + c_wind * x_wind[r] + c_bat * x_bat[r] for r in 1:R)
        + sum(y_trans_400[p] * c_trans_400 * dist[p] * cap_trans_400
            + y_trans_225[p] * c_trans_225 * dist[p] * cap_trans_225 for p in pairs)
    )

    # Constraints
    for r in 1:R, d in 1:D
        @constraint(model, e_intra[r, d, 0] == 0)
        for t in 1:T
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
            @constraint(model, e_intra[r, d, t] == e_intra[r, d, t-1] + b_charge[r, d, t] - b_discharge[r, d, t])
            @constraint(model, e_intra[r, d, t] <= e_intra_max[r, d])
            @constraint(model, e_intra[r, d, t] >= e_intra_min[r, d])
            @constraint(model, b_charge[r, d, t]    <= x_bat[r] / 4.0)
            @constraint(model, b_discharge[r, d, t] <= x_bat[r] / 4.0)
        end
    end

    for r in 1:R
        @constraint(model, sum(cluster_size[d] * e_intra[r, d, T] for d in 1:D) == 0)
    end

    for r in 1:R, i in 1:N-1
        @constraint(model, e_inter[r, i+1] == e_inter[r, i] + e_intra[r, day_to_typday[i], T])
    end

    for r in 1:R, i in 1:N
        d = day_to_typday[i]
        @constraint(model, e_inter[r, i] + e_intra_max[r, d] <= x_bat[r])
        @constraint(model, e_inter[r, i] + e_intra_min[r, d] >= 0)
    end

    for (rA, rB) in pairs
        cap_new = cap[(rA,rB)] + y_trans_400[(rA,rB)] * cap_trans_400 + y_trans_225[(rA,rB)] * cap_trans_225
        for d in 1:D, t in 1:T
            @constraint(model, flow[(rA,rB), d, t] <=  cap_new)
            @constraint(model, flow[(rA,rB), d, t] >= -cap_new)
        end
    end

    optimize!(model)

    status  = termination_status(model)
    cost    = objective_value(model) / 1e6
    solar   = sum(value(x_solar[r]) for r in 1:R)
    wind    = sum(value(x_wind[r])  for r in 1:R)
    bat     = sum(value(x_bat[r])   for r in 1:R)
    trans   = sum(round(Int, value(y_trans_400[p])) * cap_trans_400
                + round(Int, value(y_trans_225[p])) * cap_trans_225 for p in pairs)

    @printf("  Status      : %s\n",    status)
    @printf("  Total cost  : %.1f M€/yr\n", cost)
    @printf("  Solar       : %.0f MW\n",    solar)
    @printf("  Wind        : %.0f MW\n",    wind)
    @printf("  Battery     : %.0f MWh\n",   bat)
    @printf("  Trans. new  : %.0f MW\n",    trans)

    push!(sens_results, (rate * 100, cost, solar, wind, bat, trans))
end

# =============================================================================
# Save CSV
# =============================================================================
csv_path = joinpath(DATA_RESULTS, "sensitivity_demand_growth.csv")
CSV.write(csv_path, sens_results; delim=';')
println("\nResults saved: $csv_path")
println(sens_results)

# =============================================================================
# Chart — national level
# Grouped bars: Solar (MW), Wind (MW), Battery (MWh) every 4%
# Total cost annotated above each group
# =============================================================================
selected_pct = [0, 4, 8, 12, 16, 20]
sub = filter(row -> round(Int, row.growth_pct) in selected_pct, sens_results)
sub_labels = string.(round.(Int, sub.growth_pct)) .* "%"
n_sub = length(sub_labels)

solar_vals = sub.solar_MW
wind_vals  = sub.wind_MW
bat_vals   = sub.battery_MWh ./ 4.0   # MWh → MW power (4-hour battery)
cost_vals  = sub.total_cost    # M€/yr

w  = 0.22
xs = collect(1:n_sub)

y_max     = maximum(max.(solar_vals, wind_vals, bat_vals))
group_top = max.(solar_vals, wind_vals, bat_vals) .+ y_max * 0.06   # fixed gap above tallest bar

fig = bar(xs .- w, solar_vals;
    label="Solar (MW)", color="#f28e2b", bar_width=w*0.92,
    xticks=(xs, sub_labels), xlabel="Demand growth",
    ylabel="Added capacity (MW)",
    title="\nOptimal added capacity mix — national total",   # leading \n pushes title down
    legend=:topleft, xlims=(0.4, n_sub + 0.6),
    ylims=(0, y_max * 1.20),
    size=(1000, 580), margin=10Plots.mm, titlefontsize=12)
bar!(xs,      wind_vals; label="Wind (MW)",     color="#59a14f", bar_width=w*0.92)
bar!(xs .+ w, bat_vals;  label="Battery, 4-hour (MW)", color="#7b5ea6", bar_width=w*0.92)

for i in 1:n_sub
    annotate!(xs[i], group_top[i],
        text(string(round(cost_vals[i] / 1000, digits=1)) * " Md€", 10, :center, :bottom))
end

fig_path = joinpath(FIG_DIR, "sensitivity_demand_growth.png")
savefig(fig, fig_path)
println("Figure saved: $fig_path")
