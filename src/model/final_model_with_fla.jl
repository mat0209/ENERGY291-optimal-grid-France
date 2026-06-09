# Comparison: Flamanville 3 with gas vs without gas.
# Solves two models (only difference: thermal included or excluded from gen_reduced).
# Flamanville 3 is present in both scenarios.
#
# Flamanville 3 (EPR, Normandy):
#   - Rated capacity : 1,630 MW  (EDF official)
#   - Capacity factor: 75%       (new EPR in 2030 > 2024 fleet average ~67%,
#                                  source: RTE Bilan Électrique 2024, 361.7 TWh / 61.4 GW,
#                                  https://analysesetdonnees.rte-france.com/bilan-electrique-2024/production)
#   - Profile        : flat (baseload, not an optimization variable)
#
# Inputs: data/final/; Output: data/results/ and figures/results/.

using JuMP, CSV, DataFrames, Printf, Dates, Plots, HiGHS

# =============================================================================
# Paths
# =============================================================================
const DATA         = joinpath(@__DIR__, "..", "..", "data", "final")
const DATA_results = joinpath(@__DIR__, "..", "..", "data", "results")
const FIG_DIR      = joinpath(@__DIR__, "..", "..", "figures", "results")
mkpath(FIG_DIR)

# =============================================================================
# Load data
# =============================================================================
gen_df  = CSV.read(joinpath(DATA, "gen_reduced_days.csv"),
                   DataFrame; delim=';')
pv_df   = CSV.read(joinpath(DATA, "ninja_pv_new_representative_days_regions.csv"),
                   DataFrame)
wind_df = CSV.read(joinpath(DATA, "ninja_wind_new_representative_days_regions.csv"),
                   DataFrame)
cap_df  = CSV.read(joinpath(DATA, "capacites_interregionales.csv"), DataFrame)
cluster_df = CSV.read(joinpath(DATA, "day_cluster_assignment.csv"), DataFrame)
sort!(cluster_df, :actual_date)

# =============================================================================
# Sets
# =============================================================================
regions = sort(unique(gen_df.Région))
dates   = sort(unique(gen_df.Date))

R = length(regions)   # 12
D = length(dates)     # 10
T = 24
N = nrow(cluster_df)  # 366 actual days

region_idx = Dict(r => i for (i, r) in enumerate(regions))
date_idx   = Dict(string(d) => i for (i, d) in enumerate(dates))

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
pairs_in  = [[(rA,rB) for (rA,rB) in pairs if rB == r] for r in 1:R]
pairs_out = [[(rA,rB) for (rA,rB) in pairs if rA == r] for r in 1:R]

day_to_typday = [date_idx[string(row.representative_date)] for row in eachrow(cluster_df)]
cluster_size  = zeros(Int, D)
for d in day_to_typday; cluster_size[d] += 1; end

# =============================================================================
# Parameters — build two versions of gen_reduced
# =============================================================================
demand          = zeros(R, D, T)
gen_reduced_gas = zeros(R, D, T)   # with gas (thermal kept)
gen_reduced_nog = zeros(R, D, T)   # without gas (thermal removed)
cf_solar        = zeros(R, D, T)
cf_wind         = zeros(R, D, T)

for row in eachrow(gen_df)
    r          = region_idx[row.Région]
    d          = date_idx[string(row.Date)]
    t          = Dates.hour(row.Heure) + 1
    thermal_MW = coalesce(row[Symbol("Thermique (MW)")], 0.0)
    base_MW    = coalesce(row.gen_reduced_MW, 0.0)
    demand[r, d, t]          = coalesce(row[Symbol("Consommation (MW)")], 0.0)
    gen_reduced_gas[r, d, t] = base_MW
    gen_reduced_nog[r, d, t] = base_MW - thermal_MW
end

demand .*= 1.08  # +8% demand growth

const FLA3_MW   = 1_630.0 * 0.75   # 1 630 MW × 75 % CF = 1 222.5 MW
r_normandie     = region_idx["Normandie"]
for d in 1:D, t in 1:T
    gen_reduced_gas[r_normandie, d, t] += FLA3_MW
    gen_reduced_nog[r_normandie, d, t] += FLA3_MW
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
# Costs
# =============================================================================
c_solar     = 84207.18832
c_wind      = 170873.2996
c_bat       = 25529.34076
c_trans_400 = 61.09615194
c_trans_225 = 101.8269199
cap_trans_400 = 1500.0
cap_trans_225 = 400.0

# =============================================================================
# Solve function — same model structure, different gen_reduced
# =============================================================================
function solve_scenario(gen_reduced)
    m = Model(HiGHS.Optimizer)
    set_silent(m)

    @variable(m, x_solar[1:R] >= 0)
    @variable(m, x_wind[1:R]  >= 0)
    @variable(m, x_bat[1:R]   >= 0)
    @variable(m, y_trans_400[pairs] >= 0, Int)
    @variable(m, y_trans_225[pairs] >= 0, Int)
    @variable(m, b_charge[1:R, 1:D, 1:T]    >= 0)
    @variable(m, b_discharge[1:R, 1:D, 1:T] >= 0)
    @variable(m, flow[pairs, 1:D, 1:T])
    @variable(m, e_intra[1:R, 1:D, 0:T])
    @variable(m, e_intra_max[1:R, 1:D])
    @variable(m, e_intra_min[1:R, 1:D])
    @variable(m, e_inter[1:R, 1:N] >= 0)

    @objective(m, Min,
        sum(c_solar * x_solar[r] + c_wind * x_wind[r] + c_bat * x_bat[r] for r in 1:R)
        + sum(y_trans_400[p] * c_trans_400 * dist[p] * cap_trans_400
            + y_trans_225[p] * c_trans_225 * dist[p] * cap_trans_225 for p in pairs)
    )

    for r in 1:R, d in 1:D
        @constraint(m, e_intra[r, d, 0] == 0)
        for t in 1:T
            @constraint(m,
                gen_reduced[r, d, t]
                + cf_solar[r, d, t] * x_solar[r]
                + cf_wind[r, d, t]  * x_wind[r]
                + b_discharge[r, d, t] - b_charge[r, d, t]
                + sum(flow[p, d, t] for p in pairs_in[r];  init=AffExpr(0))
                - sum(flow[p, d, t] for p in pairs_out[r]; init=AffExpr(0))
                >= demand[r, d, t]
            )
            @constraint(m, e_intra[r, d, t] == e_intra[r, d, t-1] + b_charge[r, d, t] - b_discharge[r, d, t])
            @constraint(m, e_intra[r, d, t] <= e_intra_max[r, d])
            @constraint(m, e_intra[r, d, t] >= e_intra_min[r, d])
            @constraint(m, b_charge[r, d, t]    <= x_bat[r] / 4.0)
            @constraint(m, b_discharge[r, d, t] <= x_bat[r] / 4.0)
        end
    end

    for r in 1:R
        @constraint(m, sum(cluster_size[d] * e_intra[r, d, T] for d in 1:D) == 0)
    end
    for r in 1:R, i in 1:N-1
        @constraint(m, e_inter[r, i+1] == e_inter[r, i] + e_intra[r, day_to_typday[i], T])
    end
    for r in 1:R, i in 1:N
        d = day_to_typday[i]
        @constraint(m, e_inter[r, i] + e_intra_max[r, d] <= x_bat[r])
        @constraint(m, e_inter[r, i] + e_intra_min[r, d] >= 0)
    end
    for (rA, rB) in pairs
        cap_new = cap[(rA,rB)] + y_trans_400[(rA,rB)] * cap_trans_400 + y_trans_225[(rA,rB)] * cap_trans_225
        for d in 1:D, t in 1:T
            @constraint(m, flow[(rA,rB), d, t] <=  cap_new)
            @constraint(m, flow[(rA,rB), d, t] >= -cap_new)
        end
    end

    optimize!(m)
    return m, x_solar, x_wind, x_bat, y_trans_400, y_trans_225
end

# =============================================================================
# Run both scenarios
# =============================================================================
println("\n=== Scénario 1 : Flamanville 3 + avec gaz ===")
m_gas, xs_gas, xw_gas, xb_gas, yt400_gas, yt225_gas = solve_scenario(gen_reduced_gas)
println("Status    : ", termination_status(m_gas))
println("Objective : ", round(objective_value(m_gas) / 1e6, digits=1), " M€/yr")

println("\n=== Scénario 2 : Flamanville 3 + sans gaz ===")
m_nog, xs_nog, xw_nog, xb_nog, yt400_nog, yt225_nog = solve_scenario(gen_reduced_nog)
println("Status    : ", termination_status(m_nog))
println("Objective : ", round(objective_value(m_nog) / 1e6, digits=1), " M€/yr")

# =============================================================================
# Print capacity tables
# =============================================================================
region_name = Dict(v => k for (k, v) in region_idx)

for (label, m, xs, xw, xb) in [
        ("With gas",  m_gas, xs_gas, xw_gas, xb_gas),
        ("No fossil", m_nog, xs_nog, xw_nog, xb_nog)]
    println("\n--- Optimal capacities — $label ---")
    @printf("%-32s %10s %10s %14s\n", "Region", "Solar(MW)", "Wind(MW)", "Battery(MWh)")
    println("-" ^ 70)
    for r_name in regions
        r = region_idx[r_name]
        @printf("%-32s %10.1f %10.1f %14.1f\n", r_name,
            value(xs[r]), value(xw[r]), value(xb[r]))
    end
    println("-" ^ 70)
    @printf("%-32s %10.1f %10.1f %14.1f\n", "TOTAL",
        sum(value(xs[r]) for r in 1:R),
        sum(value(xw[r]) for r in 1:R),
        sum(value(xb[r]) for r in 1:R))
end

# =============================================================================
# Save CSVs
# =============================================================================
for (suffix, m, xs, xw, xb, yt400, yt225) in [
        ("fla_with_gas",   m_gas, xs_gas, xw_gas, xb_gas, yt400_gas, yt225_gas),
        ("fla_without_gas", m_nog, xs_nog, xw_nog, xb_nog, yt400_nog, yt225_nog)]

    cap_csv = DataFrame(
        Région      = regions,
        Solar_MW    = [value(xs[r]) for r in 1:R],
        Wind_MW     = [value(xw[r]) for r in 1:R],
        Battery_MWh = [value(xb[r]) for r in 1:R],
    )
    CSV.write(joinpath(DATA_results, "capacity_results_$(suffix).csv"), cap_csv; delim=';')

    trans_csv = DataFrame(
        Region_A        = [region_name[p[1]] for p in pairs],
        Region_B        = [region_name[p[2]] for p in pairs],
        Dist_km         = [dist[p]           for p in pairs],
        Cap_exist_MW    = [cap[p]            for p in pairs],
        New_lines_400kV = [round(Int, value(yt400[p])) for p in pairs],
        New_lines_225kV = [round(Int, value(yt225[p])) for p in pairs],
        New_cap_400_MW  = [round(Int, value(yt400[p])) * cap_trans_400 for p in pairs],
        New_cap_225_MW  = [round(Int, value(yt225[p])) * cap_trans_225 for p in pairs],
        Total_cap_MW    = [cap[p] + round(Int, value(yt400[p])) * cap_trans_400 + round(Int, value(yt225[p])) * cap_trans_225 for p in pairs],
    )
    CSV.write(joinpath(DATA_results, "transmission_results_$(suffix).csv"), trans_csv; delim=';')
    println("Saved: capacity_results_$(suffix).csv + transmission_results_$(suffix).csv")

    open(joinpath(DATA_results, "cost_$(suffix).txt"), "w") do f
        write(f, string(round(objective_value(m) / 1e6, digits=2)))
    end
end

# =============================================================================
# Comparison panel (GW)
# =============================================================================
ABBREV = Dict(
    "Auvergne-Rhône-Alpes"       => "ARA",
    "Bourgogne-Franche-Comté"    => "BFC",
    "Bretagne"                   => "BRE",
    "Centre-Val de Loire"        => "CVL",
    "Grand Est"                  => "GES",
    "Hauts-de-France"            => "HDF",
    "Île-de-France"              => "IDF",
    "Normandie"                  => "NOR",
    "Nouvelle-Aquitaine"         => "NAQ",
    "Occitanie"                  => "OCC",
    "Pays de la Loire"           => "PDL",
    "Provence-Alpes-Côte d'Azur" => "PAC",
)
reg_labels = [get(ABBREV, r, r) for r in regions]

cost_gas = round(objective_value(m_gas) / 1e9, digits=1)
cost_nog = round(objective_value(m_nog) / 1e9, digits=1)

solar_gas = [value(xs_gas[r]) / 1e3 for r in 1:R]
wind_gas  = [value(xw_gas[r]) / 1e3 for r in 1:R]
bat_gas   = [value(xb_gas[r]) / 4e3 for r in 1:R]

solar_nog = [value(xs_nog[r]) / 1e3 for r in 1:R]
wind_nog  = [value(xw_nog[r]) / 1e3 for r in 1:R]
bat_nog   = [value(xb_nog[r]) / 4e3 for r in 1:R]

y_max = max(maximum(solar_gas .+ wind_gas .+ bat_gas),
            maximum(solar_nog .+ wind_nog .+ bat_nog)) * 1.25

function make_subplot(reg, sol, win, bat, title_str)
    p = bar(reg, sol;
        label="Solar (GW)", color="#f28e2b", bar_width=0.7,
        title=title_str, titlefontsize=11,
        xlabel="Region", ylabel="Added capacity (GW)",
        xrotation=0, legend=:topright,
        ylims=(0, y_max), bottom_margin=6Plots.mm, left_margin=6Plots.mm)
    bar!(p, reg, win; label="Wind (GW)",    color="#59a14f", bottom=sol)
    bar!(p, reg, bat; label="Battery (GW)", color="#7b5ea6", bottom=sol .+ win)
    return p
end

p1 = make_subplot(reg_labels, solar_gas, wind_gas, bat_gas,
                  "With gas + Flamanville 3 — Cost: $(cost_gas) B€/yr")
p2 = make_subplot(reg_labels, solar_nog, wind_nog, bat_nog,
                  "No fossil + Flamanville 3 — Cost: $(cost_nog) B€/yr")

panel = plot(p1, p2; layout=(1, 2), size=(1520, 600),
             plot_title="Flamanville 3 — optimal investment with and without gas",
             plot_titlefontsize=12, top_margin=12Plots.mm)

panel_path = joinpath(FIG_DIR, "comparison_fla_gas_vs_no_gas.png")
savefig(panel, panel_path)
println("Saved comparison panel to ", panel_path)
