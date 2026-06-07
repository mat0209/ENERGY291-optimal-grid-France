# Four-scenario comparison panel: ±Gas × ±Flamanville 3
# Reads pre-computed CSVs from data/results/.
# Required inputs:
#   final_model.jl       → capacity_results_demand_growth_8pct.csv + cost_with_gas.txt
#   model_without_gas.jl → capacity_results_no_fossil.csv          (cost saved if re-run)
#   model_with_fla.jl    → capacity_results_fla_with_gas.csv       + cost_fla_with_gas.txt
#                        → capacity_results_fla_without_gas.csv    + cost_fla_without_gas.txt

using CSV, DataFrames, Plots, Statistics

const DATA_RESULTS = joinpath(@__DIR__, "..", "..", "data", "results")
const DATA_FINAL   = joinpath(@__DIR__, "..", "..", "data", "final")
const FIG_DIR      = joinpath(@__DIR__, "..", "..", "figures", "results")
mkpath(FIG_DIR)

# =============================================================================
# Region abbreviations (consistent with map.py)
# =============================================================================
const ABBREV = Dict(
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

# =============================================================================
# Helpers
# =============================================================================
THRESHOLD_GW = 0.05   # values below 50 MW treated as 0

clean(x) = x < THRESHOLD_GW ? 0.0 : round(x, digits = 2)

function read_cost(filename)
    p = joinpath(DATA_RESULTS, filename)
    isfile(p) ? round(parse(Float64, read(p, String)) / 1000, digits = 1) : NaN
end

function load_scenario(cap_file, cost_file, regions_order)
    df = CSV.read(joinpath(DATA_RESULTS, cap_file), DataFrame; delim = ';')
    by_region = Dict(row.Région => row for row in eachrow(df))
    solar = [clean(get(by_region, r, (Solar_MW=0.0,)).Solar_MW    / 1e3) for r in regions_order]
    wind  = [clean(get(by_region, r, (Wind_MW=0.0,)).Wind_MW      / 1e3) for r in regions_order]
    bat   = [clean(get(by_region, r, (Battery_MWh=0.0,)).Battery_MWh / 4e3) for r in regions_order]
    cost  = read_cost(cost_file)
    return solar, wind, bat, cost
end

# =============================================================================
# Region order (alphabetical, matching model output)
# =============================================================================
gen_df = CSV.read(joinpath(DATA_FINAL, "gen_reduced_days.csv"), DataFrame; delim = ';')
regions = sort(unique(gen_df.Région))
reg_abbr = [get(ABBREV, r, r) for r in regions]

# =============================================================================
# Gas and FLA3 context figures (shown in figure footer)
# =============================================================================
# Average national thermal (gas) generation removed in no-gas scenarios
thermal_national = combine(
    groupby(gen_df, [:Date, :Heure]),
    Symbol("Thermique (MW)") => (x -> sum(coalesce.(x, 0.0))) => :thermal_MW
)
avg_gas_gw = round(mean(thermal_national.thermal_MW) / 1e3, digits = 1)

fla3_gw = round(1_630.0 * 0.75 / 1e3, digits = 2)   # 1 630 MW × 75% CF

# =============================================================================
# Load all four scenarios
# Layout: rows = FLA3 status, columns = gas status
#   top-left  = with gas, no FLA3
#   top-right = no gas,   no FLA3
#   bot-left  = with gas, FLA3
#   bot-right = no gas,   FLA3
# =============================================================================
sol1, win1, bat1, cost1 = load_scenario(
    "capacity_results_demand_growth_8pct.csv", "cost_with_gas.txt",          regions)
sol2, win2, bat2, cost2 = load_scenario(
    "capacity_results_no_fossil.csv",           "cost_no_fossil.txt",         regions)
sol3, win3, bat3, cost3 = load_scenario(
    "capacity_results_fla_with_gas.csv",        "cost_fla_with_gas.txt",      regions)
sol4, win4, bat4, cost4 = load_scenario(
    "capacity_results_fla_without_gas.csv",     "cost_fla_without_gas.txt",   regions)

fmt_cost(c) = isnan(c) ? "—" : "$(c) B€/yr"

# =============================================================================
# Y-axis limit — computed from data, rounded up to nearest 5 GW
# =============================================================================
all_stacks = vcat(
    sol1 .+ win1 .+ bat1, sol2 .+ win2 .+ bat2,
    sol3 .+ win3 .+ bat3, sol4 .+ win4 .+ bat4
)
y_max = ceil(maximum(all_stacks) * 1.12 / 5) * 5

# =============================================================================
# Plot helpers
# =============================================================================
C_SOLAR = "#f28e2b"
C_WIND  = "#4e9a6f"
C_BAT   = "#6f5ea6"

function make_sub(reg, sol, win, bat, title_str; show_legend = false, show_ylabel = false)
    p = bar(reg, sol;
        label        = (show_legend ? "Solar (GW)"       : false),
        color        = C_SOLAR,
        bar_width    = 0.72,
        title        = title_str,
        titlefontsize = 10,
        xlabel       = "",
        ylabel       = (show_ylabel ? "Added capacity (GW)" : ""),
        ylims        = (0, y_max),
        xrotation    = 0,
        xtickfontsize = 8,
        ytickfontsize = 9,
        legend       = (show_legend ? :topleft : false),
        legendfontsize = 9,
        left_margin  = (show_ylabel ? 12 : 4) * Plots.mm,
        right_margin = 4Plots.mm,
        top_margin   = 4Plots.mm,
        bottom_margin = 2Plots.mm,
        grid         = true,
        gridcolor    = :lightgray,
        gridalpha    = 0.5,
        framestyle   = :box,
    )
    bar!(p, reg, win; label = (show_legend ? "Wind (GW)"         : false), color = C_WIND,  bottom = sol)
    bar!(p, reg, bat; label = (show_legend ? "4h Battery (GW)"   : false), color = C_BAT,   bottom = sol .+ win)
    return p
end

# =============================================================================
# Build subplots
# =============================================================================
p1 = make_sub(reg_abbr, sol1, win1, bat1,
    "With gas | No FLA3\nCost: $(fmt_cost(cost1))";
    show_legend = true, show_ylabel = true)

p2 = make_sub(reg_abbr, sol2, win2, bat2,
    "No gas | No FLA3\nCost: $(fmt_cost(cost2))";
    show_legend = false, show_ylabel = false)

p3 = make_sub(reg_abbr, sol3, win3, bat3,
    "With gas | FLA3 (+$(fla3_gw) GW, NOR)\nCost: $(fmt_cost(cost3))";
    show_legend = false, show_ylabel = true)

p4 = make_sub(reg_abbr, sol4, win4, bat4,
    "No gas | FLA3 (+$(fla3_gw) GW, NOR)\nCost: $(fmt_cost(cost4))";
    show_legend = false, show_ylabel = false)

# =============================================================================
# Blank title subplot (workaround for plot_title top-clipping in Plots.jl)
# =============================================================================
title_note = "Gas removed: ~$(avg_gas_gw) GW national avg.  |  FLA3: +$(fla3_gw) GW baseload (NOR, 75% CF)"
p_title = plot(
    title      = "Optimal capacity expansion — four scenarios (±Gas, ±Flamanville 3)",
    titlefontsize = 13,
    annotation = (0.5, 0.2, text(title_note, 9, :center, :gray)),
    framestyle = :none,
    showaxis   = false,
    bottom_margin = -8Plots.mm,
)

# =============================================================================
# Assemble panel
# =============================================================================
l = @layout [t{0.10h}; a b; c d]

panel = plot(p_title, p1, p2, p3, p4;
    layout = l,
    size   = (1480, 880),
)

out_path = joinpath(FIG_DIR, "comparison_4scenarios.png")
savefig(panel, out_path)
println("Saved: $out_path")
