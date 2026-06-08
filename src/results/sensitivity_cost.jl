# Sensitivity analysis: 5×5 grid sweep of wind and battery cost multipliers.
# Baseline: +8% demand growth (same as final_model.jl). Solar and transmission costs fixed.
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
# Load data once
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

R = length(regions)
D = length(dates)
T = 24

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
pairs_in  = [[(rA, rB) for (rA, rB) in pairs if rB == r] for r in 1:R]
pairs_out = [[(rA, rB) for (rA, rB) in pairs if rA == r] for r in 1:R]

# =============================================================================
# Demand + generation arrays (+8% demand growth)
# =============================================================================
demand      = zeros(R, D, T)
gen_reduced = zeros(R, D, T)
cf_solar    = zeros(R, D, T)
cf_wind     = zeros(R, D, T)

for row in eachrow(gen_df)
    r = region_idx[row.Région]
    d = date_idx[string(row.Date)]
    t = Dates.hour(row.Heure) + 1
    demand[r, d, t]      = coalesce(row[Symbol("Consommation (MW)")], 0.0)
    gen_reduced[r, d, t] = coalesce(row.gen_reduced_MW, 0.0)
end
demand .*= 1.08

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
# Kotzur cluster assignment
# =============================================================================
cluster_df = CSV.read(joinpath(DATA, "day_cluster_assignment.csv"), DataFrame)
sort!(cluster_df, :actual_date)
N = nrow(cluster_df)

day_to_typday = [date_idx[string(row.representative_date)] for row in eachrow(cluster_df)]

cluster_size = zeros(Int, D)
for d in day_to_typday
    cluster_size[d] += 1
end

# =============================================================================
# Base costs and grid
# =============================================================================
c_solar_base  = 84207.18832    # €/MW/yr   (fixed across all runs)
c_wind_base   = 170873.2996    # €/MW/yr   (swept)
c_bat_base    = 25529.34076    # €/MWh/yr  (swept)
c_trans_400   = 61.09615194    # €/MW/km/yr (fixed)
c_trans_225   = 101.8269199    # €/MW/km/yr (fixed)
cap_trans_400 = 1500.0
cap_trans_225 = 400.0

wind_mults = [0.7, 0.85, 1.0, 1.15, 1.3]
bat_mults  = [0.7, 0.85, 1.0, 1.15, 1.3]
n = length(wind_mults)

# Results: [i_bat, i_wind] → row = bat_mult (y-axis), col = wind_mult (x-axis)
cost_mat  = zeros(n, n)   # M€/yr
wind_mat  = zeros(n, n)   # GW
bat_mat   = zeros(n, n)   # GWh
solar_mat = zeros(n, n)   # GW

# =============================================================================
# 5×5 grid sweep
# =============================================================================
let total_runs = n * n, run_idx = 0
    for (iw, wm) in enumerate(wind_mults)
        for (ib, bm) in enumerate(bat_mults)
            run_idx += 1
            println("\n─────────────────────────────────────────────────")
            @printf("  [%2d/%2d]  wind×%.1f  bat×%.1f\n", run_idx, total_runs, wm, bm)
            println("─────────────────────────────────────────────────")

            local cw = c_wind_base * wm
            local cb = c_bat_base  * bm

            local m = Model(HiGHS.Optimizer)
            set_silent(m)

            @variable(m, xs[1:R] >= 0)
            @variable(m, xw[1:R] >= 0)
            @variable(m, xb[1:R] >= 0)
            @variable(m, yt400[pairs] >= 0, Int)
            @variable(m, yt225[pairs] >= 0, Int)

            @variable(m, bch[1:R, 1:D, 1:T]  >= 0)
            @variable(m, bdis[1:R, 1:D, 1:T] >= 0)
            @variable(m, fl[pairs, 1:D, 1:T])

            @variable(m, ei[1:R, 1:D, 0:T])
            @variable(m, ei_max[1:R, 1:D])
            @variable(m, ei_min[1:R, 1:D])
            @variable(m, ee[1:R, 1:N] >= 0)

            @objective(m, Min,
                sum(c_solar_base * xs[r] + cw * xw[r] + cb * xb[r] for r in 1:R)
                + sum(yt400[p] * c_trans_400 * dist[p] * cap_trans_400
                    + yt225[p] * c_trans_225 * dist[p] * cap_trans_225 for p in pairs)
            )

            for r in 1:R, d in 1:D
                @constraint(m, ei[r, d, 0] == 0)
                for t in 1:T
                    @constraint(m,
                        gen_reduced[r, d, t]
                        + cf_solar[r, d, t] * xs[r]
                        + cf_wind[r, d, t]  * xw[r]
                        + bdis[r, d, t]
                        - bch[r, d, t]
                        + sum(fl[p, d, t] for p in pairs_in[r];  init=AffExpr(0))
                        - sum(fl[p, d, t] for p in pairs_out[r]; init=AffExpr(0))
                        >= demand[r, d, t]
                    )
                    @constraint(m, ei[r, d, t] == ei[r, d, t-1] + bch[r, d, t] - bdis[r, d, t])
                    @constraint(m, ei[r, d, t] <= ei_max[r, d])
                    @constraint(m, ei[r, d, t] >= ei_min[r, d])
                    @constraint(m, bch[r, d, t]  <= xb[r] / 4.0)
                    @constraint(m, bdis[r, d, t] <= xb[r] / 4.0)
                end
            end

            for r in 1:R
                @constraint(m, sum(cluster_size[d] * ei[r, d, T] for d in 1:D) == 0)
            end

            for r in 1:R, i in 1:N-1
                @constraint(m, ee[r, i+1] == ee[r, i] + ei[r, day_to_typday[i], T])
            end

            for r in 1:R, i in 1:N
                d = day_to_typday[i]
                @constraint(m, ee[r, i] + ei_max[r, d] <= xb[r])
                @constraint(m, ee[r, i] + ei_min[r, d] >= 0)
            end

            for (rA, rB) in pairs
                cap_new = cap[(rA,rB)] + yt400[(rA,rB)] * cap_trans_400 + yt225[(rA,rB)] * cap_trans_225
                for d in 1:D, t in 1:T
                    @constraint(m, fl[(rA,rB), d, t] <=  cap_new)
                    @constraint(m, fl[(rA,rB), d, t] >= -cap_new)
                end
            end

            optimize!(m)

            local cost  = objective_value(m) / 1e9
            local wind  = sum(value(xw[r]) for r in 1:R) / 1e3
            local bat   = sum(value(xb[r]) for r in 1:R) / 4e3   # MWh → GW (4-hour battery)
            local solar = sum(value(xs[r]) for r in 1:R) / 1e3

            @printf("  Status : %s\n",         termination_status(m))
            @printf("  Cost   : %.2f B€/yr\n", cost)
            @printf("  Wind   : %.2f GW\n",    wind)
            @printf("  Battery: %.2f GW\n",    bat)
            @printf("  Solar  : %.2f GW\n",    solar)

            cost_mat[ib, iw]  = cost
            wind_mat[ib, iw]  = wind
            bat_mat[ib, iw]   = bat
            solar_mat[ib, iw] = solar
        end
    end
end

# =============================================================================
# Save CSV
# =============================================================================
csv_rows = [(wind_mult=wm, bat_mult=bm,
             total_cost_B€=cost_mat[ib, iw],
             wind_GW=wind_mat[ib, iw],
             bat_GW=bat_mat[ib, iw],
             solar_GW=solar_mat[ib, iw])
            for (iw, wm) in enumerate(wind_mults)
            for (ib, bm) in enumerate(bat_mults)]
csv_df   = DataFrame(csv_rows)
csv_path = joinpath(DATA_RESULTS, "sensitivity_cost_2d.csv")
CSV.write(csv_path, csv_df; delim=';')
println("\nResults saved: $csv_path")
println(csv_df)

# =============================================================================
# Heatmaps  (3 subplots: total cost, wind capacity, battery capacity)
# Layout: [i_bat, i_wind] → y-axis = bat_mult, x-axis = wind_mult
# =============================================================================
x_ticks = (wind_mults, ["×$(m)" for m in wind_mults])
y_ticks = (bat_mults,  ["×$(m)" for m in bat_mults])

function cell_label(v)
    v >= 10 ? string(round(Int, v)) : string(round(v, digits=1))
end

function make_heatmap(mat, title_str, cmap; light_at_low=false)
    mn, mx = minimum(mat), maximum(mat)
    span   = mx > mn ? mx - mn : 1.0
    p = heatmap(wind_mults, bat_mults, mat;
        color=cmap,
        xlabel="Wind cost multiplier",
        ylabel="Battery cost multiplier",
        title=title_str,
        xticks=x_ticks,
        yticks=y_ticks,
        titlefontsize=11,
        legend=:none)
    anns = Tuple{Float64,Float64,Plots.PlotText}[]
    for (iw, wm) in enumerate(wind_mults), (ib, bm) in enumerate(bat_mults)
        v    = mat[ib, iw]
        norm = (v - mn) / span
        # black text on light cells, white text on dark cells
        is_light = light_at_low ? norm < 0.40 : norm > 0.60
        clr = is_light ? :black : :white
        push!(anns, (wm, bm, text(cell_label(v), 8, clr, :center, :middle)))
    end
    annotate!(p, anns)
    return p
end

p1 = make_heatmap(cost_mat,  "Total system cost (B€/yr)",        :viridis; light_at_low=false)
p2 = make_heatmap(wind_mat,  "Optimal wind capacity (GW)",      :YlGn;    light_at_low=true)
p3 = make_heatmap(bat_mat,   "Optimal battery capacity (GW)",   :BuPu;    light_at_low=true)
p4 = make_heatmap(solar_mat, "Optimal solar capacity (GW)",     :YlOrRd;  light_at_low=true)

fig = plot(p1, p2, p3, p4;
    layout=(2, 2),
    size=(1100, 900),
    margin=10Plots.mm)

fig_path = joinpath(FIG_DIR, "sensitivity_cost_heatmap.png")
savefig(fig, fig_path)
println("Figure saved: $fig_path")
