# Optimal Siting and Sizing of Solar, Wind, and Battery Storage to Compensate for Nuclear Decommissioning in France

**Matthieu Hautsch, Anna-Sherazade Sylla**  
ENERGY 291 Project

---

## Overview

Nuclear power currently represents **67.1% of French electricity (2024)**, with a fleet of 57 reactors totaling **62.9 GW**. This strong reliance is unique in Europe but the fleet is aging, with most reactors approaching or exceeding their original 40-year design lifetime.

Under a strict retirement scenario, France could lose approximately **10 GW of nuclear capacity by 2030**, raising the question of how to replace this baseload generation while maintaining reliability and low carbon intensity.

---

## Research Question

> How much solar, wind, and battery storage is required to compensate for nuclear decommissioning in France, and where should these assets be located?

---

## Approach

We build a **grid-based capacity expansion model of the French power system** that jointly optimizes:

- Solar PV and wind capacity deployment
- Seasonal battery storage (Kotzur formulation)
- Endogenous transmission investment
- Spatial allocation across 12 administrative regions

The model is a MILP solved with HiGHS (via JuMP/Julia). It uses 10 representative days selected by k-medoids clustering, hourly demand and capacity factor data, and techno-economic cost assumptions from RTE Futurs énergétiques 2050.

**Battery storage** follows the Kotzur et al. (2018) seasonal decomposition: the state of charge is split into an intra-period component (reset to zero each typical day) and an inter-period component (carried forward across all 366 actual days in chronological order). This reduces capacity constraints by a factor of 24 compared to a naïve formulation. Reference: Kotzur L. et al. (2018), *A modeler's guide to handle complexity in energy systems optimization*, arXiv:1710.07593.

---

## Key Assumptions

- Reference year: 2024
- No imports/exports (isolated grid)
- No new nuclear except Flamanville 3 (1.6 GW)
- 50-year nuclear lifetime assumption → ~10 GW decommissioned
- Seasonal battery storage (Kotzur model)
- Utility-scale solar only
- +8% demand growth baseline
- Endogenous transmission with distance-based costs

---

## Scenarios

| Scenario | Description |
| --- | --- |
| **No fossil** | Main scenario — solar, wind, storage, new transmission, no gas |
| **With Flamanville** | Adds Flamanville 3 (1.6 GW) to the no-fossil baseline |
| **With gas** | Allows gas peakers alongside renewables |
| **Without gas** | No gas, with Flamanville |
| **Unlimited transmission** | Removes interregional capacity constraints |
| **Demand growth +8%** | Final model — baseline +8% demand, endogenous transmission |

---

## Sensitivity Analysis

- **Cost sensitivity**: 5×5 grid sweep of wind and battery cost multipliers (±50% around baseline), with solar and transmission costs fixed.
- **Demand growth sensitivity**: Sweep over demand growth rates, holding technology costs constant.

---

## Data Sources

- RTE Eco2Mix regional (load & generation by region): https://www.rte-france.com/eco2mix
- RTE Open Data API (actual generation per nuclear unit): https://data.rte-france.com
- RTE Futurs énergétiques 2050 (technology costs): https://www.rte-france.com
- IAEA PRIS (nuclear fleet): https://pris.iaea.org
- ODRE (transmission line data): https://odre.opendatasoft.com
- Renewables.ninja (capacity factors PV & wind): https://www.renewables.ninja
- IGN / data.gouv.fr (French communes & region centroids): https://www.data.gouv.fr

---

## Repository Structure

```text
src/
  model/          → JuMP/Julia optimization models
    final_model.jl                          → main model (+8% demand, endogenous tx, Kotzur battery)
    battery_model.jl                        → standalone Kotzur battery model
    model_with_fla.jl                       → Flamanville 3 variant
    model_without_gas.jl                    → no-gas variant
    with_demand_growth.jl                   → demand growth sweep
    with_transmission_constraint.jl         → constrained transmission
    capacity_model_unlimited_transmission.jl

  results/        → post-processing and visualization
    plotting_results.py                     → results charts
    comparison_4scenarios.jl                → cross-scenario comparison (Julia)
    comparison_4scenarios.py                → cross-scenario comparison (Python)
    sensitivity_cost.jl                     → 5×5 wind/battery cost sweep
    sensitivity_demand.jl                   → demand growth sensitivity
    map.py                                  → geographic capacity map

  clustering/     → representative day selection
    kmedoids.py / kmedoids_2.py / kmedoids_final.py

  data/           → data processing scripts
    rte_cleaning.py
    gen_unit_nuclear.py                     → nuclear generation per unit via RTE API
    gen_reduced_days.py                     → reduced-form dispatch per region × day
    filter_eco2mix_representative_days.py   → filter Eco2Mix to 10 representative days
    filter_ninja_representative_days.py     → filter Renewables.ninja to 10 representative days
    cleaning_solar_pv.py
    transmission.py
    regions.py
    cities.py
    plot_transmission.py
    visualization.py                        → national mix and consumption plots

data/
  raw/            → raw inputs (RTE, IAEA, ODRE, renewables.ninja)
  processed/      → intermediate outputs
  final/          → production-ready inputs for the model (hourly, 10 rep days)
  results/        → optimization outputs (CSV per scenario)

figures/
  results/        → output charts and maps
  gen representative days/ → capacity factor and demand plots
  clustering/     → representative day clustering plots
  transmission/   → transmission network plots

reports/          → project reports and documentation
```

---

## Running the Model

**Requirements**: Julia with `JuMP`, `HiGHS`, `CSV`, `DataFrames`, `Plots`, `Dates`; Python with packages in `requirements.txt`.

```bash
# Run the final model
julia src/model/final_model.jl

# Run cost sensitivity sweep (5×5 wind × battery cost grid)
julia src/results/sensitivity_cost.jl

# Run demand growth sensitivity
julia src/results/sensitivity_demand.jl

# Cross-scenario comparison
julia src/results/comparison_4scenarios.jl

# Geographic capacity map
python src/results/map.py
```

---

## Goal

Quantify the optimal mix of solar, wind, and storage required to replace declining nuclear capacity while minimizing total annualized system cost (CAPEX) under spatial and temporal constraints.
