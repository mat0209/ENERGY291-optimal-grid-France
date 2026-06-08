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

The model is a MILP solved with HiGHS (via JuMP/Julia). It uses 10 representative days (k-medoids clustering), hourly demand and capacity factor data, and techno-economic cost assumptions.

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

- RTE Eco2Mix (load & generation): https://www.rte-france.com
- IAEA PRIS (nuclear fleet): https://pris.iaea.org
- ODRE (grid data): https://odre.opendatasoft.com
- Renewables.ninja (capacity factors): https://www.renewables.ninja
- Copernicus Land Use: https://land.copernicus.eu

---

## Repository Structure

```text
src/
  model/          → JuMP/Julia optimization models
    final_model.jl                      → main model (+8% demand, endogenous tx)
    model_with_fla.jl                   → Flamanville 3 variant
    model_without_gas.jl                → no-gas variant
    with_demand_growth.jl               → demand growth sweep
    with_transmission_constraint.jl     → constrained transmission
    capacity_model_unlimited_transmission.jl
    battery_model.jl
    sensitivity_cost.jl                 → 5×5 wind/battery cost sweep
    sensitivity_demand.jl               → demand growth sensitivity
    comparison_4scenarios.jl            → cross-scenario comparison (Julia)
    comparison_4scenarios.py            → cross-scenario comparison (Python)
    map.py                              → geographic result map

  clustering/     → representative day selection
    kmedoids.py / kmedoids_2.py

  data/           → data processing scripts
    rte_cleaning.py
    gen_unit_nuclear.py
    gen_reduced_days.py
    filter_eco2mix_representative_days.py
    filter_ninja_representative_days.py
    cleaning_solar_pv.py
    transmission.py
    regions.py
    cities.py
    plot_transmission.py
    visualization.py

  results/        → result processing and plotting
    plotting_results.py

data/
  raw/            → raw inputs (RTE, IAEA, ODRE, renewables.ninja)
  processed/      → intermediate outputs
  final/          → production-ready inputs for the model
  results/        → optimization outputs (CSV per scenario)

figures/
  results/        → output charts and maps
  gen representative days/ → capacity factor and demand plots
  clustering/     → representative day clustering plots
  transmission/   → transmission network plots

reports/          → project reports (PDF)
```

---

## Running the Model

**Requirements**: Julia with `JuMP`, `HiGHS`, `CSV`, `DataFrames`, `Plots`, `Dates`; Python with packages in `requirements.txt`.

```bash
# Run the final model
julia src/model/final_model.jl

# Run cost sensitivity sweep (5×5 grid)
julia src/model/sensitivity_cost.jl

# Run demand growth sensitivity
julia src/model/sensitivity_demand.jl
```

---

## Goal

Quantify the optimal mix of solar, wind, and storage required to replace declining nuclear capacity while minimizing total annualized system cost (CAPEX) under spatial and temporal constraints.
