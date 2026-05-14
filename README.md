# ENERGY291-optimal-grid-France

# Optimal Siting and Sizing of Solar, Wind, and Battery Storage to Compensate for Nuclear Decommissioning in France

**Matthieu Hautsch, Anna-Sherazade Sylla**  
ENERGY 291 Project

---

## Overview

Nuclear power currently represents **67.1% of French electricity (2024)**, with a fleet of 57 reactors totaling **62.9 GW**. This strong reliance is unique in Europe but the fleet is aging, with most reactors approaching or exceeding their original 40-year design lifetime.

Under a strict retirement scenario, France could lose approximately **18 GW of nuclear capacity by 2040**, raising the question of how to replace this baseload generation while maintaining reliability and low carbon intensity.

---

## Research Question

> How much solar, wind, and battery storage is required to compensate for nuclear decommissioning in France, and where should these assets be located?

---

## Approach

We develop a **grid-based optimization model of the French power system** that jointly optimizes:

- Solar and wind capacity expansion  
- Battery storage deployment  
- Spatial allocation of resources  
- Simplified transmission constraints  

The model uses hourly demand and generation data, representative day clustering, and techno-economic cost assumptions.

---

## Key Assumptions

- Reference year: 2024  
- No imports/exports (isolated grid)  
- No new nuclear except Flamanville 3  
- 40–50 year nuclear lifetime assumption  
- 4-hour battery storage model  
- Utility-scale solar only  
- Simplified transmission network  

---

## Data Sources

- RTE Eco2Mix (load & generation): https://www.rte-france.com  
- IAEA PRIS (nuclear fleet): https://pris.iaea.org  
- ODRE (grid data): https://odre.opendatasoft.com  
- Renewables.ninja (capacity factors): https://www.renewables.ninja  
- Copernicus Land Use: https://land.copernicus.eu  

---

## Structure


src/ → models and optimization
data/ → datasets
notebooks/ → analysis
results/ → outputs
docs/ → reports


---

## Goal

Quantify the optimal mix of solar, wind, and storage required to replace declining nuclear capacity while minimizing total system cost under spatial and temporal constraints.
