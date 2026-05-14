# CSAM Deployment Optimization

**Multi-period multi-commodity network flow model with Benders decomposition** for optimal deployment of mobile Cold-Spray Additive Manufacturing (CSAM) facilities to supplement traditional repair facilities.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org)
[![PuLP](https://img.shields.io/badge/PuLP-2.x-orange.svg)](https://coin-or.github.io/pulp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project models the deployment of a limited number of CSAM (`l1`) mobile facilities across 10 locations (`m1`–`m10`) over 2 time periods. Traditional repair (`l2`) capacity exists only at type-specific fixed facilities (`m1`–`m5` for vehicle types `k1`–`k5`).

**Current active model**: `scripts/fleet_flow_gr9_c_bd.py` (Benders decomposition – master on deployments `y`, subproblem on flows).

**Key features**:
- Time-expanded network with node splitting (in → queue → repair → out).
- Inter-facility travel, queue carryover (t=1→t=2), dummy nodes for unmet demand.
- Fixed deployment costs + variable routing/repair/penalty costs.
- Benders optimality cuts for better scalability.

## How to Run the Model & Visualizations (Local Only)

All scripts must be run **locally** (large PuLP models).

### 1. Run the Optimization Model
```bash
cd scripts
python fleet_flow_gr9_c_bd.py

Mathematical Formulation
Sets and Indices

$M = {m1,\dots,m10}$ — candidate CSAM locations
$L = {l1,l2}$ — repair types (CSAM / traditional)
$K = {k1,\dots,k5}$ — vehicle types
$C = L \times K$ — commodities
$T = {1,2}$ — time periods

Nodes: source, m_in, m_q_l{p}, m_r_l{p}, m_out_l{p}, sink, dummy (t=2), ss (super-sink).
Decision Variables

$y_m \in {0,1}$ — deploy CSAM at location $m$
$x_{ij,t,c} \geq 0$ — flow on regular arcs
$x_{ij,t,c,t'} \geq 0$ — queue-carryover flows

Objective
$$\min \sum_{m \in M} F_m \, y_m + \sum_{\text{arcs}} c_{ij} \, x_{ij,\dots}$$
(Deployment + travel + queue entry + repair (l1 cheaper than l2) + carryover + dummy penalty)
Main Constraints

Flow conservation at every node (source = demand injection, ss = total demand, others balanced).
CSAM capacity:$$\sum_{c:c[0]=l1} x_{(m\_q\_l1 \to m\_r\_l1),t,c} \leq U_{l1} \cdot y_m \quad \forall m,t$$
Traditional (l2) capacity at fixed locations.
Deployment limit: $\sum_m y_m \leq$ max facilities.

Full detailed formulation is in main.tex.
Benders Decomposition

Master: Binary $y$ + continuous $\theta$ (subproblem cost approximation) + optimality/feasibility cuts.
Subproblem: LP flow problem with fixed $y$; duals from l1-capacity constraints generate cuts.
Manual loop in PuLP for academic transparency.

Repository Structure
textCopy.
├── scripts/
│   ├── fleet_flow_gr9_c_3.py
│   ├── fleet_flow_gr9_c_bd.py          ← Current
│   └── Graphs_bar_charts_gr9_c_bd.py   ← New visualization parser
├── output/
├── viz_output/
├── main.tex
└── README.md
Latest Results Example

CSAM Deployed: m5, m8, m9 (example)
High unmet demand penalty remains — good area for tuning.

See output/ and charts in viz_output/.
Ongoing Work & Roadmap

 Stabilize Benders cuts / convergence
 Update visualization script for Benders output
 Multi-seed experiments + statistics
 Stochastic / CVaR extensions
 Refresh main.tex with latest figures

Pulling into Overleaf
Copy sections directly, or:
BashCopypandoc README.md -o model.tex --from markdown+tex_math_dollars
Contributing / Contact
PhD project by David Dunham (Northeastern University, Advisor: Prof. Ozlem Ergun).
Issues and PRs welcome!