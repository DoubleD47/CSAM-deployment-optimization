CSAM Deployment Optimization
Multi-period multi-commodity network flow model with Benders decomposition for optimal deployment of mobile Cold-Spray Additive Manufacturing (CSAM) facilities to supplement traditional repair facilities.



Overview
This project models the deployment of a limited number of CSAM (l1) mobile facilities across 10 locations (m1–m10) over 2 time periods. Traditional repair (l2) capacity exists only at type-specific fixed facilities (m1–m5 for vehicle types k1–k5).
Current active model: scripts/fleet_flow_gr9_c_bd.py (Benders decomposition – master on deployments y, subproblem on flows).
Key features:

Time-expanded network with node splitting (in → queue → repair → out).
Inter-facility travel, queue carryover (t=1→t=2), dummy nodes for unmet demand.
Fixed deployment costs + variable routing/repair/penalty costs.
Benders optimality cuts for better scalability.

How to Run the Model & Visualizations (Local Only)
All scripts must be run locally (large PuLP models).
1. Run the Optimization Model
BashCopycd scripts
python fleet_flow_gr9_c_bd.py
Main output: output/output_gr9_c_3_benders.txt + CSVs.
2. Generate Visualizations
BashCopycd scripts
python Graphs_bar_charts_gr9_c_bd.py
Outputs → visualizations/Graphs_bar_charts_gr9_c_bd/
Mathematical Formulation
Sets and Indices

$  M = \{m1,\dots,m10\}  $ — candidate CSAM locations
$  L = \{l1,l2\}  $ — repair types (CSAM / traditional)
$  K = \{k1,\dots,k5\}  $ — vehicle types
$  C = L \times K  $ — commodities
$  T = \{1,2\}  $ — time periods

Nodes: source, m_in, m_q_l{p}, m_r_l{p}, m_out_l{p}, sink, dummy (t=2), ss (super-sink).
Decision Variables

$  y_m \in \{0,1\}  $ — deploy CSAM at location $  m  $
$  x_{ij,t,c} \geq 0  $ — flow on regular arcs
$  x_{ij,t,c,t'} \geq 0  $ — queue-carryover flows

Objective
$$\min \sum_{m \in M} F_m \, y_m + \sum_{\text{arcs}} c_{ij} \, x_{ij,\dots}$$
(Deployment + travel + queue entry + repair (l1 cheaper than l2) + carryover + dummy penalty)
Main Constraints

Flow conservation at every node (source = demand injection, ss = total demand, others balanced).
CSAM capacity:$$\sum_{c:c[0]=l1} x_{(m\_q\_l1 \to m\_r\_l1),t,c} \leq U_{l1} \cdot y_m \quad \forall m,t$$
Traditional (l2) capacity at fixed locations.
Deployment limit: $  \sum_m y_m \leq  $ max facilities.

Full detailed formulation is in main.tex.
Benders Decomposition

Master: Binary $  y  $ + continuous $  \theta  $ (subproblem cost approximation) + optimality/feasibility cuts.
Subproblem: LP flow problem with fixed $  y  $; duals from l1-capacity constraints generate cuts.
Manual loop in PuLP for academic transparency.

Repository Structure
textCopy.
├── scripts/
│   ├── fleet_flow_gr9_c_bd.py          ← Current model
│   └── Graphs_bar_charts_gr9_c_bd.py   ← Visualization parser
├── output/
├── visualizations/
├── main.tex
└── README.md
Latest Results (seed 456)

- **csam_flows.csv** — Actual CSAM (l1) repair throughput (q_l1 → r_l1)
- **traditional_flows.csv** — Repair at traditional (l2) sites; may include l1 commodities using l2 capacity ("jumping")
- **inq_flows.csv** — Demand entering repair queues from the entry node
- **qq_flows.csv** — Queue carry-over from t=1 to t=2
- **in_carry_flows.csv** — Demand held at the *entry* node (m_in t=1 → m_in t=2). Not yet in any queue. Shown as gray hatched bars.
- **travel_flows.csv** — Inter-facility movement (mX_in → mY_in)
- **dummy_flows.csv** — Unmet demand (total penalty base)

All 10 CSAM facilities deployed (max limit reached).
Objective: 362072.35
Significant CSAM repair usage; some unmet demand remains.

See output/ and charts in visualizations/Graphs_bar_charts_gr9_c_bd/.
Ongoing Work & Roadmap

 Stabilize Benders cuts / convergence
 Multi-seed experiments + statistics
 Stochastic / CVaR extensions
 Refresh main.tex with latest figures

Pulling into Overleaf
Copy sections directly, or use Pandoc:
BashCopypandoc README.md -o model.tex --from markdown+tex_math_dollars
Contributing / Contact
PhD project by David Dunham (Northeastern University, Advisor: Prof. Ozlem Ergun).
Issues and PRs welcome!