# This is the experimental version of Benders' decomposition CSAM deployment optimization, exploring different parameters and seed settings.
# Implements manual loop in PuLP for academic explainability.
# Original monolithic model decomposed: Master for y (deployments), Sub-problems are operational flows.
# For systematic multi-seed / parameter experiments with clean output folders

import os
import sys
import csv
import datetime
import json
from pathlib import Path
import shutil
import pandas as pd
from pulp import *
import numpy as np
from collections import defaultdict
import time

# ====================== EXPERIMENT CONFIG ======================
EXPERIMENT_NAME = "maxCSAM3_single"          # ← Change this per experiment
MAX_CSAM_FACILITIES = 3
NUM_SEEDS = 1                                # Set to >1 later for multi-seed
BASE_SEED = 100

# Create unique experiment folder
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
run_id = f"{timestamp}_{EXPERIMENT_NAME}"
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
exp_dir = Path(repo_root) / "experiments" / run_id
exp_dir.mkdir(parents=True, exist_ok=True)

print(f"Experiment directory created: {exp_dir}")
print(f"MAX_CSAM_FACILITIES = {MAX_CSAM_FACILITIES}")

# Legacy output folder for CSVs (used by viz scripts)
output_dir = os.path.join(repo_root, 'output')
os.makedirs(output_dir, exist_ok=True)

# Capture all prints
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(exp_dir / "full_log.txt", 'w')
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, log_file)

# Set random seed
SEED = BASE_SEED
np.random.seed(SEED)
print(f"Random seed set to: {SEED}")

# ====================== MODEL DATA ======================
M = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10']
traditional_m_dict = {'k1': 'm1', 'k2': 'm2', 'k3': 'm3', 'k4': 'm4', 'k5': 'm5'}
L = ['l1', 'l2']
K = ['k1', 'k2', 'k3', 'k4', 'k5']
C = [(l, k) for l in L for k in K]
T = [1, 2]

##### Nodes #####
nodes = []
for t in T:
    for c in C:
        nodes.append(('source', t, c))
        nodes.append(('sink', t, c))
        if t == 2:
            nodes.append(('dummy', t, c))
        nodes.append(('ss', None, c))
        for m in M:
            nodes.append((f'{m}_in', t, c))
            nodes.extend([(f'{m}_q_l1', t, c), (f'{m}_r_l1', t, c), (f'{m}_out_l1', t, c)])
            traditional_m_for_k = traditional_m_dict.get(c[1])
            if m == traditional_m_for_k:
                nodes.extend([(f'{m}_q_l2', t, c), (f'{m}_r_l2', t, c), (f'{m}_out_l2', t, c)])

##### Arcs #####
regular_arcs = []
qq_arcs = []

# Carryover arcs at the 'in' node level
in_carryover_arcs = []
for c in C:
    for m in M:
        in_carryover_arcs.append((f'{m}_in', f'{m}_in', 1, c, 2)) # t=1 to t=2

for t in T:
    for c in C:
        # source-to-in arcs
        for m in M:
            regular_arcs.append(('source', f'{m}_in', t, c))
        # in-to-in arcs (travel between nodes)
        for m1 in M:
            for m2 in M:
                if m1 != m2:
                    regular_arcs.append((f'{m1}_in', f'{m2}_in', t, c))
        # in-to-q_lp arcs
        for m in M:
            if c[0] != 'l2':
                regular_arcs.append((f'{m}_in', f'{m}_q_l1', t, c))
            traditional_m_for_k = traditional_m_dict.get(c[1])
            if m == traditional_m_for_k:
                regular_arcs.append((f'{m}_in', f'{m}_q_l2', t, c))
        # q_lp-to-r_lp arcs
        for m in M:
            regular_arcs.append((f'{m}_q_l1', f'{m}_r_l1', t, c))
            traditional_m_for_k = traditional_m_dict.get(c[1])
            if m == traditional_m_for_k:
                regular_arcs.append((f'{m}_q_l2', f'{m}_r_l2', t, c))
        # r_lp-to-out_lp arcs
        for m in M:
            regular_arcs.append((f'{m}_r_l1', f'{m}_out_l1', t, c))
            traditional_m_for_k = traditional_m_dict.get(c[1])
            if m == traditional_m_for_k:
                regular_arcs.append((f'{m}_r_l2', f'{m}_out_l2', t, c))
        # out_lp-to-sink arcs
        for m in M:
            regular_arcs.append((f'{m}_out_l1', 'sink', t, c))
            traditional_m_for_k = traditional_m_dict.get(c[1])
            if m == traditional_m_for_k:
                regular_arcs.append((f'{m}_out_l2', 'sink', t, c))
        # sink-to-ss arcs
        regular_arcs.append(('sink', 'ss', t, c))
        # q_lp-to-q_lp arcs (carryover)
        if t == 1:
            for m in M:
                qq_arcs.append((f'{m}_q_l1', f'{m}_q_l1', t, c, t+1))
                traditional_m_for_k = traditional_m_dict.get(c[1])
                if m == traditional_m_for_k:
                    qq_arcs.append((f'{m}_q_l2', f'{m}_q_l2', t, c, t+1))
        # q_lp-to-dummy and dummy-to-ss (t=2)
        if t == 2:
            for m in M:
                regular_arcs.append((f'{m}_q_l1', 'dummy', t, c))
                traditional_m_for_k = traditional_m_dict.get(c[1])
                if m == traditional_m_for_k:
                    regular_arcs.append((f'{m}_q_l2', 'dummy', t, c))
            for m in M:
                regular_arcs.append((f'{m}_in', 'dummy', t, c))
            regular_arcs.append(('dummy', 'ss', t, c))

# Demands
D = {}
for m in M:
    for t in T:
        for c in C:
            D[(m, t, c)] = np.random.uniform(5, 15)

print("\nDemands:")
for (m, t, c), d in D.items():
    print(f"D({m}, t={t}, {c}) = {d:.1f}")

# Parameters
F = {m: 1000 for m in M}  # Deployment cost
C_in_in = 1  # Travel
C_in_q = 10  # Queue entry
C_q_r_l1 = 100  # CSAM repair
C_q_r_l2 = 200  # Traditional repair
C_q_q = 5  # Carryover
C_dummy = 1000  # Dummy penalty
U_l1 = 50  # CSAM capacity per m per t
U_l2 = {k: 100 for k in K}  # Traditional per k per t
max_csam_facilities = 3

# Benders' Parameters
EPS = 1e-4
max_iter = 100

# Master Problem
master = LpProblem("Master_CSAM_Deployment", LpMinimize)
y = LpVariable.dicts("y", [(m, 'l1') for m in M], cat='Binary')
theta = LpVariable("theta", lowBound=0)
master += lpSum(F[m] * y[(m, 'l1')] for m in M) + theta
master += lpSum(y[(m, 'l1')] for m in M) <= max_csam_facilities, "max_csam_facilities"

lb, ub = -np.inf, np.inf
iter_count = 0
best_y = None
best_sub_cost = np.inf
best_sub_vars = None  # To store final flows for output

start_time = time.time()

# Begin Benders' loop
while ub - lb > EPS and iter_count < max_iter:
    iter_count += 1
    print(f"\nIteration {iter_count}: Solving Master...")
    master.solve()
    lb = value(master.objective)
    print(f"Master LB: {lb:.2f}")

    # Fix y for subproblem
    fixed_y = {m: value(y[(m, 'l1')]) for m in M}
    print("Fixed y:", {m: fixed_y[m] for m in M if fixed_y[m] > 0.5})

    # Subproblem (flow LP with fixed capacities)
    sub = LpProblem("Subproblem_Flow", LpMinimize)
    x_regular = LpVariable.dicts("flow_regular", regular_arcs, lowBound=0, cat='Continuous')
    x_qq = LpVariable.dicts("flow_qq", qq_arcs, lowBound=0, cat='Continuous')
    x_in_carryover = LpVariable.dicts("flow_in_carry", in_carryover_arcs, lowBound=0, cat='Continuous')

    # Sub Objective (variable costs only, no deployment)
    sub += (
        lpSum(C_in_in * x_regular[a] for a in regular_arcs if '_in' in a[0] and '_in' in a[1]) +
        lpSum(C_in_q * x_regular[a] for a in regular_arcs if '_in' in a[0] and '_q_' in a[1]) +
        lpSum(C_q_r_l1 * x_regular[a] for a in regular_arcs if '_q_l1' in a[0] and '_r_l1' in a[1]) +
        lpSum(C_q_r_l2 * x_regular[a] for a in regular_arcs if '_q_l2' in a[0] and '_r_l2' in a[1]) +
        lpSum(C_q_q * x_qq[a] for a in qq_arcs) +
        lpSum(0.1 * x_regular[a] for a in regular_arcs if '_r_' in a[0] and '_out_' in a[1]) +
        lpSum(0.1 * x_regular[a] for a in regular_arcs if '_out_' in a[0] and 'sink' in a[1]) +
        lpSum(0.1 * x_regular[a] for a in regular_arcs if 'sink' in a[0] and 'ss' in a[1]) +
        lpSum(C_dummy * x_regular[a] for a in regular_arcs if ('_q_' in a[0] or '_in' in a[0]) and 'dummy' in a[1]) +
        lpSum(0.1 * x_regular[a] for a in regular_arcs if 'dummy' in a[0] and 'ss' in a[1]) +
        lpSum(C_q_q * x_in_carryover[a] for a in in_carryover_arcs)
    )

    # Demand injection
    for m in M:
        for t in T:
            for c in C:
                a = ('source', f'{m}_in', t, c)
                if a in x_regular:
                    sub += x_regular[a] == D.get((m, t, c), 0), f"demand_inject_{m}_{t}_{c[0]}_{c[1]}"

    # Flow conservation (same as original)
    constraint_names = set()
    constraint_counter = 0
    unique_nodes = set(nodes)
    for n, t_node, comm in unique_nodes:
        incoming = [a for a in regular_arcs if a[1] == n and a[2] == t_node and a[3] == comm]
        outgoing = [a for a in regular_arcs if a[0] == n and a[2] == t_node and a[3] == comm]
        incoming_qq = [a for a in qq_arcs if a[1] == n and a[4] == t_node and a[3] == comm]
        outgoing_qq = [a for a in qq_arcs if a[0] == n and a[2] == t_node and a[3] == comm]
        incoming_in_carry = [a for a in in_carryover_arcs if a[1] == n and a[4] == t_node and a[3] == comm]
        outgoing_in_carry = [a for a in in_carryover_arcs if a[0] == n and a[2] == t_node and a[3] == comm]

        if not (incoming or outgoing or incoming_qq or outgoing_qq or incoming_in_carry or outgoing_in_carry):
            continue

        constraint_name = f"flow_conservation_{constraint_counter}_{n.replace('_', '-')}_{t_node if t_node else 'None'}_{comm[0]}_{comm[1]}"
        if constraint_name in constraint_names:
            raise ValueError(f"Duplicate constraint name: {constraint_name}")
        constraint_names.add(constraint_name)
        constraint_counter += 1

        if n == 'source':
            total_demand_t_c = sum(D.get((m, t_node, comm), 0) for m in M)
            constraint = lpSum(x_regular[a] for a in outgoing) + lpSum(x_qq[a] for a in outgoing_qq) + lpSum(x_in_carryover[a] for a in outgoing_in_carry) == total_demand_t_c
        elif n == 'ss' and t_node is None:
            total_demand_c = sum(D.get((m, ti, comm), 0) for m in M for ti in T)
            constraint = lpSum(x_regular[a] for a in incoming) + lpSum(x_qq[a] for a in incoming_qq) + lpSum(x_in_carryover[a] for a in incoming_in_carry) == total_demand_c
        else:
            constraint = (
                lpSum(x_regular[a] for a in incoming) + 
                lpSum(x_qq[a] for a in incoming_qq) + 
                lpSum(x_in_carryover[a] for a in incoming_in_carry) ==
                lpSum(x_regular[a] for a in outgoing) + 
                lpSum(x_qq[a] for a in outgoing_qq) + 
                lpSum(x_in_carryover[a] for a in outgoing_in_carry)
            )
        sub += constraint, constraint_name

    # Capacity constraints (fixed y)
    l1_capacity_cons = {}  # Store for duals
    for m in M:
        for t in T:
            cons_name = f"capacity_l1_{m}_{t}"
            cons = lpSum(x_regular[(f'{m}_q_l1', f'{m}_r_l1', t, c)] for c in C if (f'{m}_q_l1', f'{m}_r_l1', t, c) in x_regular) <= U_l1 * fixed_y[m]
            sub += cons, cons_name
            l1_capacity_cons[(m, t)] = cons_name

    for k in K:
        if k in traditional_m_dict:
            traditional_m = traditional_m_dict[k]
            for t in T:
                sub += lpSum(x_regular[(f'{traditional_m}_q_l2', f'{traditional_m}_r_l2', t, c)] for c in C if c[1] == k and (f'{traditional_m}_q_l2', f'{traditional_m}_r_l2', t, c) in x_regular) <= U_l2[k], f"capacity_l2_{traditional_m}_{k}_{t}"

    # Solve sub
    print("Solving Subproblem...")
    status = sub.solve()
    print("Sub Status:", LpStatus[status])

    if LpStatus[status] == 'Optimal':
        sub_cost = value(sub.objective)
        deployment_cost = sum(F[m] * fixed_y[m] for m in M)
        total_cost = deployment_cost + sub_cost
        ub = min(ub, total_cost)
        print(f"Sub Cost: {sub_cost:.2f}, Total UB: {ub:.2f}")

        # Update best if improved
        if total_cost < deployment_cost + best_sub_cost:
            best_y = fixed_y.copy()
            best_sub_cost = sub_cost
            best_sub_vars = {
                'x_regular': {a: value(x_regular[a]) for a in regular_arcs},
                'x_qq': {a: value(x_qq[a]) for a in qq_arcs},
                'x_in_carryover': {a: value(x_in_carryover[a]) for a in in_carryover_arcs}
            }

        # Get duals for l1 capacities
        pi = {(m, t): sub.constraints[l1_capacity_cons[(m, t)]].pi for m in M for t in T}

        # Add optimality cut to master
        cut_name = f"opt_cut_{iter_count}"
        cut = theta >= sub_cost + lpSum(pi[(m, t)] * U_l1 * (y[(m, 'l1')] - fixed_y[m]) for m in M for t in T)
        master += cut, cut_name
        print(f"Added optimality cut: {cut_name}")

    else:
        # Infeasible: Add simple feasibility cut (force more deployments; customize as needed)
        print("Subproblem infeasible! Adding feasibility cut.")
        cut_name = f"feas_cut_{iter_count}"
        min_deploy = sum(fixed_y.values()) + 1  # Force at least one more
        master += lpSum(y[(m, 'l1')] for m in M) >= min_deploy, cut_name

#^^^ End of Benders' loop ^^^

# Record time stats
runtime_seconds = time.time() - start_time
print(f"Total runtime: {runtime_seconds:.2f} seconds")

print("\nConverged after", iter_count, "iterations. Final UB:", ub)

# ====================== DETAILED PRINTING & SAVING ======================
print("Objective Value:", ub)

# CSAM Deployments
print("\nCSAM Deployments:")
for m in M:
    if best_y and best_y.get(m, 0) > 0.5:
        print(f"y[{m}, 'l1'] = {best_y[m]:.0f}")

# Print positive CSAM l1 flows
print("\nPositive CSAM l1 flows (q_l1 to r_l1):")
with open(os.path.join(output_dir, 'csam_flows.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Facility', 'Time', 'Commodity', 'Flow'])
    for m in M:
        for t in T:
            for c in C:
                a = (f'{m}_q_l1', f'{m}_r_l1', t, c)
                if a in best_sub_vars['x_regular']:
                    flow = best_sub_vars['x_regular'][a]
                    if flow > 1e-6:
                        print(f"Arc ({m}_q_l1 -> {m}_r_l1), t={t}, commodity={c}: flow={flow:.1f}")
                        writer.writerow([m, t, str(c), flow])

# Print traditional l2 flows
print("\nPositive traditional l2 flows (q_l2 to r_l2):")
with open(os.path.join(output_dir, 'traditional_flows.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Type', 'Facility', 'Time', 'Commodity', 'Flow'])
    for k in K:
        if k in traditional_m_dict:
            traditional_m = traditional_m_dict[k]
            print(f"\nFor k={k} at {traditional_m}:")
            for t in T:
                for c in C:
                    if c[1] != k: continue
                    a = (f'{traditional_m}_q_l2', f'{traditional_m}_r_l2', t, c)
                    if a in best_sub_vars['x_regular']:
                        flow = best_sub_vars['x_regular'][a]
                        if flow > 1e-6:
                            jumping = " (jumping if 'l1')" if c[0] == 'l1' else ""
                            print(f"Arc ({traditional_m}_q_l2 -> {traditional_m}_r_l2), t={t}, commodity={c}: flow={flow:.1f}{jumping}")
                            writer.writerow([k, traditional_m, t, str(c), flow])

# Print travel flows (in-to-in)
print("\nPositive inter-facility travel flows (in-to-in):")
with open(os.path.join(output_dir, 'travel_flows.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['From Node', 'To Node', 'Time', 'Commodity', 'Flow'])
    for m1 in M:
        for m2 in M:
            if m1 == m2: continue
            for t in T:
                for c in C:
                    a = (f'{m1}_in', f'{m2}_in', t, c)
                    if a in best_sub_vars['x_regular']:
                        flow = best_sub_vars['x_regular'][a]
                        if flow > 1e-6:
                            print(f"Arc ({m1}_in -> {m2}_in), t={t}, commodity={c}: flow={flow:.1f}")
                            writer.writerow([m1, m2, t, str(c), flow])

# Print dummy flows
print("\nPositive flows on dummy arcs (unmet demand in t=2):")
with open(os.path.join(output_dir, 'dummy_flows.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Node', 'Path', 'Commodity', 'Flow'])
    for m in M:
        for lp in L:
            for c in C:
                a = (f'{m}_q_{lp}', 'dummy', 2, c)
                if a in best_sub_vars['x_regular']:
                    flow = best_sub_vars['x_regular'][a]
                    if flow > 1e-6:
                        print(f"Arc ({m}_q_{lp} -> dummy), t=2, commodity={c}: flow={flow:.1f}")
                        writer.writerow([m, lp, str(c), flow])
        for c in C:
            a = (f'{m}_in', 'dummy', 2, c)
            if a in best_sub_vars['x_regular']:
                flow = best_sub_vars['x_regular'][a]
                if flow > 1e-6:
                    print(f"Arc ({m}_in -> dummy), t=2, commodity={c}: flow={flow:.1f}")
                    writer.writerow([m, 'in', str(c), flow])

# === CRITICAL FOR QUEUE GRAPH ===
print("\nPositive in-to-q flows (queue entries):")
with open(os.path.join(output_dir, 'inq_flows.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Facility', 'Level', 'Time', 'Commodity', 'Flow'])
    for m in M:
        for lp in L:
            for t in T:
                for c in C:
                    valid = False
                    if lp == 'l1' and c[0] != 'l2':
                        valid = True
                    elif lp == 'l2' and m == traditional_m_dict.get(c[1]):
                        valid = True
                    if valid:
                        a = (f'{m}_in', f'{m}_q_{lp}', t, c)
                        if a in best_sub_vars['x_regular']:
                            flow = best_sub_vars['x_regular'][a]
                            if flow > 1e-6:
                                print(f"Arc ({m}_in -> {m}_q_{lp}), t={t}, commodity={c}: flow={flow:.1f}")
                                writer.writerow([m, lp, t, str(c), flow])

print("\nPositive queue carryover flows (q_l? to q_l?):")
with open(os.path.join(output_dir, 'qq_flows.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Facility', 'Level', 'Commodity', 'Time_From', 'Time_To', 'Flow'])
    for m in M:
        for c in C:
            for lp in L:
                a = (f'{m}_q_{lp}', f'{m}_q_{lp}', 1, c, 2)
                if a in best_sub_vars['x_qq']:
                    flow = best_sub_vars['x_qq'][a]
                    if flow > 1e-6:
                        print(f"Arc ({m}_q_{lp} -> {m}_q_{lp}), t=1 to 2, commodity={c}: flow={flow:.1f}")
                        writer.writerow([m, lp, str(c), flow])

print("\nPositive in carryover flows (in to in across time):")
with open(os.path.join(output_dir, 'in_carry_flows.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Facility', 'Commodity', 'Time_From', 'Time_To', 'Flow'])
    for m in M:
        for c in C:
            a = (f'{m}_in', f'{m}_in', 1, c, 2)
            if a in best_sub_vars['x_in_carryover']:
                flow = best_sub_vars['x_in_carryover'][a]
                if flow > 1e-6:
                    print(f"Arc ({m}_in -> {m}_in), t=1 to 2, commodity={c}: flow={flow:.1f}")
                    writer.writerow([m, str(c), flow])

# Objective component sums
print("\nObjective Component Sums:")
deployment_cost = sum(F[m] * best_y[m] for m in M)
travel_cost = sum(C_in_in * best_sub_vars['x_regular'].get(a, 0) for a in regular_arcs if '_in' in a[0] and '_in' in a[1])
queue_entry_cost = sum(C_in_q * best_sub_vars['x_regular'].get(a, 0) for a in regular_arcs if '_in' in a[0] and '_q_' in a[1])
repair_l1_cost = sum(C_q_r_l1 * best_sub_vars['x_regular'].get(a, 0) for a in regular_arcs if '_q_l1' in a[0] and '_r_l1' in a[1])
repair_l2_cost = sum(C_q_r_l2 * best_sub_vars['x_regular'].get(a, 0) for a in regular_arcs if '_q_l2' in a[0] and '_r_l2' in a[1])
carryover_cost = sum(C_q_q * best_sub_vars['x_qq'].get(a, 0) for a in qq_arcs)
in_carryover_cost = sum(C_q_q * best_sub_vars['x_in_carryover'].get(a, 0) for a in in_carryover_arcs)
dummy_cost = sum(C_dummy * best_sub_vars['x_regular'].get(a, 0) for a in regular_arcs if ('_q_' in a[0] or '_in' in a[0]) and 'dummy' in a[1])

print("Deployment (CSAM):", deployment_cost)
print("Travel (in-in):", travel_cost)
print("Queue Entry (in-q):", queue_entry_cost)
print("Repair l1 (CSAM):", repair_l1_cost)
print("Repair l2 (TM):", repair_l2_cost)
print("Carryover (q-q):", carryover_cost)
print("In Carryover (in-in):", in_carryover_cost)
print("Dummy penalty:", dummy_cost)

# ====================== STRUCTURED SUMMARY ======================
summary = {
    "run_id": run_id,
    "timestamp": timestamp,
    "experiment": EXPERIMENT_NAME,
    "max_csam_facilities": MAX_CSAM_FACILITIES,
    "seed": SEED,
    "runtime_seconds": float(runtime_seconds),
    "objective": float(ub) if 'ub' in locals() else None,
    "deployed_count": int(sum(1 for v in (best_y or {}).values() if v > 0.5)),
    "deployed_facilities": [m for m in M if (best_y or {}).get(m, 0) > 0.5],
    "unmet_demand": float(sum(best_sub_vars['x_regular'].get(a, 0) for a in regular_arcs if 'dummy' in str(a))) if best_sub_vars else 0,
    "iterations": int(iter_count) if 'iter_count' in locals() else 0,
    "deployment_cost": float(sum(F[m] * (best_y or {}).get(m, 0) for m in M)),
    "dummy_cost": float(sum(C_dummy * best_sub_vars['x_regular'].get(a, 0) for a in regular_arcs if 'dummy' in str(a))) if best_sub_vars else 0,
}

with open(exp_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Experiment Summary ===")
print(json.dumps(summary, indent=2))

# Copy CSVs
for csv_name in ["csam_flows.csv", "traditional_flows.csv", "inq_flows.csv", 
                 "qq_flows.csv", "in_carry_flows.csv", "dummy_flows.csv", "travel_flows.csv"]:
    src = Path(output_dir) / csv_name
    if src.exists():
        shutil.copy(src, exp_dir / csv_name)

# Restore stdout
sys.stdout = original_stdout
log_file.close()

print(f"\nExperiment completed. All results saved in: {exp_dir}")