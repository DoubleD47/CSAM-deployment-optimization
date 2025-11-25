# Model: 5 commodities and 10 nodes. Adding print statements that are collected in an output file so that I can track larger runs without scrolling endlessly through the console.
# Adding CSAM deployment limits. 
# Also generating CSV output files for investigation and later graphing. 

import os
import sys
import csv
from pulp import *
import numpy as np
from collections import defaultdict

# Capture all prints to file while printing to console
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Ensure real-time output
    def flush(self):
        for f in self.files:
            f.flush()

# Create output directory at repo root if it doesn't exist
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Go up one level from 'scripts' to root
output_dir = os.path.join(repo_root, 'output')


# Open log file in the output directory
log_file_path = os.path.join(output_dir, 'output_gr9_c_3.txt')
log_file = open(log_file_path, 'w')
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, log_file)

# Set random seed
SEED = 456
np.random.seed(SEED)
print(f"Random seed set to: {SEED}")

# Define sets
M = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10']
traditional_m_dict = {'k1': 'm1', 'k2': 'm2', 'k3': 'm3', 'k4': 'm4', 'k5': 'm5'}  # Type-specific traditional facilities at these nodes
L = ['l1', 'l2']
K = ['k1', 'k2', 'k3', 'k4', 'k5']
C = [(l, k) for l in L for k in K]
T = [1, 2]

##### Nodes #####
nodes = []
for t in T:
    for c in C:
        # Source and sink per t, crumble
        nodes.append(('source', t, c))
        nodes.append(('sink', t, c))
        # Dummy and ss per c
        if t == 2:
            nodes.append(('dummy', t, c))
        nodes.append(('ss', None, c))
        for m in M:
            # Common in node per m, t, c
            nodes.append((f'{m}_in', t, c))
            # l1 path nodes for all m
            nodes.extend([
                (f'{m}_q_l1', t, c),
                (f'{m}_r_l1', t, c),
                (f'{m}_out_l1', t, c),
            ])
            # l2 path nodes only at k-specific traditional_m
            traditional_m_for_k = traditional_m_dict.get(c[1])
            if m == traditional_m_for_k:
                nodes.extend([
                    (f'{m}_q_l2', t, c),
                    (f'{m}_r_l2', t, c),
                    (f'{m}_out_l2', t, c),
                ])

##### Arcs #####
regular_arcs = []
qq_arcs = []

# New: carryover arcs at the 'in' node level
in_carryover_arcs = []
for c in C:
    for m in M:
        in_carryover_arcs.append((f'{m}_in', f'{m}_in', 1, c, 2))


for t in T:
    for c in C:
        # source-to-in arcs (will force flow == D[m,t,c])
        for m in M:
            regular_arcs.append(('source', f'{m}_in', t, c))
        # in-to-in arcs (travel between facilities)
        for m1 in M:
            for m2 in M:
                if m1 != m2:
                    regular_arcs.append((f'{m1}_in', f'{m2}_in', t, c))
        # in-to-q_lp arcs (asymmetric for l1 commodities)
        for m in M:
            # to l1 path (only if not l2 commodity)
            if c[0] != 'l2':
                regular_arcs.append((f'{m}_in', f'{m}_q_l1', t, c))
            # to l2 path (only at k-specific traditional_m, for all c)
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

# PuLP Model
model = LpProblem("CSAM_Deployment", LpMinimize)

# Variables
x_regular = LpVariable.dicts("flow_regular", regular_arcs, lowBound=0, cat='Continuous')
x_qq = LpVariable.dicts("flow_qq", qq_arcs, lowBound=0, cat='Continuous')
x_in_carryover = LpVariable.dicts("flow_in_carry", in_carryover_arcs, lowBound=0, cat='Continuous')
y = LpVariable.dicts("open_l1", [(m, 'l1') for m in M], cat='Binary')

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
C_in_in = 1  # Travel cost per unit
C_in_q = 10  # Queue entry cost per unit
C_q_r_l1 = 100  # CSAM repair cost per unit
C_q_r_l2 = 200  # Traditional repair cost per unit
C_q_q = 5  # Carryover cost per unit
C_dummy = 1000  # Dummy penalty per unit
U_l1 = 50  # CSAM capacity per m per t
U_l2 = {k: 100 for k in K}  # Traditional capacity per k per t
max_csam_facilities = 3

# Objective
model += (
    lpSum(F[m] * y[(m, 'l1')] for m in M) +
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

# Force demand injection at origin m_in
for m in M:
    for t in T:
        for c in C:
            a = ('source', f'{m}_in', t, c)
            if a in x_regular:
                model += x_regular[a] == D.get((m, t, c), 0), f"demand_inject_{m}_{t}_{c[0]}_{c[1]}"

# Flow conservation
constraint_names = set()
constraint_counter = 0
unique_nodes = set(nodes)  # Avoid duplicates
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
    model += constraint, constraint_name

# Capacity constraints
for m in M:
    for t in T:
        # l1 capacity (aggregated across all c for each m, t)
        model += lpSum(x_regular[(f'{m}_q_l1', f'{m}_r_l1', t, c)] for c in C if (f'{m}_q_l1', f'{m}_r_l1', t, c) in x_regular) <= U_l1 * y[(m, 'l1')], f"capacity_l1_{m}_{t}"

for k in K:
    if k in traditional_m_dict:
        traditional_m = traditional_m_dict[k]
        for t in T:
            model += lpSum(x_regular[(f'{traditional_m}_q_l2', f'{traditional_m}_r_l2', t, c)] for c in C if c[1] == k and (f'{traditional_m}_q_l2', f'{traditional_m}_r_l2', t, c) in x_regular) <= U_l2[k], f"capacity_l2_{traditional_m}_{k}_{t}"

# CSAM deployment limit
model += lpSum(y[(m, 'l1')] for m in M) <= max_csam_facilities, "max_csam_facilities"

# Solve and capture solver output
solver_output = []
original_write = sys.stdout.write
def capture_solver_output(obj):
    solver_output.append(obj)
    original_write(obj)
sys.stdout.write = capture_solver_output
model.solve()
sys.stdout.write = original_write

# Append solver output to log file
print("\nSolver Performance Metrics:")
for line in solver_output:
    print(line.strip())

# Print results (adapted from original)
print("Status:", LpStatus[model.status])
print("Objective Value:", value(model.objective))

# Print open CSAM facilities
print("\nCSAM Deployments:")
for m in M:
    if value(y[(m, 'l1')]) > 0.5:
        print(f"y[{m}, 'l1'] = {value(y[(m, 'l1')]):.0f}")

# Print positive CSAM l1 flows
print("\nPositive CSAM l1 flows (q_l1 to r_l1):")
with open('csam_flows.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Facility', 'Time', 'Commodity', 'Flow'])
    for m in M:
        for t in T:
            for c in C:
                a = (f'{m}_q_l1', f'{m}_r_l1', t, c)
                if a in x_regular:
                    flow = value(x_regular[a])
                    if flow > 1e-6:
                        print(f"Arc ({m}_q_l1 -> {m}_r_l1), t={t}, commodity={c}: flow={flow:.1f}")
                        writer.writerow([m, t, str(c), flow])

# Print positive traditional l2 flows, noting if jumping
print("\nPositive traditional l2 flows (q_l2 to r_l2):")
with open('trad_flows.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Type', 'Facility', 'Time', 'Commodity', 'Flow'])
    for k in K:
        if k in traditional_m_dict:
            traditional_m = traditional_m_dict[k]
            print(f"\nFor k={k} at {traditional_m}:")
            for t in T:
                for c in C:
                    if c[1] != k: continue  # Only matching k
                    a = (f'{traditional_m}_q_l2', f'{traditional_m}_r_l2', t, c)
                    if a in x_regular:
                        flow = value(x_regular[a])
                        if flow > 1e-6:
                            jumping = " (jumping if 'l1')" if c[0] == 'l1' else ""
                            print(f"Arc ({traditional_m}_q_l2 -> {traditional_m}_r_l2), t={t}, commodity={c}: flow={flow:.1f}{jumping}")
                            writer.writerow([k, traditional_m, t, str(c), flow])

# Print travel flows
print("\nPositive inter-facility travel flows (in-to-in):")
with open('travel_flows.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['From Node', 'To Node', 'Time', 'Commodity', 'Flow'])
    for m1 in M:
        for m2 in M:
            if m1 == m2: continue
            for t in T:
                for c in C:
                    a = (f'{m1}_in', f'{m2}_in', t, c)
                    if a in x_regular:
                        flow = value(x_regular[a])
                        if flow > 1e-6:
                            print(f"Arc ({m1}_in -> {m2}_in), t={t}, commodity={c}: flow={flow:.1f}")
                            writer.writerow([m1, m2, t, str(c), flow])

# Print dummy flows
print("\nPositive flows on dummy arcs (unmet demand in t=2):")
with open('dummy_flows.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Node', 'Path', 'Commodity', 'Flow'])
    for m in M:
        for lp in L:
            for c in C:
                a = (f'{m}_q_{lp}', 'dummy', 2, c)
                if a in x_regular:
                    flow = value(x_regular[a])
                    if flow > 1e-6:
                        print(f"Arc ({m}_q_{lp} -> dummy), t=2, commodity={c}: flow={flow:.1f}")
                        writer.writerow([m, lp, str(c), flow])
        for c in C:
            a = (f'{m}_in', 'dummy', 2, c)
            if a in x_regular:
                flow = value(x_regular[a])
                if flow > 1e-6:
                    print(f"Arc ({m}_in -> dummy), t=2, commodity={c}: flow={flow:.1f}")
                    writer.writerow([m, 'in', str(c), flow])
                        
# Print positive in-to-q flows (queue entries)
print("\nPositive in-to-q flows (queue entries):")
with open('inq_flows.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Facility', 'Level', 'Time', 'Commodity', 'Flow'])
    for m in M:
        for lp in L:
            for t in T:
                for c in C:
                    # Validate for l1 path: only if commodity not l2
                    if lp == 'l1' and c[0] != 'l2':
                        valid = True
                    # Validate for l2 path: only at traditional m for the k
                    elif lp == 'l2' and m == traditional_m_dict.get(c[1]):
                        valid = True
                    else:
                        valid = False
                    if valid:
                        a = (f'{m}_in', f'{m}_q_{lp}', t, c)
                        if a in x_regular:
                            flow = value(x_regular[a])
                            if flow > 1e-6:
                                print(f"Arc ({m}_in -> {m}_q_{lp}), t={t}, commodity={c}: flow={flow:.1f}")
                                writer.writerow([m, lp, t, str(c), flow])

# Print positive in carryover flows
print("\nPositive in carryover flows (in to in across time):")
with open('in_carry_flows.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Facility', 'Commodity', 'Flow'])
    for m in M:
        for c in C:
            a = (f'{m}_in', f'{m}_in', 1, c, 2)
            if a in x_in_carryover:
                flow = value(x_in_carryover[a])
                if flow > 1e-6:
                    print(f"Arc ({m}_in -> {m}_in), t=1 to 2, commodity={c}: flow={flow:.1f}")
                    writer.writerow([m, str(c), flow])

# Objective component sums (after solve)
print("\nObjective Component Sums:")
deployment_cost = sum(value(F[m] * y[(m, 'l1')]) for m in M)
travel_cost = sum(value(C_in_in * x_regular[a]) for a in regular_arcs if '_in' in a[0] and '_in' in a[1])
queue_entry_cost = sum(value(C_in_q * x_regular[a]) for a in regular_arcs if '_in' in a[0] and '_q_' in a[1])
repair_l1_cost = sum(value(C_q_r_l1 * x_regular[a]) for a in regular_arcs if '_q_l1' in a[0] and '_r_l1' in a[1])
repair_l2_cost = sum(value(C_q_r_l2 * x_regular[a]) for a in regular_arcs if '_q_l2' in a[0] and '_r_l2' in a[1])
carryover_cost = sum(value(C_q_q * x_qq[a]) for a in qq_arcs)
in_carryover_cost = sum(value(C_q_q * x_in_carryover[a]) for a in in_carryover_arcs)
r_out_cost = sum(value(0.1 * x_regular[a]) for a in regular_arcs if '_r_' in a[0] and '_out_' in a[1])
out_sink_cost = sum(value(0.1 * x_regular[a]) for a in regular_arcs if '_out_' in a[0] and 'sink' in a[1])
sink_ss_cost = sum(value(0.1 * x_regular[a]) for a in regular_arcs if 'sink' in a[0] and 'ss' in a[1])
dummy_cost = sum(value(C_dummy * x_regular[a]) for a in regular_arcs if ('_q_' in a[0] or '_in' in a[0]) and 'dummy' in a[1])
dummy_ss_cost = sum(value(0.1 * x_regular[a]) for a in regular_arcs if 'dummy' in a[0] and 'ss' in a[1])

print("Deployment (CSAM):", deployment_cost)
print("Travel (in-in):", travel_cost)
print("Queue Entry (in-q):", queue_entry_cost)
print("Repair l1 (CSAM):", repair_l1_cost)
print("Repair l2 (TM):", repair_l2_cost)
print("Carryover (q-q):", carryover_cost)
print("In Carryover (in-in):", in_carryover_cost)
print("R to Out:", r_out_cost)
print("Out to Sink:", out_sink_cost)
print("Sink to SS:", sink_ss_cost)
print("Dummy (q-dummy):", dummy_cost)
print("Dummy to SS:", dummy_ss_cost)

# ... (Remaining cost breakdowns and verifications from original can be adapted similarly if needed)

# Restore original stdout and close log file
sys.stdout = original_stdout
log_file.close()

print("Script completed. All output logged to 'output_gr9_c_3.txt' and CSVs for graphing.")