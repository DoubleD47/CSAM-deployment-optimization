from pulp import *
import numpy as np

# Set random seed
SEED = 123
np.random.seed(SEED)
print(f"Random seed set to: {SEED}")

# Define sets
M = ['m1', 'm2', 'm3']
L = ['l1', 'l2']
K = ['k1', 'k2']
C = [(l, k) for l in L for k in K]
T = [1, 2]

##### Nodes #####
nodes = []
for m in M:
    for t in T:
        for c in C:
            nodes.append((f'{m}_in', t, c))
            for lp in L:
                nodes.extend([
                    (f'{m}_q_{lp}', t, c),
                    (f'{m}_r_{lp}', t, c),
                    (f'{m}_out_{lp}', t, c),
                ])
            nodes.append(('sink', t, c))
for c in C:
    nodes.extend([('dummy', 2, c), ('ss', None, c)])
for t in T:
    for c in C:
        nodes.append(('source', t, c))

# Print nodes for debugging (optional)
# print("\nNodes:")
# for n, t, c in nodes:
#     print(f"Node: {n}, t={t}, commodity={c}")

##### Arcs #####
regular_arcs = []
qq_arcs = []
for t in T:
    for c in C:
        # source-to-in arcs
        for m in M:
            regular_arcs.append(('source', f'{m}_in', t, c))
        # in-to-in arcs
        for m1 in M:
            for m2 in M:
                if m1 != m2:
                    regular_arcs.append((f'{m1}_in', f'{m2}_in', t, c))
        # in-to-q_lp arcs (asymmetric: l2 c cannot go to l1 path)
        for m in M:
            for lp in L:
                if c[0] == 'l2' and lp == 'l1':
                    continue
                regular_arcs.append((f'{m}_in', f'{m}_q_{lp}', t, c))
        # q_lp-to-r_lp arcs
        for m in M:
            for lp in L:
                regular_arcs.append((f'{m}_q_{lp}', f'{m}_r_{lp}', t, c))
        # r_lp-to-out_lp arcs
        for m in M:
            for lp in L:
                regular_arcs.append((f'{m}_r_{lp}', f'{m}_out_{lp}', t, c))
        # out_lp-to-sink arcs
        for m in M:
            for lp in L:
                regular_arcs.append((f'{m}_out_{lp}', 'sink', t, c))
        # sink-to-ss arcs
        regular_arcs.append(('sink', 'ss', t, c))
        # q_lp-to-q_lp arcs (carryover per path)
        if t == 1:
            for m in M:
                for lp in L:
                    qq_arcs.append((f'{m}_q_{lp}', f'{m}_q_{lp}', t, c, t+1))
        # q_lp-to-dummy and dummy-to-ss arcs (per path)
        if t == 2:
            for m in M:
                for lp in L:
                    regular_arcs.append((f'{m}_q_{lp}', 'dummy', t, c))
            regular_arcs.append(('dummy', 'ss', t, c))

# Print edges for debugging (optional)
# print("\nEdges:")
# for i, j, t, c in regular_arcs:
#     print(f"Edge: ({i} -> {j}), t={t}, commodity={c}")
# for i, j, t, c, t2 in qq_arcs:
#     print(f"Edge: ({i} -> {j}), t={t} to t={t2}, commodity={c}")

# Parameters
D = {(m, t, c): np.random.uniform(1, 10) for m in M for t in T for c in C}
print("\nDemands:")
for (m, t, c), d in D.items():
    print(f"D({m}, t={t}, {c}) = {d:.1f}")
F = {m: np.random.uniform(800, 1200) for m in M}
C_in_in = np.random.uniform(2, 8)
C_in_q = np.random.uniform(1, 5)
C_q_r_l1 = np.random.uniform(2, 5)  # Cost for l1 path
C_q_r_l2 = np.random.uniform(0.5, 2)  # Cost for l2 path
C_q_q = np.random.uniform(8, 12)
C_r_out = np.random.uniform(0.5, 2)
C_out_sink = 0.1
C_sink_ss = 0.1
C_q_dummy = np.random.uniform(50, 150)
C_dummy_ss = 0.1
U_l1 = 10  # Small to force some l1 jumping to l2
U_l2 = 150

# PuLP Model
model = LpProblem("Facility_Location_MultiCommodity", LpMinimize)

# Variables
x_regular = LpVariable.dicts("flow", regular_arcs, lowBound=0, cat='Continuous')
x_qq = LpVariable.dicts("flow_qq", qq_arcs, lowBound=0, cat='Continuous')
y = LpVariable.dicts("open_l1", [(m, 'l1') for m in M], cat='Binary')

# Objective Function (costs based on path l_prime)
model += (
    lpSum(F[m] * y[(m, 'l1')] for m in M) +
    lpSum(C_in_in * x_regular[a] for a in regular_arcs if '_in' in a[0] and '_in' in a[1]) +
    lpSum(C_in_q * x_regular[a] for a in regular_arcs if '_in' in a[0] and '_q_' in a[1]) +
    lpSum(C_q_r_l1 * x_regular[a] for a in regular_arcs if '_q_l1' in a[0] and '_r_l1' in a[1]) +
    lpSum(C_q_r_l2 * x_regular[a] for a in regular_arcs if '_q_l2' in a[0] and '_r_l2' in a[1]) +
    lpSum(C_q_q * x_qq[a] for a in qq_arcs) +
    lpSum(C_r_out * x_regular[a] for a in regular_arcs if '_r_' in a[0] and '_out_' in a[1]) +
    lpSum(C_out_sink * x_regular[a] for a in regular_arcs if '_out_' in a[0] and 'sink' in a[1]) +
    lpSum(C_sink_ss * x_regular[a] for a in regular_arcs if 'sink' in a[0] and 'ss' in a[1]) +
    lpSum(C_q_dummy * x_regular[a] for a in regular_arcs if '_q_' in a[0] and 'dummy' in a[1]) +
    lpSum(C_dummy_ss * x_regular[a] for a in regular_arcs if 'dummy' in a[0] and 'ss' in a[1])
)

# Constraints
print("\nDefining constraints...")
constraint_names = set()
constraint_counter = 0
for node, t_node, comm in nodes:
    incoming = [a for a in regular_arcs if a[1] == node and a[2] == t_node and a[3] == comm]
    outgoing = [a for a in regular_arcs if a[0] == node and a[2] == t_node and a[3] == comm]
    incoming_qq = [a for a in qq_arcs if a[1] == node and a[4] == t_node and a[3] == comm]
    outgoing_qq = [a for a in qq_arcs if a[0] == node and a[2] == t_node and a[3] == comm]
    
    constraint_name = f"flow_conservation_{constraint_counter}_{node.replace('_', '-')}_{t_node}_{comm[0]}_{comm[1]}"
    if constraint_name in constraint_names:
        raise ValueError(f"Duplicate constraint name: {constraint_name}")
    constraint_names.add(constraint_name)
    constraint_counter += 1
    
    if node == 'source':
        total_demand_t_c = sum(D.get((m, ti, ci), 0) for m in M for ti in T for ci in C if ti == t_node and ci == comm)
        constraint = (
            lpSum(x_regular[a] for a in outgoing) +
            lpSum(x_qq[a] for a in outgoing_qq) == total_demand_t_c
        )
    elif node == 'ss' and t_node is None:
        total_demand_c = sum(D.get((m, ti, ci), 0) for m in M for ti in T for ci in C if ci == comm)
        constraint = (
            lpSum(x_regular[a] for a in regular_arcs if a[1] == node and a[3] == comm) == total_demand_c
        )
    else:
        constraint = (
            lpSum(x_regular[a] for a in incoming) +
            lpSum(x_qq[a] for a in incoming_qq) ==
            lpSum(x_regular[a] for a in outgoing) +
            lpSum(x_qq[a] for a in outgoing_qq)
        )
    model += constraint, constraint_name

# Capacity constraints (aggregated per path)
for m in M:
    for t in T:
        constraint_name_l1 = f"capacity_l1_{m}_{t}"
        model += lpSum(x_regular[(f'{m}_q_l1', f'{m}_r_l1', t, c)] for c in C if (f'{m}_q_l1', f'{m}_r_l1', t, c) in x_regular) <= U_l1 * y[(m, 'l1')], constraint_name_l1
        constraint_name_l2 = f"capacity_l2_{m}_{t}"
        model += lpSum(x_regular[(f'{m}_q_l2', f'{m}_r_l2', t, c)] for c in C if (f'{m}_q_l2', f'{m}_r_l2', t, c) in x_regular) <= U_l2, constraint_name_l2

# Solve
print("Solving model...")
model.solve()
print("Status:", LpStatus[model.status])
print("Objective:", value(model.objective))

# Check feasibility
if model.status != 1:
    print("Error: Model is not optimal.")
    exit(1)

# Print deployments and flows to verify jumping
print("\nFacility openings:")
for m in M:
    print(f"y[{m}, 'l1'] = {value(y[(m, 'l1')])}")

print("\nPositive flows on CSAM (l1) repair paths:")
for m in M:
    for t in T:
        for c in C:
            a = (f'{m}_q_l1', f'{m}_r_l1', t, c)
            if a in x_regular:
                flow = value(x_regular[a])
                if flow > 1e-6:
                    print(f"Arc ({m}_q_l1 -> {m}_r_l1), t={t}, commodity={c}: flow={flow:.1f} (flexible if 'l1', invalid if 'l2')")

print("\nPositive flows on traditional (l2) repair paths by flexible (l1) commodities (jumping):")
for m in M:
    for t in T:
        for c in C:
            if c[0] == 'l1':
                a = (f'{m}_q_l2', f'{m}_r_l2', t, c)
                if a in x_regular:
                    flow = value(x_regular[a])
                    if flow > 1e-6:
                        print(f"Arc ({m}_q_l2 -> {m}_r_l2), t={t}, commodity={c}: flow={flow:.1f}")

# Verify total flow
total_demand = sum(D.values())
total_ss_inflow = sum(value(x_regular[a]) for a in regular_arcs if a[1] == 'ss')
print(f"Total demand: {total_demand:.1f}")
print(f"Total inflow to ss: {total_ss_inflow:.1f}")

print("Script completed")