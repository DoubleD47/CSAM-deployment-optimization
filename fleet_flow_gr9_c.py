# The model has 15 commodities and 50 nodes.

from pulp import *
import numpy as np

# Set random seed
SEED = 123
np.random.seed(SEED)
print(f"Random seed set to: {SEED}")

# Define sets
M = [f'm{i}' for i in range(1, 51)]  # 50 nodes: m1 to m50
traditional_m_dict = {f'k{i}': f'm{i}' for i in range(1, 16)}  # Type-specific traditional facilities for k1-m1 to k15-m15
L = ['l1', 'l2']
K = [f'k{i}' for i in range(1, 16)]  # 15 commodity types: k1 to k15
C = [(l, k) for l in L for k in K]  # 30 commodities
T = [1, 2]

##### Nodes #####
nodes = []
for t in T:
    for c in C:
        # Source and sink per t, c
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
            regular_arcs.append(('dummy', 'ss', t, c))

# Parameters
D = {(m, t, c): np.random.uniform(1, 10) for m in M for t in T for c in C}
print("\nDemands:")
for (m, t, c), d in D.items():
    print(f"D({m}, t={t}, {c}) = {d:.1f}")
F = {m: np.random.uniform(100, 300) for m in M}  # Lower to encourage deployment
C_in_in = np.random.uniform(2, 8)  # Travel cost
C_in_q = np.random.uniform(1, 5)
C_q_r_l1 = np.random.uniform(0.5, 2)  # Cheaper for CSAM
C_q_r_l2 = np.random.uniform(2, 5)  # Higher for traditional
C_q_q = np.random.uniform(8, 12)
C_r_out = np.random.uniform(0.5, 2)
C_out_sink = 0.1
C_sink_ss = 0.1
C_q_dummy = np.random.uniform(50, 150)
C_dummy_ss = 0.1
U_l1 = 50  # Adjusted
U_l2 = {k: 150 for k in K}  # Type-specific for l2

# PuLP Model
model = LpProblem("Facility_Location_MultiCommodity", LpMinimize)

# Variables
x_regular = LpVariable.dicts("flow", regular_arcs, lowBound=0, cat='Continuous')
x_qq = LpVariable.dicts("flow_qq", qq_arcs, lowBound=0, cat='Continuous')
y = LpVariable.dicts("open_l1", [(m, 'l1') for m in M], cat='Binary')

# Objective Function
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
    
    if not (incoming or outgoing or incoming_qq or outgoing_qq):
        continue
    
    constraint_name = f"flow_conservation_{constraint_counter}_{n.replace('_', '-')}_{t_node if t_node else 'None'}_{comm[0]}_{comm[1]}"
    if constraint_name in constraint_names:
        raise ValueError(f"Duplicate constraint name: {constraint_name}")
    constraint_names.add(constraint_name)
    constraint_counter += 1
    
    if n == 'source':
        total_demand_t_c = sum(D.get((m, t_node, comm), 0) for m in M)
        constraint = lpSum(x_regular[a] for a in outgoing) + lpSum(x_qq[a] for a in outgoing_qq) == total_demand_t_c
    elif n == 'ss' and t_node is None:
        total_demand_c = sum(D.get((m, ti, comm), 0) for m in M for ti in T)
        constraint = lpSum(x_regular[a] for a in incoming) + lpSum(x_qq[a] for a in incoming_qq) == total_demand_c
    else:
        constraint = (
            lpSum(x_regular[a] for a in incoming) + 
            lpSum(x_qq[a] for a in incoming_qq) ==
            lpSum(x_regular[a] for a in outgoing) + 
            lpSum(x_qq[a] for a in outgoing_qq)
        )
    model += constraint, constraint_name

# Capacity constraints
for m in M:
    for t in T:
        # l1 capacity (aggregated across all c for each m, t)
        model += lpSum(x_regular[(f'{m}_q_l1', f'{m}_r_l1', t, c)] for c in C if (f'{m}_q_l1', f'{m}_r_l1', t, c) in x_regular) <= U_l1 * y[(m, 'l1')], f"capacity_l1_{m}_{t}"

for t in T:
    for k in K:
        traditional_m = traditional_m_dict[k]
        # l2 capacity per k, t at k's traditional_m (sum over l in c[0])
        model += lpSum(x_regular[(f'{traditional_m}_q_l2', f'{traditional_m}_r_l2', t, c)] for c in C if c[1] == k and (f'{traditional_m}_q_l2', f'{traditional_m}_r_l2', t, c) in x_regular) <= U_l2[k], f"capacity_l2_{k}_{t}"

# Solve
print("Solving model...")
model.solve()
print("Status:", LpStatus[model.status])
print("Objective:", value(model.objective))

if model.status != 1:
    print("Error: Model is not optimal.")
else:
    # Print deployments
    print("\nFacility openings (CSAM l1):")
    for m in M:
        print(f"y[{m}, 'l1'] = {value(y[(m, 'l1')])}")

    # Print positive repair flows
    print("\nPositive flows on CSAM (l1) repair paths (only flexible l1 commodities):")
    for m in M:
        for t in T:
            for c in C:
                if c[0] == 'l2': continue  # No l2 on l1
                a = (f'{m}_q_l1', f'{m}_r_l1', t, c)
                if a in x_regular:
                    flow = value(x_regular[a])
                    if flow > 1e-6:
                        print(f"Arc ({m}_q_l1 -> {m}_r_l1), t={t}, commodity={c}: flow={flow:.1f}")

    print("\nPositive flows on traditional (l2) repair paths:")
    for k in K:
        traditional_m = traditional_m_dict[k]
        print(f"\nFor k={k} at {traditional_m}:")
        for t in T:
            for c in C:
                if c[1] != k: continue  # Only matching k
                a = (f'{traditional_m}_q_l2', f'{traditional_m}_r_l2', t, c)
                if a in x_regular:
                    flow = value(x_regular[a])
                    if flow > 1e-6:
                        print(f"Arc ({traditional_m}_q_l2 -> {traditional_m}_r_l2), t={t}, commodity={c}: flow={flow:.1f} (jumping if 'l1')")

    # Print travel flows
    print("\nPositive inter-facility travel flows (in-to-in):")
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

    # Print dummy flows
    print("\nPositive flows on dummy arcs (unmet demand in t=2):")
    for m in M:
        for lp in L:
            for c in C:
                a = (f'{m}_q_{lp}', 'dummy', 2, c)
                if a in x_regular:
                    flow = value(x_regular[a])
                    if flow > 1e-6:
                        print(f"Arc ({m}_q_{lp} -> dummy), t=2, commodity={c}: flow={flow:.1f}")

    # Verify total
    total_demand = sum(D.values())
    total_ss_inflow = sum(value(x_regular[a]) for a in regular_arcs if a[1] == 'ss')
    print(f"\nTotal demand: {total_demand:.1f}")
    print(f"Total inflow to ss: {total_ss_inflow:.1f}")

print("Script completed")