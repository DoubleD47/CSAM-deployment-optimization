from pulp import *
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import csv

# Set random seed
SEED = 123
np.random.seed(SEED)
print(f"Random seed set to: {SEED}")

# Define sets
M = ['m1', 'm2', 'm3'] # main nodes, m
L = ['l1', 'l2'] # repair types, l1 = CSAM, l2 = traditional repair
K = ['k1', 'k2'] # vehicle types, each k# is a different vehicle
C = [(l, k) for l in L for k in K] # The C set is a tuple crossing L and K sets
T = [1, 2] # These are the different time periods modeled

##### Nodes #####
# Define node splits based on sets
nodes = []
for m in M:
    for t in T:
        for c in C:
            nodes.extend([
                (f'{m}_in', t, c),
                (f'{m}_q', t, c),
                (f'{m}_r', t, c),
                (f'{m}_out', t, c),
                ('sink', t, c) # we need a sink node for each tuple, C in each time period, T that will transfer all met demand to a super-sink where all demand terminates
            ])
for c in C:
    nodes.extend([('dummy', 2, c), ('ss', None, c)]) # we need a dummy arc for each tuple in C for unmet demand to go to in the final time period (because in the previous time periods it is passed queue-to-queue), where we have the super sink for each tuple
for t in T:
    for c in C:
        nodes.append(('source', t, c)) # we need a source node for all demand in every time period in T and every tuple in C

# in our first stage decisions, we decide where to deploy CSAM mobile facilities based on the demand we see prior to deployment (or based on probability distributions)
# in our second stage decisions, we decide where to allocate the demand that we see in future periods

# Print nodes for debugging
print("\nNodes:")
for n, t, c in nodes:
    print(f"Node: {n}, t={t}, commodity={c}")

##### Arcs #####
# Manually define arcs that we also need in this network to model possible paths
regular_arcs = []
qq_arcs = []
for t in T:
    for c in C:
        # source-to-in arcs, makes a node from the source tupie in each time T to each m_in node
        for m in M:
            regular_arcs.append(('source', f'{m}_in', t, c))
        # in-to-in arcs, adds the arcs from m1 to m2 so that demand can travel between nodes...need a better way to code this that is less static
        for m1 in M:
            for m2 in M:
                if m1 != m2:
                    regular_arcs.append((f'{m1}_in', f'{m2}_in', t, c))
        # in-to-q arcs, this brings the demand from the m_in to the queue nodes for each time in T and each tuple C...last time OE said I needed to able to jump from (l1,k#) to (l2,k#) processes,
        for m in M:
            regular_arcs.append((f'{m}_in', f'{m}_q', t, c))
        # q-to-r arcs, these bring the demand from queue to the repair nodes
        for m in M:
            regular_arcs.append((f'{m}_q', f'{m}_r', t, c))
        # r-to-out arcs, this brings the demand from the repair nodes to the m_out nodes
        for m in M:
            regular_arcs.append((f'{m}_r', f'{m}_out', t, c))
        # out-to-sink arcs, bringing completed demand from the m_out nodes to the sink nodes for each time, T and tuple, C
        for m in M:
            regular_arcs.append((f'{m}_out', 'sink', t, c))
        # sink-to-ss arcs, this brings the completed demand to the super sink in each time period (why do I need a super sink in each time period?)
        regular_arcs.append(('sink', 'ss', t, c))
        # q-to-q arcs, bringing demand from one queue period to the next (t to t+1)
        if t == 1:
            for m in M:
                qq_arcs.append((f'{m}_q', f'{m}_q', t, c, t+1))
        # q-to-dummy and dummy-to-ss arcs, these nodes take unmet demand in the final time period and take them to the super sink
        if t == 2:
            for m in M:
                regular_arcs.append((f'{m}_q', 'dummy', t, c))
            regular_arcs.append(('dummy', 'ss', t, c))

# Print edges for debugging
print("\nEdges:")
for i, j, t, c in regular_arcs:
    print(f"Edge: ({i} -> {j}), t={t}, commodity={c}")
for i, j, t, c, t2 in qq_arcs:
    print(f"Edge: ({i} -> {j}), t={t} to t={t2}, commodity={c}")

# Parameters
D = {(m, t, c): np.random.uniform(1, 10) for m in M for t in T for c in C} # Demand for each main node, time period, and commodity 
print("\nDemands:")
for (m, t, c), d in D.items(): # print the demand for each main node, time period, and commodity
    print(f"D({m}, t={t}, {c}) = {d:.1f}")
F = {m: np.random.uniform(800, 1200) for m in M} # Fixed costs for opening a facility m
C_in_in = np.random.uniform(2, 8)
C_in_q = np.random.uniform(1, 5)
C_q_r = {(m, t, c): np.random.uniform(2, 5) if c[0] == 'l1' else np.random.uniform(0.5, 2)
         for m in M for t in T for c in C}
C_q_q = np.random.uniform(8, 12)
C_r_out = np.random.uniform(0.5, 2)
C_out_sink = 0.1
C_sink_ss = 0.1
C_q_dummy = np.random.uniform(50, 150)
C_dummy_ss = 0.1
U_l1 = 50
U_l2 = 150

# PuLP Model
model = LpProblem("Facility_Location_MultiCommodity", LpMinimize)

# Variables
x_regular = LpVariable.dicts("flow", [(i, j, t, c) for i, j, t, c in regular_arcs], lowBound=0, cat='Continuous')
x_qq = LpVariable.dicts("flow_qq", [(i, j, t, c, t2) for i, j, t, c, t2 in qq_arcs], lowBound=0, cat='Continuous')
y = LpVariable.dicts("open_l1", [(m, 'l1') for m in M], cat='Binary')

# Objective Function: F[m] is the fixed cost for opening a facility m, C_in_in, C_in_q, etc. are costs per unit flow, y[m, 'l1'] is a binary variable indicating if facility m is open for l1 repair, and the C_q_r, C_q_q, C_r_out, C_out_sink, C_sink_ss, C_q_dummy, and C_dummy_ss are costs associated with the respective arcs.
model += (
    lpSum(F[m] * y[(m, 'l1')] for m in M) +
    lpSum(C_in_in * x_regular[(i, j, t, c)] for i, j, t, c in regular_arcs if '_in' in i and '_in' in j) +
    lpSum(C_in_q * x_regular[(i, j, t, c)] for i, j, t, c in regular_arcs if '_in' in i and '_q' in j) +
    lpSum(C_q_r[(m, t, c)] * x_regular[(f'{m}_q', f'{m}_r', t, c)] for m in M for t in T for c in C) +
    lpSum(C_q_q * x_qq[(i, j, t, c, t2)] for i, j, t, c, t2 in qq_arcs) +
    lpSum(C_r_out * x_regular[(i, j, t, c)] for i, j, t, c in regular_arcs if '_r' in i and '_out' in j) +
    lpSum(C_out_sink * x_regular[(i, j, t, c)] for i, j, t, c in regular_arcs if '_out' in i and 'sink' in j) +
    lpSum(C_sink_ss * x_regular[(i, j, t, c)] for i, j, t, c in regular_arcs if 'sink' in i and 'ss' in j) +
    lpSum(C_q_dummy * x_regular[(i, j, t, c)] for i, j, t, c in regular_arcs if '_q' in i and 'dummy' in j) +
    lpSum(C_dummy_ss * x_regular[(i, j, t, c)] for i, j, t, c in regular_arcs if 'dummy' in i and 'ss' in j)
)

# Constraints: We need to ensure flow conservation at each node, capacity constraints for arcs, and that the binary variables y[m, 'l1'] are set correctly based on the flow.
print("\nDefining constraints...")
# Flow conservation constraints
constraint_names = set()
constraint_counter = 0
for n, t, c in nodes:
    incoming = [(i, j, ti, ci) for i, j, ti, ci in regular_arcs if j == n and ti == t and ci == c]
    outgoing = [(i, j, ti, ci) for i, j, ti, ci in regular_arcs if i == n and ti == t and ci == c]
    incoming += [(i, j, qi, ci, t2) for i, j, qi, ci, t2 in qq_arcs if j == n and t2 == t and ci == c]
    outgoing += [(i, j, qi, ci, t2) for i, j, qi, ci, t2 in qq_arcs if i == n and qi == t and ci == c]
    
    constraint_name = f"flow_conservation_{constraint_counter}_{n.replace('_', '-')}_{t}_{c[0]}_{c[1]}"
    if constraint_name in constraint_names:
        raise ValueError(f"Duplicate constraint name: {constraint_name}")
    constraint_names.add(constraint_name)
    constraint_counter += 1
    
    if n == 'source':
        total_demand_t_c = sum(D[(m, ti, ci)] for m, ti, ci in D if ti == t and ci == c)
        constraint = (
            lpSum(x_regular[a] for a in outgoing if a in x_regular) +
            lpSum(x_qq[a] for a in outgoing if a in x_qq) == total_demand_t_c
        )
    elif n == 'ss' and t is None:
        total_demand_c = sum(D[(m, ti, ci)] for m, ti, ci in D if ci == c)
        constraint = (
            lpSum(x_regular[(i, j, ti, ci)] for i, j, ti, ci in regular_arcs if j == n and ci == c) +
            lpSum(x_qq[a] for a in incoming if a in x_qq) == total_demand_c
        )
    else:
        constraint = (
            lpSum(x_regular[a] for a in incoming if a in x_regular) +
            lpSum(x_qq[a] for a in incoming if a in x_qq) ==
            lpSum(x_regular[a] for a in outgoing if a in x_regular) +
            lpSum(x_qq[a] for a in outgoing if a in x_qq)
        )
    model += constraint, constraint_name

# Capacity constraints for q-to-r arcs
for m in M:
    for t in T:
        for c in C:
            constraint_name = f"capacity_q_r_{constraint_counter}_{m}_{t}_{c[0]}_{c[1]}"
            if constraint_name in constraint_names:
                raise ValueError(f"Duplicate constraint name: {constraint_name}")
            constraint_names.add(constraint_name)
            constraint_counter += 1
            if c[0] == 'l1':
                model += x_regular[(f'{m}_q', f'{m}_r', t, c)] <= U_l1 * y[(m, 'l1')], constraint_name #This ensures flow on CSAM repair arcs is bounded by U_l1 only when y[(m, 'l1')] = 1; otherwise, it is forced to 0.
            else:
                model += x_regular[(f'{m}_q', f'{m}_r', t, c)] <= U_l2, constraint_name

# Solve
print("Solving model...")
model.solve()
print("Status:", LpStatus[model.status])
print("Objective:", value(model.objective))

# Check feasibility
if model.status != 1:
    print("Error: Model is not optimal, skipping visualization.")
    print("\nChecking constraint violations:")
    for constraint in model.constraints.values():
        if constraint.value() is not None and abs(constraint.value()) > 1e-6:
            print(f"Constraint {constraint.name}: {constraint} = {constraint.value():.2f}")
    exit(1)

# Print all arc flows
print("\nAll arc flows (Serviced via sink, Unmet via dummy):")
for a in x_regular:
    i, j, t, c = a
    flow = x_regular[a].varValue if x_regular[a].varValue is not None else 0
    flow_type = 'Serviced' if 'sink' in j or 'ss' in j else 'Unmet' if 'dummy' in j else 'Other'
    print(f"Arc ({i} -> {j}), t={t}, commodity={c}, type={flow_type}: flow={flow:.1f}")
for a in x_qq:
    i, j, t, c, t2 = a
    flow = x_qq[a].varValue if x_qq[a].varValue is not None else 0
    print(f"Arc ({i} -> {j}), t={t} to t={t2}, commodity={c}, type=Queue-to-queue: flow={flow:.1f}")

print("\nFacility openings:")
for m in M:
    print(f"y[{m}, 'l1'] = {value(y[(m, 'l1')])}")

print("\nPositive q-to-r flows for l1 commodities:")
for m in M:
    for t in T:
        for c in C:
            if c[0] == 'l1':
                a = (f'{m}_q', f'{m}_r', t, c)
                flow = value(x_regular[a])
                if flow > 1e-6:
                    print(f"Arc ({m}_q -> {m}_r), t={t}, commodity={c}: flow={flow:.1f}")
                    
""" # Save flows to CSV (if you want a separate file to examine the output)
output_dir = r'C:/Users/david/OneDrive/Documents/Academics'
with open(os.path.join(output_dir, 'arc_flows.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Source', 'Destination', 'Time', 'Commodity', 'Type', 'Flow'])
    for a in x_regular:
        i, j, t, c = a
        flow = x_regular[a].varValue if x_regular[a].varValue is not None else 0
        flow_type = 'Serviced' if 'sink' in j or 'ss' in j else 'Unmet' if 'dummy' in j else 'Other'
        writer.writerow([i, j, t, str(c), flow_type, f"{flow:.1f}"])
    for a in x_qq:
        i, j, t, c, t2 = a
        flow = x_qq[a].varValue if x_qq[a].varValue is not None else 0
        writer.writerow([i, j, f"{t}->{t2}", str(c), 'Queue-to-queue', f"{flow:.1f}"])
print(f"Arc flows saved to: {os.path.join(output_dir, 'arc_flows.csv')}") """

# Verify flow conservation
total_demand = sum(D[(m, t, c)] for m, t, c in D)
total_ss_inflow = sum(x_regular[(i, j, t, c)].varValue for i, j, t, c in regular_arcs
                      if j == 'ss' and x_regular[(i, j, t, c)].varValue is not None)
print(f"Total demand: {total_demand:.1f}")
print(f"Total inflow to ss: {total_ss_inflow:.1f}")
""" 
# Visualization (not working)
print(f"\nSaving plots to: {output_dir}")

figures = []
commodity_colors = {
    ('l1', 'k1'): 'blue',
    ('l1', 'k2'): 'green',
    ('l2', 'k1'): 'red',
    ('l2', 'k2'): 'purple'
}

for t in T:
    print(f"Generating plot for t={t}")
    try:
        fig = plt.figure(figsize=(20, 14))
        figures.append(fig)
        G = nx.DiGraph()
        
        # Add nodes
        nodes_t = [(n, nt, nc) for n, nt, nc in nodes if nt == t or (n == 'ss' and nt is None)]
        if t == 2:
            nodes_t += [('dummy', 2, nc) for nc in C]
        
        for n, nt, nc in nodes_t:
            node_type = n.split('_')[1] if '_' in n else n
            node_label = f"{n}_{nc[0]}_{nc[1]}" if node_type not in ['ss', 'sink', 'dummy', 'source'] else n
            G.add_node((n, nc if node_type not in ['ss', 'sink', 'dummy', 'source'] else None),
                       type=node_type, label=node_label)
        
        # Debug node attributes
        print(f"\nNode attributes for t={t}:")
        for node in G.nodes():
            print(f"Node: {node}, Attributes: {G.nodes[node]}")
        
        # Add arcs
        arcs_t = [(i, j, ci) for i, j, ti, ci in regular_arcs if ti == t or (j == 'ss' and ti is not None)]
        if t < 2:
            arcs_t += [(i, j, ci) for i, j, qi, ci, t2 in qq_arcs if qi == t]
        if t == 2:
            arcs_t += [(i, j, ci) for i, j, qi, ci, t2 in qq_arcs if t2 == t]
        
        edge_colors = []
        for i, j, ci in arcs_t:
            arc_type = ('source-in' if 'source' in i and '_in' in j else
                        'in-in' if '_in' in i and '_in' in j else
                        'in-q' if '_in' in i and '_q' in j else
                        'q-r' if '_q' in i and '_r' in j else
                        'r-out' if '_r' in i and '_out' in j else
                        'out-sink' if '_out' in i and 'sink' in j else
                        'sink-ss' if 'sink' in i and 'ss' in j else
                        'q-q' if '_q' in i and '_q' in j else
                        'q-dummy' if '_q' in i and 'dummy' in j else
                        'dummy-ss' if 'dummy' in i and 'ss' in j else 'Other')
            flow = (x_regular[(i, j, t, ci)].varValue if (i, j, t, ci) in x_regular else
                    x_qq[(i, j, t, ci, t+1)].varValue if (i, j, t, ci, t+1) in x_qq else 0)
            if flow > 0:
                G.add_edge((i, ci), (j, ci), label=f'{arc_type}: {flow:.1f} ({ci[0]},{ci[1]})',
                           weight=flow, color=commodity_colors[ci])
                edge_colors.append(commodity_colors[ci])
        
        # Layout
        pos = {}
        layer_x = {'source': -2, 'in': 0, 'q': 2, 'r': 4, 'out': 6, 'sink': 8, 'ss': 10, 'dummy': 8}
        y_pos = {'m1': 8, 'm2': 4, 'm3': 0}
        commodity_offset = {c: idx * 0.2 for idx, c in enumerate(C)}
        for n, nc in G.nodes():
            node_type = G.nodes[(n, nc)]['type']
            x = layer_x.get(node_type, 10)
            if node_type in ['in', 'q', 'r', 'out']:
                loc = n.split('_')[0]
                y = y_pos[loc]
                x += commodity_offset[nc] if nc else 0
            elif node_type in ['sink', 'source']:
                y = 4
            elif node_type == 'ss':
                y = 2
            elif node_type == 'dummy':
                y = -4
            pos[(n, nc)] = (x, y)
        
        node_colors = {
            'source': 'lightpurple',
            'in': 'lightblue',
            'q': 'lightgreen',
            'r': 'lightcoral',
            'out': 'lightyellow',
            'sink': 'cyan',
            'ss': 'lightgrey',
            'dummy': 'lightpink'
        }
        colors = [node_colors[G.nodes[(n, nc)]['type']] for n, nc in G.nodes()]
        
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
                node_color=colors, node_size=1200, font_size=8, arrows=True,
                width=[G[i][j]['weight']/5 for i, j in G.edges()], edge_color=edge_colors)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
        
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=color, lw=2, label=f'{c[0]},{c[1]}')
                           for c, color in commodity_colors.items()]
        plt.legend(handles=legend_elements, loc='upper right', title='Commodities')
        
        plt.title(f'Network for Time t={t}, All Commodities')
        
        save_path = os.path.join(output_dir, f'network_t{t}.png')
        plt.savefig(save_path)
        print(f"Saved plot: {save_path}")
        plt.close(fig)
    
    except Exception as e:
        print(f"Error generating plot for t={t}: {e}")
 """
print("Script completed")