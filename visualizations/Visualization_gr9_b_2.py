# This script will parse the output from the optimization model and create visualizations

import re
import networkx as nx
import matplotlib.pyplot as plt

# The output text
# Read output from file 
try:
    output_text = open(r'C:\Git\CSAM-deployment-optimization\visualizations\output_gr9_b.txt', 'r').read()
except FileNotFoundError:
    print("Error: File not found at C:\\Git\\CSAM-deployment-optimization\\visualizations\output_gr9_b.txt")
    raise

# Parse deployments
deploy_pattern = r"y\[(\w+), 'l1'\] = ([\d.]+)"
deploy_matches = re.findall(deploy_pattern, output_text)
deployments = {m: float(val) for m, val in deploy_matches}

# Parse arcs
arc_pattern = r"Arc \(([\w_]+) -> ([\w_]+)\), t=([12]), commodity=\(('l[12]'), '([kv]\d)'\): flow=([\d.]+)"
arc_matches = re.findall(arc_pattern, output_text)

edges = []
for src, dst, t, l, k, flow in arc_matches:
    flow = float(flow)
    path = None
    type_ = None
    if '_in' in src and '_in' in dst:
        fac_src = src.split('_')[0]
        fac_dst = dst.split('_')[0]
        type_ = 'travel'
    elif '_q_' in src and '_r_' in dst:
        fac_src = src.split('_')[0]
        fac_dst = fac_src
        type_ = 'repair'
        path = src.split('_')[2]  # l1 or l2
    else:
        continue
    edges.append({
        'fac_src': fac_src,
        'fac_dst': fac_dst,
        't': int(t),
        'c': (l, k),
        'flow': flow,
        'type': type_,
        'path': path
    })

# Compute sum l1 flows per m per t
from collections import defaultdict
l1_sums = defaultdict(float)
for edge in edges:
    if edge['type'] == 'repair' and edge['path'] == 'l1':
        l1_sums[(edge['fac_src'], edge['t'])] += edge['flow']

# Generate plots for each t
for t in [1, 2]:
    G = nx.MultiDiGraph()

    # Add summed l1 repair self-loop
    for (m, tt), sum_flow in l1_sums.items():
        if tt == t and sum_flow > 1e-6:
            G.add_edge(m, m, key='l1_sum', label=f"CSAM l1 sum: {sum_flow:.1f}", weight=sum_flow, path='l1')

    # Add l2 repair self-loops (one per traditional per t)
    for edge in edges:
        if edge['t'] == t and edge['type'] == 'repair' and edge['path'] == 'l2':
            m = edge['fac_src']
            label = f"Trad l2 {edge['c']}: {edge['flow']:.1f}"
            G.add_edge(m, m, key=edge['c'][1], label=label, weight=edge['flow'], path='l2')

    # Add travel edges
    for edge in edges:
        if edge['t'] == t and edge['type'] == 'travel':
            label = f"{edge['c']}: {edge['flow']:.1f}"
            G.add_edge(edge['fac_src'], edge['fac_dst'], key=edge['c'][1], label=label, weight=edge['flow'], path=None)

    # Ensure all nodes are added
    for m in [f'm{i}' for i in range(1, 11)]:
        G.add_node(m)

    # Layout: shell with traditional in inner, others outer
    inner_shell = ['m1', 'm2', 'm3', 'm4', 'm5']
    outer_shell = ['m6', 'm7', 'm8', 'm9', 'm10']
    pos = nx.shell_layout(G, [inner_shell, outer_shell])

    # Node colors
    node_colors = ['green' if deployments.get(m, 0) == 1 else 'lightblue' for m in G.nodes()]

    # Prepare edgelists
    travel_edgelist = [(u, v, k) for u, v, k in G.edges(keys=True) if G[u][v][k]['path'] is None]
    l1_self_edgelist = [(u, v, k) for u, v, k in G.edges(keys=True) if u == v and G[u][v][k]['path'] == 'l1']
    l2_self_edgelist = [(u, v, k) for u, v, k in G.edges(keys=True) if u == v and G[u][v][k]['path'] == 'l2']

    # Draw
    plt.figure(figsize=(14, 14))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Travel edges (blue, arrows)
    if travel_edgelist:
        travel_widths = [G[u][v][k]['weight'] / 10 for u, v, k in travel_edgelist]
        nx.draw_networkx_edges(G, pos, edgelist=travel_edgelist, width=travel_widths, arrowstyle='->', arrowsize=10, edge_color='blue')

    # l1 self-loops (green, curved)
    if l1_self_edgelist:
        l1_widths = [G[u][v][k]['weight'] / 10 for u, v, k in l1_self_edgelist]
        nx.draw_networkx_edges(G, pos, edgelist=l1_self_edgelist, width=l1_widths, connectionstyle='arc3,rad=0.3', arrows=False, edge_color='green')

    # l2 self-loops (red, curved opposite)
    if l2_self_edgelist:
        l2_widths = [G[u][v][k]['weight'] / 10 for u, v, k in l2_self_edgelist]
        nx.draw_networkx_edges(G, pos, edgelist=l2_self_edgelist, width=l2_widths, connectionstyle='arc3,rad=-0.3', arrows=False, edge_color='red')

    # Edge labels
    edge_labels = {(u, v, k): d['label'] for u, v, k, d in G.edges(data=True, keys=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, label_pos=0.5)

    plt.title(f'CSAM Deployment Network Flow for t={t}\n(Blue: Travel for l2, Green: CSAM l1 repair sum, Red: Traditional l2 repair)')
    plt.axis('off')
    plt.savefig(f'csam_network_t{t}.png')
    plt.show()