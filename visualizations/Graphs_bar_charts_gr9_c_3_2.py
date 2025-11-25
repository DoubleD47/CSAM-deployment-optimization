# This is the file that will create visualization bar charts for the fleet_flow_gr9_c_3 model. This is DIFFERENT than the visualizations Graphs_bar_charts_gr9_c_3.py, which is graphing the fleet_flow_gr9_c_2.py model.
# This update attempts to correct issues with the queue graph, in order to display everything total volume that's waiting at that node, but get a different color code for the commodities that didn't move. In other words, if we chose not to move something, it should all be the same color, but if it did move, it's getting identified as its specific tuple at that graph. 
# This script will parse the output from the optimization model and create bar charts
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Dynamically create subfolder based on the script file name (e.g., 'Graphs_bar_charts_gr9_c_3' from 'Graphs_bar_charts_gr9_c_3.py')
script_name = os.path.splitext(os.path.basename(__file__))[0]  # e.g., 'Graphs_bar_charts_gr9_c_3'
subfolder = script_name  # Use full script_name to include the version '_3'

# Create viz_output subfolder path with the dynamic subfolder
output_dir = os.path.join(os.path.dirname(__file__), 'viz_output', subfolder)
os.makedirs(output_dir, exist_ok=True)

# The output text
try:
    output_text = open(r'C:\Git\CSAM-deployment-optimization\output\output_gr9_c_3.txt', 'r').read()
except FileNotFoundError:
    print("Error: File not found at C:\\Git\\CSAM-deployment-optimization\\output\\output_gr9_c_3.txt")
    raise

# Parse deployments
deploy_pattern = r"y\[(\w+), 'l1'\] = ([\d.]+)"
deploy_matches = re.findall(deploy_pattern, output_text)
deployments = {m: float(val) for m, val in deploy_matches}

# Parse CSAM l1 flows
l1_pattern = r"Arc \((\w+)_q_l1 -> \1_r_l1\), t=([12]), commodity=\(('l1', 'k\d')\): flow=([\d.]+)"
l1_matches = re.findall(l1_pattern, output_text)
l1_df = pd.DataFrame(l1_matches, columns=['Facility', 'Time', 'Commodity', 'CSAM_l1_Flow'])
l1_df['Time'] = l1_df['Time'].astype(int)
l1_df['CSAM_l1_Flow'] = l1_df['CSAM_l1_Flow'].astype(float)
l1_agg = l1_df.groupby(['Facility', 'Time'])['CSAM_l1_Flow'].sum().reset_index()

# Parse traditional l2 flows (with optional jumping)
l2_pattern = r"Arc \((\w+)_q_l2 -> \1_r_l2\), t=([12]), commodity=\(('l[12]', 'k\d')\): flow=([\d.]+)( \(jumping if 'l1'\))?"
l2_matches = re.findall(l2_pattern, output_text)
l2_df = pd.DataFrame(l2_matches, columns=['Facility', 'Time', 'Commodity', 'Traditional_l2_Flow', 'Jumping'])
l2_df['Time'] = l2_df['Time'].astype(int)
l2_df['Traditional_l2_Flow'] = l2_df['Traditional_l2_Flow'].astype(float)
l2_agg = l2_df.groupby(['Facility', 'Time'])['Traditional_l2_Flow'].sum().reset_index()

# Parse dummy flows (unmet demand in t=2)
dummy_pattern = r"Arc \((\w+)_(\w+) -> dummy\), t=2, commodity=\(('l[12]', 'k\d')\): flow=([\d.]+)"
dummy_matches = re.findall(dummy_pattern, output_text)
dummy_df = pd.DataFrame(dummy_matches, columns=['Facility', 'Path', 'Commodity', 'Unmet_Flow'])
dummy_df['Time'] = 2
dummy_df['Unmet_Flow'] = dummy_df['Unmet_Flow'].astype(float)
dummy_agg = dummy_df.groupby(['Facility', 'Time'])['Unmet_Flow'].sum().reset_index()

# Create base df with all m, t
facilities = [f'm{i}' for i in range(1,11)]
times = [1,2]
base_df = pd.DataFrame([(f, t) for f in facilities for t in times], columns=['Facility', 'Time'])

# Merge l1
df = base_df.merge(l1_agg, on=['Facility', 'Time'], how='left').fillna({'CSAM_l1_Flow': 0})

# Merge l2 (summed)
df = df.merge(l2_agg, on=['Facility', 'Time'], how='left').fillna({'Traditional_l2_Flow': 0})

# Add unmet
df = df.merge(dummy_agg, on=['Facility', 'Time'], how='left').fillna({'Unmet_Flow': 0})
df = df.rename(columns={'Unmet_Flow': 'Unmet_Dummy'})

# Parse demands
demand_pattern = r"D\((\w+), t=([12]), \(('l[12]', 'k\d')\)\) = ([\d.]+)"
demand_matches = re.findall(demand_pattern, output_text)
demand_df = pd.DataFrame(demand_matches, columns=['Facility', 'Time', 'Commodity', 'Demand'])
demand_df['Time'] = demand_df['Time'].astype(int)
demand_df['Demand'] = demand_df['Demand'].astype(float)
demand_agg = demand_df.groupby(['Facility', 'Time'])['Demand'].sum().reset_index()

# Merge demands to df
df = df.merge(demand_agg, on=['Facility', 'Time'], how='left').fillna(0)

# Sorted facility_time list for consistent ordering
sorted_facility_time = [f'{m}_t{t}' for m in facilities for t in times]

# ... (all the other graphs remain exactly the same as in the previous version you had)

# Queue sizes by commodity tuple and time (stacked) - UPDATED TO INCLUDE "DID NOT MOVE" (in carryover) IN GRAY
# Parse incoming to q arcs (queue entries)
inq_pattern = r"Arc \((\w+)_in -> \1_q_(l[12])\), t=([12]), commodity=\(('l[12]', 'k\d')\): flow=([\d.]+)"
inq_matches = re.findall(inq_pattern, output_text)
inq_df = pd.DataFrame(inq_matches, columns=['Facility', 'Level', 'Time', 'Commodity', 'InQ_Flow'])
inq_df['Time'] = inq_df['Time'].astype(int)
inq_df['InQ_Flow'] = inq_df['InQ_Flow'].astype(float)
inq_agg = inq_df.groupby(['Facility', 'Time', 'Commodity'])['InQ_Flow'].sum().reset_index()

# Parse carryover qq arcs (from t=1 q to t=2 q)
qq_pattern = r"Arc \((\w+)_q_(l[12]) -> \1_q_\2\), t=1 to 2, commodity=\(('l[12]', 'k\d')\): flow=([\d.]+)"
qq_matches = re.findall(qq_pattern, output_text)
qq_df = pd.DataFrame(qq_matches, columns=['Facility', 'Level', 'Commodity', 'Carryover'])
qq_df['Time'] = 2
qq_df['Carryover'] = qq_df['Carryover'].astype(float)

# Combine for total queue size (InQ + Carryover for t=2)
queue_df = inq_agg.rename(columns={'InQ_Flow': 'Queue_Size'}).merge(qq_df[['Facility', 'Time', 'Commodity', 'Carryover']], on=['Facility', 'Time', 'Commodity'], how='left').fillna({'Carryover': 0})
queue_df['Queue_Size'] += queue_df['Carryover']
queue_df = queue_df.drop(columns=['Carryover'])
queue_df['Facility_Time'] = queue_df['Facility'] + '_t' + queue_df['Time'].astype(str)
queue_df = queue_df.sort_values(['Facility', 'Time', 'Commodity'])

# Parse in-carryover (this is the "did not move in t=1" volume)
in_carry_pattern = r"Arc \((\w+)_in -> \1_in\), t=1 to 2, commodity=\(('l[12]', 'k\d')\): flow=([\d.]+)"
in_carry_matches = re.findall(in_carry_pattern, output_text)
in_carry_df = pd.DataFrame(in_carry_matches, columns=['Facility', 'Commodity', 'In_Carry_Flow'])
in_carry_df['In_Carry_Flow'] = in_carry_df['In_Carry_Flow'].astype(float)
in_carry_agg = in_carry_df.groupby('Facility')['In_Carry_Flow'].sum().reset_index()
in_carry_agg['Facility_Time'] = in_carry_agg['Facility'] + '_t2'
in_carry_series = in_carry_agg.set_index('Facility_Time')['In_Carry_Flow'].reindex(sorted_facility_time, fill_value=0)

commodities = sorted(queue_df['Commodity'].unique())

fig, ax = plt.subplots(figsize=(14, 8))
bottom = pd.Series(0.0, index=sorted_facility_time)

# First: "did not move in t=1" (held at entry across periods) — light gray with dense hatching
if in_carry_series.sum() > 0:
    ax.bar(
        sorted_facility_time,
        in_carry_series,
        bottom=bottom,
        label="Held at Entry (did not move in t=1)",
        color='lightgray',          # very light gray background
        hatch='////',              # dense diagonal hatching
        edgecolor='dimgray',        # dark edge so the hatching is clearly visible
        linewidth=0.8
    )
bottom += in_carry_series  # always add (zero where no carryover)

# Then the normal queued / committed to queue in their commodity-tuple colors on top
for comm in commodities:
    comm_df = queue_df[queue_df['Commodity'] == comm].groupby('Facility_Time')['Queue_Size'].sum().reset_index().set_index('Facility_Time')
    comm_series = comm_df.reindex(sorted_facility_time, fill_value=0)['Queue_Size']
    ax.bar(sorted_facility_time, comm_series, bottom=bottom, label=comm)
    bottom += comm_series

ax.set_title('Queue Sizes by Commodity Tuple, Facility, and Time\n(Held at Entry across periods shown in gray w/ hatching)')
ax.set_xlabel('Facility_Time')
ax.set_ylabel('Volume')
ax.set_xticks(range(len(sorted_facility_time)))
ax.set_xticklabels(sorted_facility_time, rotation=45, ha='right')
ax.legend(title='Commodity Tuple', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'queue_sizes_by_commodity_tuple_with_held_gray.png'))
# plt.show()  # Removed to avoid displaying the plot