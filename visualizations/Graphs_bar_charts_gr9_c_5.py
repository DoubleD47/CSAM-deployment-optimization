# This script will parse the output from the optimization model and create bar charts
# This is a further update to some of the stacked bars and an attempt at a matrix style output
# to better visualize the travel flows per commodity tuple.

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# Dynamically create subfolder based on the script file name (e.g., 'fleet_flow_gr9' from 'fleet_flow_gr9_c.py')
script_name = os.path.splitext(os.path.basename(__file__))[0]  # e.g., 'fleet_flow_gr9_c'
subfolder = script_name.rsplit('_', 1)[0] if '_' in script_name else script_name  # e.g., 'fleet_flow_gr9'

# Create viz_output subfolder path with the dynamic subfolder
output_dir = os.path.join(os.path.dirname(__file__), 'viz_output', subfolder)
os.makedirs(output_dir, exist_ok=True)

# The output text
# Read output from file 
try:
    output_text = open(r'C:\Git\CSAM-deployment-optimization\output\output_gr9_c.txt', 'r').read()
except FileNotFoundError:
    print("Error: File not found at C:\\Git\\output_gr9_c.txt")
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
dummy_pattern = r"Arc \((\w+)_q_(\w+) -> dummy\), t=2, commodity=\(('l[12]', 'k\d')\): flow=([\d.]+)"
dummy_matches = re.findall(dummy_pattern, output_text)
dummy_df = pd.DataFrame(dummy_matches, columns=['Facility', 'Path', 'Commodity', 'Unmet_Flow'])
dummy_df['Time'] = 2
dummy_df['Unmet_Flow'] = dummy_df['Unmet_Flow'].astype(float)
dummy_agg = dummy_df.groupby(['Facility', 'Time'])['Unmet_Flow'].sum().reset_index()

# Since no dummy flows, unmet=0 for all
# Create base df with all m, t
facilities = [f'm{i}' for i in range(1,11)]
times = [1,2]
base_df = pd.DataFrame([(f, t) for f in facilities for t in times], columns=['Facility', 'Time'])

# Merge l1
df = base_df.merge(l1_agg, on=['Facility', 'Time'], how='left').fillna({'CSAM_l1_Flow': 0})

# Merge l2 (summed)
df = df.merge(l2_agg, on=['Facility', 'Time'], how='left').fillna({'Traditional_l2_Flow': 0})

# Add unmet (0)
# Remove: df['Unmet_Dummy'] = 0.0
# Add merge instead:
df = df.merge(dummy_agg, on=['Facility', 'Time'], how='left').fillna({'Unmet_Flow': 0})
df = df.rename(columns={'Unmet_Flow': 'Unmet_Dummy'})

# Parse demands for comparison (optional chart)
demand_pattern = r"D\((\w+), t=([12]), \(('l[12]', 'k\d')\)\) = ([\d.]+)"
demand_matches = re.findall(demand_pattern, output_text)
demand_df = pd.DataFrame(demand_matches, columns=['Facility', 'Time', 'Commodity', 'Demand'])
demand_df['Time'] = demand_df['Time'].astype(int)
demand_df['Demand'] = demand_df['Demand'].astype(float)
demand_agg = demand_df.groupby(['Facility', 'Time'])['Demand'].sum().reset_index()

# Merge demands to df
df = df.merge(demand_agg, on=['Facility', 'Time'], how='left').fillna(0)

# Stacked bar for flows per facility/time (proper stacking using manual bar plot)
df_melt = df.melt(id_vars=['Facility', 'Time'], value_vars=['CSAM_l1_Flow', 'Traditional_l2_Flow', 'Unmet_Dummy'], var_name='Flow_Type', value_name='Flow')
df_melt['Facility_Time'] = df_melt['Facility'] + '_t' + df_melt['Time'].astype(str)
df_melt = df_melt.sort_values(['Facility_Time', 'Flow_Type'])  # Sort for consistent stacking order

flow_types = sorted(df_melt['Flow_Type'].unique())
colors = {'CSAM_l1_Flow': 'green', 'Traditional_l2_Flow': 'orange', 'Unmet_Dummy': 'blue'}

fig, ax = plt.subplots(figsize=(14, 8))
bottom = pd.Series(0, index=df_melt['Facility_Time'].unique())

for ft in flow_types:
    ft_df = df_melt[df_melt['Flow_Type'] == ft].set_index('Facility_Time')
    ax.bar(ft_df.index, ft_df['Flow'], bottom=bottom.reindex(ft_df.index, fill_value=0), label=ft, color=colors[ft])
    bottom += ft_df['Flow'].reindex(bottom.index, fill_value=0)

ax.set_title('Stacked Repair Flows by Facility and Time')
ax.set_xlabel('Facility_Time')
ax.set_ylabel('Flow Volume')
ax.tick_params(axis='x', rotation=90)
ax.legend(title='Flow Type')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'csam_flow_stacked_bars.png'))  # Save in viz_output folder
# plt.show()  # Removed to avoid displaying the plot

# Separate bar for deployments
deploy_df = pd.DataFrame(list(deployments.items()), columns=['Facility', 'Opened'])
plt.figure(figsize=(10, 6))
sns.barplot(x='Facility', y='Opened', data=deploy_df, palette='viridis')
plt.title('CSAM Facility Deployments (1=Opened)')
plt.ylabel('Opened (Binary)')
plt.savefig(os.path.join(output_dir, 'csam_deployments_bar.png'))  # Save in viz_output folder
# plt.show()  # Removed to avoid displaying the plot

# Optional: Demands vs Total Fulfilled (CSAM + Trad) - grouped, no change needed
df['Total_Fulfilled'] = df['CSAM_l1_Flow'] + df['Traditional_l2_Flow']
df_melt_comp = df.melt(id_vars=['Facility', 'Time'], value_vars=['Demand', 'Total_Fulfilled'], var_name='Type', value_name='Volume')
df_melt_comp['Facility_Time'] = df_melt_comp['Facility'] + '_t' + df_melt_comp['Time'].astype(str)

plt.figure(figsize=(14, 8))
sns.barplot(data=df_melt_comp, x='Facility_Time', y='Volume', hue='Type')
plt.title('Demands vs Fulfilled Flows by Facility and Time')
plt.xlabel('Facility_Time')
plt.ylabel('Volume')
plt.xticks(rotation=90)
plt.legend(title='Type')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'demands_vs_fulfilled_bars.png'))  # Save in viz_output folder
# plt.show()  # Removed to avoid displaying the plot

# Travel flows outgoing per facility per t - no hue, no change needed
travel_pattern = r"Arc \((\w+)_in -> (\w+)_in\), t=([12]), commodity=\(('l[12]', 'k\d')\): flow=([\d.]+)"
travel_matches = re.findall(travel_pattern, output_text)
travel_df = pd.DataFrame(travel_matches, columns=['From', 'To', 'Time', 'Commodity', 'Travel_Flow'])
travel_df['Time'] = travel_df['Time'].astype(int)
travel_df['Travel_Flow'] = travel_df['Travel_Flow'].astype(float)
travel_out = travel_df.groupby(['From', 'Time'])['Travel_Flow'].sum().reset_index().rename(columns={'From': 'Facility', 'Travel_Flow': 'Outgoing_Travel'})

plt.figure(figsize=(12, 6))
travel_out['Facility_Time'] = travel_out['Facility'] + '_t' + travel_out['Time'].astype(str)
sns.barplot(data=travel_out, x='Facility_Time', y='Outgoing_Travel', palette='muted')
plt.title('Outgoing Inter-Facility Travel Flows by Facility and Time')
plt.xlabel('Facility_Time')
plt.ylabel('Travel Flow Volume')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'outgoing_travel_bars.png'))  # Save in viz_output folder
# plt.show()  # Removed to avoid displaying the plot

# Fulfilled flows stacked by commodity tuple (manual stacking)
df_fulfilled = pd.concat([
    l1_df.rename(columns={'CSAM_l1_Flow': 'Flow'})[['Facility', 'Time', 'Commodity', 'Flow']],
    l2_df.rename(columns={'Traditional_l2_Flow': 'Flow'})[['Facility', 'Time', 'Commodity', 'Flow']]
])
df_fulfilled['Facility_Time'] = df_fulfilled['Facility'] + '_t' + df_fulfilled['Time'].astype(str)
df_fulfilled = df_fulfilled.sort_values(['Facility_Time', 'Commodity'])  # Sort for consistent stacking order

commodities = sorted(df_fulfilled['Commodity'].unique())

fig, ax = plt.subplots(figsize=(14, 8))
bottom = pd.Series(0, index=df_fulfilled['Facility_Time'].unique())

for comm in commodities:
    comm_df = df_fulfilled[df_fulfilled['Commodity'] == comm].set_index('Facility_Time')
    ax.bar(comm_df.index, comm_df['Flow'], bottom=bottom.reindex(comm_df.index, fill_value=0), label=comm)
    bottom += comm_df['Flow'].reindex(bottom.index, fill_value=0)

ax.set_title('Fulfilled Flows by Commodity Tuple, Facility, and Time')
ax.set_xlabel('Facility_Time')
ax.set_ylabel('Flow Volume')
ax.tick_params(axis='x', rotation=90)
ax.legend(title='Commodity Tuple', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fulfilled_by_commodity_tuple.png'))  # Save in viz_output subfolder
# plt.show()  # Removed to avoid displaying the plot

# Demands stacked by commodity tuple (manual stacking)
demand_df['Facility_Time'] = demand_df['Facility'] + '_t' + demand_df['Time'].astype(str)  # If not already added
demand_df = demand_df.sort_values(['Facility_Time', 'Commodity'])

commodities = sorted(demand_df['Commodity'].unique())

fig, ax = plt.subplots(figsize=(14, 8))
bottom = pd.Series(0, index=demand_df['Facility_Time'].unique())

for comm in commodities:
    comm_df = demand_df[demand_df['Commodity'] == comm].set_index('Facility_Time')
    ax.bar(comm_df.index, comm_df['Demand'], bottom=bottom.reindex(comm_df.index, fill_value=0), label=comm)
    bottom += comm_df['Demand'].reindex(bottom.index, fill_value=0)

ax.set_title('Demands by Commodity Tuple, Facility, and Time')
ax.set_xlabel('Facility_Time')
ax.set_ylabel('Demand Volume')
ax.tick_params(axis='x', rotation=90)
ax.legend(title='Commodity Tuple', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'demands_by_commodity_tuple.png'))  # Save in viz_output subfolder
# plt.show()  # Removed to avoid displaying the plot

# Outgoing travel flows stacked by commodity tuple (manual stacking)
travel_df['From_Time'] = travel_df['From'] + '_t' + travel_df['Time'].astype(str)
travel_df = travel_df.sort_values(['From_Time', 'Commodity'])

commodities = sorted(travel_df['Commodity'].unique())

fig, ax = plt.subplots(figsize=(14, 8))
bottom = pd.Series(0, index=travel_df['From_Time'].unique())

for comm in commodities:
    comm_df = travel_df[travel_df['Commodity'] == comm].set_index('From_Time')
    ax.bar(comm_df.index, comm_df['Travel_Flow'], bottom=bottom.reindex(comm_df.index, fill_value=0), label=comm)
    bottom += comm_df['Travel_Flow'].reindex(bottom.index, fill_value=0)

ax.set_title('Outgoing Inter-Facility Travel Flows by Commodity Tuple, From Facility, and Time')
ax.set_xlabel('From_Facility_Time')
ax.set_ylabel('Travel Flow Volume')
ax.tick_params(axis='x', rotation=90)
ax.legend(title='Commodity Tuple', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'outgoing_travel_by_commodity_tuple.png'))  # Save in viz_output subfolder
# plt.show()  # Removed to avoid displaying the plot

# CSAM l1 flows stacked by commodity tuple (manual stacking, with capacity reference line at 50)
df_l1_only = l1_df.rename(columns={'CSAM_l1_Flow': 'Flow'})[['Facility', 'Time', 'Commodity', 'Flow']]
df_l1_only['Facility_Time'] = df_l1_only['Facility'] + '_t' + df_l1_only['Time'].astype(str)
df_l1_only = df_l1_only.sort_values(['Facility_Time', 'Commodity'])

commodities = sorted(df_l1_only['Commodity'].unique())

fig, ax = plt.subplots(figsize=(14, 8))
bottom = pd.Series(0, index=df_l1_only['Facility_Time'].unique())

for comm in commodities:
    comm_df = df_l1_only[df_l1_only['Commodity'] == comm].set_index('Facility_Time')
    ax.bar(comm_df.index, comm_df['Flow'], bottom=bottom.reindex(comm_df.index, fill_value=0), label=comm)
    bottom += comm_df['Flow'].reindex(bottom.index, fill_value=0)

ax.axhline(y=50, color='r', linestyle='--', label='CSAM Capacity Limit')  # Add reference line for capacity
ax.set_title('CSAM l1 Flows by Commodity Tuple, Facility, and Time')
ax.set_xlabel('Facility_Time')
ax.set_ylabel('l1 Flow Volume')
ax.tick_params(axis='x', rotation=90)
ax.legend(title='Commodity Tuple', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'csam_l1_by_commodity_tuple.png'))  # Save in viz_output subfolder
# plt.show()  # Removed to avoid displaying the plot

# Queue sizes stacked by commodity tuple (manual stacking)
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
qq_df['Time'] = 2  # Carryover into t=2
qq_df['Carryover'] = qq_df['Carryover'].astype(float)

# Combine for total queue size (InQ + Carryover for t=2)
queue_df = inq_agg.rename(columns={'InQ_Flow': 'Queue_Size'}).merge(qq_df[['Facility', 'Time', 'Commodity', 'Carryover']], on=['Facility', 'Time', 'Commodity'], how='left').fillna({'Carryover': 0})
queue_df['Queue_Size'] += queue_df['Carryover']  # Add carryover to t=2 queue
queue_df = queue_df.drop(columns=['Carryover'])
queue_df['Facility_Time'] = queue_df['Facility'] + '_t' + queue_df['Time'].astype(str)
queue_df = queue_df.sort_values(['Facility_Time', 'Commodity'])

commodities = sorted(queue_df['Commodity'].unique())

fig, ax = plt.subplots(figsize=(14, 8))
bottom = pd.Series(0, index=queue_df['Facility_Time'].unique())

for comm in commodities:
    comm_df = queue_df[queue_df['Commodity'] == comm].set_index('Facility_Time')
    ax.bar(comm_df.index, comm_df['Queue_Size'], bottom=bottom.reindex(comm_df.index, fill_value=0), label=comm)
    bottom += comm_df['Queue_Size'].reindex(bottom.index, fill_value=0)

ax.set_title('Queue Sizes by Commodity Tuple, Facility, and Time')
ax.set_xlabel('Facility_Time')
ax.set_ylabel('Queue Size Volume')
ax.tick_params(axis='x', rotation=90)
ax.legend(title='Commodity Tuple', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'queue_sizes_by_commodity_tuple.png'))  # Save in viz_output subfolder
# plt.show()  # Removed to avoid displaying the plot

# Crossover l1 flows to l2 processes (l1 commodities on traditional l2 paths) - manual stacking
# Filter existing l2_df to l1 commodities
crossover_df = l2_df[l2_df['Commodity'].str.startswith("'l1'")]
crossover_df['Facility_Time'] = crossover_df['Facility'] + '_t' + crossover_df['Time'].astype(str)
crossover_df = crossover_df.sort_values(['Facility_Time', 'Commodity'])

commodities = sorted(crossover_df['Commodity'].unique())

fig, ax = plt.subplots(figsize=(14, 8))
bottom = pd.Series(0, index=crossover_df['Facility_Time'].unique())

for comm in commodities:
    comm_df = crossover_df[crossover_df['Commodity'] == comm].set_index('Facility_Time')
    ax.bar(comm_df.index, comm_df['Traditional_l2_Flow'], bottom=bottom.reindex(comm_df.index, fill_value=0), label=comm)
    bottom += comm_df['Traditional_l2_Flow'].reindex(bottom.index, fill_value=0)

ax.set_title('Crossover l1 Flows to l2 Processes by Commodity Tuple, Facility, and Time')
ax.set_xlabel('Facility_Time')
ax.set_ylabel('Crossover Flow Volume')
ax.tick_params(axis='x', rotation=90)
ax.legend(title='Commodity Tuple (l1 only)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'crossover_l1_to_l2_by_tuple.png'))  # Save in viz_output subfolder
# plt.show()  # Removed to avoid displaying the plot

# New: Total demands by commodity tuple and time (sum over facilities) - no stacking needed, no change
total_demand_per_ct = demand_df.groupby(['Commodity', 'Time'])['Demand'].sum().reset_index()
total_demand_per_ct = total_demand_per_ct.sort_values(['Commodity', 'Time'])
total_demand_per_ct['Commodity_Time'] = total_demand_per_ct['Commodity'].str.replace(r"[()' ]", "", regex=True).str.replace(",", "_") + '_t' + total_demand_per_ct['Time'].astype(str)

plt.figure(figsize=(14, 8))
sns.barplot(data=total_demand_per_ct, x='Commodity_Time', y='Demand', palette='tab10')
plt.title('Total Demands by Commodity Tuple and Time')
plt.xlabel('Commodity_Time')
plt.ylabel('Demand Volume')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'total_demands_by_commodity_time.png'))  # Save in viz_output subfolder
# plt.show()  # Removed to avoid displaying the plot