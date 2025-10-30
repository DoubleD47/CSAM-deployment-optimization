# This script will parse the output from the optimization model and create bar charts
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

# Parse traditional l2 flows
l2_pattern = r"Arc \((\w+)_q_l2 -> \1_r_l2\), t=([12]), commodity=\(('l2', 'k\d')\): flow=([\d.]+)"
l2_matches = re.findall(l2_pattern, output_text)
l2_df = pd.DataFrame(l2_matches, columns=['Facility', 'Time', 'Commodity', 'Traditional_l2_Flow'])
l2_df['Time'] = l2_df['Time'].astype(int)
l2_df['Traditional_l2_Flow'] = l2_df['Traditional_l2_Flow'].astype(float)

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

# Merge l2 (only for traditional m1-m5)
df = df.merge(l2_df[['Facility', 'Time', 'Traditional_l2_Flow']], on=['Facility', 'Time'], how='left').fillna({'Traditional_l2_Flow': 0})

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

# Stacked bar for flows per facility/time
df_melt = df.melt(id_vars=['Facility', 'Time'], value_vars=['CSAM_l1_Flow', 'Traditional_l2_Flow', 'Unmet_Dummy'], var_name='Flow_Type', value_name='Flow')
df_melt['Facility_Time'] = df_melt['Facility'] + '_t' + df_melt['Time'].astype(str)

plt.figure(figsize=(14, 8))
sns.barplot(data=df_melt, x='Facility_Time', y='Flow', hue='Flow_Type', palette='Set2')
plt.title('Stacked Repair Flows by Facility and Time')
plt.xlabel('Facility_Time')
plt.ylabel('Flow Volume')
plt.xticks(rotation=90)
plt.legend(title='Flow Type')
plt.tight_layout()
# plt.savefig('csam_flow_stacked_bars.png')
plt.savefig(os.path.join(output_dir, 'csam_flow_stacked_bars.png'))  # Save in viz_output folder
plt.show()

# Separate bar for deployments
deploy_df = pd.DataFrame(list(deployments.items()), columns=['Facility', 'Opened'])
plt.figure(figsize=(10, 6))
sns.barplot(x='Facility', y='Opened', data=deploy_df, palette='viridis')
plt.title('CSAM Facility Deployments (1=Opened)')
plt.ylabel('Opened (Binary)')
# plt.savefig('csam_deployments_bar.png')
plt.savefig(os.path.join(output_dir, 'csam_deployments_bar.png'))  # Save in viz_output folder
plt.show()

# Optional: Demands vs Total Fulfilled (CSAM + Trad)
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
# plt.savefig('demands_vs_fulfilled_bars.png')
plt.savefig(os.path.join(output_dir, 'demands_vs_fulfilled_bars.png'))  # Save in viz_output folder
plt.show()

# Travel flows outgoing per facility per t
travel_pattern = r"Arc \((\w+)_in -> (\w+)_in\), t=([12]), commodity=\(('l2', 'k\d')\): flow=([\d.]+)"
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
# plt.savefig('outgoing_travel_bars.png')
plt.savefig(os.path.join(output_dir, 'outgoing_travel_bars.png'))  # Save in viz_output folder
plt.show()

# Fulfilled flows stacked by commodity tuple
df_fulfilled = pd.concat([
    l1_df.rename(columns={'CSAM_l1_Flow': 'Flow'})[['Facility', 'Time', 'Commodity', 'Flow']],
    l2_df.rename(columns={'Traditional_l2_Flow': 'Flow'})[['Facility', 'Time', 'Commodity', 'Flow']]
])
df_fulfilled['Facility_Time'] = df_fulfilled['Facility'] + '_t' + df_fulfilled['Time'].astype(str)

plt.figure(figsize=(14, 8))
sns.barplot(data=df_fulfilled, x='Facility_Time', y='Flow', hue='Commodity', dodge=False, palette='tab10')
plt.title('Fulfilled Flows by Commodity Tuple, Facility, and Time')
plt.xlabel('Facility_Time')
plt.ylabel('Flow Volume')
plt.xticks(rotation=90)
plt.legend(title='Commodity Tuple', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fulfilled_by_commodity_tuple.png'))  # Save in viz_output subfolder
plt.show()

# Demands stacked by commodity tuple (for comparison)
demand_df['Facility_Time'] = demand_df['Facility'] + '_t' + demand_df['Time'].astype(str)  # If not already added

plt.figure(figsize=(14, 8))
sns.barplot(data=demand_df, x='Facility_Time', y='Demand', hue='Commodity', dodge=False, palette='tab10')
plt.title('Demands by Commodity Tuple, Facility, and Time')
plt.xlabel('Facility_Time')
plt.ylabel('Demand Volume')
plt.xticks(rotation=90)
plt.legend(title='Commodity Tuple', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'demands_by_commodity_tuple.png'))  # Save in viz_output subfolder
plt.show()

# Outgoing travel flows stacked by commodity tuple
travel_df['From_Time'] = travel_df['From'] + '_t' + travel_df['Time'].astype(str)

plt.figure(figsize=(14, 8))
sns.barplot(data=travel_df, x='From_Time', y='Travel_Flow', hue='Commodity', dodge=False, palette='tab10'   )
plt.title('Outgoing Inter-Facility Travel Flows by Commodity Tuple, From Facility, and Time')
plt.xlabel('From_Facility_Time')
plt.ylabel('Travel Flow Volume')
plt.xticks(rotation=90)
plt.legend(title='Commodity Tuple', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'outgoing_travel_by_commodity_tuple.png'))  # Save in viz_output subfolder
plt.show()

# CSAM l1 flows stacked by commodity tuple (filtered to l1 only)
df_l1_only = l1_df.rename(columns={'CSAM_l1_Flow': 'Flow'})[['Facility', 'Time', 'Commodity', 'Flow']]
df_l1_only['Facility_Time'] = df_l1_only['Facility'] + '_t' + df_l1_only['Time'].astype(str)

plt.figure(figsize=(14, 8))
sns.barplot(data=df_l1_only, x='Facility_Time', y='Flow', hue='Commodity', dodge=False, palette='tab10')
plt.title('CSAM l1 Flows by Commodity Tuple, Facility, and Time')
plt.xlabel('Facility_Time')
plt.ylabel('l1 Flow Volume')
plt.xticks(rotation=90)
plt.legend(title='Commodity Tuple', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'csam_l1_by_commodity_tuple.png'))  # Save in viz_output subfolder
plt.show()

# Queue sizes (incoming to q arcs + carryover qq if applicable) stacked by commodity tuple
# Note: Assumes model script prints positive in-to-q and qq flows; add prints if needed (e.g., similar to travel flows)
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

plt.figure(figsize=(14, 8))
sns.barplot(data=queue_df, x='Facility_Time', y='Queue_Size', hue='Commodity', dodge=False, palette='tab10')
plt.title('Queue Sizes by Commodity Tuple, Facility, and Time')
plt.xlabel('Facility_Time')
plt.ylabel('Queue Size Volume')
plt.xticks(rotation=90)
plt.legend(title='Commodity Tuple', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'queue_sizes_by_commodity_tuple.png'))  # Save in viz_output subfolder
plt.show()

# Crossover l1 flows to l2 processes (l1 commodities on traditional l2 paths)
# Updated pattern to handle optional "(jumping if 'l1')"
l2_pattern = r"Arc \((\w+)_q_l2 -> \1_r_l2\), t=([12]), commodity=\(('l[12]', 'k\d')\): flow=([\d.]+)( \(jumping if 'l1'\))?"
l2_matches = re.findall(l2_pattern, output_text)
l2_df = pd.DataFrame(l2_matches, columns=['Facility', 'Time', 'Commodity', 'Traditional_l2_Flow', 'Jumping'])
l2_df['Time'] = l2_df['Time'].astype(int)
l2_df['Traditional_l2_Flow'] = l2_df['Traditional_l2_Flow'].astype(float)
crossover_df = l2_df[l2_df['Commodity'].str.startswith("'l1'")]  # Filter to l1 commodities on l2 paths (adjusted for string format)
crossover_df['Facility_Time'] = crossover_df['Facility'] + '_t' + crossover_df['Time'].astype(str)

plt.figure(figsize=(14, 8))
sns.barplot(data=crossover_df, x='Facility_Time', y='Traditional_l2_Flow', hue='Commodity', dodge=False, palette='tab10')
plt.title('Crossover l1 Flows to l2 Processes by Commodity Tuple, Facility, and Time')
plt.xlabel('Facility_Time')
plt.ylabel('Crossover Flow Volume')
plt.xticks(rotation=90)
plt.legend(title='Commodity Tuple (l1 only)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'crossover_l1_to_l2_by_tuple.png'))  # Save in viz_output subfolder
plt.show()

