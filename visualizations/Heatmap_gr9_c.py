# This script will parse the output from the optimization model and create heatmaps and store them in the viz_output folder
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Capacities from model code
U_l1 = 50  # For CSAM l1 per m per t
U_l2 = 150  # For traditional l2 per k per t (at corresponding m)

# Parse CSAM l1 flows (aggregated per m, t)
l1_pattern = r"Arc \((\w+)_q_l1 -> \1_r_l1\), t=([12]), commodity=\(('l1', 'k\d')\): flow=([\d.]+)"
l1_matches = re.findall(l1_pattern, output_text)
l1_df = pd.DataFrame(l1_matches, columns=['Facility', 'Time', 'Commodity', 'l1_Flow'])
l1_df['Time'] = l1_df['Time'].astype(int)
l1_df['l1_Flow'] = l1_df['l1_Flow'].astype(float)
l1_agg = l1_df.groupby(['Facility', 'Time'])['l1_Flow'].sum().reset_index()
l1_agg['l1_Util'] = (l1_agg['l1_Flow'] / U_l1) * 100

# Pivot for heatmap (rows: Facilities, columns: Time)
l1_pivot = l1_agg.pivot(index='Facility', columns='Time', values='l1_Util').fillna(0)
l1_pivot = l1_pivot.sort_index()  # Sort by m1 to m10

# Heatmap for CSAM l1 utilization
plt.figure(figsize=(6, 10))
sns.heatmap(l1_pivot, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'Utilization %'})
plt.title('CSAM l1 Capacity Utilization (%) by Facility and Time')
plt.xlabel('Time Period')
plt.ylabel('Facility')
# plt.savefig('csam_l1_util_heatmap.png')
plt.savefig(os.path.join(output_dir, 'csam_l1_util_heatmap.png'))  # Save in viz_output folder
plt.show()

# Parse traditional l2 flows (per traditional m, t)
l2_pattern = r"Arc \((\w+)_q_l2 -> \1_r_l2\), t=([12]), commodity=\(('l2', '(k\d)')\): flow=([\d.]+)"
l2_matches = re.findall(l2_pattern, output_text)
l2_df = pd.DataFrame(l2_matches, columns=['Facility', 'Time', 'Commodity', 'k', 'l2_Flow'])
l2_df['Time'] = l2_df['Time'].astype(int)
l2_df['l2_Flow'] = l2_df['l2_Flow'].astype(float)
l2_df['l2_Util'] = (l2_df['l2_Flow'] / U_l2) * 100

# Pivot for heatmap (rows: Facility (m1-m5), columns: Time)
l2_pivot = l2_df.pivot(index='Facility', columns='Time', values='l2_Util').fillna(0)
l2_pivot = l2_pivot.sort_index()

# Heatmap for traditional l2 utilization
plt.figure(figsize=(6, 5))
sns.heatmap(l2_pivot, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Utilization %'})
plt.title('Traditional l2 Capacity Utilization (%) by Facility and Time')
plt.xlabel('Time Period')
plt.ylabel('Traditional Facility')
# plt.savefig('trad_l2_util_heatmap.png')
plt.savefig(os.path.join(output_dir, 'trad_l2_util_heatmap.png'))  # Save in viz_output folder
plt.show()

# Parse inter-facility travels for flow matrix (sum over t, c)
travel_pattern = r"Arc \((\w+)_in -> (\w+)_in\), t=([12]), commodity=\(('l2', '(k\d)')\): flow=([\d.]+)"
travel_matches = re.findall(travel_pattern, output_text)
travel_df = pd.DataFrame(travel_matches, columns=['From', 'To', 'Time', 'Commodity', 'k', 'Flow'])
travel_df['Flow'] = travel_df['Flow'].astype(float)

# Aggregate total flow from-to (sum over t, c)
travel_matrix = travel_df.groupby(['From', 'To'])['Flow'].sum().unstack(fill_value=0)
facilities = [f'm{i}' for i in range(1,11)]
travel_matrix = travel_matrix.reindex(index=facilities, columns=facilities, fill_value=0)

# Heatmap for inter-facility travel flows
plt.figure(figsize=(10, 10))
sns.heatmap(travel_matrix, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Total Flow'})
plt.title('Inter-Facility Travel Flow Matrix (Sum over Time and Commodities)')
plt.xlabel('To Facility')
plt.ylabel('From Facility')
# plt.savefig('travel_flow_matrix_heatmap.png')
plt.savefig(os.path.join(output_dir, 'travel_flow_matrix_heatmap.png'))  # Save in viz_output folder
plt.show()