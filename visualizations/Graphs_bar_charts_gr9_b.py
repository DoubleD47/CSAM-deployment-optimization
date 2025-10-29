# This script will parse the output from the optimization model and create bar charts
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create viz_output subfolder path to store visualizations
output_dir = os.path.join(os.path.dirname(__file__), 'viz_output')


# The output text
# Read output from file 
try:
    output_text = open(r'C:\Git\CSAM-deployment-optimization\output\output_gr9_b.txt', 'r').read()
except FileNotFoundError:
    print("Error: File not found at C:\\Git\\output_gr9_b.txt")
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
df['Unmet_Dummy'] = 0.0

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