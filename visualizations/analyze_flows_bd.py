# visualizations/analyze_flows_bd.py
# All facilities shown + sorted + travel heatmaps

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# ====================== CONFIG ======================
if len(sys.argv) > 1:
    exp_folder = Path(sys.argv[1])
else:
    experiments_root = Path(__file__).parent.parent / "experiments"
    exp_folder = max(experiments_root.glob("*/"), key=os.path.getmtime, default=None)

if not exp_folder or not exp_folder.exists():
    print("Error: Could not find experiment folder.")
    sys.exit(1)

print(f"Analyzing flows from: {exp_folder}")

output_dir = exp_folder / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)

# All facilities in order
all_facilities = [f'm{i}' for i in range(1, 11)]

# ====================== LOAD DATA ======================
def load_csv(name):
    path = exp_folder / name
    if path.exists():
        df = pd.read_csv(path)
        print(f"Loaded {name} ({len(df)} rows)")
        return df
    else:
        print(f"Warning: {name} not found")
        return pd.DataFrame()

csam = load_csv("csam_flows.csv")
traditional = load_csv("traditional_flows.csv")
inq = load_csv("inq_flows.csv")
qq = load_csv("qq_flows.csv")
travel = load_csv("travel_flows.csv")
dummy = load_csv("dummy_flows.csv")

for df in [csam, traditional, inq, qq, dummy, travel]:
    if not df.empty and 'Commodity' in df.columns:
        df['Commodity_Tuple'] = df['Commodity'].astype(str)

# Helper to ensure all facilities
def ensure_all_facilities(df, facility_col='Facility'):
    if df.empty:
        return df
    df[facility_col] = pd.Categorical(df[facility_col], categories=all_facilities, ordered=True)
    return df

# ====================== 1. REPAIR THROUGHPUT (Node + Tuple) ======================
data = []
if not csam.empty:
    data.append(ensure_all_facilities(csam.assign(Type='CSAM_l1')))
if not traditional.empty:
    data.append(ensure_all_facilities(traditional.assign(Type='Traditional_l2')))

if data:
    repair_df = pd.concat(data, ignore_index=True)
    pivot = repair_df.pivot_table(index='Facility', columns='Commodity_Tuple', 
                                  values='Flow', aggfunc='sum', fill_value=0)
    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Flow Volume'})
    plt.title('Repair Throughput by Facility and Commodity Tuple')
    plt.ylabel('Facility')
    plt.xlabel('Commodity Tuple')
    plt.tight_layout()
    plt.savefig(output_dir / 'repair_by_node_and_tuple_heatmap.png', dpi=150)
    plt.close()

# ====================== 2. QUEUE LEVELS (Node + Tuple) ======================
if not inq.empty:
    queue = ensure_all_facilities(inq.copy())
    if not qq.empty:
        qq_temp = ensure_all_facilities(qq.copy())
        queue = pd.merge(queue, qq_temp, on=['Facility', 'Level', 'Commodity_Tuple'], 
                         how='left', suffixes=('', '_carry'))
        queue['Queue_Size'] = queue['Flow'] + queue.get('Flow_carry', 0).fillna(0)
    else:
        queue['Queue_Size'] = queue['Flow']

    plt.figure(figsize=(16, 9))
    sns.barplot(data=queue, x='Facility', y='Queue_Size', hue='Commodity_Tuple', 
                errorbar=None, dodge=True)
    plt.title('Queue Levels by Facility and Commodity Tuple')
    plt.ylabel('Total Queue Volume')
    plt.xticks(rotation=45)
    plt.legend(title='Commodity Tuple', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'queue_by_node_and_tuple.png', dpi=150)
    plt.close()

    # Heatmap version
    pivot_q = queue.pivot_table(index='Facility', columns='Commodity_Tuple', 
                                values='Queue_Size', aggfunc='sum', fill_value=0)
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_q, annot=True, fmt='.1f', cmap='YlOrBr')
    plt.title('Queue Heatmap by Facility and Commodity Tuple')
    plt.savefig(output_dir / 'queue_heatmap_by_node_tuple.png', dpi=150)
    plt.close()

# ====================== 3. UNMET DEMAND (Node + Tuple) ======================
if not dummy.empty:
    if 'Time' not in dummy.columns:
        dummy['Time'] = 2
    dummy = ensure_all_facilities(dummy, facility_col='Node')
    unmet = dummy.groupby(['Node', 'Commodity_Tuple', 'Time'])['Flow'].sum().reset_index()

    plt.figure(figsize=(14, 8))
    sns.barplot(data=unmet, x='Node', y='Flow', hue='Commodity_Tuple', errorbar=None, dodge=True)
    plt.title('Unmet Demand by Node and Commodity Tuple')
    plt.ylabel('Unmet Volume')
    plt.xticks(rotation=45)
    plt.legend(title='Commodity Tuple', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig(output_dir / 'unmet_by_node_and_tuple.png', dpi=150)
    plt.close()

# ====================== 4. TRAVEL HEATMAPS (l1 vs l2 aggregated) ======================
if not travel.empty:
    travel = ensure_all_facilities(travel, facility_col='From Node')
    
    # Overall travel
    travel_agg = travel.groupby(['From Node', 'To Node'])['Flow'].sum().unstack(fill_value=0)
    plt.figure(figsize=(12, 10))
    sns.heatmap(travel_agg, annot=True, fmt='.1f', cmap='Greens', cbar_kws={'label': 'Total Travel Flow'})
    plt.title('All Travel Flows: From → To')
    plt.tight_layout()
    plt.savefig(output_dir / 'travel_heatmap_all.png', dpi=150)
    plt.close()

    # l1 travel only
    travel_l1 = travel[travel['Commodity_Tuple'].str.contains("'l1'", na=False)]
    if not travel_l1.empty:
        agg_l1 = travel_l1.groupby(['From Node', 'To Node'])['Flow'].sum().unstack(fill_value=0)
        plt.figure(figsize=(12, 10))
        sns.heatmap(agg_l1, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'l1 Travel Flow'})
        plt.title('Travel Flows - l1 Commodities Only')
        plt.tight_layout()
        plt.savefig(output_dir / 'travel_heatmap_l1.png', dpi=150)
        plt.close()

    # l2 travel only
    travel_l2 = travel[travel['Commodity_Tuple'].str.contains("'l2'", na=False)]
    if not travel_l2.empty:
        agg_l2 = travel_l2.groupby(['From Node', 'To Node'])['Flow'].sum().unstack(fill_value=0)
        plt.figure(figsize=(12, 10))
        sns.heatmap(agg_l2, annot=True, fmt='.1f', cmap='Oranges', cbar_kws={'label': 'l2 Travel Flow'})
        plt.title('Travel Flows - l2 Commodities Only')
        plt.tight_layout()
        plt.savefig(output_dir / 'travel_heatmap_l2.png', dpi=150)
        plt.close()

print(f"\n✅ All visualizations (including l1/l2 travel heatmaps) saved to: {output_dir}")
print(f"\n✅ All visualizations saved to: {output_dir}")