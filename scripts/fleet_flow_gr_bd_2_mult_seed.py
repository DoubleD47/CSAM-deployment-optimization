# Benders' decomposition - Multi-seed experiment version
import os
import sys
import csv
import datetime
import json
from pathlib import Path
import shutil
import time as timer
from pulp import *
import numpy as np
from collections import defaultdict
import subprocess

# ====================== EXPERIMENT CONFIG ======================
EXPERIMENT_NAME = "maxCSAM3_multi_seed_Cdummy10000"
MAX_CSAM_FACILITIES = 3
NUM_SEEDS = 10                     # ← Change this as needed
BASE_SEED = 100

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
run_id = f"{timestamp}_{EXPERIMENT_NAME}"
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
exp_dir = Path(repo_root) / "experiments" / run_id
exp_dir.mkdir(parents=True, exist_ok=True)

output_dir = os.path.join(repo_root, 'output')
os.makedirs(output_dir, exist_ok=True)

print(f"Multi-seed experiment started: {exp_dir}")
print(f"Seeds: {NUM_SEEDS} | Max CSAM: {MAX_CSAM_FACILITIES}")

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(exp_dir / "full_log.txt", 'w')
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, log_file)

# ====================== MODEL SETUP (shared) ======================
M = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10']
traditional_m_dict = {'k1': 'm1', 'k2': 'm2', 'k3': 'm3', 'k4': 'm4', 'k5': 'm5'}
L = ['l1', 'l2']
K = ['k1', 'k2', 'k3', 'k4', 'k5']
C = [(l, k) for l in L for k in K]
T = [1, 2]

# ... (keep your full nodes/arcs/D/parameters section here - same as before) ...

C_dummy = 10000   # High penalty as requested

all_summaries = []

for seed_idx in range(NUM_SEEDS):
    SEED = BASE_SEED + seed_idx
    np.random.seed(SEED)
    print(f"\n{'='*60}")
    print(f"Running seed {SEED} ({seed_idx+1}/{NUM_SEEDS})")
    print(f"{'='*60}")

    start_time = timer.time()

    # ====================== BENDERS LOOP (your existing code) ======================
    # Paste your full Benders loop here (master, sub, while loop, etc.)

    # After the loop:
    runtime_seconds = timer.time() - start_time

    # Save CSVs + generate visualizations (using the new script)
    # ... (your existing printing blocks + the viz call from previous message) ...

    # Record summary for this seed
    summary = {
        "seed": SEED,
        "objective": float(ub),
        "deployed_count": int(sum(1 for v in best_y.values() if v > 0.5)),
        "deployed_facilities": [m for m in M if best_y.get(m, 0) > 0.5],
        "unmet_demand": float(sum(best_sub_vars['x_regular'].get(a, 0) for a in regular_arcs if 'dummy' in str(a))),
        "runtime_seconds": float(runtime_seconds),
        "iterations": int(iter_count),
    }
    all_summaries.append(summary)

    # Copy CSVs + run viz for this seed
    seed_dir = exp_dir / f"seed_{SEED}"
    seed_dir.mkdir(exist_ok=True)
    for csv_name in ["csam_flows.csv", "traditional_flows.csv", "inq_flows.csv", 
                     "qq_flows.csv", "dummy_flows.csv", "travel_flows.csv"]:
        src = Path(output_dir) / csv_name
        if src.exists():
            shutil.copy(src, seed_dir / csv_name)

    # Run visualizations
    viz_script = os.path.join(repo_root, "visualizations", "analyze_flows_bd.py")
    try:
        subprocess.run(["python", viz_script, str(seed_dir)], cwd=repo_root, timeout=30)
    except:
        pass

# Final summary table
pd.DataFrame(all_summaries).to_csv(exp_dir / "summary_all_runs.csv", index=False)
print("\n=== Multi-seed experiment completed ===")
print(pd.DataFrame(all_summaries))

sys.stdout = original_stdout
log_file.close()