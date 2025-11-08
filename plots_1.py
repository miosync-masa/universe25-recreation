import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# =========================
# Parameters
# =========================
BASE_DIR = "/content/outputs"
TAG = "U25_FULL"
RUN_TAG = "N10"
N_RUNS = 10

# =========================
# Load Data
# =========================
print("📂 Loading data...")
all_ts = []
for run_id in range(N_RUNS):
    ts_path = os.path.join(BASE_DIR, f"{TAG}_{RUN_TAG}_run{run_id:02d}", "timeseries.csv")
    if os.path.exists(ts_path):
        all_ts.append(pd.read_csv(ts_path))
        print(f"  ✓ Run {run_id:02d}")
    else:
        print(f"  ✗ Run {run_id:02d} not found")

if len(all_ts) == 0:
    print("❌ No data loaded!")
    exit()

print(f"\n✅ Loaded {len(all_ts)}/{N_RUNS} runs")

# =========================
# Aggregate Metrics
# =========================
print("\n📊 Aggregating metrics...")
required_cols = ["alive", "avg_fit", "avg_coop", "rep_rate", "lcc_frac",
                 "bo_strategic", "bo_survivor", "bo_dying"]

metrics = {}
for col in required_cols:
    if col in all_ts[0].columns:
        stacked = np.vstack([ts[col].to_numpy() for ts in all_ts])
        metrics[col] = {
            "mean": np.nanmean(stacked, axis=0),
            "std": np.nanstd(stacked, axis=0),
            "sem": np.nanstd(stacked, axis=0) / np.sqrt(len(all_ts))
        }
        print(f"  ✓ {col}")
    else:
        print(f"  ✗ {col} not found")

x = np.arange(len(metrics["alive"]["mean"]))

# =========================
# Phase Detection (5-phase version)
# =========================
print("\n🔍 Detecting phases...")

def detect_phases_with_zombie(alive, rep_rate, lcc_frac, bo_dying):
    """5-phase detection: splits Phase 4 into Zombie/Extinction."""
    phases = np.zeros(len(alive), dtype=int)

    peak_idx = np.argmax(alive)
    peak_alive = alive[peak_idx]

    # Phase 1→2: reaches 80% of peak
    phase1_end = peak_idx
    for t in range(len(alive)):
        if alive[t] >= peak_alive * 0.80:
            phase1_end = t
            break

    # Phase 2→3: drops 10% from peak
    phase2_end = peak_idx
    for t in range(peak_idx, len(alive)):
        if alive[t] < peak_alive * 0.90:
            phase2_end = t
            break

    # Phase 3→4: Alive < 0.15 (late collapse → zombie state)
    phase3_end = len(alive) - 1
    for t in range(phase2_end, len(alive)):
        if alive[t] < 0.15:
            phase3_end = t
            break

    # Phase 4→5: Alive < 0.05 (zombie → extinction)
    phase4_end = len(alive) - 1
    for t in range(phase3_end, len(alive)):
        if alive[t] < 0.05:
            phase4_end = t
            break

    # Assign intervals
    phases[0:phase1_end] = 1          # Growth
    phases[phase1_end:phase2_end] = 2  # Prosperity
    phases[phase2_end:phase3_end] = 3  # Collapse
    phases[phase3_end:phase4_end] = 4  # Zombie State
    phases[phase4_end:] = 5            # Extinction

    return phases

phases = detect_phases_with_zombie(
    metrics["alive"]["mean"],
    metrics["rep_rate"]["mean"],
    metrics["lcc_frac"]["mean"],
    metrics["bo_dying"]["mean"]
)

# =========================
# Phase Statistics (integrated version)
# =========================
phase_names = {
    1: "Growth",
    2: "Prosperity",
    3: "Collapse",
    4: "Zombie",
    5: "Extinction"
}

phase_durations = {p: int(np.sum(phases == p)) for p in [1, 2, 3, 4, 5]}

print("\n" + "="*60)
print("📊 PHASE DURATION SUMMARY")
print("="*60)

total_steps = len(phases)
for p in [1, 2, 3, 4, 5]:
    dur = phase_durations[p]
    pct = 100 * dur / total_steps
    print(f"  Phase {p} ({phase_names[p]:12s}): {dur:3d} steps ({pct:5.1f}%)")

# Phase transition points
print("\n" + "-"*60)
print("🔄 PHASE TRANSITIONS")
print("-"*60)

for p in range(2, 6):
    if p in phases:
        trans_step = np.where(phases == p)[0][0]
        prev_name = phase_names[p-1]
        curr_name = phase_names[p]
        print(f"  {prev_name:12s} → {curr_name:12s}: step {trans_step:3d}")
    else:
        print(f"  Phase {p-1}→{p}: N/A")

# Special statistics for Zombie period
if 4 in phases and 5 in phases:
    zombie_dur = phase_durations[4]
    extinction_dur = phase_durations[5]
    total_phase45 = zombie_dur + extinction_dur
    zombie_pct_of_phase45 = 100 * zombie_dur / total_phase45 if total_phase45 > 0 else 0

    print("\n" + "-"*60)
    print("🧟 ZOMBIE STATE ANALYSIS")
    print("-"*60)
    print(f"  Zombie duration:      {zombie_dur} steps")
    print(f"  Extinction duration:  {extinction_dur} steps")
    print(f"  Total Phase 4+5:      {total_phase45} steps")
    print(f"  Zombie ratio:         {zombie_pct_of_phase45:.1f}% of Phase 4+5")
    print(f"  Combined ratio:       {100*(total_phase45)/total_steps:.1f}% of total")

print("="*60 + "\n")

# =========================
# Phase Background Colors (5 colors)
# =========================
phase_colors = {
    1: 'lightgreen',   # Growth
    2: 'lightyellow',  # Prosperity
    3: 'lightcoral',   # Collapse
    4: 'plum',         # Zombie
    5: 'lightgray'     # Extinction
}

def add_phase_background(ax, phases):
    """Add phase background color to the plot"""
    start = 0
    cur = phases[0]
    for t in range(1, len(phases) + 1):
        if t == len(phases) or phases[t] != cur:
            ax.axvspan(start, t, alpha=0.2, color=phase_colors.get(cur, 'white'), linewidth=0)
            if t < len(phases):
                start = t
                cur = phases[t]

# =========================
# Normalization Function
# =========================
def normalize_multi(means_list, sems_list):
    """Normalize multiple series to 0-1 (min-max)"""
    all_data = np.vstack(means_list)
    vmin = np.min(all_data)
    vmax = np.max(all_data)
    denom = vmax - vmin if vmax > vmin else 1.0

    means_norm = [(m - vmin) / denom for m in means_list]
    sems_norm = [s / denom for s in sems_list]
    cis_norm = [1.96 * s for s in sems_norm]

    return means_norm, cis_norm

# =========================
# Output Directory
# =========================
out_dir = os.path.join(BASE_DIR, f"{TAG}_ensemble")
os.makedirs(out_dir, exist_ok=True)

print(f"📁 Output directory: {out_dir}\n")

# =========================
# Plot 1: BO 3-types
# =========================
print("📈 Plotting BO 3-types...")

if all(col in metrics for col in ["bo_strategic", "bo_survivor", "bo_dying"]):
    means_list = [
        metrics["bo_strategic"]["mean"],
        metrics["bo_survivor"]["mean"],
        metrics["bo_dying"]["mean"]
    ]
    sems_list = [
        metrics["bo_strategic"]["sem"],
        metrics["bo_survivor"]["sem"],
        metrics["bo_dying"]["sem"]
    ]
    labels = ["BO Strategic", "BO Survivor", "BO Dying"]

    means_norm, cis_norm = normalize_multi(means_list, sems_list)

    plt.figure(figsize=(10, 4.6))
    ax = plt.gca()
    add_phase_background(ax, phases)

    line_objs = []
    for idx, lab in enumerate(labels):
        line, = ax.plot(x, means_norm[idx], linewidth=2.5, label=lab)
        line_objs.append(line)

    for idx, line in enumerate(line_objs):
        c = line.get_color()
        lo = np.maximum(0, means_norm[idx] - cis_norm[idx])
        hi = np.minimum(1, means_norm[idx] + cis_norm[idx])
        ax.fill_between(x, lo, hi, alpha=0.18, color=c)

    ax.set_xlim(0, len(x) - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (steps)", fontsize=12)
    ax.set_ylabel("BO Subtypes (normalized 0-1)", fontsize=12)
    ax.set_title(f"Beautiful Ones: Three-Type Classification (N={len(all_ts)})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()

    out_bo3 = os.path.join(out_dir, f"{RUN_TAG}_BO_three_types_phase.pdf")
    plt.savefig(out_bo3, dpi=300)
    plt.show()
    print(f"  ✅ Saved: {out_bo3}")
else:
    print("  ⚠️  BO columns not found, skipping")

# =========================
# Plot 2: Alive
# =========================
print("\n📈 Plotting Alive dynamics...")

if "alive" in metrics:
    alive_mean = metrics["alive"]["mean"]
    alive_sem = metrics["alive"]["sem"]
    ci_alive = 1.96 * alive_sem

    plt.figure(figsize=(10, 4.6))
    ax = plt.gca()
    add_phase_background(ax, phases)

    ax.plot(x, alive_mean, linewidth=2.5, label='Alive Fraction', color='steelblue')
    ax.fill_between(x, np.maximum(0, alive_mean - ci_alive),
                    np.minimum(1, alive_mean + ci_alive),
                    alpha=0.25, color='steelblue', label='95% CI')

    ax.set_xlim(0, len(x) - 1)
    ax.set_xlabel("Time (steps)", fontsize=12)
    ax.set_ylabel("Alive Fraction", fontsize=12)
    ax.set_title(f"Population Dynamics with Phase Transitions (N={len(all_ts)})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()

    out_alive_phase = os.path.join(out_dir, f"{RUN_TAG}_alive_phase.pdf")
    plt.savefig(out_alive_phase, dpi=300)
    plt.show()
    print(f"  ✅ Saved: {out_alive_phase}")

# =========================
# Plot 3: Multi-panel
# =========================
print("\n📈 Plotting multi-panel...")

if "alive" in metrics and all(col in metrics for col in ["bo_strategic", "bo_survivor", "bo_dying"]):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: Alive
    ax = axes[0]
    add_phase_background(ax, phases)
    ax.plot(x, alive_mean, linewidth=2.5, color='steelblue')
    ax.fill_between(x, np.maximum(0, alive_mean - ci_alive),
                    np.minimum(1, alive_mean + ci_alive),
                    alpha=0.25, color='steelblue')
    ax.set_ylabel("Alive Fraction", fontsize=11)
    ax.set_title(f"Universe 25 Collapse Dynamics (N={len(all_ts)})", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Bottom: BO 3-types
    ax = axes[1]
    add_phase_background(ax, phases)
    for idx, lab in enumerate(labels):
        ax.plot(x, means_norm[idx], linewidth=2, label=lab)
        c = line_objs[idx].get_color()
        lo = np.maximum(0, means_norm[idx] - cis_norm[idx])
        hi = np.minimum(1, means_norm[idx] + cis_norm[idx])
        ax.fill_between(x, lo, hi, alpha=0.15, color=c)

    ax.set_xlim(0, len(x) - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (steps)", fontsize=11)
    ax.set_ylabel("BO Subtypes (norm.)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    out_multi = os.path.join(out_dir, f"{RUN_TAG}_multipanel_phase.pdf")
    plt.savefig(out_multi, dpi=300)
    plt.show()
    print(f"  ✅ Saved: {out_multi}")

# =========================
# Save phase statistics as JSON
# =========================
phase_summary = {
    "total_steps": int(total_steps),
    "phases": {
        phase_names[p]: {
            "duration": int(phase_durations[p]),
            "percentage": float(100 * phase_durations[p] / total_steps)
        } for p in [1, 2, 3, 4, 5]
    },
    "transitions": {
        f"{phase_names[p-1]}_to_{phase_names[p]}": int(np.where(phases == p)[0][0])
        if p in phases else None
        for p in range(2, 6)
    }
}

if 4 in phases and 5 in phases:
    phase_summary["zombie_analysis"] = {
        "zombie_duration": int(phase_durations[4]),
        "extinction_duration": int(phase_durations[5]),
        "total_phase45": int(phase_durations[4] + phase_durations[5]),
        "zombie_percentage_of_phase45": float(zombie_pct_of_phase45),
        "phase45_percentage_of_total": float(100 * (phase_durations[4] + phase_durations[5]) / total_steps)
    }

json_path = os.path.join(out_dir, f"{RUN_TAG}_phase_summary.json")
with open(json_path, "w") as f:
    json.dump(phase_summary, f, indent=2)

print(f"\n💾 Phase summary saved: {json_path}")

# =========================
# Done
# =========================
print("\n" + "="*60)
print("✅ ALL PLOTS AND ANALYSES COMPLETE")
print("="*60)
print(f"📁 Output directory: {out_dir}")
print(f"📊 Generated files:")
print(f"   - {RUN_TAG}_BO_three_types_phase.pdf")
print(f"   - {RUN_TAG}_alive_phase.pdf")
print(f"   - {RUN_TAG}_multipanel_phase.pdf")
print(f"   - {RUN_TAG}_phase_summary.json")
print("="*60)
