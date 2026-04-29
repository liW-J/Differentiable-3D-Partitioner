'''
Collect stability experiment results and generate visualizations.

Steps
-----
1. Parse the DEF netlist once to build net → cell mapping.
2. Compute true cutsize for:
   - hMETIS  : every part.*.txt file in HMETIS_DIR
   - Ours    : each seed result in results/jpeg-stability-seedN/
   - TritonPart: placeholder (skip if directory absent)
3. Print raw data + per-method statistics (avg, max, min, std).
4. Save:
   - stability_raw.csv
   - stability_distribution.png  (strip plot + box overlay)
   - stability_runs.png          (run-index line plot with mean ± std band)

Usage (from project root):
    python -m examples.collect_stability
'''

import os
import sys
import csv
import glob
import re
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Make sure sibling scripts are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_cutsize import (
    parse_def_netlist,
    compute_cutsize,
    load_hmetis_partition,
    load_ours_partition,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEF_PATH     = PROJECT_ROOT / "benchmarks/lefdef/asap7/jpeg/jpeg.def"
HMETIS_DIR   = PROJECT_ROOT / "results/partition-2d/jpeg"
OUR_SEEDS    = list(range(1, 6))          # seeds 1..5
RESULT_DIR   = PROJECT_ROOT / "results"
OUTPUT_DIR   = PROJECT_ROOT / "results/stability"

# TritonPart directory – set to None if not yet available
TRITONPART_DIR: Optional[Path] = None    # e.g. PROJECT_ROOT / "results/partition-triton/jpeg"


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_hmetis(netlist: dict) -> list[dict]:
    '''Return list of {label, run_id, cutsize} for each hMETIS partition file.'''
    part_files = sorted(glob.glob(str(HMETIS_DIR / "part.*.txt")))
    rows = []
    for i, pf in enumerate(part_files, start=1):
        fname = Path(pf).stem            # e.g. part.ub1.000000.seed1
        partition = load_hmetis_partition(pf)
        cs = compute_cutsize(netlist, partition)
        rows.append({'method': 'hMETIS', 'run_id': i, 'label': fname,
                     'cutsize': cs})
        print(f"  hMETIS {fname}: cutsize = {cs}")
    return rows


def collect_ours(netlist: dict) -> list:
    '''Return list of {label, run_id, cutsize} for each seed of our method.'''
    rows = []
    for seed in OUR_SEEDS:
        design_name = f"jpeg-stability-seed{seed}"
        run_dir = RESULT_DIR / design_name
        binary  = run_dir / "binary_assignment.txt"
        names   = run_dir / "node_names.txt"

        if not binary.exists():
            print(f"  [WARN] Ours seed={seed}: binary_assignment.txt not found, skipping")
            continue
        if not names.exists():
            print(f"  [WARN] Ours seed={seed}: node_names.txt not found, skipping")
            continue

        partition = load_ours_partition(str(binary), str(names))
        cs = compute_cutsize(netlist, partition)
        rows.append({'method': 'Ours', 'run_id': seed, 'label': f'seed{seed}',
                     'cutsize': cs})
        print(f"  Ours seed={seed}: cutsize = {cs}")
    return rows


def collect_tritonpart(netlist: dict) -> list:
    '''Placeholder for TritonPart. Returns empty list if dir is absent.'''
    if TRITONPART_DIR is None or not Path(TRITONPART_DIR).exists():
        print("  [INFO] TritonPart results not available, skipping.")
        return []

    part_files = sorted(glob.glob(str(TRITONPART_DIR / "part.*.txt")))
    rows = []
    for i, pf in enumerate(part_files, start=1):
        fname = Path(pf).stem
        partition = load_hmetis_partition(pf)   # same format assumed
        cs = compute_cutsize(netlist, partition)
        rows.append({'method': 'TritonPart', 'run_id': i, 'label': fname,
                     'cutsize': cs})
        print(f"  TritonPart {fname}: cutsize = {cs}")
    return rows


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def statistics(values: list) -> dict:
    arr = np.array(values, dtype=float)
    return {
        'n':    len(arr),
        'avg':  float(np.mean(arr)),
        'std':  float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'min':  float(np.min(arr)),
        'max':  float(np.max(arr)),
        'median': float(np.median(arr)),
    }


def print_and_save(all_rows: list):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group by method
    methods = {}
    for r in all_rows:
        methods.setdefault(r['method'], []).append(r['cutsize'])

    print('\n' + '=' * 70)
    print('Stability Experiment -- True Cutsize Statistics (JPEG, ASAP7)')
    print('=' * 70)
    hdr = f"{'Method':<14} {'N':>3} {'Avg':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Median':>8}"
    print(hdr)
    print('-' * 70)
    stats_rows = []
    for mth, vals in methods.items():
        s = statistics(vals)
        print(f"{mth:<14} {s['n']:>3} {s['avg']:>8.1f} {s['std']:>8.1f} "
              f"{s['min']:>8.0f} {s['max']:>8.0f} {s['median']:>8.1f}")
        stats_rows.append({'method': mth, **s})
    print('=' * 70)

    # Save raw CSV
    raw_csv = OUTPUT_DIR / 'stability_raw.csv'
    with open(raw_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'run_id', 'label', 'cutsize'])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nRaw data saved to {raw_csv}')

    # Save stats CSV
    stats_csv = OUTPUT_DIR / 'stability_stats.csv'
    with open(stats_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'n', 'avg', 'std', 'min', 'max', 'median'])
        writer.writeheader()
        writer.writerows(stats_rows)
    print(f'Stats saved to {stats_csv}')

    return methods


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Color palette – one per method
METHOD_COLORS = {
    'hMETIS':     '#E07B54',   # orange
    'TritonPart': '#6BAED6',   # blue
    'Ours':       '#74C476',   # green
}
METHOD_ORDER = ['hMETIS', 'TritonPart', 'Ours']


def plot_distribution(methods: dict):
    '''Strip plot + box overlay for each method.'''
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    present = [m for m in METHOD_ORDER if m in methods and methods[m]]
    if not present:
        print('[WARN] No data to plot in distribution figure.')
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    rng = np.random.default_rng(42)
    for xi, mth in enumerate(present):
        vals = methods[mth]
        color = METHOD_COLORS.get(mth, '#888888')

        # Box plot (thin, no caps)
        bp = ax.boxplot(vals, positions=[xi], widths=0.35,
                        patch_artist=True, notch=False, showcaps=False,
                        whiskerprops=dict(linewidth=1.2, color=color, linestyle='--'),
                        medianprops=dict(linewidth=2.0, color='white'),
                        boxprops=dict(facecolor=color, alpha=0.30, linewidth=1.2, edgecolor=color),
                        flierprops=dict(marker=''),
                        manage_ticks=False)

        # Strip / jitter plot
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter([xi + j for j in jitter], vals,
                   color=color, alpha=0.90, s=60, zorder=5,
                   edgecolors='white', linewidths=0.5)

        # Mean marker
        ax.scatter([xi], [np.mean(vals)], marker='D', color='white',
                   edgecolors=color, linewidths=1.5, s=80, zorder=6)

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, fontsize=13)
    ax.set_ylabel('True Cutsize (#cut nets)', fontsize=12)
    ax.set_title('Partitioner Stability: Cutsize Distribution\n'
                 '(JPEG, ASAP7, 5 Runs)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=METHOD_COLORS.get(m, '#888'),
                           markersize=9, label=m) for m in present]
    handles.append(plt.Line2D([0], [0], marker='D', color='gray',
                               markerfacecolor='white', markersize=8,
                               markeredgewidth=1.5, label='Mean', linestyle=''))
    ax.legend(handles=handles, fontsize=10, loc='upper right')

    plt.tight_layout()
    out = OUTPUT_DIR / 'stability_distribution.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Distribution plot saved to {out}')


def plot_runs(methods: dict):
    '''Line plot of cutsize per run index with mean ± std band.'''
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    present = [m for m in METHOD_ORDER if m in methods and methods[m]]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for mth in present:
        vals = np.array(methods[mth], dtype=float)
        color = METHOD_COLORS.get(mth, '#888888')
        xs = np.arange(1, len(vals) + 1)

        ax.plot(xs, vals, 'o-', color=color, linewidth=2,
                markersize=7, label=mth, zorder=4)

        if len(vals) > 1:
            mean = vals.mean()
            std  = vals.std(ddof=1)
            ax.axhline(mean, color=color, linewidth=1.0, linestyle='--', alpha=0.7)
            ax.fill_between(xs, mean - std, mean + std,
                            color=color, alpha=0.12, zorder=3)

    # X-axis: use the maximum run count
    max_runs = max(len(methods[m]) for m in present)
    ax.set_xticks(range(1, max_runs + 1))
    ax.set_xticklabels([f'Run {i}' for i in range(1, max_runs + 1)], fontsize=10)
    ax.set_ylabel('True Cutsize (#cut nets)', fontsize=12)
    ax.set_xlabel('Run Index', fontsize=12)
    ax.set_title('Partitioner Stability: Cutsize per Run\n'
                 '(JPEG, ASAP7; dashed = mean, band = ±1 std)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / 'stability_runs.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Run-index plot saved to {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f'Parsing DEF netlist: {DEF_PATH}')
    netlist = parse_def_netlist(str(DEF_PATH))
    print(f'  {len(netlist)} nets loaded\n')

    print('Collecting hMETIS results ...')
    rows = collect_hmetis(netlist)

    print('\nCollecting our method results ...')
    rows += collect_ours(netlist)

    print('\nCollecting TritonPart results ...')
    rows += collect_tritonpart(netlist)

    if not rows:
        print('[ERROR] No results collected. Run experiments first.')
        return

    methods = print_and_save(rows)
    plot_distribution(methods)
    plot_runs(methods)

    print('\nDone. Outputs in:', OUTPUT_DIR)


if __name__ == '__main__':
    main()
