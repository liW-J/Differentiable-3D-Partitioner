'''
Sweep top/bottom threshold_factor for any design.

Scans threshold pairs from (start, 1-start) to (0.50, 0.50) in steps of 0.05,
runs the partitioner for each, records the true cutsize, and saves the best
result to the standard result directory.

Tasks are distributed across multiple GPUs in parallel via multiprocessing.

Usage examples:
  # jpeg on asap7 — parallel across 5 GPUs
  python examples/sweep_threshold.py \
      --dreamplace benchmarks/lefdef/asap7/jpeg/dreamplace.json \
      --config configs/openroad/asap7/jpeg.yaml \
      --gpus 2,3,4,5,6

  # aes on nangate45 — single GPU (sequential)
  python examples/sweep_threshold.py \
      --dreamplace benchmarks/lefdef/nangate45/aes/dreamplace.json \
      --config configs/openroad/nangate45/aes.yaml \
      --gpus 2

  # swerv_wrapper on asap7_nangate45
  python examples/sweep_threshold.py \
      --dreamplace benchmarks/lefdef/asap7_nangate45/swerv_wrapper/dreamplace.json \
      --config configs/openroad/asap7_nangate45/swerv_wrapper.yaml \
      --gpus 2,3,4,5,6
'''

import argparse
import copy
import gc
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import traceback

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_sweep_points(start=0.20, end=0.50, step=0.05):
    """Generate (top_factor, bottom_factor) pairs where bottom = 1 - top."""
    points = []
    top = start
    while top <= end + 1e-9:
        top_r = round(top, 2)
        bot_r = round(1.0 - top_r, 2)
        points.append((top_r, bot_r))
        top += step
    return points


def worker(args):
    """Self-contained worker: parse design, run flow, return metrics."""
    import torch
    from partitioner import Differentiable3DPartitionerFlow, DreamplaceParser

    top_factor, bottom_factor, dreamplace_path, base_config, design_name, gpu_id = args

    tag = f"[GPU {gpu_id}] top={top_factor:.2f}, bot={bottom_factor:.2f}"
    try:
        if torch.cuda.is_available() and gpu_id is not None:
            torch.cuda.set_device(gpu_id)

        print(f"{tag}  Parsing design ...")
        parser = DreamplaceParser()
        parser.parse_design(dreamplace_path)
        print(f"{tag}  Design parsed.")

        config = copy.deepcopy(base_config)
        config['partitioner']['balance_loss']['top_threshold_factor'] = top_factor
        config['partitioner']['balance_loss']['bottom_threshold_factor'] = bottom_factor
        suffix = f"sweep_top{top_factor:.2f}_bot{bottom_factor:.2f}"
        config['design']['name'] = f"{design_name}/{suffix}"

        fd, tmp_yaml = tempfile.mkstemp(suffix='.yaml', prefix='sweep_')
        with os.fdopen(fd, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        try:
            flow = Differentiable3DPartitionerFlow(
                num_nodes=parser.num_nodes,
                num_nets=parser.num_nets,
                num_pins=parser.num_pins,
                node_pos=parser.node_pos,
                pin_pos=parser.pin_pos,
                flat_net2pin_map=parser.flat_net2pin_map,
                flat_net2pin_start_map=parser.flat_net2pin_start_map,
                pin2node_map=parser.pin2node_map,
                node_size_x=parser.node_size_x,
                node_size_y=parser.node_size_y,
                dreamplace_basic=parser.dreamplace_basic,
                config_path=tmp_yaml,
                die_xl=parser.die_xl,
                die_yl=parser.die_yl,
                die_xh=parser.die_xh,
                die_yh=parser.die_yh,
            )
            metrics = flow.run()
        finally:
            del flow, parser
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if os.path.exists(tmp_yaml):
                os.remove(tmp_yaml)

        entry = {
            'top_threshold_factor': top_factor,
            'bottom_threshold_factor': bottom_factor,
            'true_cutsize': metrics.get('true_cutsize'),
            'true_d2d_hpwl': metrics.get('true_d2d_hpwl'),
            'final_total_loss': metrics.get('final_total_loss'),
            'final_cutsize': metrics.get('final_cutsize'),
            'final_balance': metrics.get('final_balance'),
            'top_cells': metrics.get('top_cells'),
            'bottom_cells': metrics.get('bottom_cells'),
        }
        print(f"{tag}  Done — true_cutsize = {entry['true_cutsize']}")
        return entry

    except Exception as e:
        print(f"{tag}  FAILED: {e}")
        traceback.print_exc()
        return {
            'top_threshold_factor': top_factor,
            'bottom_threshold_factor': bottom_factor,
            'true_cutsize': None,
            'true_d2d_hpwl': None,
            'final_total_loss': None,
            'final_cutsize': None,
            'final_balance': None,
            'top_cells': None,
            'bottom_cells': None,
            'error': str(e),
        }


def main():
    ap = argparse.ArgumentParser(
        description="Sweep threshold_factor for any design (parallel across GPUs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--dreamplace', required=True,
                    help='Path to dreamplace.json (relative to project root)')
    ap.add_argument('--config', required=True,
                    help='Path to config YAML (relative to project root)')
    ap.add_argument('--gpus', required=True,
                    help='Comma-separated GPU ids, e.g. "2,3,4,5,6"')
    ap.add_argument('--start', type=float, default=0.20,
                    help='Starting top_threshold_factor (default 0.20)')
    ap.add_argument('--end', type=float, default=0.50,
                    help='Ending top_threshold_factor (default 0.50)')
    ap.add_argument('--step', type=float, default=0.05,
                    help='Step size for top_threshold_factor (default 0.05)')
    args = ap.parse_args()

    gpu_list = [int(g.strip()) for g in args.gpus.split(',')]

    config_abs = os.path.join(PROJECT_ROOT, args.config)
    dreamplace_abs = os.path.join(PROJECT_ROOT, args.dreamplace)

    with open(config_abs, 'r') as f:
        base_config = yaml.safe_load(f)

    design_name = base_config['design']['name']

    sweep_points = build_sweep_points(args.start, args.end, args.step)

    print(f"\n{'='*60}")
    print(f"Design:          {design_name}")
    print(f"Config:          {args.config}")
    print(f"DREAMPlace JSON: {args.dreamplace}")
    print(f"GPUs:            {gpu_list}")
    print(f"Sweep points:    {len(sweep_points)}")
    print(f"Parallelism:     {min(len(sweep_points), len(gpu_list))} workers")
    for i, (top_f, bot_f) in enumerate(sweep_points):
        gpu_id = gpu_list[i % len(gpu_list)]
        print(f"  top={top_f:.2f}, bot={bot_f:.2f}  ->  GPU {gpu_id}")
    print(f"{'='*60}\n")

    result_base = os.path.join(PROJECT_ROOT, "results", design_name)
    os.makedirs(result_base, exist_ok=True)

    tasks = []
    for i, (top_f, bot_f) in enumerate(sweep_points):
        gpu_id = gpu_list[i % len(gpu_list)]
        tasks.append((top_f, bot_f, dreamplace_abs, base_config, design_name,
                       gpu_id))

    num_workers = min(len(tasks), len(gpu_list))
    print(f"Launching {num_workers} parallel workers for {len(tasks)} tasks ...\n")

    with mp.Pool(processes=num_workers) as pool:
        all_results = pool.map(worker, tasks)

    # --- Find best ---
    valid = [r for r in all_results if r.get('true_cutsize') is not None]
    if not valid:
        print("\nERROR: no valid results collected.")
        for r in all_results:
            if r.get('error'):
                print(f"  top={r['top_threshold_factor']:.2f}: {r['error']}")
        sys.exit(1)

    best = min(valid, key=lambda r: r['true_cutsize'])

    # --- Copy best binary_assignment to the main result dir ---
    best_suffix = (f"sweep_top{best['top_threshold_factor']:.2f}"
                   f"_bot{best['bottom_threshold_factor']:.2f}")
    best_src_dir = os.path.join(result_base, best_suffix)
    for fname in ('binary_assignment.pt', 'binary_assignment.txt'):
        src = os.path.join(best_src_dir, fname)
        dst = os.path.join(result_base, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # --- Write sweep summary ---
    summary = {
        'design': design_name,
        'config': args.config,
        'dreamplace': args.dreamplace,
        'gpus': gpu_list,
        'sweep_config': {
            'start': args.start,
            'end': args.end,
            'step': args.step,
        },
        'best': best,
        'all_results': all_results,
    }
    summary_path = os.path.join(result_base, 'sweep_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # --- Print summary table ---
    print(f"\n{'='*60}")
    print(f"SWEEP SUMMARY  —  {design_name}")
    print(f"{'='*60}")
    print(f"{'top':>6s}  {'bot':>6s}  {'cutsize':>10s}  {'d2d_hpwl':>12s}  "
          f"{'top_cells':>10s}  {'bot_cells':>10s}")
    print('-' * 66)
    for r in all_results:
        if r.get('true_cutsize') is None:
            print(f"{r['top_threshold_factor']:6.2f}  "
                  f"{r['bottom_threshold_factor']:6.2f}  "
                  f"{'FAILED':>10s}")
            continue
        marker = " *" if r is best else ""
        print(f"{r['top_threshold_factor']:6.2f}  "
              f"{r['bottom_threshold_factor']:6.2f}  "
              f"{r['true_cutsize']:>10}  "
              f"{r['true_d2d_hpwl']:>12.2f}  "
              f"{r['top_cells']:>10}  "
              f"{r['bottom_cells']:>10}"
              f"{marker}")
    print('-' * 66)
    print(f"\nBest: top={best['top_threshold_factor']:.2f}, "
          f"bot={best['bottom_threshold_factor']:.2f}, "
          f"true_cutsize={best['true_cutsize']}")
    print(f"\nBest binary_assignment copied to: {result_base}/")
    print(f"Full sweep summary saved to:      {summary_path}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
