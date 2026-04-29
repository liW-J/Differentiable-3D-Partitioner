'''
Stability experiment: run the differentiable 3D partitioner 5 times with
different random seeds on the JPEG benchmark.

Each run is launched as an independent subprocess (avoids CUDA fork deadlock).
All output (including C-level DREAMPlace logs) is captured to per-seed log files.

Usage (from project root):
    python -m examples.run_stability
'''

import subprocess
import sys
import os
import time

DREAMPLACE_JSON = "benchmarks/lefdef/asap7/jpeg/dreamplace.json"

SEEDS = [1, 2, 3, 4, 5]   # matches configs/stability/jpeg-seedN.yaml


def get_gpu_count():
    """Detect GPU count without initializing CUDA in the launcher process."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-c",
             "import torch; print(torch.cuda.device_count())"],
            timeout=30).decode().strip()
        return max(int(out), 1)
    except Exception:
        return 1


def main():
    log_dir = "results/logs/stability"
    os.makedirs(log_dir, exist_ok=True)

    num_gpus = get_gpu_count()
    print(f"Detected {num_gpus} GPU(s).")

    processes = []
    for i, seed_idx in enumerate(SEEDS):
        config_path = f"configs/stability/jpeg-seed{seed_idx}.yaml"
        log_path    = os.path.join(log_dir, f"jpeg-seed{seed_idx}.log")
        gpu_id      = i % num_gpus

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        log_f = open(log_path, "w")
        cmd = [
            sys.executable, "-m", "examples.run_single_exp",
            "--dreamplace", DREAMPLACE_JSON,
            "--config", config_path,
            "--gpu", "0",
        ]
        print(f"[{i+1}/{len(SEEDS)}] Launching seed={seed_idx}  "
              f"GPU={gpu_id}  log={log_path}")
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        processes.append((p, log_f, seed_idx))

    print(f"\nAll {len(processes)} experiments launched. Waiting for completion...")
    t0 = time.time()

    for p, log_f, seed_idx in processes:
        p.wait()
        log_f.close()
        elapsed = time.time() - t0
        status = "OK" if p.returncode == 0 else f"FAILED (rc={p.returncode})"
        print(f"  seed={seed_idx}  {status}  ({elapsed:.0f}s elapsed)")

    print(f"\nAll experiments finished in {time.time() - t0:.0f}s.")
    print(f"Logs saved to {log_dir}/")


if __name__ == "__main__":
    main()
