#!/usr/bin/env python3
"""
Sequential local runner for RoBERTa and DistilBERT hyperparameter configs.

Runs configs one at a time to avoid GPU OOM (11.69 GB GPU, batch_size <= 16).
Results are written to gcp_results/<experiment_id>/.

Usage:
  nohup python run_local_alt_models.py > run_local_alt_models.log 2>&1 &
  tail -f run_local_alt_models.log
"""

import sys
import time
from datetime import datetime
from pathlib import Path

from gcp_parallel_search import launch_local


def gpu_has_training_process() -> bool:
    """Check if any train_design_classifier.py process is currently using the GPU."""
    import subprocess
    from pathlib import Path
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            text=True, timeout=10,
        )
        gpu_pids = {int(line.strip()) for line in out.strip().splitlines() if line.strip()}
    except Exception:
        return False
    for pid in gpu_pids:
        try:
            cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
            if 'train_design_classifier' in cmdline:
                return True
        except Exception:
            continue
    return False

OUTPUT_DIR = Path('./gcp_results')

CONFIGS = [
    # ── RoBERTa ──────────────────────────────────────────────────────────────
    {
        'name': 'roberta_conservative',
        'model': 'roberta-base',
        'learning_rate': 1e-5,
        'batch_size': 16,
        'epochs': 3,
        'dropout': 0.1,
    },
    {
        'name': 'roberta_standard',
        'model': 'roberta-base',
        'learning_rate': 2e-5,
        'batch_size': 16,
        'epochs': 3,
        'dropout': 0.1,
    },
    {
        'name': 'roberta_very_conservative',
        'model': 'roberta-base',
        'learning_rate': 5e-6,
        'batch_size': 16,
        'epochs': 5,
        'dropout': 0.1,
    },
    # ── DistilBERT ───────────────────────────────────────────────────────────
    {
        'name': 'distilbert_conservative',
        'model': 'distilbert-base-uncased',
        'learning_rate': 1e-5,
        'batch_size': 16,
        'epochs': 3,
        'dropout': 0.1,
    },
    {
        'name': 'distilbert_standard',
        'model': 'distilbert-base-uncased',
        'learning_rate': 2e-5,
        'batch_size': 16,
        'epochs': 3,
        'dropout': 0.1,
    },
    {
        'name': 'distilbert_very_conservative',
        'model': 'distilbert-base-uncased',
        'learning_rate': 5e-6,
        'batch_size': 16,
        'epochs': 5,
        'dropout': 0.1,
    },
]


def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


def main():
    timestamp = datetime.now().strftime('%m%d_%H%M')
    log("=" * 60)
    log("Local alt-model runner started")
    log(f"  Configs: {[c['name'] for c in CONFIGS]}")
    log(f"  Output:  {OUTPUT_DIR}")
    log("=" * 60)

    results = []

    for config in CONFIGS:
        experiment_id = f"{config['name']}_{timestamp}"
        log("")
        log(f"Starting: {experiment_id}")
        log(f"  model={config['model']}  lr={config['learning_rate']}"
            f"  batch={config['batch_size']}  epochs={config['epochs']}")

        # Wait if GPU is busy (shouldn't happen in sequential mode, but just in case)
        while gpu_has_training_process():
            log("  GPU busy — waiting 60s ...")
            time.sleep(60)

        proc = launch_local(config, experiment_id, OUTPUT_DIR, mode='full')
        if proc is None:
            log(f"  FAILED to start {experiment_id}")
            results.append((experiment_id, 'launch_failed'))
            continue

        start = time.time()
        proc.wait()
        elapsed = time.time() - start
        rc = proc.returncode
        flag = 'OK' if rc == 0 else f'FAILED (exit {rc})'
        log(f"  [{flag}] {experiment_id}  ({elapsed/60:.1f} min)")
        results.append((experiment_id, flag))

    log("")
    log("=" * 60)
    log("All configs finished:")
    for eid, status in results:
        log(f"  {status:<20} {eid}")
    log("=" * 60)


if __name__ == '__main__':
    main()
