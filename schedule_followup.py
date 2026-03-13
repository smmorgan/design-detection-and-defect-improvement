#!/usr/bin/env python3
"""
Watch the running GCP VM experiment and local training process, then dispatch
the 4 follow-up BERT configs to whichever worker frees up first.

This is a single long-running process — it monitors both workers, dispatches
follow-up configs one at a time as slots open, and exits when all are done.

Usage:
  nohup python schedule_followup.py &
  tail -f schedule_followup.log
"""

import os
import sys
import json
import time
import signal
import subprocess
from datetime import datetime
from pathlib import Path

# ── Import helpers from gcp_parallel_search ──────────────────────────────────
from gcp_parallel_search import (
    TARGETED_CONFIGS,
    check_completion,
    launch_vm,
    launch_local,
    make_startup_script,
    gsutil,
    download_results,
    upload_transfer_data,
    upload_code,
    DEFAULT_TAWOS_PATH,
    DEFAULT_MANUALLY_LABELLED_DIR,
)

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT = os.environ.get('GCP_PROJECT', 'project-23587fbc-5e61-4f58-880')
BUCKET = os.environ.get('GCP_BUCKET', 'design-detection-data')
ZONE = 'us-east1-c'
OUTPUT_DIR = Path('./gcp_results')
POLL_INTERVAL = 120  # seconds between checks
MODE = 'full'
LOGFILE = Path('./schedule_followup.log')

# The existing jobs we're watching
MANIFEST_PATH = Path('./gcp_results/launch_manifest.json')
LOCAL_PID = None  # set to a PID to watch an existing local job before dispatching

# Follow-up configs to dispatch (by name from TARGETED_CONFIGS)
# bert_conservative_longer already running from previous scheduler launch
FOLLOWUP_NAMES = {
    'bert_baseline',
    'bert_conservative_large_batch',
    'bert_very_conservative',
}

# ── Logging ──────────────────────────────────────────────────────────────────

def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, 'a') as f:
        f.write(line + '\n')


# ── Helpers ──────────────────────────────────────────────────────────────────

def read_gcp_experiment_ids() -> list[str]:
    """Read experiment IDs from the launch manifest."""
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH) as f:
        data = json.load(f)
    return [entry['experiment_id'] for entry in data]


def pid_is_alive(pid: int) -> bool:
    """Check if a PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def get_followup_configs() -> list[dict]:
    return [c for c in TARGETED_CONFIGS if c['name'] in FOLLOWUP_NAMES]


def gpu_has_training_process() -> bool:
    """Check if any train_design_classifier.py process is using the GPU.

    Uses nvidia-smi to find CUDA processes, then checks if any of those PIDs
    are running train_design_classifier.py.  This prevents OOM from launching
    a second training job while the GPU is already occupied.
    """
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            text=True, timeout=10,
        )
        gpu_pids = {int(line.strip()) for line in out.strip().splitlines() if line.strip()}
    except Exception:
        return False  # can't check — assume free

    for pid in gpu_pids:
        try:
            cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
            if 'train_design_classifier' in cmdline:
                return True
        except Exception:
            continue
    return False


def find_best_config(output_dir: Path, bucket: str) -> dict | None:
    """Find the best config by F1 across all GCS and local results.

    Downloads GCS results first, then scans training_metrics.csv files
    under output_dir.  Falls back to training_history.json (validation F1)
    for runs where training_metrics.csv was never written (df_clean bug).

    Returns {'experiment_id', 'model', 'learning_rate', 'batch_size',
    'epochs', 'dropout', 'test_f1', ...} or None.
    """
    import csv

    # Pull GCS results into output_dir so we have everything in one place
    log("Downloading all results from GCS...")
    try:
        download_results(bucket, output_dir)
    except Exception as e:
        log(f"  WARNING: GCS download failed: {e}")

    # Also check local targeted-search results
    local_search_dir = Path('./models/targeted_search')
    search_dirs = [output_dir]
    if local_search_dir.is_dir():
        search_dirs.append(local_search_dir)

    best = None

    def _update_best(candidate: dict):
        nonlocal best
        if best is None or candidate['test_f1'] > best['test_f1']:
            best = candidate

    for search_dir in search_dirs:
        # ── Primary: training_metrics.csv (has test metrics) ──────────
        for metrics_file in sorted(search_dir.glob('*/training_metrics.csv')):
            experiment_id = metrics_file.parent.name
            try:
                with open(metrics_file) as f:
                    reader = csv.DictReader(f)
                    records = list(reader)
                if not records:
                    continue
                last = records[-1]
                f1 = float(last.get('test_f1_score', 0))
                _update_best({
                    'experiment_id': experiment_id,
                    'model': last.get('model_name', 'bert-base-uncased'),
                    'learning_rate': float(last.get('learning_rate', 2e-5)),
                    'batch_size': int(last.get('batch_size', 16)),
                    'epochs': int(last.get('num_epochs', 3)),
                    'dropout': float(last.get('dropout_rate', 0.1)),
                    'test_f1': f1,
                    'test_auc': last.get('test_auc', '?'),
                    'test_acc': last.get('test_accuracy', '?'),
                    'source': 'training_metrics.csv',
                })
            except Exception:
                continue

        # ── Fallback: training_history.json (validation F1 only) ──────
        for history_file in sorted(search_dir.glob('*/training_history.json')):
            experiment_id = history_file.parent.name
            # Skip if we already have CSV metrics for this experiment
            if (history_file.parent / 'training_metrics.csv').exists():
                continue
            try:
                with open(history_file) as f:
                    data = json.load(f)
                val_metrics = data.get('val_metrics', [])
                if not val_metrics:
                    continue
                # Pick best epoch by validation F1
                best_epoch = max(val_metrics, key=lambda m: m.get('f1_score', 0))
                f1 = float(best_epoch.get('f1_score', 0))
                if f1 == 0:
                    continue
                # Infer config from experiment_id name
                config = _infer_config_from_experiment(experiment_id)
                _update_best({
                    'experiment_id': experiment_id,
                    'model': config.get('model', 'bert-base-uncased'),
                    'learning_rate': config.get('learning_rate', 2e-5),
                    'batch_size': config.get('batch_size', 16),
                    'epochs': config.get('epochs', 3),
                    'dropout': config.get('dropout', 0.1),
                    'test_f1': f1,
                    'test_auc': best_epoch.get('auc', '?'),
                    'test_acc': best_epoch.get('accuracy', '?'),
                    'source': 'training_history.json (val F1)',
                })
            except Exception:
                continue

    return best


def _infer_config_from_experiment(experiment_id: str) -> dict:
    """Try to match an experiment_id back to its TARGETED_CONFIGS entry."""
    for config in TARGETED_CONFIGS:
        if experiment_id.startswith(config['name']):
            return config
    return {}


def launch_transfer_run(best: dict, project: str, bucket: str, zone: str) -> str | None:
    """Launch a transfer-learning GCP VM using the best config. Returns experiment_id."""
    config = {
        'name': 'transfer_best',
        'model': best['model'],
        'learning_rate': best['learning_rate'],
        'batch_size': best['batch_size'],
        'epochs': best['epochs'],
        'dropout': best['dropout'],
    }
    timestamp = datetime.now().strftime('%m%d_%H%M')
    eid = f"transfer_best_{timestamp}"

    # Ensure transfer data is uploaded to GCS
    log("Uploading transfer learning data to GCS...")
    try:
        upload_transfer_data(bucket, DEFAULT_TAWOS_PATH, DEFAULT_MANUALLY_LABELLED_DIR)
    except Exception as e:
        log(f"  WARNING: transfer data upload failed: {e}")

    # Also re-upload code in case it changed
    try:
        upload_code(bucket, Path('.'))
    except Exception as e:
        log(f"  WARNING: code upload failed: {e}")

    log(f"Launching transfer VM: {eid}")
    log(f"  model={config['model']}  lr={config['learning_rate']}  "
        f"batch={config['batch_size']}  epochs={config['epochs']}")
    try:
        launch_vm(project, zone, config, bucket, eid,
                  dry_run=False, mode='transfer', max_n_labels=1500)
        return eid
    except Exception as e:
        log(f"  FAILED to launch transfer VM: {e}")
        return None


# ── Main scheduler ───────────────────────────────────────────────────────────

def main():
    followup_queue = get_followup_configs()
    if not followup_queue:
        log("ERROR: No follow-up configs found in TARGETED_CONFIGS")
        sys.exit(1)

    gcp_eids = read_gcp_experiment_ids()
    timestamp = datetime.now().strftime('%m%d_%H%M')

    log("=" * 60)
    log("Follow-up scheduler started")
    log(f"  Project:    {PROJECT}")
    log(f"  Bucket:     {BUCKET}")
    log(f"  Zone:       {ZONE}")
    log(f"  GCP watch:  {gcp_eids}")
    log(f"  Local PID:  {LOCAL_PID}")
    log(f"  Queue:      {[c['name'] for c in followup_queue]}")
    log(f"  Poll:       {POLL_INTERVAL}s")
    log("=" * 60)

    # Track which workers are busy with the EXISTING jobs
    gcp_busy = bool(gcp_eids)        # GCP VM running the targeted search
    local_busy = LOCAL_PID is not None and pid_is_alive(LOCAL_PID)

    # Track active follow-up jobs we've launched
    active_gcp_eid = None       # experiment_id of follow-up on GCP
    active_local_proc = None    # Popen of follow-up running locally
    active_local_eid = None     # experiment_id of follow-up on local

    launched = []

    def dispatch_to_gcp():
        nonlocal active_gcp_eid
        if not followup_queue:
            return
        config = followup_queue.pop(0)
        eid = f"{config['name']}_{timestamp}"
        log(f"  Dispatching to GCP: {eid}")
        try:
            launch_vm(PROJECT, ZONE, config, BUCKET, eid,
                      dry_run=False, mode=MODE, max_n_labels=1500)
            active_gcp_eid = eid
            launched.append({'experiment_id': eid, 'type': 'gcp', 'config': config})
        except Exception as e:
            log(f"  GCP launch FAILED for {config['name']}: {e}")
            followup_queue.insert(0, config)  # put back

    def dispatch_to_local():
        nonlocal active_local_proc, active_local_eid
        if not followup_queue:
            return
        if gpu_has_training_process():
            log("  Local dispatch deferred — GPU already has a training process")
            return
        config = followup_queue.pop(0)
        eid = f"{config['name']}_{timestamp}"
        log(f"  Dispatching to LOCAL: {eid}")
        proc = launch_local(config, eid, OUTPUT_DIR, mode=MODE,
                            max_n_labels=1500, dry_run=False)
        active_local_proc = proc
        active_local_eid = eid
        launched.append({'experiment_id': eid, 'type': 'local', 'config': config})

    log("")
    log("Watching existing jobs...")

    while followup_queue or active_gcp_eid or active_local_proc:
        time.sleep(POLL_INTERVAL)
        now = datetime.now().strftime('%H:%M:%S')

        # ── Check if the ORIGINAL GCP experiments finished ───────────────
        if gcp_busy and gcp_eids:
            status = check_completion(BUCKET, gcp_eids)
            all_done = all(code is not None for code in status.values())
            if all_done:
                for eid, code in status.items():
                    flag = 'OK' if code == 0 else f'FAILED({code})'
                    log(f"  [{now}] GCP original [{flag}] {eid}")
                gcp_busy = False
                log(f"  [{now}] GCP slot is now FREE")
                if followup_queue and active_gcp_eid is None:
                    dispatch_to_gcp()

        # ── Check if the ORIGINAL local process finished ─────────────────
        if local_busy:
            if not pid_is_alive(LOCAL_PID):
                log(f"  [{now}] Local original DONE (PID {LOCAL_PID})")
                local_busy = False
                log(f"  [{now}] Local slot is now FREE")
                if followup_queue and active_local_proc is None:
                    dispatch_to_local()

        # ── Check follow-up GCP job ──────────────────────────────────────
        if active_gcp_eid:
            status = check_completion(BUCKET, [active_gcp_eid])
            code = status.get(active_gcp_eid)
            if code is not None:
                flag = 'OK' if code == 0 else f'FAILED({code})'
                log(f"  [{now}] GCP follow-up [{flag}] {active_gcp_eid}")
                active_gcp_eid = None
                if followup_queue:
                    dispatch_to_gcp()

        # ── Check follow-up local job ────────────────────────────────────
        if active_local_proc is not None:
            rc = active_local_proc.poll()
            if rc is not None:
                flag = 'OK' if rc == 0 else f'FAILED({rc})'
                log(f"  [{now}] Local follow-up [{flag}] {active_local_eid}")
                active_local_proc = None
                active_local_eid = None
                if followup_queue:
                    dispatch_to_local()

        # ── Try to fill idle slots with queued configs ───────────────────
        if followup_queue:
            if not gcp_busy and active_gcp_eid is None:
                dispatch_to_gcp()
            if not local_busy and active_local_proc is None:
                dispatch_to_local()

        # ── Status line ──────────────────────────────────────────────────
        gcp_state = 'original' if gcp_busy else (active_gcp_eid or 'idle')
        local_state = f'original(PID {LOCAL_PID})' if local_busy else (active_local_eid or 'idle')
        log(f"  [{now}] gcp={gcp_state}  local={local_state}  queued={len(followup_queue)}")

    # ── Phase 2 done ────────────────────────────────────────────────────
    log("")
    log("=" * 60)
    log("Phase 2: All follow-up configs completed!")
    log(f"  Launched: {len(launched)}")
    for item in launched:
        log(f"    [{item['type']}] {item['experiment_id']}")
    log("=" * 60)

    # Save manifest
    manifest_path = OUTPUT_DIR / 'followup_manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(launched, f, indent=2)
    log(f"Manifest saved to: {manifest_path}")

    # ── Phase 3: Find best config, launch transfer learning on GCP ────
    log("")
    log("=" * 60)
    log("Phase 3: Selecting best config for transfer learning")
    log("=" * 60)

    best = find_best_config(OUTPUT_DIR, BUCKET)
    if best is None:
        log("ERROR: Could not find any results (training_metrics.csv or training_history.json)")
        sys.exit(1)

    log(f"  Best experiment: {best['experiment_id']}")
    log(f"    source:   {best.get('source', 'unknown')}")
    log(f"    model:    {best['model']}")
    log(f"    lr:       {best['learning_rate']}")
    log(f"    batch:    {best['batch_size']}")
    log(f"    epochs:   {best['epochs']}")
    log(f"    dropout:  {best['dropout']}")
    log(f"    F1:       {best['test_f1']}")
    log(f"    AUC:      {best['test_auc']}")
    log(f"    Acc:      {best['test_acc']}")

    transfer_eid = launch_transfer_run(best, PROJECT, BUCKET, ZONE)
    if transfer_eid is None:
        log("Transfer learning launch failed. Exiting.")
        sys.exit(1)

    log(f"Transfer learning VM launched: {transfer_eid}")
    log("Waiting for transfer learning to complete...")

    while True:
        time.sleep(POLL_INTERVAL)
        now = datetime.now().strftime('%H:%M:%S')
        status = check_completion(BUCKET, [transfer_eid])
        code = status.get(transfer_eid)
        if code is not None:
            flag = 'OK' if code == 0 else f'FAILED({code})'
            log(f"  [{now}] Transfer learning [{flag}] {transfer_eid}")
            break
        log(f"  [{now}] Transfer learning running... ({transfer_eid})")

    # Download final results
    log("")
    log("Downloading final results (including transfer learning)...")
    try:
        download_results(BUCKET, OUTPUT_DIR)
    except Exception as e:
        log(f"  WARNING: final download failed: {e}")

    log("")
    log("=" * 60)
    log("ALL PHASES COMPLETE")
    log(f"  Phase 1: Original targeted search")
    log(f"  Phase 2: Follow-up BERT configs ({len(launched)} experiments)")
    log(f"  Phase 3: Transfer learning with best config")
    log(f"    Best: {best['experiment_id']} (F1={best['test_f1']})")
    log(f"    Transfer: {transfer_eid}")
    log(f"  Results: {OUTPUT_DIR}")
    log("=" * 60)


if __name__ == '__main__':
    main()
