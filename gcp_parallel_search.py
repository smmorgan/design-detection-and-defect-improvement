"""
Parallel Hyperparameter Search on Google Cloud Compute Engine
=============================================================

Each experiment config gets its own VM with a T4 GPU.
VMs self-terminate after training. Results are collected in a GCS bucket.

Prerequisites:
  1. gcloud CLI installed and authenticated:
       gcloud auth login
       gcloud config set project YOUR_PROJECT_ID
  2. Compute Engine API and Cloud Storage API enabled in your project.
  3. A GCS bucket created:
       gsutil mb gs://YOUR_BUCKET_NAME

Workflow:
  # Step 1 — upload code and training data (one-time setup):
  python gcp_parallel_search.py \\
      --project my-project --bucket my-bucket \\
      --upload --train_data ./data/data/train_data/raw/combined.csv

  # Step 2 — launch all configs via a work queue (1 GCP VM + local in parallel):
  python gcp_parallel_search.py \\
      --project my-project --bucket my-bucket --zone us-central1-a \\
      --targeted_search --local

  # GCP-only (serialize all configs through 1 VM, no local worker):
  python gcp_parallel_search.py \\
      --project my-project --bucket my-bucket --zone us-central1-a \\
      --targeted_search

  # Or launch specific configs only:
  python gcp_parallel_search.py \\
      --project my-project --bucket my-bucket --zone us-central1-a \\
      --configs bert_conservative,bert_baseline

  # Step 3 — monitor progress:
  python gcp_parallel_search.py \\
      --project my-project --bucket my-bucket --monitor

  # Step 4 — download results when done:
  python gcp_parallel_search.py \\
      --project my-project --bucket my-bucket \\
      --download --output_dir ./gcp_results

Costs (approximate):
  L4 GPU VM (g2-standard-4): ~$0.70/hr
  12 configs serialized through 1 VM (~60 min each) → ~$8.40 total
  Add --local to run one config locally in parallel, saving ~1 VM-slot worth of time
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# ── Experiment configurations (mirrors hyperparameter_search.py) ──────────────

TARGETED_CONFIGS = [
    {
        'name': 'bert_conservative',
        'model': 'bert-base-uncased',
        'learning_rate': 1e-5,
        'batch_size': 16,
        'epochs': 3,
        'dropout': 0.1,
    },
    {
        'name': 'bert_standard',
        'model': 'bert-base-uncased',
        'learning_rate': 2e-5,
        'batch_size': 16,
        'epochs': 3,
        'dropout': 0.1,
    },
    {
        'name': 'bert_longer',
        'model': 'bert-base-uncased',
        'learning_rate': 2e-5,
        'batch_size': 16,
        'epochs': 5,
        'dropout': 0.1,
    },
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
        'name': 'distilbert_longer',
        'model': 'distilbert-base-uncased',
        'learning_rate': 2e-5,
        'batch_size': 16,
        'epochs': 5,
        'dropout': 0.1,
    },
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
    # ── BERT follow-up: baseline + conservative variants ──────────────────────
    # Textbook BERT baseline: batch=32, lr=2e-5, 3 epochs — standard HF recipe
    {
        'name': 'bert_baseline',
        'model': 'bert-base-uncased',
        'learning_rate': 2e-5,
        'batch_size': 32,
        'epochs': 3,
        'dropout': 0.1,
    },
    # Conservative LR extended: same as bert_conservative but 5 epochs
    {
        'name': 'bert_conservative_longer',
        'model': 'bert-base-uncased',
        'learning_rate': 1e-5,
        'batch_size': 16,
        'epochs': 5,
        'dropout': 0.1,
    },
    # Conservative LR + larger batch: reduces gradient noise at low LR
    {
        'name': 'bert_conservative_large_batch',
        'model': 'bert-base-uncased',
        'learning_rate': 1e-5,
        'batch_size': 32,
        'epochs': 3,
        'dropout': 0.1,
    },
    # Very conservative LR: below the targeted search floor, sanity-checks collapse
    {
        'name': 'bert_very_conservative',
        'model': 'bert-base-uncased',
        'learning_rate': 5e-6,
        'batch_size': 16,
        'epochs': 5,
        'dropout': 0.1,
    },
]

# Source files to package and upload to GCS
CODE_FILES = [
    'train_design_classifier.py',
    'requirements.txt',
]

# Default paths for transfer mode data
DEFAULT_TAWOS_PATH = './data/data/manually_labelled_data/tawos_labeled_SERVER.tsv'
DEFAULT_MANUALLY_LABELLED_DIR = './data/data/manually_labelled_data'

# ── GCS helpers ───────────────────────────────────────────────────────────────

def gsutil(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(['gsutil'] + args, check=check, text=True,
                          capture_output=True)


def gcloud(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(['gcloud'] + args, check=check, text=True,
                          capture_output=True)


def upload_code(bucket: str, repo_dir: Path):
    """Upload training code to gs://bucket/code/."""
    print(f"Uploading code to gs://{bucket}/code/ ...")
    for fname in CODE_FILES:
        src = repo_dir / fname
        if not src.exists():
            print(f"  WARNING: {fname} not found, skipping")
            continue
        r = gsutil(['-q', 'cp', str(src), f'gs://{bucket}/code/{fname}'])
        print(f"  Uploaded {fname}")


def upload_data(bucket: str, train_data: str):
    """Upload training CSV to gs://bucket/data/."""
    print(f"Uploading training data to gs://{bucket}/data/ ...")
    src = Path(train_data)
    if not src.exists():
        print(f"ERROR: {train_data} not found.")
        sys.exit(1)
    # Preserve directory structure under data/
    dest = f'gs://{bucket}/data/data/train_data/raw/{src.name}'
    gsutil(['-q', 'cp', str(src), dest])
    print(f"  Uploaded {src.name}")


def upload_transfer_data(bucket: str, tawos_path: str, manually_labelled_dir: str):
    """Upload TAWOS file and manually-labelled TSVs for transfer mode."""
    print(f"Uploading transfer learning data to gs://{bucket}/data/ ...")

    tawos = Path(tawos_path)
    if not tawos.exists():
        print(f"ERROR: tawos_path not found: {tawos_path}")
        sys.exit(1)
    dest = f'gs://{bucket}/data/data/manually_labelled_data/{tawos.name}'
    gsutil(['-q', 'cp', str(tawos), dest])
    print(f"  Uploaded {tawos.name}")

    ml_dir = Path(manually_labelled_dir)
    if not ml_dir.is_dir():
        print(f"ERROR: manually_labelled_dir not found: {manually_labelled_dir}")
        sys.exit(1)
    gsutil(['-m', '-q', 'rsync', '-r', str(ml_dir),
            f'gs://{bucket}/data/data/manually_labelled_data/'])
    print(f"  Uploaded manually_labelled_data/ ({len(list(ml_dir.glob('*.tsv')))} TSVs)")


# ── Startup script generation ─────────────────────────────────────────────────

def make_startup_script(config: Dict, bucket: str, zone: str,
                        experiment_id: str, mode: str = 'full',
                        max_n_labels: int = 1500) -> str:
    """Generate the bash startup script that runs inside the VM."""

    if mode == 'transfer':
        train_cmd_parts = [
            'python3 train_design_classifier.py',
            '--mode transfer',
            f'--model {config["model"]}',
            '--stackoverflow_path ./data/data/train_data/raw/combined.csv',
            '--tawos_path ./data/data/manually_labelled_data/tawos_labeled_SERVER.tsv',
            '--manually_labelled_dir ./data/data/manually_labelled_data',
            f'--max_n_labels {max_n_labels}',
            '--output_dir ./output',
            f'--epochs {config["epochs"]}',
            f'--learning_rate {config["learning_rate"]}',
            f'--batch_size {config["batch_size"]}',
            f'--dropout {config["dropout"]}',
        ]
    else:
        train_cmd_parts = [
            'python3 train_design_classifier.py',
            '--mode full',
            '--no_preprocess',
            f'--model {config["model"]}',
            '--stackoverflow_path ./data/data/train_data/raw/combined.csv',
            '--output_dir ./output',
            f'--epochs {config["epochs"]}',
            f'--learning_rate {config["learning_rate"]}',
            f'--batch_size {config["batch_size"]}',
            f'--dropout {config["dropout"]}',
        ]
    train_cmd = ' \\\n    '.join(train_cmd_parts)

    extra_data_download = ''
    if mode == 'transfer':
        extra_data_download = '''
# Download TAWOS and manually-labelled data for transfer mode
echo "=== Downloading transfer learning data ==="
mkdir -p data/data/manually_labelled_data
gsutil -m rsync -r gs://{bucket}/data/data/manually_labelled_data/ ./data/data/manually_labelled_data/
'''.format(bucket=bucket)

    return f'''#!/bin/bash
# Auto-generated startup script for experiment: {experiment_id}
set -e
LOGFILE=/var/log/training_{experiment_id}.log
exec >> "$LOGFILE" 2>&1

echo "=== [{experiment_id}] Starting at $(date) ==="
echo "Config: mode={mode} model={config['model']} lr={config['learning_rate']} epochs={config['epochs']} batch={config['batch_size']}"

WORKDIR=/opt/training_{experiment_id}
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Download training code
echo "=== Downloading code ==="
gsutil -m cp -r gs://{bucket}/code/* .

# Install dependencies from requirements.txt
echo "=== Installing dependencies ==="
pip3 install -q -r requirements.txt

# Download training data
echo "=== Downloading training data ==="
mkdir -p data/data/train_data/raw
gsutil -m cp gs://{bucket}/data/data/train_data/raw/combined.csv ./data/data/train_data/raw/
{extra_data_download}
# Run training
echo "=== Starting training ==="
{train_cmd}

TRAIN_EXIT=$?
echo "=== Training finished with exit code $TRAIN_EXIT at $(date) ==="

# Upload results
echo "=== Uploading results ==="
gsutil -m cp -r ./output/* gs://{bucket}/results/{experiment_id}/

# Upload the training log
gsutil cp "$LOGFILE" gs://{bucket}/results/{experiment_id}/training.log

# Write completion marker (include exit code so monitor can detect failures)
echo "$TRAIN_EXIT" | gsutil cp - gs://{bucket}/results/{experiment_id}/done.txt

echo "=== Done. Shutting down VM. ==="
# Self-terminate this VM
INSTANCE_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
INSTANCE_ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F/ '{{print $NF}}')
gcloud compute instances stop "$INSTANCE_NAME" --zone="$INSTANCE_ZONE" --quiet
'''


# ── VM lifecycle ──────────────────────────────────────────────────────────────

def launch_vm(project: str, zone: str, config: Dict, bucket: str,
              experiment_id: str, dry_run: bool = False,
              mode: str = 'full', max_n_labels: int = 1500) -> str:
    """Create a Compute Engine VM for one experiment. Returns instance name."""

    instance_name = f'hparam-{experiment_id.replace("_", "-")[:50]}'
    startup_script = make_startup_script(config, bucket, zone, experiment_id,
                                         mode=mode, max_n_labels=max_n_labels)

    # Write startup script to a temp file (avoids shell quoting issues)
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False)
    tmp.write(startup_script)
    tmp.flush()

    cmd = [
        'gcloud', 'compute', 'instances', 'create', instance_name,
        f'--project={project}',
        f'--zone={zone}',
        '--machine-type=g2-standard-4',
        '--accelerator=type=nvidia-l4,count=1',
        '--maintenance-policy=TERMINATE',          # required for GPU VMs
        '--restart-on-failure',                    # don't restart on crash
        '--no-restart-on-failure',                 # override: don't loop on failure
        '--image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570',
        '--image-project=deeplearning-platform-release',
        '--boot-disk-size=100GB',
        '--boot-disk-type=pd-balanced',
        f'--metadata-from-file=startup-script={tmp.name}',
        '--scopes=cloud-platform',                 # allows gsutil + gcloud inside VM
        '--format=json',
    ]

    print(f"  Launching VM: {instance_name}")
    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd[:8])} ...")
        os.unlink(tmp.name)
        return instance_name

    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        os.unlink(tmp.name)
        print(f"  VM created: {instance_name}")
        return instance_name
    except subprocess.CalledProcessError as e:
        os.unlink(tmp.name)
        print(f"  ERROR creating VM {instance_name}:")
        print(f"  {e.stderr}")
        raise


def list_running_experiments(project: str, zone: str) -> List[str]:
    """List hparam-* VMs that are still running."""
    result = gcloud([
        'compute', 'instances', 'list',
        f'--project={project}',
        f'--filter=name:hparam-* AND zone:{zone} AND status:RUNNING',
        '--format=value(name)',
    ], check=False)
    names = [n.strip() for n in result.stdout.splitlines() if n.strip()]
    return names


# ── Results ───────────────────────────────────────────────────────────────────

def check_completion(bucket: str, experiment_ids: List[str]) -> Dict[str, Optional[int]]:
    """
    Check which experiments have finished.
    Returns dict of {experiment_id: exit_code or None if still running}.
    """
    status = {}
    for eid in experiment_ids:
        result = gsutil(['cat', f'gs://{bucket}/results/{eid}/done.txt'], check=False)
        if result.returncode == 0:
            try:
                status[eid] = int(result.stdout.strip())
            except ValueError:
                status[eid] = 0
        else:
            status[eid] = None  # still running
    return status


def download_results(bucket: str, output_dir: Path):
    """Download all results from GCS to local output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading results from gs://{bucket}/results/ to {output_dir} ...")
    gsutil(['-m', 'rsync', '-r', f'gs://{bucket}/results/', str(output_dir)])
    print("Download complete.")

    # Print summary table
    _print_summary(output_dir)


def _print_summary(output_dir: Path):
    """Print a simple comparison of metrics across experiments."""
    import csv

    rows = []
    for metrics_file in sorted(output_dir.glob('*/training_metrics.csv')):
        experiment_id = metrics_file.parent.name
        try:
            with open(metrics_file) as f:
                reader = csv.DictReader(f)
                records = list(reader)
            if records:
                last = records[-1]
                rows.append({
                    'experiment': experiment_id,
                    'model': last.get('model_name', '?'),
                    'lr': last.get('learning_rate', '?'),
                    'epochs': last.get('num_epochs', '?'),
                    'test_f1': last.get('test_f1_score', '?'),
                    'test_auc': last.get('test_auc', '?'),
                    'test_acc': last.get('test_accuracy', '?'),
                })
        except Exception:
            pass

    if not rows:
        print("No training_metrics.csv files found yet.")
        return

    rows.sort(key=lambda r: float(r['test_f1']) if r['test_f1'] not in ('?', '') else 0,
              reverse=True)

    header = f"{'Experiment':<35} {'Model':<25} {'LR':<8} {'Ep':<4} {'F1':<8} {'AUC':<8} {'Acc':<8}"
    print(f"\n{'='*len(header)}")
    print("RESULTS SUMMARY (sorted by F1)")
    print('='*len(header))
    print(header)
    print('-'*len(header))
    for r in rows:
        print(f"{r['experiment']:<35} {r['model']:<25} {r['lr']:<8} {r['epochs']:<4} "
              f"{r['test_f1']:<8} {r['test_auc']:<8} {r['test_acc']:<8}")
    print('='*len(header))


# ── Local worker ──────────────────────────────────────────────────────────────

def launch_local(config: Dict, experiment_id: str, output_dir: Path,
                 mode: str = 'full', max_n_labels: int = 1500,
                 dry_run: bool = False) -> Optional[subprocess.Popen]:
    """Start a local training subprocess. Returns Popen (or None for dry_run)."""

    local_out = output_dir / experiment_id
    local_out.mkdir(parents=True, exist_ok=True)

    if mode == 'transfer':
        cmd = [
            sys.executable, 'train_design_classifier.py',
            '--mode', 'transfer',
            '--model', config['model'],
            '--stackoverflow_path', './data/data/train_data/raw/combined.csv',
            '--tawos_path', './data/data/manually_labelled_data/tawos_labeled_SERVER.tsv',
            '--manually_labelled_dir', './data/data/manually_labelled_data',
            '--max_n_labels', str(max_n_labels),
            '--output_dir', str(local_out),
            '--epochs', str(config['epochs']),
            '--learning_rate', str(config['learning_rate']),
            '--batch_size', str(config['batch_size']),
            '--dropout', str(config['dropout']),
        ]
    else:
        cmd = [
            sys.executable, 'train_design_classifier.py',
            '--mode', 'full',
            '--no_preprocess',
            '--model', config['model'],
            '--stackoverflow_path', './data/data/train_data/raw/combined.csv',
            '--output_dir', str(local_out),
            '--epochs', str(config['epochs']),
            '--learning_rate', str(config['learning_rate']),
            '--batch_size', str(config['batch_size']),
            '--dropout', str(config['dropout']),
        ]

    print(f"  Launching LOCAL: {experiment_id}")
    if dry_run:
        print(f"  [DRY RUN] Would run: {' '.join(cmd[:5])} ...")
        return None

    log_path = local_out / 'training.log'
    log_file = open(log_path, 'w')
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"  Local process PID {proc.pid} → log: {log_path}")
    return proc


# ── Work-queue dispatcher ──────────────────────────────────────────────────────

def run_work_queue(configs: List[Dict], project: str, zone: str, bucket: str,
                   mode: str, max_n_labels: int, dry_run: bool,
                   output_dir: Path, max_vms: int = 1,
                   use_local: bool = False) -> List[Dict]:
    """
    Dispatch configs to workers (GCP VMs and/or local) as slots open.

    At any moment there are at most max_vms GCP VMs and at most 1 local
    process running.  When a worker finishes, the next config from the
    queue is immediately dispatched to that slot.
    """
    timestamp = datetime.now().strftime('%m%d_%H%M')
    queue = list(configs)

    # active: experiment_id -> {'type': 'gcp'|'local', 'config': ..., 'process': Popen|None}
    active: Dict[str, Dict] = {}
    launched: List[Dict] = []

    def gcp_active_count() -> int:
        return sum(1 for v in active.values() if v['type'] == 'gcp')

    def local_is_busy() -> bool:
        return any(v['type'] == 'local' for v in active.values())

    def dispatch_next(prefer_local: bool = False) -> bool:
        """Try to dispatch the next queued config to an available slot.

        Tries slots in priority order; falls through to the next slot if the
        preferred one fails (e.g. GCP quota exceeded).  Returns True if a
        config was dispatched, False if nothing could be started.
        """
        if not queue:
            return False

        slots = []
        if not prefer_local:
            if gcp_active_count() < max_vms:
                slots.append('gcp')
            if use_local and not local_is_busy():
                slots.append('local')
        else:
            if use_local and not local_is_busy():
                slots.append('local')
            if gcp_active_count() < max_vms:
                slots.append('gcp')

        if not slots:
            return False

        config = queue.pop(0)
        experiment_id = f"{config['name']}_{timestamp}"

        for slot in slots:
            if slot == 'gcp':
                try:
                    instance = launch_vm(project, zone, config, bucket,
                                         experiment_id, dry_run, mode, max_n_labels)
                    active[experiment_id] = {'type': 'gcp', 'config': config,
                                             'instance': instance, 'process': None}
                    launched.append({'experiment_id': experiment_id,
                                     'type': 'gcp', 'config': config})
                    return True
                except Exception as e:
                    print(f"  GCP launch failed for {config['name']}: {e} — "
                          f"{'trying local' if 'local' in slots else 'will retry later'}")
                    # fall through to local if available
            else:  # local
                proc = launch_local(config, experiment_id, output_dir, mode,
                                    max_n_labels, dry_run)
                active[experiment_id] = {'type': 'local', 'config': config,
                                         'instance': None, 'process': proc}
                launched.append({'experiment_id': experiment_id,
                                 'type': 'local', 'config': config})
                return True

        # All slots failed — put config back and stop trying for now
        queue.insert(0, config)
        return False

    # Initial fill: dispatch until no slots remain or all slots fail
    while queue:
        if not dispatch_next():
            break

    if dry_run or not active:
        return launched

    # Poll loop
    print(f"\nPolling for completion (Ctrl+C to stop polling and exit) ...")
    try:
        while active:
            time.sleep(60)
            now = datetime.now().strftime('%H:%M:%S')
            completed = []

            # Check local process
            for eid, info in list(active.items()):
                if info['type'] == 'local' and info['process'] is not None:
                    rc = info['process'].poll()
                    if rc is not None:
                        flag = 'OK' if rc == 0 else f'FAILED (exit {rc})'
                        print(f"  [{now}] LOCAL [{flag}] {eid}")
                        completed.append((eid, 'local'))

            # Check GCP completions
            gcp_eids = [k for k, v in active.items() if v['type'] == 'gcp']
            if gcp_eids:
                status = check_completion(bucket, gcp_eids)
                for eid, code in status.items():
                    if code is not None:
                        flag = 'OK' if code == 0 else f'FAILED (exit {code})'
                        print(f"  [{now}] GCP [{flag}] {eid}")
                        completed.append((eid, 'gcp'))

            # Free slots and dispatch next
            for eid, worker_type in completed:
                del active[eid]
                dispatch_next(prefer_local=(worker_type == 'local'))

            # Also try to fill any open slot that wasn't triggered by a
            # completion (e.g. GCP VM stopped externally, quota freed up)
            if queue:
                dispatch_next()

            print(f"  [{now}] {len(active)} active "
                  f"(gcp={gcp_active_count()}, local={int(local_is_busy())}), "
                  f"{len(queue)} queued")

    except KeyboardInterrupt:
        print("\nStopped polling. Active workers will continue running.")

    return launched


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Launch parallel hyperparameter search on GCP Compute Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # GCP settings
    parser.add_argument('--project', type=str, help='GCP project ID')
    parser.add_argument('--bucket', type=str, help='GCS bucket name (no gs:// prefix)')
    parser.add_argument('--zone', type=str, default='us-central1-a',
                        help='GCP zone (default: us-central1-a)')

    # Actions
    parser.add_argument('--upload', action='store_true',
                        help='Upload code and training data to GCS (one-time setup)')
    parser.add_argument('--train_data', type=str,
                        default='./data/data/train_data/raw/combined.csv',
                        help='Local path to training CSV (used with --upload)')
    parser.add_argument('--targeted_search', action='store_true',
                        help='Run all configs in TARGETED_CONFIGS via the work queue')
    parser.add_argument('--configs', type=str,
                        help='Comma-separated config names to run '
                             '(e.g. bert_conservative,bert_baseline)')
    parser.add_argument('--max_vms', type=int, default=1,
                        help='Max concurrent GCP VMs (default: 1)')
    parser.add_argument('--local', action='store_true',
                        help='Also run one config locally in parallel with the GCP VM')
    parser.add_argument('--monitor', action='store_true',
                        help='Check completion status of running experiments')
    parser.add_argument('--experiment_ids', type=str,
                        help='Comma-separated experiment IDs to monitor/download')
    parser.add_argument('--download', action='store_true',
                        help='Download results from GCS to --output_dir')
    parser.add_argument('--output_dir', type=str, default='./gcp_results',
                        help='Local directory for downloaded results (default: ./gcp_results)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be done without actually creating VMs')
    parser.add_argument('--transfer', action='store_true',
                        help='Use --mode transfer instead of --mode full')
    parser.add_argument('--tawos_path', type=str, default=DEFAULT_TAWOS_PATH,
                        help='Local path to TAWOS TSV (used with --upload --transfer)')
    parser.add_argument('--manually_labelled_dir', type=str,
                        default=DEFAULT_MANUALLY_LABELLED_DIR,
                        help='Local path to manually-labelled TSV dir (used with --upload --transfer)')
    parser.add_argument('--max_n_labels', type=int, default=1500,
                        help='Max N (non-design) samples for Stage 3 (default: 1500)')

    args = parser.parse_args()

    # ── Upload ────────────────────────────────────────────────────────────────
    if args.upload:
        if not args.bucket:
            parser.error('--bucket is required for --upload')
        repo_dir = Path(__file__).parent
        upload_code(args.bucket, repo_dir)
        upload_data(args.bucket, args.train_data)
        if args.transfer:
            upload_transfer_data(args.bucket, args.tawos_path, args.manually_labelled_dir)
        print("\nUpload complete. Ready to launch experiments.")
        return

    # ── Monitor ───────────────────────────────────────────────────────────────
    if args.monitor:
        if not args.bucket:
            parser.error('--bucket is required for --monitor')

        if args.experiment_ids:
            eids = [e.strip() for e in args.experiment_ids.split(',')]
        else:
            # List all result folders in GCS
            result = gsutil(['ls', f'gs://{args.bucket}/results/'], check=False)
            eids = [p.rstrip('/').split('/')[-1]
                    for p in result.stdout.splitlines() if p.strip()]

        if not eids:
            print("No experiments found in GCS results folder.")
            return

        print(f"\nChecking {len(eids)} experiment(s)...\n")
        status = check_completion(args.bucket, eids)
        done = [(e, c) for e, c in status.items() if c is not None]
        running = [e for e, c in status.items() if c is None]

        print(f"  Done    ({len(done)}): ")
        for eid, code in done:
            flag = 'OK' if code == 0 else f'FAILED (exit {code})'
            print(f"    [{flag}] {eid}")
        print(f"  Running ({len(running)}): ")
        for eid in running:
            print(f"    {eid}")
        return

    # ── Download ──────────────────────────────────────────────────────────────
    if args.download:
        if not args.bucket:
            parser.error('--bucket is required for --download')
        download_results(args.bucket, Path(args.output_dir))
        return

    # ── Launch experiments ────────────────────────────────────────────────────
    if not args.project:
        parser.error('--project is required to launch experiments')
    if not args.bucket:
        parser.error('--bucket is required to launch experiments')

    # Select configs to run
    if args.targeted_search:
        configs = TARGETED_CONFIGS
    elif args.configs:
        names = {c.strip() for c in args.configs.split(',')}
        configs = [c for c in TARGETED_CONFIGS if c['name'] in names]
        if not configs:
            print(f"No matching configs found for: {args.configs}")
            print("Available:", ', '.join(c['name'] for c in TARGETED_CONFIGS))
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)

    mode = 'transfer' if args.transfer else 'full'
    output_dir = Path(args.output_dir)
    print(f"\nWork queue: {len(configs)} config(s), "
          f"max_vms={args.max_vms}, local={'yes' if args.local else 'no'}, "
          f"mode={mode}\n")

    launched = run_work_queue(
        configs=configs,
        project=args.project,
        zone=args.zone,
        bucket=args.bucket,
        mode=mode,
        max_n_labels=args.max_n_labels,
        dry_run=args.dry_run,
        output_dir=output_dir,
        max_vms=args.max_vms,
        use_local=args.local,
    )

    if not launched:
        print("No experiments launched.")
        return

    # Save launch manifest
    manifest_path = output_dir / 'launch_manifest.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(launched, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")

    if args.dry_run:
        return

    print("\nAll experiments complete. Downloading results ...")
    download_results(args.bucket, output_dir)


if __name__ == '__main__':
    main()
