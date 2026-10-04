#!/bin/bash
# Multi-seed LOPO sweep: measures run-to-run variance of the key configurations
# so they can be reported as mean ± SD instead of a single run. Seed-major order
# keeps completed runs balanced across configs if the sweep is stopped early;
# existing result files are skipped, so re-running the script resumes it.
set -u
cd "$(dirname "$0")"

PY=.venv/bin/python
OUT=gcp_results/seed_sweep
SEEDS="${SEEDS:-42 43 44 45 46}"
ROBERTA=gcp_results/roberta_conservative_0309_0738
MODERNBERT=gcp_results/modernbert_standard_0831_1935
mkdir -p "$OUT/logs"

# name|args -- each reproduces the code path of the run reported in the paper
CONFIGS=(
  "base_meta_sqrtw|--pretrained_model $ROBERTA --metadata --class_weighting sqrt --threshold tuned --val_selection balance"
  "base_meta|--pretrained_model $ROBERTA --metadata --class_weighting inverse --threshold fixed --val_selection balance"
  "baseline|--pretrained_model $ROBERTA --class_weighting inverse --threshold fixed --val_selection smallest"
  "modernbert_meta_sqrtw|--pretrained_model $MODERNBERT --metadata --class_weighting sqrt --threshold tuned --val_selection balance --batch_size 8"
  # Remaining tab:meta rows. Aug250 adds the top 250 heuristic D and N labels
  # to each fold's training set. Note the paper's row names understate what
  # they do: the "SqrtW" rows also tune a per-fold threshold, while the
  # "Thresh" rows tune one on top of full inverse-frequency weights.
  "aug250_meta|--pretrained_model $ROBERTA --metadata --augment --aug_max 250 --class_weighting inverse --threshold fixed --val_selection balance"
  "aug250_meta_sqrtw|--pretrained_model $ROBERTA --metadata --augment --aug_max 250 --class_weighting sqrt --threshold tuned --val_selection balance"
  "base_meta_thresh|--pretrained_model $ROBERTA --metadata --class_weighting inverse --threshold tuned --val_selection balance"
  "aug250_meta_thresh|--pretrained_model $ROBERTA --metadata --augment --aug_max 250 --class_weighting inverse --threshold tuned --val_selection balance"
)

for seed in $SEEDS; do
  for cfg in "${CONFIGS[@]}"; do
    name="${cfg%%|*}"
    args="${cfg#*|}"
    out="$OUT/${name}_seed${seed}_results.json"
    if [ -f "$out" ]; then
      echo "[$(date +%T)] skip $name seed=$seed (exists)"
      continue
    fi
    echo "[$(date +%T)] start $name seed=$seed"
    # shellcheck disable=SC2086
    if $PY eval_lopo.py $args --seed "$seed" --out "$out" > "$OUT/logs/${name}_seed${seed}.log" 2>&1; then
      echo "[$(date +%T)] done  $name seed=$seed  $(grep -o '"mean_f1": [0-9.]*' "$out")"
    else
      echo "[$(date +%T)] FAIL  $name seed=$seed (see $OUT/logs/${name}_seed${seed}.log)"
    fi
  done
done
echo "[$(date +%T)] sweep finished"
