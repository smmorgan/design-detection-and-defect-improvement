#!/bin/bash
# Multi-seed sweep for tab:lopo (RoBERTa, text-only, no metadata) -- the
# augmentation-level and layer-freezing ablation. Mirrors run_seed_sweep.sh's
# pattern. "baseline" (no augmentation, no freezing) is already fully swept
# at gcp_results/seed_sweep/baseline_seed{42-46}_results.json (used both here
# and as tab:meta's no-metadata comparison point), so it is not repeated.
# All configs use the flags SEED_VARIANCE_FINDINGS.md confirms reproduce the
# original tab:lopo protocol: no --metadata, inverse class weights, fixed 0.5
# threshold, smallest-project validation selection.
set -u
cd "$(dirname "$0")"

PY=.venv/bin/python
OUT=gcp_results/seed_sweep
SEEDS="${SEEDS:-42 43 44 45 46}"
ROBERTA=gcp_results/roberta_conservative_0309_0738
COMMON="--pretrained_model $ROBERTA --class_weighting inverse --threshold fixed --val_selection smallest"
mkdir -p "$OUT/logs"

# name|args -- each reproduces one row of tab:lopo (manuscript.tex)
CONFIGS=(
  "lopo_aug250|$COMMON --augment --aug_max 250"
  "lopo_aug500|$COMMON --augment --aug_max 500"
  "lopo_aug750|$COMMON --augment --aug_max 750"
  "lopo_aug1000|$COMMON --augment --aug_max 1000"
  "lopo_augall|$COMMON --augment"
  "lopo_freeze10|$COMMON --freeze_layers 10"
  "lopo_freeze10_aug|$COMMON --augment --freeze_layers 10"
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
echo "[$(date +%T)] lopo aug sweep finished"
