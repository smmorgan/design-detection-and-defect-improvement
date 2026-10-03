#!/bin/bash
# Runs the 4 component-unit cells (drop/bucket x raw/corrected) plus both
# primary-unit ratios in parallel. BLAS threads are capped per process so the
# six jobs don't oversubscribe the machine.
set -e
cd /home/smorgan/repos/design-detection-and-defect-improvement
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
pids=()
for panel in drop bucket; do
  for ratio in raw corrected; do
    .venv/bin/python fit_rq2_component_models.py \
      --panel data/rq2_component_panel_${panel}.csv \
      --ratio_col design_ratio_${ratio} \
      > logs/component_${panel}_${ratio}.log 2>&1 &
    pids+=($!)
  done
done
for ratio in raw corrected; do
  .venv/bin/python fit_rq2_models.py --ratio_col design_ratio_${ratio} \
    > logs/primary_${ratio}.log 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "ALL DONE (status=$status) $(date)"
exit $status
