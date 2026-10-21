#!/bin/bash
# Re-specified H1 (design_ratio_nonbug, RQ2_PREREGISTRATION.md §11) at all
# three units: model fits/CV/H3 via the fit scripts (chi-square only), and the
# shuffle + circular permutation nulls, MDE/TOST and the H2 circular null via
# power_equivalence_rq2.py. Six jobs in parallel, BLAS threads capped.
set -e
cd /home/smorgan/repos/design-detection-and-defect-improvement
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
R=design_ratio_nonbug
pids=()
.venv/bin/python fit_rq2_models.py --ratio_col $R --skip_permutation > logs/primary_nonbug.log 2>&1 & pids+=($!)
for panel in drop bucket; do
  .venv/bin/python fit_rq2_component_models.py --panel data/rq2_component_panel_${panel}.csv \
    --ratio_col $R --skip_permutation > logs/component_${panel}_nonbug.log 2>&1 & pids+=($!)
done
for unit in primary drop bucket; do
  .venv/bin/python power_equivalence_rq2.py --unit $unit --ratio $R > logs/power_equivalence_${unit}_nonbug.log 2>&1 & pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "ALL DONE (status=$status) $(date)"
exit $status
