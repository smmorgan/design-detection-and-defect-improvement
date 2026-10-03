#!/usr/bin/env python3
"""
RQ2 plan step 2 (continued): ensemble the 5-seed Base+Meta predictions into a
single frozen, versioned label set for the full per-project populations.

Reads gcp_results/rq2_labels/<PROJECT>_seed<seed>_full_predictions.csv (one
per project x seed, from generate_versioned_labels.py), averages prob_design
across the 5 seeds per issue_key, thresholds at 0.5 (per SEED_VARIANCE_FINDINGS.md,
per-fold/pooled threshold tuning doesn't help and a plain 0.5 is at least as
good), and joins back onto the full per-project issue metadata
(data/tawos_full/<PROJECT>.csv) so the output is panel-ready.

Output: data/tawos_full/labeled/<PROJECT>_labeled.csv -- one row per issue,
all original columns plus prob_design_ensemble, pred_design, n_seeds.
This is the FROZEN label set referenced by paper/RQ2_PREREGISTRATION.md §6.
Once written, it is not regenerated.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECTS = [
    'CONFSERVER', 'DM', 'DNN', 'FAB', 'JRASERVER',
    'MESOS', 'MULE', 'NEXUS', 'SERVER', 'TIMOB',
]
SEEDS = [42, 43, 44, 45, 46]
LABELS_DIR = Path('gcp_results/rq2_labels')
FULL_POP_DIR = Path('data/tawos_full')
OUT_DIR = FULL_POP_DIR / 'labeled'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = []

    for project in PROJECTS:
        frames = []
        for seed in SEEDS:
            path = LABELS_DIR / f'{project}_seed{seed}_full_predictions.csv'
            if not path.exists():
                missing.append(str(path))
                continue
            df = pd.read_csv(path).rename(columns={'prob_design': f'prob_seed{seed}'})
            frames.append(df.set_index('issue_key'))

        if len(frames) < len(SEEDS):
            logger.warning(f"{project}: only {len(frames)}/{len(SEEDS)} seeds present, skipping for now")
            continue

        merged = pd.concat(frames, axis=1)
        prob_cols = [f'prob_seed{s}' for s in SEEDS]
        merged['prob_design_ensemble'] = merged[prob_cols].mean(axis=1)
        merged['pred_design'] = (merged['prob_design_ensemble'] >= 0.5).astype(int)
        merged['n_seeds'] = merged[prob_cols].notna().sum(axis=1)
        merged = merged.reset_index()[['issue_key', 'prob_design_ensemble', 'pred_design', 'n_seeds']]

        full_df = pd.read_csv(FULL_POP_DIR / f'{project}.csv')
        labeled = full_df.merge(merged, on='issue_key', how='left')
        n_unlabeled = labeled['pred_design'].isna().sum()
        if n_unlabeled:
            logger.warning(f"{project}: {n_unlabeled} issues have no ensemble prediction (join failure?)")

        out_path = OUT_DIR / f'{project}_labeled.csv'
        labeled.to_csv(out_path, index=False)
        design_rate = labeled['pred_design'].mean()
        logger.info(
            f"{project}: {len(labeled)} issues, design rate={design_rate:.3f} -> {out_path}"
        )

    if missing:
        logger.warning(f"{len(missing)} project/seed prediction files still missing, e.g. {missing[:3]}")
        logger.warning("Re-run once generate_versioned_labels.py finishes all (project, seed) pairs.")


if __name__ == '__main__':
    main()
