#!/usr/bin/env python3
"""
Per-project sensitivity/specificity from the frozen Base+Meta LOPO predictions.

RQ2 plan (paper/RQ2_COLLAPSED_PLAN.md) step 3: these feed the Rogan-Gladen
prevalence correction (blocker 2) and the 10- vs 39-project scope decision
(blocker 4) -- a project whose classifier sensitivity/specificity we can't
estimate can't have its design ratio corrected, so it can't be safely added
outside the 10 labelled projects without new hand-labelling.

Uses the already-completed 5-seed sweep (gcp_results/seed_sweep/base_meta_seed*)
at the frozen config: RoBERTa Stage-1 pretrained, metadata, inverse class
weights, fixed 0.5 threshold, balanced validation project. No retraining needed.
"""

import glob
import json

import pandas as pd

SEEDS = [42, 43, 44, 45, 46]
PRED_PATTERN = 'gcp_results/seed_sweep/base_meta_seed{seed}_results_predictions.csv'
OUT_JSON = 'gcp_results/rq2_sensitivity_specificity.json'


def main():
    frames = []
    for seed in SEEDS:
        df = pd.read_csv(PRED_PATTERN.format(seed=seed))
        df['seed'] = seed
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    rows = []
    for project, g in all_df.groupby('project'):
        tp = ((g['label'] == 1) & (g['pred'] == 1)).sum()
        fn = ((g['label'] == 1) & (g['pred'] == 0)).sum()
        tn = ((g['label'] == 0) & (g['pred'] == 0)).sum()
        fp = ((g['label'] == 0) & (g['pred'] == 1)).sum()
        sensitivity = tp / (tp + fn) if (tp + fn) else float('nan')
        specificity = tn / (tn + fp) if (tn + fp) else float('nan')
        true_prevalence = (g['label'] == 1).mean()
        apparent_prevalence = (g['pred'] == 1).mean()
        rows.append({
            'project': project,
            'n_labeled_x_seeds': int(len(g)),
            'n_labeled': int(len(g) / len(SEEDS)),
            'tp': int(tp), 'fn': int(fn), 'tn': int(tn), 'fp': int(fp),
            'sensitivity': round(float(sensitivity), 4),
            'specificity': round(float(specificity), 4),
            'true_prevalence': round(float(true_prevalence), 4),
            'apparent_prevalence': round(float(apparent_prevalence), 4),
        })

    rows.sort(key=lambda r: r['sensitivity'] + r['specificity'])

    print(f"{'Project':<12} {'n':>5} {'Sens':>7} {'Spec':>7} {'TruePrev':>9} {'ApptPrev':>9}")
    for r in rows:
        print(
            f"{r['project']:<12} {r['n_labeled']:>5} {r['sensitivity']:>7.4f} "
            f"{r['specificity']:>7.4f} {r['true_prevalence']:>9.4f} {r['apparent_prevalence']:>9.4f}"
        )

    with open(OUT_JSON, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == '__main__':
    main()
