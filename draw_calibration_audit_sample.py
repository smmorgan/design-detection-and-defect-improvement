"""Draw the random calibration-audit sample for RQ2 (see RQ2_INITIAL_RESULTS.md §1).

The existing manual labels (output/all_manually_labelled.csv) come only from each
project's most recent ~3,000 issues and are enriched for design-like tickets, so the
sensitivity/specificity estimated on them do not transfer to the full population.
This draws a simple random sample, stratified by project x era, from the population
the design ratio is actually computed over: every *resolved* issue (all issue types),
dated by resolution date (matches build_rq2_panel.py).

Design:
  - Era = tercile of resolution date within each project (early / middle / late), so
    calibration drift over a project's history can be tested directly.
  - Tickets already in the manual sample are excluded.
  - PER_STRATUM tickets drawn per (project, era) with a fixed seed; within each
    stratum, `audit_rank` is the random draw order. Any prefix (ranks 1..k for every
    stratum) is itself a stratified random sample, so labelling can stop early at a
    balanced point without biasing the sample.
  - The labeller-facing file omits the classifier's prediction/probability so labels
    are blind; the key file keeps them for analysis.

The sample is drawn once and frozen. Do not redraw after seeing results.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261003
PER_STRATUM = 10
ERAS = ['early', 'middle', 'late']
PROJECTS = ['CONFSERVER', 'DM', 'DNN', 'FAB', 'JRASERVER',
            'MESOS', 'MULE', 'NEXUS', 'SERVER', 'TIMOB']

LABELED_DIR = Path('data/tawos_full/labeled')
MANUAL = Path('output/all_manually_labelled.csv')
OUT_DIR = Path('data/calibration_audit')


def main():
    rng = np.random.default_rng(SEED)
    already = set(pd.read_csv(MANUAL, usecols=['issue_key'])['issue_key'])

    frames, frame_sizes = [], []
    for project in PROJECTS:
        df = pd.read_csv(LABELED_DIR / f'{project}_labeled.csv',
                         parse_dates=['creation_date', 'resolution_date'])
        df = df.dropna(subset=['resolution_date'])
        df = df[~df['issue_key'].isin(already)].copy()
        # Rank first so ties in resolution_date can't produce uneven terciles.
        df['era'] = pd.qcut(df['resolution_date'].rank(method='first'), 3, labels=ERAS)

        for era, stratum in df.groupby('era', observed=True):
            idx = rng.permutation(len(stratum))[:PER_STRATUM]
            picked = stratum.iloc[idx].copy()
            picked['audit_rank'] = np.arange(1, len(picked) + 1)
            picked['stratum_size'] = len(stratum)
            picked['era_start'] = stratum['resolution_date'].min().date()
            picked['era_end'] = stratum['resolution_date'].max().date()
            frames.append(picked)
            frame_sizes.append({'project': project, 'era': era, 'stratum_size': len(stratum),
                                'era_start': stratum['resolution_date'].min().date(),
                                'era_end': stratum['resolution_date'].max().date()})

    sample = pd.concat(frames, ignore_index=True)
    sample['sampling_weight'] = sample['stratum_size'] / PER_STRATUM

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key_cols = ['issue_key', 'project', 'era', 'audit_rank', 'issue_type', 'creation_date',
                'resolution_date', 'stratum_size', 'sampling_weight',
                'prob_design_ensemble', 'pred_design']
    sample[key_cols].to_csv(OUT_DIR / 'audit_sample_key.csv', index=False)

    # Blind labelling sheet: shuffled across strata, no classifier output.
    sheet = sample[['issue_key', 'project', 'issue_type', 'summary', 'description', 'audit_rank']]
    sheet = sheet.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sheet['label'] = ''      # 1 = design, 0 = not design (same definition as the manual set)
    sheet['notes'] = ''
    sheet.to_csv(OUT_DIR / 'audit_sample_to_label.csv', index=False)

    pd.DataFrame(frame_sizes).to_csv(OUT_DIR / 'sampling_frame.csv', index=False)
    print(f'{len(sample)} tickets drawn ({PER_STRATUM}/stratum x {len(ERAS)} eras x {len(PROJECTS)} projects)')
    print(sample.groupby(['project', 'era'], observed=True).size().unstack())
    print(f"classifier-flagged share in sample: {sample['pred_design'].mean():.3f}")


if __name__ == '__main__':
    main()
