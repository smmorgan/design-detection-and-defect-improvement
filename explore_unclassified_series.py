#!/usr/bin/env python3
"""
Exploratory follow-up to RQ2_INITIAL_RESULTS.md §6 / §7 item 2: does the
"Unclassified" series (issues with no component, bucketed per project) carry the
bucket cell's H1 signal on its own? Exploratory, outside the pre-registered Holm
family.

1. Describe the no-component issues vs. component-linked ones, per project:
   share of resolved issues, bug share, classifier-flagged design share, and how
   the no-component share changes over each project's history.
2. Sanity check: the bucket panel minus its Unclassified series is identical to
   the drop panel (so the drop/bucket difference is entirely the Unclassified
   series).
3. Fits on the bucket cell's own M2-complete sample (same rows as the reported
   bucket results), raw ratio:
     a. H1 on the Unclassified series alone (one series per project), with the
        same permutation null as the component unit.
     b. H1 on the bucket sample *without* the Unclassified series.
     c. Interaction: does the lag effect differ between Unclassified and real
        component series? LR test of M1 vs. M1 + design_ratio_lag{k}:unclassified.

Usage: .venv/bin/python explore_unclassified_series.py [--n_permutations 2000]
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd

from fit_rq2_component_models import add_lag_features, permutation_test_h1_component
from fit_rq2_models import LAGS, fit_nb_glm, nested_lr_test

PROJECTS = ['CONFSERVER', 'DM', 'DNN', 'FAB', 'JRASERVER',
            'MESOS', 'MULE', 'NEXUS', 'SERVER', 'TIMOB']
RATIO = 'design_ratio_raw'
OUT = 'gcp_results/rq2_unclassified_check.json'

M0 = 'bug_count ~ bug_count_lag1 + quarter_index + C(project_component)'
M1 = M0 + ' + ' + ' + '.join(f'design_ratio_lag{k}' for k in LAGS)
FIT_COLS_M2 = ['bug_count', 'bug_count_lag1', 'quarter_index', 'project_component', 'project',
               'log_exposure', 'gini_design_effort', 'front_loaded_index',
               'quarters_since_changepoint'] + [f'design_ratio_lag{k}' for k in LAGS]


def describe():
    rows = []
    for p in PROJECTS:
        df = pd.read_csv(f'data/tawos_full/labeled/{p}_labeled.csv',
                         usecols=['issue_type', 'components', 'pred_design', 'resolution_date'],
                         parse_dates=['resolution_date'])
        df = df.dropna(subset=['resolution_date'])
        none = df['components'].isna() | (df['components'].astype(str).str.strip() == '')
        era = pd.qcut(df['resolution_date'].rank(method='first'), 3, labels=['early', 'middle', 'late'])
        row = {'project': p, 'n_resolved': len(df),
               'no_component_share': round(none.mean(), 3),
               'bug_share_no_comp': round((df.loc[none, 'issue_type'] == 'Bug').mean(), 3),
               'bug_share_with_comp': round((df.loc[~none, 'issue_type'] == 'Bug').mean(), 3),
               'flagged_design_no_comp': round(df.loc[none, 'pred_design'].mean(), 3),
               'flagged_design_with_comp': round(df.loc[~none, 'pred_design'].mean(), 3)}
        for e in ['early', 'middle', 'late']:
            row[f'no_comp_share_{e}'] = round(none[era == e].mean(), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def h1(data, label, n_perms, panel=None):
    m0, m1 = fit_nb_glm(M0, data), fit_nb_glm(M1, data)
    stat, p, df = nested_lr_test(m0, m1)
    out = {'sample': label, 'n_rows': int(len(data)), 'n_series': int(data['project_component'].nunique()),
           'lr_stat': round(float(stat), 3), 'df': int(df), 'chi2_p': float(p),
           'm1_converged': bool(m1.mle_retvals['converged']),
           'lag_coefs': {k: round(float(v), 4) for k, v in m1.params.filter(like='design_ratio_lag').items()}}
    if n_perms:
        _, perm_p, n_ok = permutation_test_h1_component(panel, data, ratio_col=RATIO, n_perms=n_perms)
        out.update(perm_p=float(perm_p), n_perms_ok=int(n_ok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_permutations', type=int, default=2000)
    args = ap.parse_args()
    warnings.simplefilter('ignore')

    desc = describe()
    print(desc.to_string(index=False))

    bucket = pd.read_csv('data/rq2_component_panel_bucket.csv')
    drop = pd.read_csv('data/rq2_component_panel_drop.csv')
    is_unc = bucket['component'] == 'Unclassified'
    key = ['project_component', 'quarter']
    a = bucket[~is_unc].sort_values(key).reset_index(drop=True)
    b = drop.sort_values(key).reset_index(drop=True)
    num = ['resolved_total', 'resolved_design', 'resolved_nonbug', 'bug_count', 'design_ratio_raw']
    identical = a[key].equals(b[key]) and np.allclose(a[num].fillna(-1), b[num].fillna(-1))
    print(f'\nbucket minus Unclassified == drop panel: {identical}')

    panel = add_lag_features(bucket, ratio_col=RATIO)
    clean = panel.dropna(subset=FIT_COLS_M2).copy()
    clean['unclassified'] = (clean['component'] == 'Unclassified').astype(int)
    unc_n = clean.groupby('project')['unclassified'].sum().to_dict()
    print(f'Unclassified rows in bucket M2 sample, by project: {unc_n}')

    results = {'bucket_minus_unclassified_equals_drop': bool(identical),
               'unclassified_rows_in_sample_by_project': {k: int(v) for k, v in unc_n.items()}}
    results['full_bucket_sample'] = h1(clean, 'bucket, all series', 0)
    results['unclassified_only'] = h1(clean[clean['unclassified'] == 1], 'Unclassified series only',
                                      args.n_permutations, panel[panel['component'] == 'Unclassified'])
    results['without_unclassified'] = h1(clean[clean['unclassified'] == 0], 'bucket minus Unclassified', 0)

    inter = M1 + ' + ' + ' + '.join(f'design_ratio_lag{k}:unclassified' for k in LAGS)
    m1, mi = fit_nb_glm(M1, clean), fit_nb_glm(inter, clean)
    stat, p, df = nested_lr_test(m1, mi)
    results['interaction_lag_x_unclassified'] = {
        'lr_stat': round(float(stat), 3), 'df': int(df), 'chi2_p': float(p),
        'interaction_coefs': {k: round(float(v), 4) for k, v in mi.params.filter(like=':unclassified').items()}}

    results['description'] = desc.to_dict(orient='records')
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != 'description'}, indent=2))


if __name__ == '__main__':
    main()
