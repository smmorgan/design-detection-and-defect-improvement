#!/usr/bin/env python3
"""
MC-SIMEX for the RQ2 H1 design-ratio lag coefficients (RQ2_PREREGISTRATION.md §8).

The design ratio is built from misclassified binary ticket labels, so this uses
MC-SIMEX for misclassified binary variables (Küchenhoff, Mwalili & Lesaffre 2006,
Biometrics 62:85-96) instead of additive-error SIMEX:

  Pi = [[spec, 1-sens], [1-spec, sens]]   (P(observed | true); columns sum to 1)

For each lambda in LAMBDAS, each ticket's *observed* label is re-misclassified
through Pi^lambda (fractional power via eigendecomposition; needs sens+spec > 1),
the design ratio is rebuilt per series-quarter, and M1 is refit on the same rows
as the naive fit. Averaging over B replicates gives the coefficient at error level
1+lambda; a quadratic (and, as a check, linear) fit in lambda is extrapolated to
lambda = -1, i.e. no misclassification.

Misclassification inputs come from the random calibration audit
(RQ2_INITIAL_RESULTS.md §8):
  pooled       one population-weighted sens/spec for all projects
  project_spec per-project specificity from the audit (projects' false-positive
               rates differ, §8b) with pooled sensitivity (too few audit positives
               per project to estimate it separately)

Only M1 (H1's coefficients) is extrapolated. M2 also needs the change-point
feature, which would have to be re-detected per replicate. H2 was null throughout
and the M2 lag coefficients track M1's, so this is left out and documented.

Usage: .venv/bin/python simex_rq2.py --unit {primary,drop,bucket} --pi {pooled,project_spec} [--B 50]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from build_rq2_component_panel import explode_components
from fit_rq2_models import LAGS, fit_nb_glm

PROJECTS = ['CONFSERVER', 'DM', 'DNN', 'FAB', 'JRASERVER',
            'MESOS', 'MULE', 'NEXUS', 'SERVER', 'TIMOB']
LAMBDAS = [0.5, 1.0, 1.5, 2.0]
SEED = 20261003
OUT_DIR = Path('gcp_results/rq2_simex')

UNITS = {
    'primary': {'panel': 'data/rq2_panel.csv', 'series': 'project', 'missing': None},
    'drop': {'panel': 'data/rq2_component_panel_drop.csv', 'series': 'project_component', 'missing': 'drop'},
    'bucket': {'panel': 'data/rq2_component_panel_bucket.csv', 'series': 'project_component', 'missing': 'bucket'},
}
EXTRA_FIT_COLS = ['gini_design_effort', 'front_loaded_index', 'quarters_since_changepoint']


def audit_inputs():
    sys.argv = sys.argv[:1]
    import rq2_calibration as rc
    calib = pd.read_csv(rc.CALIB_OUT)
    audit = calib[calib['source'] == 'random_audit']
    sens, spec = rc.sens_spec(audit)
    per_project_spec = {}
    for p, g in audit.groupby('project'):
        neg = g[g['label'] == 0]
        per_project_spec[p] = float((neg['pred_design'] == 0).mean())
    return float(sens), float(spec), per_project_spec


def pi_power(sens, spec, lam):
    pi = np.array([[spec, 1 - sens], [1 - spec, sens]])
    vals, vecs = np.linalg.eig(pi)
    if np.min(vals.real) <= 0:
        raise ValueError(f'sens+spec-1 must be > 0 for fractional powers (sens={sens}, spec={spec})')
    return (vecs @ np.diag(vals.real ** lam) @ np.linalg.inv(vecs)).real


def load_tickets(unit_cfg, panel):
    """Resolved tickets mapped to their panel row(s). Returns (labels, ticket_project,
    row_index_per_link, ticket_index_per_link)."""
    series_col = unit_cfg['series']
    row_of = {(s, q): i for i, (s, q) in enumerate(zip(panel[series_col], panel['quarter']))}
    labels, projects, link_rows, link_tickets = [], [], [], []
    offset = 0
    for p in PROJECTS:
        df = pd.read_csv(f'data/tawos_full/labeled/{p}_labeled.csv',
                         usecols=['issue_key', 'components', 'pred_design', 'resolution_date'],
                         parse_dates=['resolution_date'])
        df = df.dropna(subset=['resolution_date']).reset_index(drop=True)
        df['tid'] = np.arange(len(df)) + offset
        df['quarter'] = df['resolution_date'].dt.to_period('Q').astype(str)
        if unit_cfg['missing'] is None:
            df['series'] = p
            links = df
        else:
            links = explode_components(df, unit_cfg['missing'])
            links['series'] = p + '::' + links['component']
        rows = [row_of.get(k, -1) for k in zip(links['series'], links['quarter'])]
        links = links.assign(row=rows)
        links = links[links['row'] >= 0]
        labels.append(df['pred_design'].to_numpy(dtype=int))
        projects.append(np.full(len(df), p))
        link_rows.append(links['row'].to_numpy())
        link_tickets.append(links['tid'].to_numpy())
        offset += len(df)
    return (np.concatenate(labels), np.concatenate(projects),
            np.concatenate(link_rows), np.concatenate(link_tickets))


def add_lags(panel, series_col, ratio):
    panel = panel.copy()
    panel['_ratio'] = ratio
    g = panel.groupby(series_col)
    for k in LAGS:
        panel[f'design_ratio_lag{k}'] = g['_ratio'].shift(k)
        panel[f'bug_count_lag{k}'] = g['bug_count'].shift(k)
    panel['log_exposure'] = np.log(panel['resolved_nonbug'] + 1)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', choices=list(UNITS), required=True)
    ap.add_argument('--pi', choices=['pooled', 'project_spec'], required=True)
    ap.add_argument('--B', type=int, default=50)
    args = ap.parse_args()
    warnings.simplefilter('ignore')

    cfg = UNITS[args.unit]
    series_col = cfg['series']
    fe = f'C({series_col})'
    m1 = (f'bug_count ~ bug_count_lag1 + quarter_index + {fe} + ' +
          ' + '.join(f'design_ratio_lag{k}' for k in LAGS))
    lag_names = [f'design_ratio_lag{k}' for k in LAGS]

    panel = pd.read_csv(cfg['panel']).sort_values([series_col, 'quarter_index']).reset_index(drop=True)
    labels, tproj, link_rows, link_tids = load_tickets(cfg, panel)
    total = np.bincount(link_rows, minlength=len(panel)).astype(float)

    def ratio_from(lbl):
        design = np.bincount(link_rows, weights=lbl[link_tids], minlength=len(panel))
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(total > 0, design / total, np.nan)

    # Validation: ticket-level rebuild must reproduce the panel exactly.
    naive_ratio = ratio_from(labels)
    assert np.allclose(total, panel['resolved_total'].to_numpy()), 'resolved_total mismatch'
    assert np.allclose(np.nan_to_num(naive_ratio, nan=-1),
                       np.nan_to_num(panel['design_ratio_raw'].to_numpy(), nan=-1)), 'ratio mismatch'

    fit_cols = (['bug_count', 'bug_count_lag1', 'quarter_index', series_col, 'project', 'log_exposure']
                + EXTRA_FIT_COLS + lag_names)
    base = add_lags(panel, series_col, naive_ratio)
    keep = base.dropna(subset=fit_cols).index  # same rows as the reported M2-complete fits

    def fit_coefs(ratio):
        d = add_lags(panel, series_col, ratio).loc[keep]
        res = fit_nb_glm(m1, d)
        return res.params[lag_names].to_numpy(), bool(res.mle_retvals['converged']), res

    naive, conv, naive_res = fit_coefs(naive_ratio)
    naive_se = naive_res.bse[lag_names].to_numpy()

    sens, spec, proj_spec = audit_inputs()
    spec_of = ({p: spec for p in PROJECTS} if args.pi == 'pooled' else proj_spec)

    rng = np.random.default_rng(SEED)
    by_lambda = {}
    n_nonconv = 0
    for lam in LAMBDAS:
        # P(new obs = 1 | current obs) per ticket, from that ticket's project's Pi^lambda.
        p_one = np.empty(len(labels))
        for p in PROJECTS:
            P = pi_power(sens, spec_of[p], lam)
            m = tproj == p
            p_one[m] = np.where(labels[m] == 1, P[1, 1], P[1, 0])
        reps = []
        for _ in range(args.B):
            new = (rng.random(len(labels)) < p_one).astype(float)
            coefs, ok, _ = fit_coefs(ratio_from(new))
            n_nonconv += (not ok)
            reps.append(coefs)
        by_lambda[lam] = np.mean(reps, axis=0)
        print(f'lambda={lam}: ' + ', '.join(f'{c:.4f}' for c in by_lambda[lam]), flush=True)

    lam_grid = np.array([0.0] + LAMBDAS)
    est = np.vstack([naive] + [by_lambda[l] for l in LAMBDAS])
    quad = [float(np.polyval(np.polyfit(lam_grid, est[:, j], 2), -1)) for j in range(len(LAGS))]
    lin = [float(np.polyval(np.polyfit(lam_grid, est[:, j], 1), -1)) for j in range(len(LAGS))]

    out = {
        'unit': args.unit, 'pi': args.pi, 'B': args.B, 'lambdas': LAMBDAS,
        'n_rows': int(len(keep)), 'naive_m1_converged': conv, 'n_replicates_nonconverged': int(n_nonconv),
        'sens_pooled': round(sens, 4), 'spec_pooled': round(spec, 4),
        'spec_by_project': {p: round(v, 4) for p, v in spec_of.items()},
        'coef_by_lambda': {str(l): dict(zip(lag_names, map(float, est[i])))
                           for i, l in enumerate(lam_grid)},
        'naive': dict(zip(lag_names, map(float, naive))),
        'naive_se_cluster': dict(zip(lag_names, map(float, naive_se))),
        'simex_quadratic': dict(zip(lag_names, quad)),
        'simex_linear': dict(zip(lag_names, lin)),
        'note': 'M1 only; no SIMEX standard errors (B and the per-replicate refits make '
                'jackknife variance estimation expensive) -- read as a point-estimate '
                'stability check, per pre-registration §8.',
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f'simex_{args.unit}_{args.pi}.json'
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ['naive', 'simex_quadratic', 'simex_linear',
                                          'n_replicates_nonconverged']}, indent=2))
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
