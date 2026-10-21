#!/usr/bin/env python3
"""
Robustness checks for the re-specified RQ2 H1/H2 (RQ2_PREREGISTRATION.md §11,
results in RQ2_INITIAL_RESULTS.md §11). Exploratory: none of these is in the
pre-registered Holm family. Each check changes one thing relative to the
re-specified baseline (design_ratio_nonbug, lags 1-4, M1-complete sample):

H1 checks (M0 vs M1, the design-lag block):
  baseline_m1_sample   design_ratio_nonbug on every row with complete M1 terms.
  baseline_m2_sample   same, restricted to rows with complete M2 terms (what
                       the grid scripts and power_equivalence_rq2.py report).
  outcome_valid        outcome = bugs not resolved as non-defects
                       (build_rq2_panel.NON_DEFECT_RESOLUTIONS), unresolved kept.
  outcome_fixed        outcome = bugs resolved with a fix-type resolution only.
  drop_last_quarter    excludes each project's final, partial quarter.
  exposure_covariate   log(non-bug resolved) as an estimated regressor instead
                       of an offset fixed at 1.
  project_trends       one linear trend per project instead of a common one.
  combined             outcome_valid + drop_last_quarter together.
  decomposed_counts    no ratio: lagged log(1+count) of design non-bug, other
                       non-bug and bug tickets resolved enter separately; the
                       test is the design-count block, net of the other two.
  original_raw_m1      the original design_ratio_raw on the M1-complete sample.

H2 checks (M1 vs M2), each with its own M1:
  raw_cp_corrected     original raw-ratio cell: change point from the clipped
                       Rogan-Gladen ratio (as reported).
  raw_cp_raw           change point from the raw ratio itself.
  nonbug_cp_nonbug     re-specified ratio and its own change point.
  nonbug_cp_causal     change point re-detected at each quarter from data up
                       to that quarter only (the nearest-change-point feature
                       looks ahead: a break detected from later quarters
                       changes earlier rows' values).
  nonbug_cp_trends     nonbug_cp_nonbug with per-project linear trends.

Inference: the chi-square LR p-value, plus a within-series circular-shift null
(fit_rq2_models.circular_offset; keeps each series' autocorrelation, which
RQ2_INITIAL_RESULTS.md §10 found makes the shuffle null anti-conservative).
For H1 the tested source column is shifted and its lags rebuilt over the full
panel, then refit on the reported rows (same construction as
fit_rq2_models.permutation_null). For H2 the change-point and Gini features
are shifted together over the fitted rows (as in power_equivalence_rq2.py).

Usage: .venv/bin/python rq2_robustness_checks.py --unit {primary,drop,bucket}
       [--n_perms 1000] [--workers 8]
"""
import argparse
import json
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import stats

import statsmodels.formula.api as smf

from fit_rq2_models import LAGS, circular_offset

UNITS = {
    'primary': ('data/rq2_panel.csv', 'project'),
    'drop': ('data/rq2_component_panel_drop.csv', 'project_component'),
    'bucket': ('data/rq2_component_panel_bucket.csv', 'project_component'),
}
OUT_DIR = Path('gcp_results/rq2_robustness')
SEED = 20261020
LAGGED = ['design_ratio_nonbug', 'design_ratio_raw', 'log_design_nonbug', 'log_other_nonbug',
          'log_bug_resolved', 'bug_count', 'bug_count_valid', 'bug_count_fixed']


def fit_nb_glm(formula, data, offset_col='log_exposure', start_from=None, start_overrides=None):
    """fit_rq2_models.fit_nb_glm (NB2 MLE, cluster-robust by project) with a
    convergence fallback some checks here need (e.g. the exposure covariate
    without an offset does not converge in 500 BFGS steps): BFGS for up to
    2000 steps, then Newton from that point if BFGS has not converged.
    start_from: a fitted nested model whose estimates seed the shared terms
    (new terms start at 0), so the larger model starts at the smaller one's
    optimum and its LR against it cannot come out negative from an early stop."""
    model = smf.negativebinomial(formula=formula, data=data, offset=data[offset_col].values)
    kw = dict(disp=0, cov_type='cluster', cov_kwds={'groups': data['project']})
    start = None
    if start_from is not None:
        names = list(model.exog_names)  # NB2's exog_names already ends with 'alpha'
        start = np.array([(start_overrides or {}).get(n, start_from.params.get(n, 0.0)) for n in names])
    res = model.fit(method='bfgs', maxiter=2000, start_params=start, **kw)
    if not res.mle_retvals.get('converged', True):
        try:
            res = model.fit(method='newton', maxiter=200, start_params=res.params.to_numpy(), **kw)
        except (np.linalg.LinAlgError, ValueError):
            pass  # rank-deficient Hessian; keep the BFGS result (its converged flag stays False)
    return res


def lag_terms(src):
    return ' + '.join(f'{src}_lag{k}' for k in LAGS)


def causal_changepoint(x):
    """Quarters since the most recent change point detected using only
    quarters 0..t, with the same PELT/L2/variance*log(n) settings as
    build_rq2_panel.changepoint_distances. Quarters before any detected break
    count from the series start; prefixes shorter than 8 are NaN, matching
    the original feature's minimum length."""
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
    out = np.full(len(x), np.nan)
    for t in range(7, len(x)):
        pre = x[:t + 1].reshape(-1, 1)
        var = float(np.var(pre))
        bps = []
        if var > 0:
            try:
                bps = [b for b in rpt.Pelt(model='l2', min_size=2).fit(pre).predict(pen=var * np.log(len(pre)))
                       if b < len(pre)]
            except Exception:
                bps = []
        out[t] = t - (max(bps) if bps else 0)
    return out


def load(unit):
    path, series = UNITS[unit]
    p = pd.read_csv(path).sort_values([series, 'quarter_index']).reset_index(drop=True)
    p['log_exposure'] = np.log(p['resolved_nonbug'] + 1)
    p['zero_offset'] = 0.0
    p['log_design_nonbug'] = np.log1p(p['resolved_design_nonbug'])
    p['log_other_nonbug'] = np.log1p(p['resolved_nonbug'] - p['resolved_design_nonbug'])
    p['log_bug_resolved'] = np.log1p(p['resolved_bug'])
    g = p.groupby(series)
    for c in LAGGED:
        for k in LAGS:
            p[f'{c}_lag{k}'] = g[c].shift(k)
    p['quarters_since_changepoint_causal'] = np.concatenate(
        [causal_changepoint(s.to_numpy()) for _, s in g['design_ratio_nonbug']])
    return p, series


def h1_specs(fe):
    base = dict(y='bug_count', src='design_ratio_nonbug', extra='', offset='log_exposure',
                trend='quarter_index', sample='m1', drop_last=False)
    s = {
        'baseline_m1_sample': {},
        'baseline_m2_sample': dict(sample='m2'),
        'outcome_valid': dict(y='bug_count_valid'),
        'outcome_fixed': dict(y='bug_count_fixed'),
        'drop_last_quarter': dict(drop_last=True),
        'exposure_covariate': dict(offset='zero_offset', extra='log_exposure'),
        'project_trends': dict(trend='C(project):quarter_index'),
        'combined': dict(y='bug_count_valid', drop_last=True),
        'decomposed_counts': dict(src='log_design_nonbug',
                                  extra=lag_terms('log_other_nonbug') + ' + ' + lag_terms('log_bug_resolved')),
        'original_raw_m1': dict(src='design_ratio_raw'),
    }
    out = {}
    for name, over in s.items():
        c = {**base, **over}
        m0 = f"{c['y']} ~ {c['y']}_lag1 + {c['trend']} + {fe}" + (f" + {c['extra']}" if c['extra'] else '')
        c.update(m0=m0, m1=m0 + ' + ' + lag_terms(c['src']))
        out[name] = c
    return out


H2_SPECS = {
    'raw_cp_corrected': ('design_ratio_raw', 'quarters_since_changepoint', 'quarter_index'),
    'raw_cp_raw': ('design_ratio_raw', 'quarters_since_changepoint_raw', 'quarter_index'),
    'nonbug_cp_nonbug': ('design_ratio_nonbug', 'quarters_since_changepoint_nonbug', 'quarter_index'),
    'nonbug_cp_causal': ('design_ratio_nonbug', 'quarters_since_changepoint_causal', 'quarter_index'),
    'nonbug_cp_trends': ('design_ratio_nonbug', 'quarters_since_changepoint_nonbug', 'C(project):quarter_index'),
}


def fitted_rows(p, formula_cols, drop_last=False):
    clean = p.dropna(subset=formula_cols)
    return clean[~clean['is_last_quarter']] if drop_last else clean


def lag_cols(src):
    return [f'{src}_lag{k}' for k in LAGS]


def run_h1(args):
    name, c, unit, n_perms = args
    p, series = load(unit)
    cols = ['bug_count_lag1', f"{c['y']}_lag1", c['y'], 'log_exposure'] + lag_cols(c['src'])
    for x in ('log_other_nonbug', 'log_bug_resolved'):
        if x in c['extra']:
            cols += lag_cols(x)
    if c['sample'] == 'm2':
        cols += ['gini_design_effort', 'front_loaded_index', 'quarters_since_changepoint_nonbug']
    clean = fitted_rows(p, cols, c['drop_last'])
    if c['offset'] == 'zero_offset':
        # Start the estimated-exposure model at the offset model's optimum
        # (log_exposure coefficient = 1, i.e. the same fitted values); from
        # the default start it diverges.
        m0_offset = fit_nb_glm(c['m0'].replace(' + log_exposure', ''), clean)
        m0 = fit_nb_glm(c['m0'], clean, offset_col='zero_offset', start_from=m0_offset,
                        start_overrides={'log_exposure': 1.0})
    else:
        m0 = fit_nb_glm(c['m0'], clean, offset_col=c['offset'])
    m1 = fit_nb_glm(c['m1'], clean, offset_col=c['offset'], start_from=m0)
    df = np.linalg.matrix_rank(m1.model.exog) - np.linalg.matrix_rank(m0.model.exog)
    lr = 2 * (m1.llf - m0.llf)
    beta = m1.params[lag_cols(c['src'])]

    rng = np.random.default_rng(SEED)
    src = c['src']
    has = p[src].notna().to_numpy()
    ser = p[series].to_numpy()
    groups = [np.flatnonzero((ser == s) & has) for s in pd.unique(ser)]
    base = p[src].to_numpy(copy=True)
    exceed, n_ok, n_nonconv = 0, 0, 0
    for _ in range(n_perms):
        new = base.copy()
        for idx in groups:
            if len(idx) > 1:
                new[idx] = np.roll(base[idx], circular_offset(rng, len(idx)))
        q = p.copy()
        q[src] = new
        for k in LAGS:
            q[f'{src}_lag{k}'] = q.groupby(series)[src].shift(k)
        q = q.loc[clean.index]
        try:
            r = fit_nb_glm(c['m1'], q, offset_col=c['offset'], start_from=m0)
        except Exception:
            continue
        n_ok += 1
        n_nonconv += not r.mle_retvals.get('converged', True)
        exceed += 2 * (r.llf - m0.llf) >= lr
    return name, {
        'test': 'H1', 'outcome': c['y'], 'predictor': src, 'n_rows': int(len(clean)),
        'n_series': int(clean[series].nunique()), 'n_projects': int(clean['project'].nunique()),
        'lr': float(lr), 'df': int(df), 'chi2_p': float(stats.chi2.sf(lr, df)),
        'circular_p': float((exceed + 1) / (n_ok + 1)), 'n_perms_ok': n_ok, 'n_perms_nonconverged': n_nonconv,
        'converged': bool(m0.mle_retvals.get('converged', True) and m1.mle_retvals.get('converged', True)),
        'beta': {k: float(v) for k, v in beta.items()}, 'beta_sum': float(beta.sum()),
        'cluster_se_lag1': float(m1.bse[f'{src}_lag1']),
        'm0': c['m0'], 'm1': c['m1'], 'offset': c['offset'], 'drop_last_quarter': c['drop_last'],
    }


def run_h2(args):
    name, (ratio, cp, trend), unit, n_perms = args
    p, series = load(unit)
    fe = f'C({series})'
    m1f = f'bug_count ~ bug_count_lag1 + {trend} + {fe} + ' + lag_terms(ratio)
    # front_loaded_index is left out: it is a per-project constant, exactly
    # collinear with the fixed effects (0 df, RQ2_PREREGISTRATION.md §9b), and
    # its redundant column breaks the Newton fallback's Hessian inversion.
    m2f = m1f + f' + gini_design_effort + {cp}'
    clean = fitted_rows(p, ['bug_count_lag1', 'log_exposure', 'gini_design_effort', 'front_loaded_index', cp]
                        + lag_cols(ratio))
    m1 = fit_nb_glm(m1f, clean)
    m2 = fit_nb_glm(m2f, clean, start_from=m1)
    df = np.linalg.matrix_rank(m2.model.exog) - np.linalg.matrix_rank(m1.model.exog)
    lr = 2 * (m2.llf - m1.llf)
    feats = ['gini_design_effort', cp]
    rng = np.random.default_rng(SEED + 1)
    pos = [np.flatnonzero(clean[series].to_numpy() == s) for s in clean[series].unique()]
    X = clean[feats].to_numpy()
    exceed, n_ok = 0, 0
    for _ in range(n_perms):
        new = X.copy()
        for idx in pos:
            if len(idx) > 1:
                new[idx] = np.roll(X[idx], circular_offset(rng, len(idx)), axis=0)
        d = clean.copy()
        d[feats] = new
        try:
            r = fit_nb_glm(m2f, d, start_from=m1)
        except Exception:
            continue
        n_ok += 1
        exceed += 2 * (r.llf - m1.llf) >= lr
    sd = float(clean[cp].std())
    return name, {
        'test': 'H2', 'ratio': ratio, 'changepoint_col': cp, 'trend': trend, 'n_rows': int(len(clean)),
        'n_series': int(clean[series].nunique()), 'n_projects': int(clean['project'].nunique()),
        'lr': float(lr), 'df': int(df), 'chi2_p': float(stats.chi2.sf(lr, df)),
        'circular_p': float((exceed + 1) / (n_ok + 1)), 'n_perms_ok': n_ok,
        'cp_coef_per_sd': float(m2.params[cp]) * sd,
        'cp_pct_per_sd': float(100 * (np.exp(m2.params[cp] * sd) - 1)),
        'gini_coef': float(m2.params['gini_design_effort']),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', choices=list(UNITS), required=True)
    ap.add_argument('--n_perms', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--only', nargs='*', help='run just these check names')
    args = ap.parse_args()
    warnings.simplefilter('ignore')
    os.environ.setdefault('OMP_NUM_THREADS', '1')

    fe = f'C({UNITS[args.unit][1]})'
    jobs = [(run_h1, (n, c, args.unit, args.n_perms)) for n, c in h1_specs(fe).items()]
    jobs += [(run_h2, (n, s, args.unit, args.n_perms)) for n, s in H2_SPECS.items()]
    if args.only:
        jobs = [j for j in jobs if j[1][0] in args.only]
    with Pool(args.workers) as pool:
        results = dict(pool.starmap(_call, jobs))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path_out = OUT_DIR / f'{args.unit}.json'
    if args.only and path_out.exists():
        results = {**json.loads(path_out.read_text()), **results}
    path_out.write_text(json.dumps({'unit': args.unit, 'n_perms': args.n_perms, 'checks': results}, indent=2))

    print(f'{"check":22s} {"rows":>6s} {"proj":>4s} {"LR":>7s} {"chi2 p":>8s} {"circ p":>7s}  coefficients')
    for n, r in results.items():
        if r['test'] == 'error':
            print(f'{n:22s} ERROR {r["error"]}')
            continue
        if r['test'] == 'H1':
            b = list(r['beta'].values())
            coef = 'lag1-4 ' + ' '.join(f'{x:+.2f}' for x in b) + f'  sum {r["beta_sum"]:+.2f}'
        else:
            coef = f'change point {r["cp_pct_per_sd"]:+.1f}% bugs per SD'
        print(f'{n:22s} {r["n_rows"]:6d} {r["n_projects"]:4d} {r["lr"]:7.2f} {r["chi2_p"]:8.4f} '
              f'{r["circular_p"]:7.4f}  {coef}')
    print(f'wrote {path_out}')


def _call(fn, a):
    try:
        return fn(a)
    except Exception as e:
        return a[0], {'test': 'error', 'error': f'{type(e).__name__}: {e}'}


if __name__ == '__main__':
    main()
