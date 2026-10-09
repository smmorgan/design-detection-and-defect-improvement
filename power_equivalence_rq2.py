#!/usr/bin/env python3
"""
Sensitivity (minimum detectable effect) and TOST equivalence analysis for RQ2,
per RQ2_PREREGISTRATION.md §9. Raw design ratio, one cell per run.

§9 asked for this "before fitting M1". It wasn't done then, so this is run
after the results are known (recorded as a deviation in §11). It therefore
reports a *sensitivity* analysis -- the effect size the design could detect,
given the panel's actual n and dependence -- and not "observed power", which
is a function of the p-value and says nothing new.

Why not the model's own SEs: with 10 clusters and autocorrelated series, the
chi-square tests were ~10x too liberal (RQ2_INITIAL_RESULTS.md §2e, §6). So
every standard error here is the largest of:
  - the cluster-robust (by project) SE from the fit;
  - the SD of the coefficient under the pre-registered within-series shuffle
    null (fit_rq2_models.permutation_null, scheme='shuffle');
  - the SD under a within-series circular-shift null (scheme='circular'),
    which keeps the design ratio's own autocorrelation and is usually wider.

H1 targets (coefficients on the raw ratio, per unit of ratio; x0.1 = per
+10 percentage points of flagged design share):
  lag1      design_ratio_lag1, the short-lag effect H1 is about.
  long_run  sum of the 4 lag coefficients: a sustained +1 in the ratio.
For each: 1-df MDE at power 0.8 for alpha 0.05 and 0.05/12 (Holm's
strictest step); MDE for the actual joint 4-df test, from a non-central
chi-square with the null covariance of the 4 lag coefficients (an
approximation of the permutation-calibrated LR test: the Wald p for the
observed fit under that covariance is reported next to the permutation p as
a check); and a TOST at alpha 0.05: the 90% CI and the smallest symmetric
bound it rejects. The pre-registration fixed no smallest effect of interest,
so TOST p-values are reported at two reference bounds, +/-5% and +/-10% bugs
per +10pp, alongside the bound-free "smallest bound rejected".

H2 targets: gini_design_effort and quarters_since_changepoint in M2, per 1 SD
of the feature (front_loaded_index is collinear with the fixed effects and
adds nothing). Null: both features circularly shifted together within each
series over the fitted rows, M2 refit. That null also gives an H2 LR
p-value calibrated to the panel's dependence, which the confirmatory
chi-square test was not (exploratory, outside the Holm family).

Usage: .venv/bin/python power_equivalence_rq2.py --unit {primary,drop,bucket}
       [--n_perms 1000] [--n_perms_h2 500]
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

import fit_rq2_component_models as fc
import fit_rq2_models as fm
from fit_rq2_models import LAGS, circular_offset, fit_nb_glm, permutation_null

RATIO = 'design_ratio_raw'
SEED = 20261004
POWER = 0.8
ALPHAS = {'0.05': 0.05, 'holm_strictest_0.05/12': 0.05 / 12}
REF_PCT = [5, 10]  # reference equivalence bounds, % change in bugs
H2_FEATURES = ['gini_design_effort', 'quarters_since_changepoint']
OUT_DIR = Path('gcp_results/rq2_power_equivalence')

UNITS = {
    'primary': ('data/rq2_panel.csv', 'project', fm.add_lag_features),
    'drop': ('data/rq2_component_panel_drop.csv', 'project_component', fc.add_lag_features),
    'bucket': ('data/rq2_component_panel_bucket.csv', 'project_component', fc.add_lag_features),
}


def pct(b, scale):
    """Coefficient -> % change in expected bugs for a `scale`-unit increase."""
    return 100 * (np.exp(b * scale) - 1)


def coef_for_pct(p, scale):
    return np.log(1 + p / 100) / scale


def tost(est, se, scale):
    lo, hi = est - 1.645 * se, est + 1.645 * se
    bound = max(abs(lo), abs(hi))
    out = {'ci90': [lo, hi], 'smallest_bound_rejected': bound,
           'smallest_bound_rejected_pct': [pct(-bound, scale), pct(bound, scale)]}
    for p in REF_PCT:
        d = coef_for_pct(p, scale)
        p_tost = max(stats.norm.sf((est + d) / se), stats.norm.cdf((est - d) / se))
        out[f'tost_p_at_{p}pct'] = float(p_tost)
    return out


def mde_1df(se):
    return {k: float((stats.norm.ppf(1 - a / 2) + stats.norm.ppf(POWER)) * se) for k, a in ALPHAS.items()}


def mde_joint(direction, cov, df):
    """Smallest b such that a Wald test of all lag coefficients (df) with
    covariance `cov` has power POWER against beta = b * direction."""
    q = float(direction @ np.linalg.solve(cov, direction))
    out = {}
    for k, a in ALPHAS.items():
        crit = stats.chi2.ppf(1 - a, df)
        nc = brentq(lambda d: stats.ncx2.sf(crit, df, d) - POWER, 1e-6, 1e4)
        out[k] = float(np.sqrt(nc / q))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', choices=list(UNITS), required=True)
    ap.add_argument('--n_perms', type=int, default=1000)
    ap.add_argument('--n_perms_h2', type=int, default=500)
    args = ap.parse_args()
    warnings.simplefilter('ignore')

    path, series, add_lags = UNITS[args.unit]
    fe = f'C({series})'
    lag_cols = [f'design_ratio_lag{k}' for k in LAGS]
    m0f = f'bug_count ~ bug_count_lag1 + quarter_index + {fe}'
    m1f = m0f + ' + ' + ' + '.join(lag_cols)
    m2f = m1f + ' + gini_design_effort + front_loaded_index + quarters_since_changepoint'

    panel = add_lags(pd.read_csv(path), ratio_col=RATIO)
    fit_cols = (['bug_count', 'bug_count_lag1', 'quarter_index', series, 'project', 'log_exposure',
                 'front_loaded_index'] + H2_FEATURES + lag_cols)
    clean = panel.dropna(subset=fit_cols)

    # ---- H1 ----
    nulls = {}
    for i, scheme in enumerate(['shuffle', 'circular']):
        obs, p, n_ok, draws, m0, m1 = permutation_null(
            panel, clean, series, RATIO, m0f, m1f, args.n_perms, SEED + i,
            scheme=scheme, return_draws=True)
        D = pd.DataFrame(draws)
        nulls[scheme] = {'perm_p': float(p), 'n_ok': int(n_ok),
                         'n_nonconverged': int((~D['converged']).sum()),
                         'lr_q95': float(D['lr'].quantile(0.95)),
                         'cov': np.cov(D[lag_cols].to_numpy(), rowvar=False),
                         'mean': D[lag_cols].mean().to_numpy()}
        print(f'{scheme}: observed LR={obs:.3f}  perm p={p:.4f}  ({n_ok} ok)  '
              f'null LR 95th pct={nulls[scheme]["lr_q95"]:.2f} (chi2_4: {stats.chi2.ppf(.95, 4):.2f})',
              flush=True)

    beta = m1.params[lag_cols].to_numpy()
    v_cl = m1.cov_params().loc[lag_cols, lag_cols].to_numpy()
    targets = {'lag1': np.eye(len(LAGS))[0], 'long_run': np.ones(len(LAGS))}
    joint_dirs = {'lag1': np.eye(len(LAGS))[0], 'long_run': np.ones(len(LAGS)) / len(LAGS)}
    h1 = {'observed_lr': float(obs), 'n_rows': int(len(clean)), 'n_series': int(clean[series].nunique()),
          'beta': dict(zip(lag_cols, map(float, beta)))}
    for s, nd in nulls.items():
        w = float(beta @ np.linalg.solve(nd['cov'], beta))
        h1[f'null_{s}'] = {'perm_p': nd['perm_p'], 'n_ok': nd['n_ok'], 'n_nonconverged': nd['n_nonconverged'],
                           'lr_q95': nd['lr_q95'], 'lr_inflation_vs_chi2': nd['lr_q95'] / stats.chi2.ppf(.95, 4),
                           'wald_p_observed': float(stats.chi2.sf(w, len(LAGS))),
                           'null_mean_beta': list(map(float, nd['mean']))}
    for name, w in targets.items():
        est = float(w @ beta)
        ses = {'cluster': float(np.sqrt(w @ v_cl @ w)),
               **{s: float(np.sqrt(w @ nd['cov'] @ w)) for s, nd in nulls.items()}}
        se = max(ses.values())
        joint = {s: mde_joint(joint_dirs[name], nd['cov'], len(LAGS)) for s, nd in nulls.items()}
        joint_max = {k: max(j[k] for j in joint.values()) for k in ALPHAS}
        h1[name] = {'estimate': est, 'estimate_pct_per_10pp': pct(est, 0.1), 'se': ses, 'se_used': se,
                    'mde_1df': mde_1df(se),
                    'mde_1df_pct_per_10pp': {k: pct(-v, 0.1) for k, v in mde_1df(se).items()},
                    'mde_joint_4df': joint_max,
                    'mde_joint_4df_pct_per_10pp': {k: pct(-v, 0.1) for k, v in joint_max.items()},
                    'tost': tost(est, se, 0.1)}

    # ---- H2 ----
    m2 = fit_nb_glm(m2f, clean)
    obs2 = 2 * (m2.llf - m1.llf)
    rng = np.random.default_rng(SEED + 2)
    pos = [np.flatnonzero(clean[series].to_numpy() == s) for s in clean[series].unique()]
    feats = clean[H2_FEATURES].to_numpy()
    draws2, exceed = [], 0
    for _ in range(args.n_perms_h2):
        new = feats.copy()
        for idx in pos:
            if len(idx) > 1:
                new[idx] = np.roll(feats[idx], circular_offset(rng, len(idx)), axis=0)
        d = clean.copy()
        d[H2_FEATURES] = new
        try:
            r = fit_nb_glm(m2f, d)
        except Exception:
            continue
        stat = 2 * (r.llf - m1.llf)
        exceed += stat >= obs2
        draws2.append([r.params[f] for f in H2_FEATURES])
    D2 = np.array(draws2)
    h2 = {'observed_lr': float(obs2), 'n_ok': len(draws2),
          'circular_perm_p': float((exceed + 1) / (len(draws2) + 1))}
    print(f'H2: observed LR={obs2:.3f}  circular-shift p={h2["circular_perm_p"]:.4f}', flush=True)
    for j, f in enumerate(H2_FEATURES):
        sd = float(clean[f].std())
        est = float(m2.params[f]) * sd
        ses = {'cluster': float(m2.bse[f]) * sd, 'circular': float(D2[:, j].std(ddof=1)) * sd}
        se = max(ses.values())
        h2[f] = {'feature_sd': sd, 'estimate_per_sd': est, 'estimate_pct_per_sd': pct(est, 1),
                 'se_per_sd': ses, 'se_used': se, 'mde_1df_per_sd': mde_1df(se),
                 'mde_1df_pct_per_sd': {k: pct(-v, 1) for k, v in mde_1df(se).items()},
                 'tost': tost(est, se, 1)}

    out = {'unit': args.unit, 'ratio': RATIO, 'power': POWER, 'n_perms': args.n_perms,
           'n_perms_h2': args.n_perms_h2, 'h1': h1, 'h2': h2}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path_out = OUT_DIR / f'{args.unit}.json'
    path_out.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
    print(f'wrote {path_out}')


if __name__ == '__main__':
    main()
