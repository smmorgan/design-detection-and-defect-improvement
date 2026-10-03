#!/usr/bin/env python3
"""
Model fitting for the project x component x quarter secondary/robustness unit
(RQ2_PREREGISTRATION.md Sec 5.1, Sec 5.4). Mirrors fit_rq2_models.py's M0/M1/M2
+ H3 structure and reuses its fit_nb_glm/nested_lr_test/blocked_time_cv_mae/
fit_h3_reverse helpers directly -- the only structural difference is the fixed
effect: C(project_component) (one dummy per (project, component) pair, per
build_rq2_component_panel.py decision 5) instead of C(project), while
cluster-robust standard errors are still clustered BY PROJECT (10 clusters),
matching the pre-registered clustering scheme (Sec 7) rather than clustering
by the finer project_component grouping.

Known scope-specific caveat worth flagging up front: this unit has ~450
(project, component) fixed-effect levels fit against ~14-17k rows with only
10 clusters for inference. This is the opposite data-shape problem from the
primary unit (there, few clusters/few FE levels risked exact or near
collinearity with a couple of controls; here, many FE levels each with a
modest number of quarters risks incidental-parameter bias in the NB MLE,
particularly for the jointly-estimated dispersion parameter alpha). This is
inherent to the secondary unit's design (subsystem-level fixed effects), not
something this script corrects for -- flagged as a limitation on this unit's
results, same treatment as the primary unit's own known caveats (Sec 9b).
"""

import argparse
import logging

import numpy as np
import pandas as pd

from fit_rq2_models import (
    LAGS,
    N_PERMUTATIONS,
    RNG_SEED,
    blocked_time_cv_mae,
    fit_h3_reverse,
    fit_nb_glm,
    nested_lr_test,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_lag_features(panel, ratio_col='design_ratio_corrected'):
    panel = panel.sort_values(['project_component', 'quarter_index']).copy()
    for k in LAGS:
        panel[f'design_ratio_lag{k}'] = panel.groupby('project_component')[ratio_col].shift(k)
        panel[f'bug_count_lag{k}'] = panel.groupby('project_component')['bug_count'].shift(k)
    panel['log_exposure'] = np.log(panel['resolved_nonbug'] + 1)
    return panel


def permutation_test_h1_component(data, ratio_col='design_ratio_corrected', n_perms=N_PERMUTATIONS, seed=RNG_SEED):
    """Same logic as fit_rq2_models.permutation_test_h1, but shuffles within
    project_component (the finer series identity) rather than project, since
    that's the unit whose own autocorrelation/level should be preserved.

    n_perms is set by --n_permutations (default 2000). The original 200 was
    chosen for runtime but turned out too coarse to resolve p-values near the
    Holm threshold -- see the --n_permutations comment in main().
    """
    rng = np.random.default_rng(seed)
    m0_formula = 'bug_count ~ bug_count_lag1 + quarter_index + C(project_component)'
    m1_terms = ' + '.join(f'design_ratio_lag{k}' for k in LAGS)
    m1_formula = f'{m0_formula} + {m1_terms}'

    fit_cols = ['bug_count', 'bug_count_lag1', 'quarter_index',
                'project_component', 'project', 'log_exposure'] + [f'design_ratio_lag{k}' for k in LAGS]
    clean = data.dropna(subset=fit_cols)

    m0 = fit_nb_glm(m0_formula, clean)
    m1 = fit_nb_glm(m1_formula, clean)
    observed_stat, _, _ = nested_lr_test(m0, m1)

    exceed, n_ok = 0, 0
    for _ in range(n_perms):
        perm = clean.copy()
        perm[ratio_col] = perm.groupby('project_component')[ratio_col].transform(
            lambda s: rng.permutation(s.values)
        )
        for k in LAGS:
            perm[f'design_ratio_lag{k}'] = perm.groupby('project_component')[ratio_col].shift(k)
        perm = perm.dropna(subset=[f'design_ratio_lag{k}' for k in LAGS])
        try:
            m1_perm = fit_nb_glm(m1_formula, perm)
            m0_perm = fit_nb_glm(m0_formula, perm)
            stat, _, _ = nested_lr_test(m0_perm, m1_perm)
        except Exception:
            continue
        n_ok += 1
        if stat >= observed_stat:
            exceed += 1

    # (exceed + 1) / (n_ok + 1): the observed statistic counts as one draw
    # from the null, so p can never be exactly 0, and failed refits shrink
    # the denominator instead of silently counting as non-exceedances.
    return observed_stat, (exceed + 1) / (n_ok + 1), n_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', required=True,
                         help='data/rq2_component_panel_drop.csv or data/rq2_component_panel_bucket.csv')
    parser.add_argument('--skip_permutation', action='store_true')
    parser.add_argument('--skip_cv', action='store_true')
    # 2000, not the original 200: with a 12-test Holm family the strictest
    # threshold is 0.05/12 ~= 0.0042, and 200 reps can't resolve a p-value
    # that small (min attainable p = 1/201 ~= 0.005). Empirically ~2.7s/rep
    # here, so ~90 min per cell.
    parser.add_argument('--n_permutations', type=int, default=2000,
                         help='See permutation_test_h1_component docstring and the '
                              'comment above for why this differs from the primary unit.')
    parser.add_argument('--ratio_col', default='design_ratio_corrected',
                         choices=['design_ratio_corrected', 'design_ratio_raw'])
    args = parser.parse_args()

    panel = pd.read_csv(args.panel)
    panel = add_lag_features(panel, ratio_col=args.ratio_col)

    m0_formula = 'bug_count ~ bug_count_lag1 + quarter_index + C(project_component)'
    m1_formula = m0_formula + ' + ' + ' + '.join(f'design_ratio_lag{k}' for k in LAGS)
    m2_formula = m1_formula + ' + gini_design_effort + front_loaded_index + quarters_since_changepoint'

    fit_cols_m2 = [
        'bug_count', 'bug_count_lag1', 'quarter_index', 'project_component', 'project',
        'log_exposure', 'gini_design_effort', 'front_loaded_index', 'quarters_since_changepoint',
    ] + [f'design_ratio_lag{k}' for k in LAGS]
    clean = panel.dropna(subset=fit_cols_m2)
    logger.info(
        f"Panel: {len(panel)} rows ({panel['project_component'].nunique()} project-component series) "
        f"-> {len(clean)} after dropping NA lag/control rows"
    )

    m0 = fit_nb_glm(m0_formula, clean)
    m1 = fit_nb_glm(m1_formula, clean)
    m2 = fit_nb_glm(m2_formula, clean)

    h1_stat, h1_p, h1_df = nested_lr_test(m0, m1)
    h2_stat, h2_p, h2_df = nested_lr_test(m1, m2)

    print(f"\nH1 (M0 vs M1) LR stat={h1_stat:.3f}, df={h1_df}, p={h1_p:.4f}")
    print(f"H2 (M1 vs M2) LR stat={h2_stat:.3f}, df={h2_df}, p={h2_p:.4f}")
    print("\nM1 design_ratio_lag coefficients:")
    print(m1.params.filter(like='design_ratio_lag'))
    print(f"M0 alpha={m0.params.get('alpha'):.4f}  M1 alpha={m1.params.get('alpha'):.4f}  "
          f"M2 alpha={m2.params.get('alpha'):.4f}")

    if not args.skip_permutation:
        logger.info(f"Running {args.n_permutations}-permutation null for H1 (this is slow with {clean['project_component'].nunique()} series)...")
        # Pass `clean` (the M2-dropna sample), not `panel` -- see
        # fit_rq2_models.py's matching comment. Using the full panel here
        # recomputes M0/M1 on a much larger, less-restricted row set (202 vs
        # 448 series filtered down further by M2's columns) and gives an
        # "observed" stat that doesn't match the h1_stat reported above at
        # all (empirically 6.1 vs 13.9 on the drop/raw panel) -- clean
        # already satisfies this function's own dropna, so this is a no-op
        # restriction, not a scope change.
        obs_stat, perm_p, n_ok = permutation_test_h1_component(
            clean, ratio_col=args.ratio_col, n_perms=args.n_permutations
        )
        print(f"H1 permutation null ({args.n_permutations} reps): observed LR stat={obs_stat:.3f}, "
              f"permutation p={perm_p:.4f} ({n_ok}/{args.n_permutations} permuted refits succeeded)")

    if not args.skip_cv:
        logger.info("Running blocked-time CV MAE for M0/M1/M2...")
        for name, formula in [('M0', m0_formula), ('M1', m1_formula), ('M2', m2_formula)]:
            mae, n_used, n_nonconverged, n_thin = blocked_time_cv_mae(
                clean, formula, min_train_projects=6, fe_col='project_component'
            )
            print(f"{name}: MAE={mae:.3f}  folds_used={n_used}  "
                  f"folds_dropped_nonconverged={n_nonconverged}  folds_skipped_thin_training={n_thin}")

    print("\n=== H3 (reverse: bug_count -> design_ratio), project_component FE ===")
    h3_formula_note = "reuses fit_h3_reverse's C(project) FE -- swapping in project_component below"
    h3 = fit_h3_reverse_component(panel, ratio_col=args.ratio_col)
    print(h3.summary())


def fit_h3_reverse_component(data, ratio_col='design_ratio_corrected'):
    """Same as fit_rq2_models.fit_h3_reverse but with C(project_component) FE
    and cluster-robust-by-project SEs, matching this unit's M0/M1/M2 spec."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    formula = (
        f'{ratio_col} ~ ' +
        ' + '.join(f'bug_count_lag{k}' for k in LAGS) +
        ' + quarter_index + C(project_component)'
    )
    fit_cols = [ratio_col, 'resolved_total', 'project'] + [f'bug_count_lag{k}' for k in LAGS] + \
        ['quarter_index', 'project_component']
    clean = data.dropna(subset=fit_cols)
    clean = clean[clean['resolved_total'] > 0]
    model = smf.glm(
        formula=formula, data=clean,
        family=sm.families.Binomial(),
        var_weights=clean['resolved_total'],
    )
    return model.fit(cov_type='cluster', cov_kwds={'groups': clean['project']})


if __name__ == '__main__':
    main()
