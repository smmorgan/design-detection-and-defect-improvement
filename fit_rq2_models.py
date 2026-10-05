#!/usr/bin/env python3
"""
RQ2 plan step 5 (model fitting): M0/M1/M2 nested negative-binomial GLMs plus
the reverse-direction H3 model, per paper/RQ2_PREREGISTRATION.md §7.

Primary specification only (project x quarter unit, Rogan-Gladen-corrected
design ratio, lags 1-4 as a single distributed-lag block). The remaining
pre-registered grid axes (project x component unit, uncorrected ratio,
missing-component handling, SIMEX) are follow-ups layered on this same
panel/model core -- not yet run, so no p-value from them is reported until
they are, per the pre-registration's Holm-correction-over-the-whole-family
rule.

M0: bug_count ~ bug_count_lag1 + quarter_index + C(project)
M1: M0 + design_ratio_corrected lagged 1..4 quarters (jointly, distributed-lag)
M2: M1 + gini_design_effort + front_loaded_index + quarters_since_changepoint
H3 (reverse): design_ratio_corrected ~ bug_count lagged 1..4 + controls,
    binomial GLM (resolved_design successes / resolved_total trials).

All NB models use log(resolved_nonbug + 1) as a fixed offset, MLE-estimated
dispersion (alpha), and cluster-robust (by project) standard errors. H1 =
joint LR test of the 4 lag coefficients in M1 vs M0. H2 = LR test of M2 vs
M1. A permutation null (shuffle design_ratio_corrected within project,
refit, repeat) accompanies the H1 LR test since the panel's within-project
autocorrelation makes the chi-square reference distribution an approximation.

NB fitting uses statsmodels' discrete NegativeBinomial MLE model (via
smf.negativebinomial), not the GLM family=NegativeBinomial() path -- that
path requires alpha to be supplied and silently defaults to a fixed alpha=1.0
otherwise (confirmed via a runtime ValueWarning), which is an arbitrary
dispersion assumption, not a fitted one. This dataset's actual MLE alpha is
~0.53.

`calendar_index` (quarters since 2000Q1) is deliberately NOT a regressor here,
despite being listed as a separate control in RQ2_PREREGISTRATION.md §4/§7.
Given C(project) fixed effects, calendar_index and quarter_index (project age)
are exactly collinear: quarter_index = calendar_index - (project's own start
quarter), and that per-project constant is already absorbed by the project
dummies, so {C(project), calendar_index, quarter_index} spans a rank-(J)
space with J+2 columns (J=10 projects) -- confirmed via
np.linalg.matrix_rank (12 cols, rank 11) and by fitting all three
combinations, which gives identical log-likelihood and identical
design_ratio_lag coefficients to 1e-6 regardless of which of the two linear
trend terms is kept. This is an age-period-cohort-style non-identification
baked into the pre-registered model design, not a small-cluster SE artifact
as originally described in §9b (that description has been corrected in the
pre-registration to point here) -- it produced HessianInversionWarning in
the NB MLE fits and an explicit SingularMatrixWarning in H3's GLM (IRLS
does an explicit WLS solve each iteration, so the rank deficiency surfaces
directly there instead of only showing up as a degenerate covariance at the
end). Keeping quarter_index over calendar_index is an arbitrary but
inconsequential choice per the above -- ties the retained control to the
plan's own project-maturity/front-loadedness narrative.
"""

import argparse
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PANEL_PATH = 'data/rq2_panel.csv'
LAGS = [1, 2, 3, 4]
N_PERMUTATIONS = 1000
RNG_SEED = 42


def add_lag_features(panel, ratio_col='design_ratio_corrected'):
    panel = panel.sort_values(['project', 'quarter_index']).copy()
    for k in LAGS:
        panel[f'design_ratio_lag{k}'] = panel.groupby('project')[ratio_col].shift(k)
        panel[f'bug_count_lag{k}'] = panel.groupby('project')['bug_count'].shift(k)
    panel['log_exposure'] = np.log(panel['resolved_nonbug'] + 1)
    return panel


def fit_nb_glm(formula, data, offset_col='log_exposure'):
    """Fit NB2 via the MLE discrete model, not statsmodels' GLM
    NegativeBinomial family. GLM's family=NegativeBinomial() requires alpha
    to be supplied and otherwise silently fixes it at 1.0 (confirmed via a
    ValueWarning) -- that is not a fitted dispersion, it's an arbitrary
    constant, and this dataset's own MLE alpha comes out to ~0.53, nowhere
    near 1. smf.negativebinomial (statsmodels.discrete.discrete_model)
    estimates alpha jointly with the mean-structure coefficients by MLE,
    which is what "negative binomial GLM" is supposed to mean here.
    """
    model = smf.negativebinomial(formula=formula, data=data, offset=data[offset_col].values)
    return model.fit(
        disp=0, method='bfgs', maxiter=500,
        cov_type='cluster', cov_kwds={'groups': data['project']},
    )


def lr_test(restricted_llf, full_llf, df_diff):
    stat = 2 * (full_llf - restricted_llf)
    from scipy.stats import chi2
    p = chi2.sf(stat, df_diff)
    return stat, p


def nested_lr_test(restricted, full):
    """LR test between two fitted nested models, with df computed from actual
    design-matrix rank rather than a hardcoded term count.

    Found via the calendar_index/quarter_index collinearity (see module
    docstring): a "new" term added going from the restricted to the full
    model can be exactly collinear with columns already in the restricted
    model (a project-constant feature alongside C(project), a redundant
    linear time trend, etc.), in which case it contributes zero real degrees
    of freedom even though it's one more named coefficient. Counting it as a
    real df anyway inflates the chi-square reference distribution and biases
    the LR test toward non-significance. rank(full.exog) - rank(restricted.exog)
    is the number of degrees of freedom actually added, whatever the cause.
    """
    df_diff = (
        np.linalg.matrix_rank(full.model.exog) - np.linalg.matrix_rank(restricted.model.exog)
    )
    return lr_test(restricted.llf, full.llf, df_diff) + (df_diff,)


def circular_offset(rng, n, min_gap=len(LAGS) + 1):
    """Random circular-shift offset for a series of length n, at least min_gap
    from 0 in both directions. A shift of 1-4 quarters would line a lag column
    back up with a neighbouring real lag and leak the true effect into the null.
    Falls back to any non-zero offset when the series is too short for that."""
    lo, hi = min_gap, n - min_gap
    return int(rng.integers(lo, hi + 1)) if hi >= lo else int(rng.integers(1, n))


def permutation_null(panel, clean, series_col, ratio_col, m0_formula, m1_formula,
                     n_perms, seed, scheme='shuffle', return_draws=False):
    """Null distribution of the H1 LR statistic (M0 vs M1), built by
    re-randomising the design ratio within each series and rebuilding its lags.

    panel: the full lagged panel (every quarter of every series).
    clean: the rows the reported fits use (a subset of panel's index).

    Each draw re-randomises the ratio over the *full* series, recomputes the
    lag columns on the full panel, then refits on exactly `clean`'s rows.

    Fixed 2026-10-04. The original version shuffled and re-lagged inside
    `clean` itself, so every permuted refit lost each series' first 4 rows:
    335 vs 367 rows at the project unit, and 2,774 vs 3,919 for
    component/drop. It also drew permuted values only from quarters after the
    first 4. This is the same class of bug as RQ2_INITIAL_RESULTS.md §2e.

    Positions where the ratio is NaN (no resolved issues) stay fixed, so
    clean's lag columns never gain NaNs. M0 doesn't involve the ratio, so it
    is fit once.

    scheme='shuffle'   within-series permutation (the pre-registered null).
    scheme='circular'  within-series circular shift by a random offset (see
                       circular_offset); keeps the ratio's own autocorrelation
                       intact, which the shuffle destroys.
    """
    rng = np.random.default_rng(seed)
    lag_cols = [f'design_ratio_lag{k}' for k in LAGS]
    m0 = fit_nb_glm(m0_formula, clean)
    m1 = fit_nb_glm(m1_formula, clean)
    observed_stat = 2 * (m1.llf - m0.llf)

    series = panel[series_col].to_numpy()
    has_ratio = panel[ratio_col].notna().to_numpy()
    groups = [np.flatnonzero((series == s) & has_ratio) for s in pd.unique(series)]
    base = panel[ratio_col].to_numpy(copy=True)

    exceed, n_ok, draws = 0, 0, []
    for _ in range(n_perms):
        new = base.copy()
        for idx in groups:
            if len(idx) < 2:
                continue
            if scheme == 'shuffle':
                new[idx] = base[rng.permutation(idx)]
            else:
                new[idx] = np.roll(base[idx], circular_offset(rng, len(idx)))
        perm = panel.copy()
        perm[ratio_col] = new
        for k in LAGS:
            perm[f'design_ratio_lag{k}'] = perm.groupby(series_col)[ratio_col].shift(k)
        perm = perm.loc[clean.index]
        assert not perm[lag_cols].isna().any().any(), 'permuted lags gained NaNs'
        try:
            m1_perm = fit_nb_glm(m1_formula, perm)
        except Exception:
            continue
        stat = 2 * (m1_perm.llf - m0.llf)
        n_ok += 1
        if stat >= observed_stat:
            exceed += 1
        if return_draws:
            draws.append({'lr': float(stat), 'converged': bool(m1_perm.mle_retvals['converged']),
                          **{c: float(m1_perm.params[c]) for c in lag_cols}})

    # (exceed + 1) / (n_ok + 1): the observed statistic counts as one draw
    # from the null, so p can never be exactly 0, and failed refits shrink
    # the denominator instead of silently counting as non-exceedances.
    out = (observed_stat, (exceed + 1) / (n_ok + 1), n_ok)
    return out + (draws, m0, m1) if return_draws else out


def permutation_test_h1(panel, clean, ratio_col='design_ratio_corrected', n_perms=N_PERMUTATIONS, seed=RNG_SEED):
    """Shuffle the design ratio within project (keeps each project's own
    level, breaks the link to bug_count), refit M0 vs M1 on the reported rows,
    and count how often the permuted LR stat reaches the observed one.
    Non-parametric check on the chi-square-based H1 p-value.
    """
    m0_formula = 'bug_count ~ bug_count_lag1 + quarter_index + C(project)'
    m1_formula = m0_formula + ' + ' + ' + '.join(f'design_ratio_lag{k}' for k in LAGS)
    return permutation_null(panel, clean, 'project', ratio_col, m0_formula, m1_formula, n_perms, seed)


def blocked_time_cv_mae(data, formula, offset_col='log_exposure', min_train_quarters=8,
                         min_train_projects=6, step=2, fe_col='project'):
    """Forward-chaining (expanding-window, blocked-by-time) CV, pooled across
    projects: at each calendar cutoff, train on every project's rows with
    calendar_index < cutoff and predict every project's row at calendar_index
    == cutoff. Per-project rolling-origin would make C(project) collinear
    with the intercept (only one level present in a single project's own
    training data) -- pooling is what lets the fixed effects mean anything.
    Returns (mae, n_folds_used, n_folds_skipped_for_convergence,
    n_folds_skipped_for_thin_training).

    `min_train_projects` (new): projects enter the panel at different
    calendar quarters (each one's own JIRA adoption date), so the earliest
    cutoffs can have training data from as few as 3 of the 10 projects --
    too few C(project) levels for a stable fit (confirmed empirically: folds
    with <6 projects present accounted for nearly all of the
    ConvergenceWarning/HessianInversionWarning noise previously attributed
    just to "small early windows"). Cutoffs before at least
    `min_train_projects` projects have entered are skipped outright, and any
    remaining fold whose fit reports `mle_retvals['converged'] is False` is
    dropped from the MAE rather than silently averaged in with an unreliable
    prediction.
    """
    data = data.dropna(subset=['calendar_index', 'bug_count']).sort_values('calendar_index')
    cutoffs = sorted(data['calendar_index'].unique())[min_train_quarters::step]
    errors = []
    n_used, n_nonconverged, n_thin = 0, 0, 0
    for cutoff in cutoffs:
        train = data[data['calendar_index'] < cutoff].dropna()
        test = data[data['calendar_index'] == cutoff].dropna(subset=[offset_col, 'bug_count'])
        if train.empty or test.empty or train['bug_count'].sum() == 0:
            continue
        if train['project'].nunique() < min_train_projects:
            n_thin += 1
            continue
        # Only score test rows whose fixed-effect level appears in training
        # (else its dummy is unseen and predict() raises). `fe_col` must match
        # the formula's FE: 'project_component' at the component unit, where
        # filtering on project alone let unseen components through and every
        # fold's predict() failed (miscounted as non-convergence).
        test = test[test[fe_col].isin(train[fe_col].unique())]
        if test.empty:
            continue
        try:
            model = fit_nb_glm(formula, train, offset_col=offset_col)
            if model.mle_retvals.get('converged') is False:
                n_nonconverged += 1
                continue
            pred = model.predict(test, offset=test[offset_col])
        except Exception:
            n_nonconverged += 1
            continue
        errors.extend(np.abs(pred.values - test['bug_count'].values).tolist())
        n_used += 1
    mae = float(np.mean(errors)) if errors else float('nan')
    return mae, n_used, n_nonconverged, n_thin


def fit_h3_reverse(data, ratio_col='design_ratio_corrected'):
    """H3: does bug_count at t predict design ratio at t+k ('crisis-driven')?
    Binomial GLM: resolved_design successes / resolved_total trials ~ lagged
    bug counts + controls. Uses the SAME lag structure as H1 for symmetry.

    Uses var_weights (precision weighting for aggregated binomial proportions),
    not freq_weights. freq_weights tells statsmodels each row was observed
    `resolved_total` times independently, which inflates Df Residuals to the
    sum of resolved_total (~195,000 instead of ~400) and produces a degenerate
    pseudo-R^2=1.000 plus a `cov_type not fully supported with freq_weights`
    warning on cluster SEs. var_weights is the correct weighting for a
    successes/trials proportion response and is compatible with cluster-robust
    covariance.
    """
    formula = (
        f'{ratio_col} ~ ' +
        ' + '.join(f'bug_count_lag{k}' for k in LAGS) +
        ' + quarter_index + C(project)'
    )
    fit_cols = [ratio_col, 'resolved_total'] + [f'bug_count_lag{k}' for k in LAGS] + \
        ['quarter_index', 'project']
    clean = data.dropna(subset=fit_cols)
    clean = clean[clean['resolved_total'] > 0]
    model = smf.glm(
        formula=formula, data=clean,
        family=sm.families.Binomial(),
        var_weights=clean['resolved_total'],
    )
    return model.fit(cov_type='cluster', cov_kwds={'groups': clean['project']})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', default=PANEL_PATH)
    parser.add_argument('--skip_permutation', action='store_true',
                         help='Skip the 1000-permutation null (slow) -- for quick smoke tests')
    parser.add_argument('--skip_cv', action='store_true')
    parser.add_argument('--ratio_col', default='design_ratio_corrected',
                         choices=['design_ratio_corrected', 'design_ratio_raw'],
                         help='Which design-ratio column feeds the lag features and H3. '
                              'design_ratio_corrected is the Rogan-Gladen-corrected primary '
                              'measure per the pre-registration; design_ratio_raw is the '
                              'uncorrected fallback -- use it if the correction is degenerate '
                              '(see COMPONENT_SCHEMA_FINDINGS.md / RQ2 status memory).')
    args = parser.parse_args()

    panel = pd.read_csv(args.panel)
    panel = add_lag_features(panel, ratio_col=args.ratio_col)

    m0_formula = 'bug_count ~ bug_count_lag1 + quarter_index + C(project)'
    m1_formula = m0_formula + ' + ' + ' + '.join(f'design_ratio_lag{k}' for k in LAGS)
    m2_formula = m1_formula + ' + gini_design_effort + front_loaded_index + quarters_since_changepoint'

    fit_cols_m2 = [
        'bug_count', 'bug_count_lag1', 'quarter_index', 'project',
        'log_exposure', 'gini_design_effort', 'front_loaded_index', 'quarters_since_changepoint',
    ] + [f'design_ratio_lag{k}' for k in LAGS]
    clean = panel.dropna(subset=fit_cols_m2)
    logger.info(f"Panel: {len(panel)} rows -> {len(clean)} after dropping NA lag/control rows")

    m0 = fit_nb_glm(m0_formula, clean)
    m1 = fit_nb_glm(m1_formula, clean)
    m2 = fit_nb_glm(m2_formula, clean)

    h1_stat, h1_p, h1_df = nested_lr_test(m0, m1)
    h2_stat, h2_p, h2_df = nested_lr_test(m1, m2)

    print("\n=== M0 (controls only) ===")
    print(m0.summary())
    print("\n=== M1 (+ design ratio, lags 1-4) ===")
    print(m1.summary())
    print("\n=== M2 (+ temporal features) ===")
    print(m2.summary())

    print(f"\nH1 (M0 vs M1) LR stat={h1_stat:.3f}, df={h1_df}, p={h1_p:.4f}")
    print(f"H2 (M1 vs M2) LR stat={h2_stat:.3f}, df={h2_df}, p={h2_p:.4f}")
    if h2_df != 3:
        print(
            f"NOTE: H2 nominally adds 3 terms (gini_design_effort, front_loaded_index, "
            f"quarters_since_changepoint) but only {h2_df} actual degree(s) of freedom is "
            f"identified -- front_loaded_index is a per-project constant (by its own "
            f"pre-registered definition: whole-history share of design tickets in the first "
            f"third of observed quarters), exactly collinear with C(project), and contributes "
            f"nothing beyond it. gini_design_effort (expanding-window, varies within project) "
            f"and quarters_since_changepoint are the real additions. The naive df=3 test "
            f"understates significance."
        )

    if not args.skip_permutation:
        logger.info(f"Running {N_PERMUTATIONS}-permutation null for H1 (this is slow)...")
        # Pass `clean` (the M2-dropna sample), not `panel`, so the permutation
        # null's own "observed" LR stat is fit on the exact same rows as the
        # h1_stat reported above -- passing the full panel silently recomputes
        # M0/M1 on a larger sample (e.g. it would put DM/FAB back in, which
        # the M2 dropna excludes for having no detected changepoint), so the
        # permutation p-value would be testing a different statistic than the
        # one being reported alongside it. The full panel is passed as well,
        # so permuted lags are rebuilt from whole series (see permutation_null).
        obs_stat, perm_p, n_ok = permutation_test_h1(panel, clean, ratio_col=args.ratio_col)
        print(f"H1 permutation null: observed LR stat={obs_stat:.3f}, permutation p={perm_p:.4f} "
              f"({n_ok}/{N_PERMUTATIONS} permuted refits succeeded)")

    if not args.skip_cv:
        logger.info("Running blocked-time CV MAE for M0/M1/M2...")
        cv_results = {
            'M0': blocked_time_cv_mae(clean, m0_formula),
            'M1': blocked_time_cv_mae(clean, m1_formula),
            'M2': blocked_time_cv_mae(clean, m2_formula),
        }
        print()
        for name, (mae, n_used, n_nonconverged, n_thin) in cv_results.items():
            print(f"{name}: MAE={mae:.3f}  folds_used={n_used}  "
                  f"folds_dropped_nonconverged={n_nonconverged}  folds_skipped_thin_training={n_thin}")

    print("\n=== H3 (reverse: bug_count -> design_ratio) ===")
    h3 = fit_h3_reverse(panel, ratio_col=args.ratio_col)
    print(h3.summary())


if __name__ == '__main__':
    main()
