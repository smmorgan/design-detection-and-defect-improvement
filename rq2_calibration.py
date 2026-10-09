"""Calibration input + time-drift check for the RQ2 design-ratio correction.

See RQ2_PREREGISTRATION.md §11 and RQ2_INITIAL_RESULTS.md §1.

Builds one calibration table from every human label we have, tagged by source:
  - manual_curated: output/all_manually_labelled.csv (recent-only, design-enriched)
  - random_audit:   data/calibration_audit/audit_sample_to_label.csv, rows whose
                    `label` has been filled in (empty until the audit is labelled)
Each row is joined to the frozen ensemble classifier probability and assigned an
era (early/middle/late) using the audit's per-project resolution-date terciles.
Adding audit labels later only means filling in the sheet and re-running this.

Then runs the drift check:
  1. Era coverage per source. The curated sample has no early-era tickets at all,
     so drift cannot be tested without the audit.
  2. Drift test on the random audit only:
       primary   label ~ logit(p) + C(project)  vs.  + C(era)
                 (does P(design | score) shift by era? df=2, robust to sparse eras)
       secondary label ~ logit(p) + C(project)  vs.  + logit(p) * C(era)
                 (also the slope; only when every era has >= MIN_ERA_CLASS of each class)
     Unweighted: the stratification variables (project, era) are covariates, so
     the sampling design is ignorable for these conditional models. Not run on
     the curated labels: their selection is confounded with era (the 38
     'middle'-era curated tickets are 100% design), which produced spurious
     drift in simulation even when none existed.
  3. Transfer check (the option-1 assumption): fit the calibration on the
     curated sample only, predict the random audit, and compare observed vs.
     predicted design rate per era (weighted to the population).
  4. Per-era sensitivity/specificity at threshold 0.5 on the random audit,
     population-weighted, with stratified bootstrap CIs.

Usage:  .venv/bin/python rq2_calibration.py [--audit PATH] [--calib_out PATH] [--out PATH]
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

PROJECTS = ['CONFSERVER', 'DM', 'DNN', 'FAB', 'JRASERVER',
            'MESOS', 'MULE', 'NEXUS', 'SERVER', 'TIMOB']
ERAS = ['early', 'middle', 'late']

LABELED_DIR = Path('data/tawos_full/labeled')
MANUAL = Path('output/all_manually_labelled.csv')
AUDIT_DIR = Path('data/calibration_audit')
# The labelled copy of audit_sample_to_label.csv (exported from Google Sheets, labels D/N).
AUDIT_SHEET = AUDIT_DIR / 'audit_sample_to_label - audit_sample_to_label.csv'
AUDIT_KEY = AUDIT_DIR / 'audit_sample_key.csv'
FRAME = AUDIT_DIR / 'sampling_frame.csv'
CALIB_OUT = AUDIT_DIR / 'calibration_labels.csv'
REPORT_OUT = Path('gcp_results/rq2_calibration_drift.json')

# Drift test: an era needs MIN_ERA_N labels to enter; the slope test also needs
# MIN_ERA_CLASS of each class in every era.
MIN_ERA_N = 20
MIN_ERA_CLASS = 5
N_BOOT = 1000
SEED = 20261003
EPS = 1e-6


def parse_label(v):
    if pd.isna(v) or str(v).strip() == '':
        return np.nan
    s = str(v).strip().lower()
    if s in ('1', '1.0', 'd', 'design', 'true', 'yes', 'y'):
        return 1.0
    if s in ('0', '0.0', 'non-design', 'not design', 'false', 'no', 'n'):
        return 0.0
    raise ValueError(f'Unrecognised label value: {v!r} (use 1/0, D/N or design/non-design)')


def load_scores():
    cols = ['issue_key', 'project', 'creation_date', 'resolution_date',
            'prob_design_ensemble', 'pred_design']
    frames = [pd.read_csv(LABELED_DIR / f'{p}_labeled.csv', usecols=cols,
                          parse_dates=['creation_date', 'resolution_date'])
              for p in PROJECTS]
    return pd.concat(frames, ignore_index=True)


def era_cutpoints():
    frame = pd.read_csv(FRAME, parse_dates=['era_start'])
    cuts = {}
    for project, g in frame.groupby('project'):
        starts = g.set_index('era')['era_start']
        cuts[project] = (starts['middle'], starts['late'])
    return cuts


def assign_era(df, cuts):
    # Era is defined on resolution date (the design ratio's time axis); unresolved
    # tickets fall back to creation date and are flagged.
    when = df['resolution_date'].fillna(df['creation_date'])
    df['era_basis'] = np.where(df['resolution_date'].notna(), 'resolution', 'creation')
    era = []
    for project, t in zip(df['project'], when):
        mid, late = cuts[project]
        era.append('early' if t < mid else 'middle' if t < late else 'late')
    df['era'] = pd.Categorical(era, categories=ERAS)
    return df


def check_stopping_rule(audit):
    """Labelled audit rows must be ranks 1..k in every stratum for the same k."""
    labelled = audit[audit['label'].notna()]
    if labelled.empty:
        return {'n_labelled': 0, 'uniform_prefix': None, 'k': None}
    strata = audit.groupby(['project', 'era'], observed=True)
    max_rank = strata.apply(lambda g: g.loc[g['label'].notna(), 'audit_rank'].max()).fillna(0)
    count = strata['label'].count()
    uniform = bool(max_rank.nunique() == 1 and (count == max_rank).all())
    if not uniform:
        warnings.warn('Audit labels are not a uniform rank prefix across all 30 strata; '
                      'the labelled subset is no longer a balanced random sample '
                      '(see RQ2_PREREGISTRATION.md §11).')
    return {'n_labelled': int(len(labelled)), 'uniform_prefix': uniform,
            'k': int(max_rank.iloc[0]) if uniform else None}


def build_calibration_table(audit_path=AUDIT_SHEET):
    scores = load_scores()
    cuts = era_cutpoints()

    manual = pd.read_csv(MANUAL, usecols=['issue_key', 'label'])
    manual['label'] = manual['label'].map(parse_label)
    manual['source'] = 'manual_curated'
    manual['sampling_weight'] = np.nan  # selection mechanism unknown

    sheet = pd.read_csv(audit_path, usecols=['issue_key', 'label'], dtype={'label': str})
    sheet['label'] = sheet['label'].map(parse_label)
    key = pd.read_csv(AUDIT_KEY, usecols=['issue_key', 'project', 'era', 'audit_rank',
                                          'sampling_weight'])
    audit_all = key.merge(sheet, on='issue_key', how='left')
    stopping = check_stopping_rule(audit_all)
    audit = audit_all.dropna(subset=['label'])[['issue_key', 'label', 'audit_rank', 'sampling_weight']]
    audit['source'] = 'random_audit'

    calib = pd.concat([manual, audit], ignore_index=True)
    calib = calib.merge(scores, on='issue_key', how='left')
    missing = calib['prob_design_ensemble'].isna().sum()
    if missing:
        raise ValueError(f'{missing} labelled tickets have no classifier score')
    calib = assign_era(calib, cuts)
    p = calib['prob_design_ensemble'].clip(EPS, 1 - EPS)
    calib['logit_p'] = np.log(p / (1 - p))
    calib['label'] = calib['label'].astype(int)

    cols = ['issue_key', 'project', 'source', 'label', 'prob_design_ensemble', 'logit_p',
            'pred_design', 'era', 'era_basis', 'audit_rank', 'sampling_weight',
            'creation_date', 'resolution_date']
    return calib[cols], stopping


def era_coverage(calib):
    out = {}
    for source, g in calib.groupby('source'):
        out[source] = {
            era: {'n': int(len(e)), 'n_design': int(e['label'].sum()),
                  'n_unresolved_creation_basis': int((e['era_basis'] == 'creation').sum())}
            for era, e in g.groupby('era', observed=False)
        }
    return out


def fit_logit(formula, data):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return smf.logit(formula, data=data).fit(disp=0, maxiter=500)


def lr_test(base_f, full_f, d):
    try:
        base, full = fit_logit(base_f, d), fit_logit(full_f, d)
    except Exception as e:  # e.g. perfect separation in a very sparse era
        return {'status': 'failed', 'reason': f'{type(e).__name__}: {e}'}
    lr = max(2 * (full.llf - base.llf), 0.0)
    df = int(full.df_model - base.df_model)
    return {'status': 'ok', 'lr_stat': round(float(lr), 4), 'df': df,
            'p_value': float(stats.chi2.sf(lr, df)),
            'converged': bool(base.mle_retvals['converged'] and full.mle_retvals['converged']),
            'era_terms': {k: round(float(v), 4) for k, v in full.params.items() if 'era' in k}}


def drift_test(audit):
    if audit.empty:
        return {'status': 'skipped', 'reason': 'no random-audit labels yet'}
    counts = audit.groupby('era', observed=False)['label'].agg(['size', 'sum'])
    eras = [e for e, r in counts.iterrows() if r['size'] >= MIN_ERA_N]
    if len(eras) < 2:
        return {'status': 'skipped', 'reason': f'fewer than 2 eras with >= {MIN_ERA_N} labels'}
    d = audit[audit['era'].isin(eras)].copy()
    d['era'] = d['era'].cat.remove_unused_categories()
    proj = ' + C(project)' if d['project'].nunique() > 1 else ''
    base = f'label ~ logit_p{proj}'
    out = {'status': 'ok', 'n': int(len(d)), 'eras': eras,
           'design_count_by_era': {e: int(counts.loc[e, 'sum']) for e in eras},
           'intercept_shift': lr_test(base, f'{base} + C(era)', d)}
    sparse = [e for e in eras
              if min(counts.loc[e, 'sum'], counts.loc[e, 'size'] - counts.loc[e, 'sum']) < MIN_ERA_CLASS]
    out['intercept_and_slope'] = (
        {'status': 'skipped', 'reason': f'era(s) {sparse} have < {MIN_ERA_CLASS} of a class'}
        if sparse else lr_test(base, f'label ~ logit_p * C(era){proj}', d))
    return out


def weighted_rate(values, weights):
    return float(np.average(values, weights=weights)) if len(values) else float('nan')


def transfer_check(calib):
    """Option-1 assumption: calibration fit on curated labels predicts random tickets."""
    manual = calib[calib['source'] == 'manual_curated']
    audit = calib[calib['source'] == 'random_audit'].copy()
    if audit.empty:
        return {'status': 'skipped', 'reason': 'no random-audit labels yet'}
    model = fit_logit('label ~ logit_p + C(project)', manual)
    audit['pred_calibrated'] = model.predict(audit)
    rows = {}
    for era, e in list(audit.groupby('era', observed=False)) + [('all', audit)]:
        if e.empty:
            continue
        w = e['sampling_weight']
        rows[era] = {'n': int(len(e)),
                     'observed_design_rate': round(weighted_rate(e['label'], w), 4),
                     'predicted_design_rate': round(weighted_rate(e['pred_calibrated'], w), 4),
                     'raw_classifier_rate': round(weighted_rate(e['pred_design'], w), 4)}
    return {'status': 'ok', 'by_era': rows}


def sens_spec(e):
    w, y, yhat = e['sampling_weight'], e['label'], e['pred_design']
    pos, neg = y == 1, y == 0
    sens = np.sum(w[pos] * yhat[pos]) / np.sum(w[pos]) if pos.any() else np.nan
    spec = np.sum(w[neg] * (1 - yhat[neg])) / np.sum(w[neg]) if neg.any() else np.nan
    return sens, spec


def per_era_sens_spec(calib):
    audit = calib[calib['source'] == 'random_audit']
    if audit.empty:
        return {'status': 'skipped', 'reason': 'no random-audit labels yet'}
    rng = np.random.default_rng(SEED)
    out = {}
    for era, e in audit.groupby('era', observed=False):
        if e.empty:
            continue
        sens, spec = sens_spec(e)
        strata = [s for _, s in e.groupby('project')]
        boots = np.array([sens_spec(pd.concat([s.iloc[rng.integers(0, len(s), len(s))]
                                               for s in strata]))
                          for _ in range(N_BOOT)], dtype=float)

        def ci(col):
            return [round(float(np.nanpercentile(boots[:, col], q)), 4) for q in (2.5, 97.5)]

        # Unweighted counts + Wilson CIs: the bootstrap percentile CI collapses to a
        # point when every positive in an era is detected (or none are).
        tp = int(((e['label'] == 1) & (e['pred_design'] == 1)).sum())
        tn = int(((e['label'] == 0) & (e['pred_design'] == 0)).sum())
        n_pos, n_neg = int((e['label'] == 1).sum()), int((e['label'] == 0).sum())

        def wilson(k, n):
            return [round(float(x), 4) for x in proportion_confint(k, n, method='wilson')]

        out[era] = {'n': int(len(e)), 'n_design': n_pos,
                    'sensitivity': round(float(sens), 4), 'sensitivity_ci95': ci(0),
                    'sensitivity_counts': f'{tp}/{n_pos}', 'sensitivity_wilson95': wilson(tp, n_pos),
                    'sensitivity_reliable': bool(n_pos >= MIN_ERA_CLASS),
                    'specificity': round(float(spec), 4), 'specificity_ci95': ci(1),
                    'specificity_counts': f'{tn}/{n_neg}', 'specificity_wilson95': wilson(tn, n_neg)}
    return {'status': 'ok', 'by_era': out,
            'note': 'sensitivity/specificity and *_ci95 are population-weighted (stratified '
                    'bootstrap); *_counts and *_wilson95 are unweighted. Sensitivity is '
                    'imprecise with ~10 labels per stratum'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit', type=Path, default=AUDIT_SHEET,
                    help='filled-in audit labelling sheet (default: %(default)s)')
    ap.add_argument('--calib_out', type=Path, default=CALIB_OUT)
    ap.add_argument('--out', type=Path, default=REPORT_OUT)
    args = ap.parse_args()

    calib, stopping = build_calibration_table(args.audit)
    args.calib_out.parent.mkdir(parents=True, exist_ok=True)
    calib.to_csv(args.calib_out, index=False)

    report = {
        'n_labels_by_source': calib['source'].value_counts().to_dict(),
        'audit_stopping_rule': stopping,
        'era_coverage': era_coverage(calib),
        'drift_test_random_audit': drift_test(calib[calib['source'] == 'random_audit']),
        'transfer_check_curated_to_random': transfer_check(calib),
        'per_era_sens_spec_random_audit': per_era_sens_spec(calib),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f'\nwrote {args.calib_out} ({len(calib)} rows) and {args.out}')


if __name__ == '__main__':
    main()
