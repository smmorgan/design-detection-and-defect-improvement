#!/usr/bin/env python3
"""
Build the RQ2 project x quarter panel (plan step 5) from the frozen label set.

Reads data/tawos_full/labeled/<PROJECT>_labeled.csv (from
ensemble_versioned_labels.py) and gcp_results/rq2_sensitivity_specificity.json
(per-project classifier sensitivity/specificity from the frozen Base+Meta
5-seed LOPO sweep), and emits one row per (project, quarter):

  project, quarter (period), quarter_index (project age, quarters since the
  project's first issue), calendar_index (quarters since 2000Q1, for the
  calendar-time trend control), resolved_total, resolved_design,
  resolved_nonbug, design_ratio_raw, design_ratio_corrected (Rogan-Gladen),
  bug_count (created that quarter), bug_count_lag1, gini_design_effort,
  front_loaded_index, quarters_since_changepoint.

See paper/RQ2_PREREGISTRATION.md for the definitions this implements
(exposure offset, controls, temporal features) and why each choice was made.
Does not fit any model -- that's fit_rq2_models.py, kept separate so the
panel itself can be inspected/audited before any hypothesis test touches it.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECTS = [
    'CONFSERVER', 'DM', 'DNN', 'FAB', 'JRASERVER',
    'MESOS', 'MULE', 'NEXUS', 'SERVER', 'TIMOB',
]
LABELED_DIR = Path('data/tawos_full/labeled')
SENS_SPEC_PATH = 'gcp_results/rq2_sensitivity_specificity.json'
OUT_PATH = 'data/rq2_panel.csv'
CALENDAR_EPOCH = pd.Period('2000Q1', freq='Q')
MIN_QUARTERS = 12
MIN_BUGS = 50


def rogan_gladen(p_apparent, sens, spec):
    denom = sens + spec - 1
    if denom <= 0:
        # Classifier at or worse than chance for this project -- correction is
        # undefined/unstable. Flagged, not silently clipped into something
        # that looks like a real estimate.
        return np.full_like(p_apparent, np.nan, dtype=float)
    corrected = (p_apparent + spec - 1) / denom
    return np.clip(corrected, 0.0, 1.0)


def gini(x):
    """Gini coefficient of a non-negative series (0 = perfectly even, 1 = all in one quarter)."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def changepoint_distances(series):
    """PELT change-point detection (piecewise-constant mean, L2 cost) on a 1D
    series; returns, for each index, the number of quarters to the nearest
    detected change point.

    Penalty is the standard BIC-style formula for an L2/Gaussian-mean PELT
    cost, `variance(series) * log(n)` -- pre-registered in
    RQ2_PREREGISTRATION.md §5.6, not tuned per-series against any downstream
    p-value. An initial attempt used an RBF cost with a flat `3*log(n)`
    penalty; that combination never found a single interior change point on
    any of the 10 real project series (RBF cost values aren't on the same
    scale that penalty assumes), so this was swapped to L2 + variance-scaled
    penalty, which was checked against all 10 real series before being
    frozen here.
    """
    x = np.nan_to_num(np.asarray(series, dtype=float), nan=0.0).reshape(-1, 1)
    n = len(x)
    if n < 8:
        return np.full(n, np.nan)
    variance = float(np.var(x))
    if variance == 0:
        return np.full(n, np.nan)
    penalty = variance * np.log(n)
    algo = rpt.Pelt(model='l2', min_size=2).fit(x)
    try:
        breakpoints = algo.predict(pen=penalty)
    except Exception:
        return np.full(n, np.nan)
    breakpoints = [b for b in breakpoints if b < n]
    if not breakpoints:
        return np.full(n, np.nan)
    idx = np.arange(n)
    dist = np.min(np.abs(idx[:, None] - np.array(breakpoints)[None, :]), axis=1)
    return dist.astype(float)


def build_project_panel(project, sens, spec):
    df = pd.read_csv(
        LABELED_DIR / f'{project}_labeled.csv',
        parse_dates=['creation_date', 'resolution_date'],
    )
    if df['pred_design'].isna().all():
        raise ValueError(f"{project}: no ensemble predictions found -- run ensemble_versioned_labels.py first")

    df['creation_q'] = df['creation_date'].dt.to_period('Q')
    df['resolution_q'] = df['resolution_date'].dt.to_period('Q')
    t0 = df['creation_q'].min()

    resolved = df.dropna(subset=['resolution_q'])
    resolved_total = resolved.groupby('resolution_q').size().rename('resolved_total')
    resolved_design = resolved.groupby('resolution_q')['pred_design'].sum().rename('resolved_design')
    resolved_nonbug = (
        resolved[resolved['issue_type'] != 'Bug'].groupby('resolution_q').size().rename('resolved_nonbug')
    )
    bug_count = df[df['issue_type'] == 'Bug'].groupby('creation_q').size().rename('bug_count')

    last_q = max(
        df['creation_q'].max(),
        resolved['resolution_q'].max() if not resolved.empty else t0,
    )
    all_quarters = pd.period_range(t0, last_q, freq='Q')

    panel = pd.DataFrame(index=all_quarters)
    panel = panel.join(resolved_total).join(resolved_design).join(resolved_nonbug).join(bug_count)
    panel[['resolved_total', 'resolved_design', 'resolved_nonbug', 'bug_count']] = (
        panel[['resolved_total', 'resolved_design', 'resolved_nonbug', 'bug_count']].fillna(0)
    )

    panel['project'] = project
    panel['quarter'] = panel.index.astype(str)
    panel['quarter_index'] = range(len(panel))
    panel['calendar_index'] = [(q.ordinal - CALENDAR_EPOCH.ordinal) for q in panel.index]

    with np.errstate(invalid='ignore', divide='ignore'):
        apparent = np.where(
            panel['resolved_total'] > 0,
            panel['resolved_design'] / panel['resolved_total'],
            np.nan,
        )
    panel['design_ratio_raw'] = apparent
    panel['design_ratio_corrected'] = rogan_gladen(apparent, sens, spec)
    panel['bug_count_lag1'] = panel['bug_count'].shift(1)

    # gini_design_effort is an expanding-window ("cumulative ... to date" per
    # RQ2_PREREGISTRATION.md Sec.4) statistic recomputed at every quarter from
    # only that project's history through the current quarter -- NOT a
    # whole-history value broadcast to every row. An earlier version computed
    # Gini once over the full series and assigned that single number to every
    # quarter; besides leaking each project's future design-effort shape into
    # its own early quarters, a per-project constant is exactly collinear
    # with C(project) in the M2 regression (fit_rq2_models.py's module
    # docstring has the collinearity proof) and was silently contributing zero
    # real degrees of freedom to the H2 test.
    panel['gini_design_effort'] = [
        gini(panel['resolved_design'].iloc[:i + 1]) for i in range(len(panel))
    ]

    # front_loaded_index is, by its own pre-registered definition ("share of
    # design tickets resolved in the first third of OBSERVED project-quarters"),
    # a whole-history summary and therefore a genuine per-project constant --
    # unlike gini_design_effort above, this one is not a bug, just inherently
    # unidentifiable alongside C(project). fit_rq2_models.py's nested_lr_test
    # already discounts it from H2's degrees of freedom automatically (via
    # design-matrix rank), so it's kept here for descriptive completeness.
    first_third_cutoff = max(1, len(panel) // 3)
    total_design_effort = panel['resolved_design'].sum()
    front_loaded = (
        panel['resolved_design'].iloc[:first_third_cutoff].sum() / total_design_effort
        if total_design_effort > 0 else np.nan
    )
    panel['front_loaded_index'] = front_loaded
    panel['quarters_since_changepoint'] = changepoint_distances(panel['design_ratio_corrected'])

    return panel.reset_index(drop=True)


def main():
    sens_spec = {r['project']: r for r in json.load(open(SENS_SPEC_PATH))}
    panels = []
    excluded = []

    for project in PROJECTS:
        path = LABELED_DIR / f'{project}_labeled.csv'
        if not path.exists():
            logger.warning(f"{project}: {path} not found, skipping (run ensemble_versioned_labels.py first)")
            continue
        sens = sens_spec[project]['sensitivity']
        spec = sens_spec[project]['specificity']
        panel = build_project_panel(project, sens, spec)

        n_quarters = len(panel)
        n_bugs = panel['bug_count'].sum()
        if n_quarters < MIN_QUARTERS or n_bugs < MIN_BUGS:
            excluded.append((project, n_quarters, n_bugs))
            logger.warning(
                f"{project}: EXCLUDED by inclusion rule "
                f"(quarters={n_quarters}, bugs={n_bugs}; need >={MIN_QUARTERS}/{MIN_BUGS})"
            )
            continue

        logger.info(
            f"{project}: {n_quarters} quarters, {n_bugs} bugs, "
            f"mean design_ratio_raw={panel['design_ratio_raw'].mean():.3f}, "
            f"mean design_ratio_corrected={panel['design_ratio_corrected'].mean():.3f}"
        )
        panels.append(panel)

    if not panels:
        logger.error("No projects produced a panel -- nothing to write.")
        return

    full_panel = pd.concat(panels, ignore_index=True)
    full_panel.to_csv(OUT_PATH, index=False)
    logger.info(f"Wrote {len(full_panel)} project-quarter rows across {len(panels)} projects -> {OUT_PATH}")
    if excluded:
        logger.info(f"Excluded by inclusion rule: {excluded}")


if __name__ == '__main__':
    main()
