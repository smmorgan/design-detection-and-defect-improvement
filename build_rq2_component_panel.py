#!/usr/bin/env python3
"""
RQ2 pre-registration Sec 5's secondary/robustness unit: project x component x
quarter. Builds on build_rq2_panel.py's per-series logic (Gini, front-loaded
index, change-point distance, Rogan-Gladen correction) but nests it one level
deeper, inside each project's components.

New decisions this file makes that RQ2_PREREGISTRATION.md does not cover --
documented here rather than decided silently, same practice as the
calendar_index/quarter_index collinearity fix and gini_design_effort
expanding-window fix in fit_rq2_models.py / build_rq2_panel.py:

1. **Multi-component issues**: an issue linked to N components is exploded
   into N (issue, component) rows and counted once per linked component.
   This means resolved_total/bug_count summed across a project's components
   can exceed that project's own project-level totals in data/rq2_panel.csv
   -- expected, not a bug, since a multi-component issue genuinely
   contributes to more than one component's series. Average components per
   linked issue is 1.04-1.26 (COMPONENT_SCHEMA_FINDINGS.md), so the
   inflation is modest.
2. **Missing-component handling** (both variants run, per
   RQ2_PREREGISTRATION.md Sec 5.4): 'drop' excludes issues with no listed
   component from this unit entirely; 'bucket' assigns them to a synthetic
   'Unclassified' component per project.
3. **Component-quarter inclusion rule** (new -- the pre-registration only
   defines a project-level rule, Sec 3: >=12 quarters / >=50 bugs): a
   (project, component) series needs >=8 quarters and >=20 bugs, scaled down
   for this finer-grained unit.
4. **Cross-project reporting gate** (RQ2_PREREGISTRATION.md Sec 5.1): the
   unit is only fit/reported if >=6 of the 10 projects retain at least one
   qualifying component.
5. **Fixed effects**: fit_rq2_component_models.py uses one dummy per
   (project, component) pair, not per project alone -- the point of this
   unit is to control for subsystem-level baseline bug-rate differences a
   project-only FE can't see.
6. **quarter_index** is defined per (project, component) pair (quarters
   since that component's own first linked issue), not per project, to
   avoid reintroducing the calendar_index/quarter_index/FE exact
   collinearity documented in fit_rq2_models.py's module docstring --
   calendar_index is again omitted as a regressor for the same reason.
7. **Rogan-Gladen correction** uses the PROJECT-level sensitivity/
   specificity (gcp_results/rq2_sensitivity_specificity.json) applied
   uniformly to every component within that project -- no per-component
   classifier calibration exists or is in scope to create.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from build_rq2_panel import CALENDAR_EPOCH, changepoint_distances, gini, rogan_gladen

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECTS = [
    'CONFSERVER', 'DM', 'DNN', 'FAB', 'JRASERVER',
    'MESOS', 'MULE', 'NEXUS', 'SERVER', 'TIMOB',
]
LABELED_DIR = Path('data/tawos_full/labeled')
SENS_SPEC_PATH = 'gcp_results/rq2_sensitivity_specificity.json'
MIN_QUARTERS = 8
MIN_BUGS = 20
MIN_QUALIFYING_PROJECTS = 6


def explode_components(df, missing_handling):
    """Split the semicolon-separated `components` column into one row per
    (issue, component) pair. missing_handling: 'drop' or 'bucket'."""
    df = df.copy()
    df['components'] = df['components'].astype('string')
    has_component = df['components'].notna() & (df['components'].str.strip() != '')

    with_comp = df[has_component].copy()
    with_comp['component'] = with_comp['components'].str.split(';')
    with_comp = with_comp.explode('component')
    with_comp['component'] = with_comp['component'].str.strip()
    with_comp = with_comp[with_comp['component'] != '']

    if missing_handling == 'drop':
        return with_comp
    elif missing_handling == 'bucket':
        no_comp = df[~has_component].copy()
        no_comp['component'] = 'Unclassified'
        return pd.concat([with_comp, no_comp], ignore_index=True)
    else:
        raise ValueError(f"unknown missing_handling: {missing_handling}")


def build_component_series(g, project, component):
    """Same per-series construction as build_rq2_panel.build_project_panel,
    just parameterized over a (project, component) group instead of a whole
    project."""
    t0 = g['creation_q'].min()
    resolved = g.dropna(subset=['resolution_q'])
    resolved_total = resolved.groupby('resolution_q').size().rename('resolved_total')
    resolved_design = resolved.groupby('resolution_q')['pred_design'].sum().rename('resolved_design')
    resolved_nonbug = (
        resolved[resolved['issue_type'] != 'Bug'].groupby('resolution_q').size().rename('resolved_nonbug')
    )
    bug_count = g[g['issue_type'] == 'Bug'].groupby('creation_q').size().rename('bug_count')

    last_q = max(g['creation_q'].max(), resolved['resolution_q'].max() if not resolved.empty else t0)
    all_quarters = pd.period_range(t0, last_q, freq='Q')

    cp = pd.DataFrame(index=all_quarters)
    cp = cp.join(resolved_total).join(resolved_design).join(resolved_nonbug).join(bug_count)
    cp[['resolved_total', 'resolved_design', 'resolved_nonbug', 'bug_count']] = (
        cp[['resolved_total', 'resolved_design', 'resolved_nonbug', 'bug_count']].fillna(0)
    )
    return cp


def build_component_panel(project, sens, spec, missing_handling):
    df = pd.read_csv(
        LABELED_DIR / f'{project}_labeled.csv',
        parse_dates=['creation_date', 'resolution_date'],
    )
    if df['pred_design'].isna().all():
        raise ValueError(f"{project}: no ensemble predictions found -- run ensemble_versioned_labels.py first")

    exploded = explode_components(df, missing_handling)
    exploded['creation_q'] = exploded['creation_date'].dt.to_period('Q')
    exploded['resolution_q'] = exploded['resolution_date'].dt.to_period('Q')

    panels = []
    for component, g in exploded.groupby('component'):
        cp = build_component_series(g, project, component)

        n_quarters = len(cp)
        n_bugs = cp['bug_count'].sum()
        if n_quarters < MIN_QUARTERS or n_bugs < MIN_BUGS:
            continue

        cp['project'] = project
        cp['component'] = component
        cp['project_component'] = f'{project}::{component}'
        cp['quarter'] = cp.index.astype(str)
        cp['quarter_index'] = range(len(cp))
        # Calendar-absolute time axis, NOT a regressor (would reintroduce the
        # calendar_index/quarter_index/FE collinearity, decision 6 above) --
        # kept only so fit_rq2_component_models.py's blocked-time CV can order
        # and split folds across series that start at different real dates.
        cp['calendar_index'] = [(q.ordinal - CALENDAR_EPOCH.ordinal) for q in cp.index]

        with np.errstate(invalid='ignore', divide='ignore'):
            apparent = np.where(cp['resolved_total'] > 0, cp['resolved_design'] / cp['resolved_total'], np.nan)
        cp['design_ratio_raw'] = apparent
        cp['design_ratio_corrected'] = rogan_gladen(apparent, sens, spec)
        cp['bug_count_lag1'] = cp['bug_count'].shift(1)

        cp['gini_design_effort'] = [
            gini(cp['resolved_design'].iloc[:i + 1]) for i in range(len(cp))
        ]
        first_third_cutoff = max(1, len(cp) // 3)
        total_design_effort = cp['resolved_design'].sum()
        front_loaded = (
            cp['resolved_design'].iloc[:first_third_cutoff].sum() / total_design_effort
            if total_design_effort > 0 else np.nan
        )
        cp['front_loaded_index'] = front_loaded
        cp['quarters_since_changepoint'] = changepoint_distances(cp['design_ratio_corrected'])

        panels.append(cp.reset_index(drop=True))

    if not panels:
        return None
    return pd.concat(panels, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--missing_handling', choices=['drop', 'bucket'], required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    out_path = args.out or f'data/rq2_component_panel_{args.missing_handling}.csv'

    sens_spec = {r['project']: r for r in json.load(open(SENS_SPEC_PATH))}
    panels = []
    qualifying_projects = set()

    for project in PROJECTS:
        path = LABELED_DIR / f'{project}_labeled.csv'
        if not path.exists():
            logger.warning(f"{project}: {path} not found, skipping")
            continue
        sens = sens_spec[project]['sensitivity']
        spec = sens_spec[project]['specificity']
        panel = build_component_panel(project, sens, spec, args.missing_handling)
        if panel is None:
            logger.warning(f"{project}: no qualifying components under '{args.missing_handling}' handling")
            continue
        n_components = panel['component'].nunique()
        logger.info(f"{project}: {n_components} qualifying components, {len(panel)} component-quarter rows")
        qualifying_projects.add(project)
        panels.append(panel)

    logger.info(
        f"{len(qualifying_projects)}/{len(PROJECTS)} projects have >=1 qualifying component "
        f"(need >={MIN_QUALIFYING_PROJECTS} per RQ2_PREREGISTRATION.md Sec 5.1 to report this unit)"
    )
    if len(qualifying_projects) < MIN_QUALIFYING_PROJECTS:
        logger.error(
            f"Component-level unit NOT reportable under '{args.missing_handling}' handling -- "
            f"writing the panel anyway for inspection, but fit_rq2_component_models.py should "
            f"not be treated as a pre-registered result in this case."
        )

    if not panels:
        logger.error("No projects produced a component panel -- nothing to write.")
        return

    full_panel = pd.concat(panels, ignore_index=True)
    full_panel.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(full_panel)} rows across {len(qualifying_projects)} qualifying projects -> {out_path}")


if __name__ == '__main__':
    main()
