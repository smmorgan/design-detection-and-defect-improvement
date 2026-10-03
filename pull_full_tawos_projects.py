#!/usr/bin/env python3
"""
Full (uncapped) TAWOS project pull for the RQ2 panel.

The IEEE DataPort export (`ieee_dataport/tawos_project_issues/*.csv`) caps each
project at 3,000 issues and spans a partial slice of project history (see
`paper/RQ2_COLLAPSED_PLAN.md` §4 blocker 6). Building a quarterly panel needs
full project histories: every issue (design classifier needs Title/Description
text; the defect-rate outcome and activity-volume exposure need every issue
regardless of text length), dated by both Creation_Date (discovery / bug
reports) and Resolution_Date (when work resolved), plus each issue's
Component links (see COMPONENT_SCHEMA_FINDINGS.md).

Usage:
    .venv/bin/python pull_full_tawos_projects.py
"""

import logging
import os
from pathlib import Path

import mysql.connector
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECTS = [
    'CONFSERVER', 'DM', 'DNN', 'FAB', 'JRASERVER',
    'MESOS', 'MULE', 'NEXUS', 'SERVER', 'TIMOB',
]
OUT_DIR = Path('data/tawos_full')

ISSUE_QUERY = """
    SELECT
        i.ID              AS id,
        i.Issue_Key       AS issue_key,
        p.Project_Key     AS project,
        i.Type            AS issue_type,
        i.Priority        AS priority,
        i.Status          AS status,
        i.Resolution      AS resolution,
        i.Title           AS summary,
        i.Description_Text AS description,
        i.Creation_Date   AS creation_date,
        i.Resolution_Date AS resolution_date,
        i.Last_Updated    AS last_updated
    FROM Issue i
    JOIN Project p ON p.ID = i.Project_ID
    WHERE p.Project_Key = %s
"""

COMPONENT_QUERY = """
    SELECT ic.Issue_ID AS issue_id, c.Name AS component
    FROM Issue_Component ic
    JOIN Component c ON c.ID = ic.Component_ID
    JOIN Issue i ON i.ID = ic.Issue_ID
    JOIN Project p ON p.ID = i.Project_ID
    WHERE p.Project_Key = %s
"""


def connect():
    return mysql.connector.connect(
        host=os.environ.get('TAWOS_DB_HOST', 'localhost'),
        port=int(os.environ.get('TAWOS_DB_PORT', '3306')),
        database=os.environ.get('TAWOS_DB_NAME', 'tawos'),
        user=os.environ.get('TAWOS_DB_USER', 'root'),
        password=os.environ.get('TAWOS_DB_PASSWORD', ''),
        charset='utf8mb4', use_unicode=True,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect()

    for project in PROJECTS:
        issues = pd.read_sql(ISSUE_QUERY, conn, params=(project,))
        components = pd.read_sql(COMPONENT_QUERY, conn, params=(project,))

        if not components.empty:
            comp_agg = (
                components.groupby('issue_id')['component']
                .apply(lambda s: '; '.join(sorted(set(s))))
                .rename('components')
            )
            issues = issues.merge(comp_agg, left_on='id', right_index=True, how='left')
        else:
            issues['components'] = pd.NA

        out_path = OUT_DIR / f'{project}.csv'
        issues.to_csv(out_path, index=False)
        n_with_comp = issues['components'].notna().sum()
        logger.info(
            f"{project}: {len(issues)} issues, "
            f"{n_with_comp} ({100*n_with_comp/max(1,len(issues)):.1f}%) with a component, "
            f"date range {issues['creation_date'].min()} .. {issues['creation_date'].max()} "
            f"-> {out_path}"
        )

    conn.close()


if __name__ == '__main__':
    main()
