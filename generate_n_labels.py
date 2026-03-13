#!/usr/bin/env python3
"""
Generate high-confidence non-design (N) labels from TAWOS using the converse
of the keyword-based design heuristic documented in LABELLING_METHODOLOGY.md.

Logic:
  - Fetch issues from the same projects used in the D-label heuristic
  - Apply the SAME keyword scoring used for D-labels
  - Select issues that score 0 (no design keywords matched at all)
  - Prioritise issue types that are strong non-design signals:
      Bug, Sub-task, Test Task (hard-excluded from D selection)
  - Also include low-scoring non-excluded types (score=0) as secondary pool
  - Exclude issues already in all_manually_labelled.csv or new_d_labels_review.csv
  - Cap output to roughly match the new_d count (~5K)

Usage:
    python generate_n_labels.py
    python generate_n_labels.py --max_labels 5000
    python generate_n_labels.py --dry_run   # just show counts, don't write
"""

import argparse
import logging
import os
import re
from pathlib import Path

import pandas as pd

from tawos_connector import TAWOSConfig, TAWOSConnector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ── Projects to query (same as D-label generation) ──────────────────────────

# Original 10 projects (from TSV files)
TSV_PROJECTS = [
    'CONFSERVER', 'DM', 'DNN', 'FAB', 'JRASERVER',
    'MESOS', 'MULE', 'NEXUS', 'SERVER', 'TIMOB',
]

# 17 additional DB projects
DB_PROJECTS = [
    'JRACLOUD', 'CONFCLOUD', 'BAM', 'CWD', 'EVG', 'XD', 'IS', 'STL',
    'JSWSERVER', 'JSWCLOUD', 'FE', 'JAVA', 'TISTUD', 'INDY', 'APSTUD',
    'COMPASS', 'MDL',
]

ALL_PROJECTS = TSV_PROJECTS + DB_PROJECTS

# ── Hard-excluded issue types (these are already excluded from D) ───────────
# These are the STRONGEST non-design signals
BUG_TYPES = {'Bug', 'Sub-task', 'Test Task', 'Sub Task', 'Technical task'}

# ── Design keyword tiers (from LABELLING_METHODOLOGY.md) ────────────────────

STRONG_KEYWORDS = [
    'architecture', 'microservices', 'microservice', 'service design',
    'api gateway', 'api versioning', 'event-driven', 'event driven',
    'message queue', 'schema design', 'data model', 'domain model',
    'service mesh', 'event sourcing', 'cqrs', 'hexagonal',
    'clean architecture', 'design pattern', 'abstraction layer',
    'service layer', 'multi-tenant', 'multi tenant', 'multitenant',
    'plugin system', 'extension point', 'plugin architecture',
    'oauth', 'saml', 'openid', 'sso', 'identity provider', 'jwt',
    'grpc', 'graphql', 'websocket', 'message broker',
    'kafka', 'rabbitmq', 'distributed', 'scalability',
    'horizontal scaling',
]

MEDIUM_KEYWORDS = [
    'refactor', 'redesign', 'rework', 'overhaul',
    'sdk', 'client library', 'public api', 'rest api', 'new api',
    'webhook', 'connector', 'integration', 'new endpoint',
    'authentication', 'authorization', 'permission model', 'access control',
    'migration', 'data migration', 'database migration',
    'caching', 'cache layer', 'indexing', 'search index',
    'asynchronous', 'async', 'background job', 'worker',
    'rate limit', 'throttle', 'circuit breaker',
]

WEAK_KEYWORDS = [
    'api', 'endpoint', 'service', 'database', 'interface',
    'protocol', 'plugin', 'module', 'component', 'framework',
]

# ── Exclusion patterns (from D methodology — issues with these are ops/docs) ─
VERSION_UPGRADE_RE = re.compile(
    r'(upgrade|bump|migrate)\s+(to\s+)?(version|v?\d)',
    re.IGNORECASE,
)
DOC_RE = re.compile(
    r'(add|update|improve|fix)\s+(documentation|javadoc|readme|changelog)',
    re.IGNORECASE,
)
STACK_TRACE_RE = re.compile(
    r'(java\.lang\.\w+Exception|at\s+\w+\.\w+\()',
    re.IGNORECASE,
)
DEPLOY_RE = re.compile(r'\bdeploy\b', re.IGNORECASE)

UI_WORDS = {
    'button', 'dropdown', 'checkbox', 'modal', 'tooltip', 'icon',
    'font', 'color', 'css', 'styling', 'layout', 'cosmetic',
    'wording', 'typo', 'rename', 'spelling',
}


def compute_design_score(summary: str, description: str) -> int:
    """Compute the design keyword score for an issue (same as D heuristic)."""
    text = f"{summary} {description}".lower()
    score = 0

    for kw in STRONG_KEYWORDS:
        if kw in text:
            score += 3
    for kw in MEDIUM_KEYWORDS:
        if kw in text:
            score += 2
    for kw in WEAK_KEYWORDS:
        if kw in text:
            score += 1

    return score


def is_ops_or_docs(summary: str, description: str) -> bool:
    """Check if issue matches ops/docs exclusion patterns."""
    text = f"{summary} {description}"
    return bool(
        VERSION_UPGRADE_RE.search(text)
        or DOC_RE.search(text)
        or DEPLOY_RE.search(text)
    )


def has_stack_trace(description: str) -> bool:
    """Check if description contains stack trace content."""
    return bool(STACK_TRACE_RE.search(description or ''))


def count_ui_words(summary: str) -> int:
    """Count UI-signal words in summary."""
    words = set(summary.lower().split())
    return len(words & UI_WORDS)


def main():
    parser = argparse.ArgumentParser(
        description='Generate high-confidence non-design labels from TAWOS',
    )
    parser.add_argument('--max_labels', type=int, default=5000,
                        help='Maximum N labels to generate')
    parser.add_argument('--out', default='output/new_n_labels.csv',
                        help='Output CSV path')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show counts without writing')
    parser.add_argument('--host', default=os.environ.get('TAWOS_DB_HOST', 'localhost'))
    parser.add_argument('--port', type=int,
                        default=int(os.environ.get('TAWOS_DB_PORT', '3306')))
    parser.add_argument('--database', default=os.environ.get('TAWOS_DB_NAME', 'tawos'))
    parser.add_argument('--user', default=os.environ.get('TAWOS_DB_USER', 'root'))
    parser.add_argument('--password', default=os.environ.get('TAWOS_DB_PASSWORD', ''))
    args = parser.parse_args()

    # Load existing labels to exclude
    existing_keys = set()
    manual_path = Path('output/all_manually_labelled.csv')
    if manual_path.exists():
        df_manual = pd.read_csv(manual_path)
        existing_keys.update(df_manual['issue_key'])
        logger.info(f"Loaded {len(df_manual)} existing manual labels")

    for p in ['output/new_d_labels_review.csv', 'output/server_new_d_labels_review.csv']:
        if Path(p).exists():
            df_d = pd.read_csv(p)
            existing_keys.update(df_d['issue_key'])
            logger.info(f"Loaded {len(df_d)} existing D labels from {p}")

    logger.info(f"Total existing keys to exclude: {len(existing_keys)}")

    # Connect to TAWOS
    config = TAWOSConfig(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password,
        projects=ALL_PROJECTS,
        min_text_length=50,
    )

    connector = TAWOSConnector(config)
    if not connector.connect():
        logger.error("Failed to connect to TAWOS database")
        return

    try:
        logger.info(f"Fetching issues from {len(ALL_PROJECTS)} projects...")
        df = connector.fetch_issues()
        logger.info(f"Fetched {len(df)} total issues")
    finally:
        connector.disconnect()

    # Remove issues already labelled
    df = df[~df['issue_key'].isin(existing_keys)]
    logger.info(f"After removing existing labels: {len(df)} issues")

    # Score each issue
    df['design_score'] = df.apply(
        lambda r: compute_design_score(
            str(r.get('summary', '')),
            str(r.get('description', '')),
        ),
        axis=1,
    )

    df['is_bug_type'] = df['issue_type'].isin(BUG_TYPES)
    df['has_stack_trace'] = df['description'].fillna('').apply(has_stack_trace)
    df['is_ops_docs'] = df.apply(
        lambda r: is_ops_or_docs(str(r.get('summary', '')), str(r.get('description', ''))),
        axis=1,
    )
    df['ui_word_count'] = df['summary'].fillna('').apply(count_ui_words)

    # ── Tier 1: Bug/Sub-task types with score=0 (highest confidence N) ──────
    tier1 = df[
        df['is_bug_type']
        & (df['design_score'] == 0)
    ].copy()
    tier1['confidence_tier'] = 1
    tier1['n_reason'] = 'bug_type+score_0'
    logger.info(f"Tier 1 (Bug/Sub-task, score=0): {len(tier1)}")

    # ── Tier 2: Non-bug types with score=0 AND ops/docs/UI signals ──────────
    tier2 = df[
        ~df['is_bug_type']
        & (df['design_score'] == 0)
        & (df['is_ops_docs'] | df['has_stack_trace'] | (df['ui_word_count'] >= 2))
    ].copy()
    tier2['confidence_tier'] = 2
    tier2['n_reason'] = 'score_0+ops_docs_ui'
    logger.info(f"Tier 2 (non-bug, score=0, ops/docs/UI): {len(tier2)}")

    # ── Tier 3: Any type with score=0, no design signal at all ──────────────
    already_selected = set(tier1['issue_key']) | set(tier2['issue_key'])
    tier3 = df[
        (df['design_score'] == 0)
        & ~df['issue_key'].isin(already_selected)
    ].copy()
    tier3['confidence_tier'] = 3
    tier3['n_reason'] = 'score_0'
    logger.info(f"Tier 3 (any type, score=0): {len(tier3)}")

    # Combine tiers, prioritise by confidence
    candidates = pd.concat([tier1, tier2, tier3], ignore_index=True)
    candidates = candidates.sort_values(
        ['confidence_tier', 'project'],
        ascending=[True, True],
    )
    logger.info(f"Total N candidates: {len(candidates)}")

    # Per-project distribution
    print(f"\n{'='*60}")
    print("N-LABEL CANDIDATES BY PROJECT AND TIER")
    print(f"{'='*60}")
    ct = pd.crosstab(candidates['project'], candidates['confidence_tier'], margins=True)
    print(ct.to_string())

    # Cap to max_labels, sampling evenly across projects (then by tier)
    if len(candidates) > args.max_labels:
        per_project = args.max_labels // len(candidates['project'].unique())
        sampled_parts = []
        for proj, grp in candidates.groupby('project'):
            # Within each project, prioritise by confidence tier
            grp_sorted = grp.sort_values('confidence_tier')
            n = min(len(grp_sorted), max(per_project, 20))
            sampled_parts.append(grp_sorted.head(n))
        sampled = pd.concat(sampled_parts, ignore_index=True)

        # If over budget, trim proportionally from largest projects
        if len(sampled) > args.max_labels:
            sampled = sampled.sample(n=args.max_labels, random_state=42)

        candidates = sampled
        logger.info(f"Sampled down to {len(candidates)} labels")

    # Prepare output
    out_cols = [
        'project', 'issue_key', 'issue_type', 'design_score',
        'confidence_tier', 'n_reason', 'summary', 'description',
    ]
    output = candidates[out_cols].copy()
    output = output.sort_values(['project', 'issue_key'])

    print(f"\n{'='*60}")
    print("FINAL N-LABEL DISTRIBUTION")
    print(f"{'='*60}")
    final_ct = pd.crosstab(output['project'], output['confidence_tier'], margins=True)
    print(final_ct.to_string())
    print(f"\nTotal: {len(output)} non-design labels")

    # Compare with D labels
    print(f"\n{'='*60}")
    print("COMPARISON WITH D LABELS (per LOPO project)")
    print(f"{'='*60}")
    for p in TSV_PROJECTS:
        n_count = len(output[output['project'] == p])
        print(f"  {p:<15} N={n_count:>4}")

    if not args.dry_run:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(out_path, index=False)
        logger.info(f"Saved {len(output)} N labels to {out_path}")
    else:
        logger.info("Dry run — no file written")


if __name__ == '__main__':
    main()
