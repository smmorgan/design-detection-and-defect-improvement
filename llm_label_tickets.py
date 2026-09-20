#!/usr/bin/env python3
"""
Label TAWOS tickets as design/non-design using an LLM (Anthropic or OpenAI),
in place of, or alongside, the keyword-based cherry-picking heuristic in
generate_n_labels.py / LABELLING_METHODOLOGY.md.

Two roles (same underlying classification call, kept distinct for bookkeeping):
  --role baseline  Zero-/few-shot classifier evaluated against human ground truth
                    (see eval_llm_baseline.py). Never used to generate training data.
  --role labeler   Generates pseudo-labels for unlabelled TAWOS tickets, written in
                    the same CSV schema eval_lopo.py already consumes via
                    --aug_source llm.

Output schema matches D_LABELS_PATH / N_LABELS_PATH in eval_lopo.py:
  project, issue_key, issue_type, summary, description, label, score,
  label_source, rationale, model, provider

Usage:
    python llm_label_tickets.py \
        --input ieee_dataport/tawos_project_issues/DM.csv \
        --provider anthropic --role labeler --mode zeroshot \
        --out output/llm_labels_DM.csv

    python llm_label_tickets.py \
        --input ieee_dataport/manual_labels/manually_labelled_issues.csv \
        --provider openai --role baseline --mode fewshot --n_shot 5 \
        --exclude_project DM --out output/llm_baseline_DM.csv
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from llm_clients import classify_issue_with_retry, estimate_cost_usd, PROVIDERS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_few_shot_examples(source_csv, exclude_project, n_shot, seed=42):
    """Draw a balanced few-shot set from human-labeled data, excluding one project
    (kept out to preserve LOPO no-leakage semantics when used as a per-fold baseline)."""
    df = pd.read_csv(source_csv)
    if exclude_project:
        df = df[df['project'] != exclude_project]
    per_class = max(1, n_shot // 2)
    examples = []
    for label in ['design', 'non-design']:
        pool = df[df['label'] == label]
        examples.extend(pool.sample(n=min(per_class, len(pool)), random_state=seed).to_dict('records'))
    return examples[:n_shot]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
                         help='CSV with project, issue_key, issue_type, summary, description[, priority]')
    parser.add_argument('--provider', required=True, choices=list(PROVIDERS))
    parser.add_argument('--model', default=None, help='Overrides provider default_model')
    parser.add_argument('--role', required=True, choices=['baseline', 'labeler'])
    parser.add_argument('--mode', default='zeroshot', choices=['zeroshot', 'fewshot'])
    parser.add_argument('--n_shot', type=int, default=5)
    parser.add_argument('--fewshot_source', default='ieee_dataport/manual_labels/manually_labelled_issues.csv')
    parser.add_argument('--exclude_project', default=None,
                         help='Project to exclude from few-shot examples (the held-out LOPO fold)')
    parser.add_argument('--max_tickets', type=int, default=0, help='0 = all rows in --input')
    parser.add_argument('--sleep', type=float, default=0.0, help='Seconds to sleep between calls (rate limiting)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume', action='store_true',
                         help='Skip issue_keys already present in --out, append new rows')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.max_tickets:
        df = df.head(args.max_tickets)

    done_keys = set()
    if args.resume and Path(args.out).exists():
        done_keys = set(pd.read_csv(args.out)['issue_key'])
        logger.info(f"Resuming: {len(done_keys)} tickets already labeled")

    few_shot_examples = None
    if args.mode == 'fewshot':
        few_shot_examples = load_few_shot_examples(
            args.fewshot_source, args.exclude_project, args.n_shot,
        )
        logger.info(f"Loaded {len(few_shot_examples)} few-shot examples "
                    f"(excluded project={args.exclude_project})")

    rows = []
    total_in_tok = total_out_tok = 0
    t_start = time.monotonic()

    for i, row in df.iterrows():
        key = row['issue_key']
        if key in done_keys:
            continue

        try:
            result = classify_issue(
                provider=args.provider,
                model=args.model,
                issue_type=row.get('issue_type'),
                priority=row.get('priority'),
                summary=row.get('summary'),
                description=row.get('description'),
                few_shot_examples=few_shot_examples,
            )
        except Exception as e:
            logger.error(f"Failed on {key}: {e}")
            continue

        total_in_tok += result.input_tokens
        total_out_tok += result.output_tokens

        rows.append({
            'project': row.get('project'),
            'issue_key': key,
            'issue_type': row.get('issue_type'),
            'summary': row.get('summary'),
            'description': row.get('description'),
            'label': result.label,
            'score': result.confidence,
            'label_source': f'llm_{args.role}',
            'rationale': result.rationale,
            'model': result.model,
            'provider': result.provider,
            'latency_s': result.latency_s,
        })

        if (i + 1) % 25 == 0:
            logger.info(f"Labeled {i + 1}/{len(df)}")
        if args.sleep:
            time.sleep(args.sleep)

    wall_s = time.monotonic() - t_start
    out_df = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out_path.exists() and not out_df.empty:
        out_df = pd.concat([pd.read_csv(out_path), out_df], ignore_index=True)
    out_df.to_csv(out_path, index=False)
    logger.info(f"Wrote {len(out_df)} labels to {out_path}")

    model_name = out_df['model'].iloc[0] if not out_df.empty else (args.model or PROVIDERS[args.provider]['default_model'])
    cost = estimate_cost_usd(model_name, total_in_tok, total_out_tok)
    cost_summary = {
        'provider': args.provider,
        'model': model_name,
        'role': args.role,
        'mode': args.mode,
        'n_tickets': len(rows),
        'input_tokens': total_in_tok,
        'output_tokens': total_out_tok,
        'estimated_cost_usd': cost,
        'wall_clock_s': wall_s,
        'mean_latency_s': wall_s / len(rows) if rows else None,
    }
    cost_path = out_path.with_suffix('.cost.json')
    with open(cost_path, 'w') as f:
        json.dump(cost_summary, f, indent=2)
    logger.info(f"Cost/latency summary written to {cost_path}: {cost_summary}")


if __name__ == '__main__':
    main()
