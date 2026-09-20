#!/usr/bin/env python3
"""
Evaluate an LLM zero-/few-shot classifier as a baseline against human-labeled
TAWOS ground truth, using the same per-project LOPO folds as eval_lopo.py and
significance_tests.py, so results plug directly into the existing comparison
pipeline (RoBERTa vs SVM vs GB vs LLM).

The LLM is NEVER evaluated against its own or the heuristic's pseudo-labels --
only against ieee_dataport/manual_labels/manually_labelled_issues.csv. This
keeps the baseline comparison free of the label-generation circularity that
would result from scoring an LLM against labels it (or a sibling call) produced.

Usage:
    python eval_llm_baseline.py --provider anthropic --mode zeroshot \
        --out gcp_results/llm_zeroshot_results.json
    python eval_llm_baseline.py --provider openai --mode fewshot --n_shot 5 \
        --out gcp_results/llm_fewshot_results.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from llm_clients import classify_issue_with_retry, estimate_cost_usd, PROVIDERS
from llm_label_tickets import load_few_shot_examples

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MANUAL_LABELS = 'ieee_dataport/manual_labels/manually_labelled_issues.csv'


def evaluate_fold(df, held_out, provider, model, mode, n_shot, fewshot_source, sleep):
    test_df = df[df['project'] == held_out]
    y_true, y_pred, y_score = [], [], []
    rows, failed_keys = [], []
    in_tok = out_tok = 0
    t0 = time.monotonic()

    few_shot_examples = None
    if mode == 'fewshot':
        few_shot_examples = load_few_shot_examples(fewshot_source, held_out, n_shot)

    for _, row in test_df.iterrows():
        try:
            result = classify_issue_with_retry(
                provider=provider,
                model=model,
                issue_type=row.get('issue_type'),
                priority=row.get('priority'),
                summary=row.get('summary'),
                description=row.get('description'),
                few_shot_examples=few_shot_examples,
            )
        except Exception as e:
            logger.error(f"[{held_out}] Giving up on {row['issue_key']} after retries: {e}")
            failed_keys.append(row['issue_key'])
            continue

        in_tok += result.input_tokens
        out_tok += result.output_tokens
        true_label = 1 if row['label'] == 'design' else 0
        pred_label = 1 if result.label == 'design' else 0
        # score = P(design), regardless of which label the model returned
        score = result.confidence if pred_label == 1 else 1 - result.confidence

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_score.append(score)
        rows.append({
            'project': held_out,
            'issue_key': row['issue_key'],
            'issue_type': row.get('issue_type'),
            'label': true_label,
            'pred': pred_label,
            'confidence': result.confidence,
            'prob_design': score,
            'rationale': result.rationale,
        })

        if sleep:
            time.sleep(sleep)

    wall_s = time.monotonic() - t0
    n = len(y_true)
    if n == 0:
        return None, rows

    metrics = {
        'project': held_out,
        'n': n,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float('nan'),
        'input_tokens': in_tok,
        'output_tokens': out_tok,
        'wall_clock_s': wall_s,
        'n_failed': len(failed_keys),
        'failed_issue_keys': failed_keys,
    }
    return metrics, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', required=True, choices=list(PROVIDERS))
    parser.add_argument('--model', default=None)
    parser.add_argument('--mode', default='zeroshot', choices=['zeroshot', 'fewshot'])
    parser.add_argument('--n_shot', type=int, default=5)
    parser.add_argument('--fewshot_source', default=MANUAL_LABELS)
    parser.add_argument('--manual_labels', default=MANUAL_LABELS)
    parser.add_argument('--sleep', type=float, default=0.0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.manual_labels)
    projects = sorted(df['project'].unique())
    logger.info(f"LOPO projects: {projects}")

    per_project, prediction_rows = [], []
    for held_out in projects:
        logger.info(f"Evaluating fold: held-out project = {held_out}")
        metrics, rows = evaluate_fold(
            df, held_out, args.provider, args.model, args.mode,
            args.n_shot, args.fewshot_source, args.sleep,
        )
        prediction_rows.extend(rows)
        if metrics:
            per_project.append(metrics)
            logger.info(f"  {held_out}: F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f} "
                        f"(n={metrics['n']})")

    f1_scores = [p['f1'] for p in per_project]
    auc_scores = [p['auc'] for p in per_project if not np.isnan(p['auc'])]
    total_in = sum(p['input_tokens'] for p in per_project)
    total_out = sum(p['output_tokens'] for p in per_project)
    model_name = args.model or PROVIDERS[args.provider]['default_model']

    output = {
        'method': f'LLM {args.mode} classifier ({args.provider}/{model_name})',
        'provider': args.provider,
        'model': model_name,
        'mode': args.mode,
        'n_shot': args.n_shot if args.mode == 'fewshot' else 0,
        'per_project': per_project,
        'mean_f1': float(np.mean(f1_scores)) if f1_scores else None,
        'std_f1': float(np.std(f1_scores)) if f1_scores else None,
        'mean_auc': float(np.mean(auc_scores)) if auc_scores else None,
        'total_input_tokens': total_in,
        'total_output_tokens': total_out,
        'estimated_cost_usd': estimate_cost_usd(model_name, total_in, total_out),
        'total_wall_clock_s': sum(p['wall_clock_s'] for p in per_project),
        'n_failed': sum(p['n_failed'] for p in per_project),
        # AUC is computed from the model's self-reported confidence, not a
        # calibrated probability -- not directly comparable to encoder AUCs.
        'auc_score_source': 'verbalized_confidence',
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    pred_path = out_path.with_name(out_path.stem + '_predictions.csv')
    pd.DataFrame(prediction_rows).to_csv(pred_path, index=False)
    logger.info(f"Per-ticket predictions saved to {pred_path}")

    logger.info(f"\nMean F1: {output['mean_f1']:.4f} | Mean AUC: {output['mean_auc']:.4f}")
    logger.info(f"Estimated cost: ${output['estimated_cost_usd']:.2f} "
                f"({total_in} in / {total_out} out tokens)")
    logger.info(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
