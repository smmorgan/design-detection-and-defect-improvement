#!/usr/bin/env python3
"""
Evaluate roberta_transfer_all_n on the new heuristic-labelled D and N datasets.

Tests how well the current best transfer model agrees with the keyword heuristic
labels, broken down by project. This validates both the model and the labels
before using them for retraining.

Usage:
    python eval_heuristic_labels.py
    python eval_heuristic_labels.py --model_dir gcp_results/roberta_transfer_all_n_0310
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

MODEL_DIR    = 'gcp_results/roberta_transfer_all_n_0310'
D_LABELS     = 'output/new_d_labels_review.csv'
D_LABELS_SVR = 'output/server_new_d_labels_review.csv'
N_LABELS     = 'output/new_n_labels.csv'
MAX_LENGTH   = 512
BATCH_SIZE   = 32

LOPO_PROJECTS = [
    'CONFSERVER', 'DM', 'DNN', 'FAB', 'JRASERVER',
    'MESOS', 'MULE', 'NEXUS', 'SERVER', 'TIMOB',
]


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, truncation=True, padding='max_length',
            max_length=max_length, return_tensors='pt',
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def predict(model, tokenizer, texts, labels, device, batch_size):
    """Run inference and return predictions + probabilities."""
    dataset = TextDataset(texts, labels, tokenizer, MAX_LENGTH)
    loader  = DataLoader(dataset, batch_size=batch_size)
    all_preds, all_probs = [], []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return all_preds, all_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default=MODEL_DIR)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--out', default='gcp_results/heuristic_eval_results.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load model
    logger.info(f"Loading model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)

    # ── Load D labels ────────────────────────────────────────────────────────
    d_parts = []
    if Path(D_LABELS).exists():
        df = pd.read_csv(D_LABELS)
        df['text'] = df['summary'].fillna('') + ' [SEP] ' + df['description'].fillna('')
        d_parts.append(df)
    if Path(D_LABELS_SVR).exists():
        df = pd.read_csv(D_LABELS_SVR)
        df['project'] = 'SERVER'
        df['text'] = df['summary'].fillna('') + ' [SEP] ' + df['description'].fillna('')
        d_parts.append(df)

    df_d = pd.concat(d_parts, ignore_index=True)
    df_d['true_label'] = 1  # design
    logger.info(f"D labels: {len(df_d)}")

    # ── Load N labels ────────────────────────────────────────────────────────
    df_n = pd.read_csv(N_LABELS)
    df_n['text'] = df_n['summary'].fillna('') + ' [SEP] ' + df_n['description'].fillna('')
    df_n['true_label'] = 0  # non-design
    logger.info(f"N labels: {len(df_n)}")

    # ── Combined evaluation ──────────────────────────────────────────────────
    df_all = pd.concat([
        df_d[['project', 'issue_key', 'text', 'true_label']],
        df_n[['project', 'issue_key', 'text', 'true_label']],
    ], ignore_index=True)

    logger.info(f"Total: {len(df_all)} (D={len(df_d)}, N={len(df_n)})")
    logger.info("Running inference...")

    preds, probs = predict(
        model, tokenizer,
        df_all['text'].tolist(),
        df_all['true_label'].tolist(),
        device, args.batch_size,
    )
    df_all['pred'] = preds
    df_all['prob'] = probs

    # ── Overall metrics ──────────────────────────────────────────────────────
    acc  = accuracy_score(df_all['true_label'], df_all['pred'])
    prec = precision_score(df_all['true_label'], df_all['pred'], zero_division=0)
    rec  = recall_score(df_all['true_label'], df_all['pred'], zero_division=0)
    f1   = f1_score(df_all['true_label'], df_all['pred'], zero_division=0)
    cm   = confusion_matrix(df_all['true_label'], df_all['pred'], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*70}")
    print("OVERALL: Model vs Heuristic Labels")
    print(f"{'='*70}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  D agreement: {tp}/{tp+fn} = {tp/(tp+fn)*100:.1f}% of heuristic D predicted as D")
    print(f"  N agreement: {tn}/{tn+fp} = {tn/(tn+fp)*100:.1f}% of heuristic N predicted as N")

    # ── Per-class breakdown ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("D LABELS: How many heuristic D does the model agree with?")
    print(f"{'='*70}")
    d_correct = df_all[(df_all['true_label'] == 1) & (df_all['pred'] == 1)]
    d_wrong   = df_all[(df_all['true_label'] == 1) & (df_all['pred'] == 0)]
    print(f"  Agreed (predicted D): {len(d_correct)}/{len(df_d)} ({len(d_correct)/len(df_d)*100:.1f}%)")
    print(f"  Disagreed (predicted N): {len(d_wrong)}/{len(df_d)} ({len(d_wrong)/len(df_d)*100:.1f}%)")
    print(f"  Mean prob for correct D: {d_correct['prob'].mean():.4f}")
    print(f"  Mean prob for missed D:  {d_wrong['prob'].mean():.4f}")

    print(f"\n{'='*70}")
    print("N LABELS: How many heuristic N does the model agree with?")
    print(f"{'='*70}")
    n_correct = df_all[(df_all['true_label'] == 0) & (df_all['pred'] == 0)]
    n_wrong   = df_all[(df_all['true_label'] == 0) & (df_all['pred'] == 1)]
    print(f"  Agreed (predicted N): {len(n_correct)}/{len(df_n)} ({len(n_correct)/len(df_n)*100:.1f}%)")
    print(f"  Disagreed (predicted D): {len(n_wrong)}/{len(df_n)} ({len(n_wrong)/len(df_n)*100:.1f}%)")
    print(f"  Mean prob for correct N: {n_correct['prob'].mean():.4f}")
    print(f"  Mean prob for false D:   {n_wrong['prob'].mean():.4f}")

    # ── Per-project breakdown (LOPO projects only) ───────────────────────────
    print(f"\n{'='*70}")
    print("PER-PROJECT BREAKDOWN (LOPO projects)")
    print(f"{'='*70}")
    hdr = f"  {'Project':<15} {'D_agree':>8} {'D_total':>8} {'D%':>6} {'N_agree':>8} {'N_total':>8} {'N%':>6}"
    print(hdr)
    print(f"  {'-'*65}")

    results = {'overall': {
        'accuracy': round(acc, 4), 'precision': round(prec, 4),
        'recall': round(rec, 4), 'f1': round(f1, 4),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'd_agreement': round(tp/(tp+fn), 4),
        'n_agreement': round(tn/(tn+fp), 4),
    }, 'per_project': []}

    for proj in sorted(df_all['project'].unique()):
        proj_df = df_all[df_all['project'] == proj]
        proj_d = proj_df[proj_df['true_label'] == 1]
        proj_n = proj_df[proj_df['true_label'] == 0]

        d_ok = len(proj_d[proj_d['pred'] == 1]) if len(proj_d) > 0 else 0
        n_ok = len(proj_n[proj_n['pred'] == 0]) if len(proj_n) > 0 else 0
        d_pct = d_ok / len(proj_d) * 100 if len(proj_d) > 0 else 0
        n_pct = n_ok / len(proj_n) * 100 if len(proj_n) > 0 else 0

        marker = ' *' if proj in LOPO_PROJECTS else ''
        print(f"  {proj:<15} {d_ok:>8}/{len(proj_d):<6} {d_pct:>5.1f}% {n_ok:>8}/{len(proj_n):<6} {n_pct:>5.1f}%{marker}")

        results['per_project'].append({
            'project': proj, 'd_agree': d_ok, 'd_total': len(proj_d),
            'd_pct': round(d_pct, 1), 'n_agree': n_ok, 'n_total': len(proj_n),
            'n_pct': round(n_pct, 1),
        })

    # Save
    out_path = Path(args.out)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
