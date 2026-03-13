#!/usr/bin/env python3
"""
Honest evaluation of the transfer-learning model on held-out Stack Overflow data.

The SO test set (30K samples) was never seen during Stage 3 TAWOS fine-tuning,
making it an unbiased measure of whether transfer learning helped or hurt
generalization compared to the Stage 1 SO-only baseline.

Also evaluates on all_manually_labelled.csv (in-sample TAWOS data) for comparison,
so we can directly see the inflation gap between the two.

Usage:
    python eval_transfer_model.py
    python eval_transfer_model.py --model_dir gcp_results/bert_conservative_transfer_0308_1515
    python eval_transfer_model.py --max_samples 5000   # faster smoke-test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ── Reproduce the exact Stage 1 train/val/test split ─────────────────────────

TRAIN_SPLIT  = 0.70
RANDOM_SEED  = 42
MAX_LENGTH   = 512
BATCH_SIZE   = 32   # inference only — larger is fine
SO_DATA_PATH = './data/data/train_data/raw/combined.csv'
TAWOS_MANUAL = './output/all_manually_labelled.csv'


# ── Dataset ───────────────────────────────────────────────────────────────────

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


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, texts, labels, desc, batch_size, device):
    dataset    = TextDataset(texts, labels, tokenizer, MAX_LENGTH)
    loader     = DataLoader(dataset, batch_size=batch_size)
    all_preds  = []
    all_probs  = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            out   = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    acc  = accuracy_score(labels, all_preds)
    prec = precision_score(labels, all_preds, zero_division=0)
    rec  = recall_score(labels, all_preds, zero_division=0)
    f1   = f1_score(labels, all_preds, zero_division=0)
    auc  = roc_auc_score(labels, all_probs)
    cm   = confusion_matrix(labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    pos_rate = sum(labels) / len(labels)

    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  n={len(labels):,}  positive_rate={pos_rate:.1%}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")

    return {
        'dataset': desc, 'n': len(labels), 'positive_rate': round(pos_rate, 4),
        'accuracy': round(acc, 4), 'precision': round(prec, 4),
        'recall': round(rec, 4), 'f1': round(f1, 4), 'auc': round(auc, 4),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_so_test_split(so_path, max_samples=None):
    """Reproduce the exact Stage 1 70/15/15 stratified test split."""
    print(f"Loading SO data from {so_path} ...")
    df = pd.read_csv(so_path)

    if max_samples and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=RANDOM_SEED)

    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    text_col  = 'text'  if 'text'  in df.columns else df.columns[0]

    texts  = df[text_col].fillna('').tolist()
    labels = (df[label_col] == 'design').astype(int).tolist()

    _, temp_t, _, temp_l = train_test_split(
        texts, labels,
        test_size=(1 - TRAIN_SPLIT),
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    _, test_t, _, test_l = train_test_split(
        temp_t, temp_l,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_l,
    )
    print(f"  SO test split: {len(test_t):,} samples")
    return test_t, test_l


def load_tawos_manual(path):
    """Load the human-labelled TAWOS set (in-sample for Stage 3)."""
    print(f"Loading TAWOS manually-labelled data from {path} ...")
    df = pd.read_csv(path)
    texts  = df['text'].fillna('').tolist()
    labels = (df['label'] == 'design').astype(int).tolist()
    print(f"  TAWOS manual: {len(texts):,} samples")
    return texts, labels


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='gcp_results/bert_conservative_transfer_0308_1515',
                        help='Path to saved model directory')
    parser.add_argument('--so_data', default=SO_DATA_PATH,
                        help='Path to combined.csv')
    parser.add_argument('--tawos_manual', default=TAWOS_MANUAL,
                        help='Path to all_manually_labelled.csv')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Cap SO dataset size for a faster smoke-test')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--out', default=None,
                        help='Optional JSON path to save results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: model directory not found: {model_dir}")
        sys.exit(1)

    print(f"\nLoading model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    print("  Model loaded.")

    results = []

    # ── Test 1: SO held-out test set (out-of-sample for Stage 3) ─────────────
    if Path(args.so_data).exists():
        so_texts, so_labels = load_so_test_split(args.so_data, args.max_samples)
        r = evaluate(model, tokenizer, so_texts, so_labels,
                     'SO test set (held-out, balanced — Stage 3 never saw this)',
                     args.batch_size, device)
        results.append(r)
    else:
        print(f"WARNING: SO data not found at {args.so_data}, skipping.")

    # ── Test 2: TAWOS manually-labelled (in-sample for Stage 3) ─────────────
    if Path(args.tawos_manual).exists():
        tawos_texts, tawos_labels = load_tawos_manual(args.tawos_manual)
        r = evaluate(model, tokenizer, tawos_texts, tawos_labels,
                     'TAWOS manually-labelled (in-sample for Stage 3)',
                     args.batch_size, device)
        results.append(r)
    else:
        print(f"WARNING: TAWOS manual data not found at {args.tawos_manual}, skipping.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    hdr = f"  {'Dataset':<45} {'F1':>6}  {'AUC':>6}  {'Acc':>6}  pos%"
    print(hdr)
    print(f"  {'-'*65}")
    for r in results:
        name = r['dataset'][:45]
        print(f"  {name:<45} {r['f1']:>6.4f}  {r['auc']:>6.4f}  {r['accuracy']:>6.4f}  {r['positive_rate']:.1%}")

    # Reference numbers from training log
    print(f"\n  Stage 1 SO val  (best epoch 3, in-training):   F1=0.9037  AUC=0.9635")
    print(f"  Stage 3 TAWOS val (best epoch 2, in-training): F1=0.9177  AUC=0.9939")

    if args.out:
        out_path = Path(args.out)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
