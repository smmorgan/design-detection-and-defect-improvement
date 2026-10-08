#!/usr/bin/env python3
"""
RQ2 plan step 2: freeze the label generator and emit a versioned label set.

Per SEED_VARIANCE_FINDINGS.md, Base+Meta (metadata, inverse class weights, fixed
0.5 threshold, balanced validation project) beats the published
Base+Meta+SqrtW and per-fold threshold tuning buys nothing -- so that is the
config frozen here, matching run_seed_sweep.sh's "base_meta" entry exactly.

For each of the 5 seeds (42-46) and each of the 10 LOPO folds, trains a fresh
model on the other 9 projects' manual labels (identical to eval_lopo.py
--metadata --class_weighting inverse --threshold fixed --val_selection
balance) and then scores the held-out project's FULL issue population
(data/tawos_full/<PROJECT>.csv, pulled uncapped from the live TAWOS DB --
see COMPONENT_SCHEMA_FINDINGS.md) rather than just its ~180 manually-labelled
tickets. Model weights are never saved (they weren't for the original seed
sweep either); only the resulting probabilities are kept.

Output: gcp_results/rq2_labels/<PROJECT>_seed<seed>_full_predictions.csv
        (issue_key, prob_design), one file per (project, seed) pair.
The ensembling step (average across 5 seeds, threshold at 0.5) is separate --
see ensemble_versioned_labels.py -- so a crashed/interrupted run here can
resume without re-doing completed (project, seed) pairs.
"""

import argparse
import copy
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
    get_linear_schedule_with_warmup, set_seed,
)
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PRETRAINED_MODEL = 'gcp_results/roberta_conservative_0309_0738'
TAWOS_MANUAL = './output/all_manually_labelled.csv'
FULL_POP_DIR = Path('data/tawos_full')
OUT_DIR = Path('gcp_results/rq2_labels')
SEEDS = [42, 43, 44, 45, 46]

MAX_LENGTH = 512
BATCH_SIZE = 16
INFERENCE_BATCH_SIZE = 64  # only affects speed, not results -- no gradients held
LR = 1e-5
EPOCHS = 5
PATIENCE = 2
DROPOUT = 0.1
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1


def prepend_metadata(texts, issue_types, priorities):
    enriched = []
    for text, it, pr in zip(texts, issue_types, priorities):
        prefix_parts = []
        if pd.notna(it) and str(it).strip():
            prefix_parts.append(f'[issue_type] {it}')
        if pd.notna(pr) and str(pr).strip():
            prefix_parts.append(f'[priority] {pr}')
        if prefix_parts:
            text = ' '.join(prefix_parts) + ' [SEP] ' + text
        enriched.append(text)
    return enriched


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
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_probs(model, loader, device, autocast=False):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=autocast):
                out = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits.float(), dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
    return all_probs


def find_optimal_threshold(probs, labels, steps=100):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, steps):
        preds = [1 if p >= t else 0 for p in probs]
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    return float(best_t), float(best_f1)


def train_fold(pretrained_path, tokenizer, train_texts, train_labels,
                val_texts, val_labels, device, fold_name):
    """Base+Meta training: inverse class weights, fixed 0.5 threshold at test time."""
    model_config = AutoConfig.from_pretrained(pretrained_path, num_labels=2)
    for attr in ('hidden_dropout_prob', 'attention_probs_dropout_prob', 'dropout', 'attention_dropout'):
        if hasattr(model_config, attr):
            setattr(model_config, attr, DROPOUT)
    if hasattr(model_config, 'reference_compile'):
        model_config.reference_compile = False

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_path, config=model_config)
    model.to(device)

    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    total = len(train_labels)
    class_weights = torch.tensor(
        [total / (2.0 * n_neg), total / (2.0 * n_pos)], dtype=torch.float,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    train_ds = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_ds = TextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=INFERENCE_BATCH_SIZE, shuffle=False)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1, best_state, no_improve = 0.0, None, 0
    t0 = time.monotonic()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labs = batch['labels'].to(device)
            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        val_probs = get_probs(model, val_loader, device)
        val_preds = [1 if p >= 0.5 else 0 for p in val_probs]
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            val_auc = 0.0
        logger.info(
            f"  [{fold_name}] epoch {epoch}/{EPOCHS} loss={epoch_loss/len(train_loader):.4f} "
            f"val_F1={val_f1:.4f} val_AUC={val_auc:.4f}"
        )
        if val_f1 > best_f1:
            best_f1, best_state, no_improve = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.info(f"  [{fold_name}] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"  [{fold_name}] trained in {time.monotonic()-t0:.1f}s, best val F1={best_f1:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--projects', nargs='+', default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    df = pd.read_csv(TAWOS_MANUAL)
    all_projects = sorted(df['project'].unique())
    held_out_projects = args.projects or all_projects
    logger.info(f"All projects: {all_projects}")
    logger.info(f"Held-out folds to run: {held_out_projects}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

    def class_balance(p):
        sub = df[df['project'] == p]
        return abs((sub['label'] == 'design').mean() - 0.5)

    for seed in args.seeds:
        for held_out in held_out_projects:
            out_path = OUT_DIR / f'{held_out}_seed{seed}_full_predictions.csv'
            if out_path.exists():
                logger.info(f"skip {held_out} seed={seed} (exists)")
                continue

            set_seed(seed)
            logger.info(f"\n{'='*60}\nseed={seed} held_out={held_out}\n{'='*60}")

            train_df = df[df['project'] != held_out]
            remaining = [p for p in all_projects if p != held_out]
            val_project = min(remaining, key=class_balance)
            val_df = train_df[train_df['project'] == val_project]
            train_df = train_df[train_df['project'] != val_project]

            train_texts = prepend_metadata(
                train_df['text'].fillna('').tolist(),
                train_df['issue_type'].tolist(),
                train_df['priority'] if 'priority' in train_df.columns else [None] * len(train_df),
            )
            train_labels = (train_df['label'] == 'design').astype(int).tolist()
            val_texts = prepend_metadata(
                val_df['text'].fillna('').tolist(),
                val_df['issue_type'].tolist(),
                val_df['priority'] if 'priority' in val_df.columns else [None] * len(val_df),
            )
            val_labels = (val_df['label'] == 'design').astype(int).tolist()

            model = train_fold(
                PRETRAINED_MODEL, tokenizer, train_texts, train_labels,
                val_texts, val_labels, device, f"seed{seed}/{held_out}",
            )

            full_df = pd.read_csv(FULL_POP_DIR / f'{held_out}.csv')
            full_texts = (
                full_df['summary'].fillna('') + ' [SEP] ' + full_df['description'].fillna('')
            ).tolist()
            full_texts = prepend_metadata(
                full_texts, full_df['issue_type'].tolist(), full_df['priority'].tolist(),
            )

            t0 = time.monotonic()
            full_ds = TextDataset(full_texts, [0] * len(full_texts), tokenizer, MAX_LENGTH)
            full_loader = DataLoader(full_ds, batch_size=INFERENCE_BATCH_SIZE, shuffle=False)
            full_probs = get_probs(model, full_loader, device, autocast=True)
            logger.info(
                f"  [{held_out}] scored {len(full_texts)} full-population issues in "
                f"{time.monotonic()-t0:.1f}s"
            )

            out_df = pd.DataFrame({
                'issue_key': full_df['issue_key'],
                'prob_design': [round(p, 6) for p in full_probs],
            })
            out_df.to_csv(out_path, index=False)
            logger.info(f"  saved {out_path}")

            del model
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
