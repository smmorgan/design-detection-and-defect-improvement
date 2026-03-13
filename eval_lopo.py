#!/usr/bin/env python3
"""
Leave-One-Project-Out (LOPO) cross-validation for the transfer learning model.

For each of the 10 TAWOS projects, trains on the other 9 projects' manually-labelled
data (using the pretrained RoBERTa Stage 1 model as initialization) and evaluates
on the held-out project. This tests true cross-project generalization.

Usage:
    python eval_lopo.py
    python eval_lopo.py --pretrained_model gcp_results/roberta_conservative_0309_0738
    python eval_lopo.py --out lopo_results.json
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────

PRETRAINED_MODEL = 'gcp_results/roberta_conservative_0309_0738'
TAWOS_MANUAL     = './output/all_manually_labelled.csv'
D_LABELS_PATH    = './output/new_d_labels_review.csv'
D_LABELS_SVR     = './output/server_new_d_labels_review.csv'
N_LABELS_PATH    = './output/new_n_labels.csv'
MAX_LENGTH       = 512
BATCH_SIZE       = 16
LR               = 1e-5
EPOCHS           = 5
PATIENCE         = 2
DROPOUT          = 0.1
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.1
RANDOM_SEED      = 42


# ── Dataset ──────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, sample_weights=None):
        self.encodings = tokenizer(
            texts, truncation=True, padding='max_length',
            max_length=max_length, return_tensors='pt',
        )
        self.labels = labels
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.sample_weights is not None:
            item['sample_weight'] = torch.tensor(self.sample_weights[idx], dtype=torch.float)
        return item


# ── Training & evaluation ────────────────────────────────────────────────────

def train_and_evaluate(
    pretrained_path, tokenizer,
    train_texts, train_labels,
    val_texts, val_labels,
    test_texts, test_labels,
    device, fold_name,
    freeze_layers=0,
    train_sample_weights=None,
):
    """Train a fresh model from pretrained weights, return test metrics."""

    # Load fresh model from pretrained checkpoint each fold
    model_config = AutoConfig.from_pretrained(
        pretrained_path, num_labels=2,
    )
    if hasattr(model_config, 'hidden_dropout_prob'):
        model_config.hidden_dropout_prob = DROPOUT
    if hasattr(model_config, 'attention_probs_dropout_prob'):
        model_config.attention_probs_dropout_prob = DROPOUT
    if hasattr(model_config, 'dropout'):
        model_config.dropout = DROPOUT
    if hasattr(model_config, 'attention_dropout'):
        model_config.attention_dropout = DROPOUT

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_path, config=model_config,
    )
    model.to(device)

    # Freeze bottom N transformer layers + embeddings
    if freeze_layers > 0:
        # Freeze embeddings
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False
        # Freeze bottom N encoder layers
        for i in range(min(freeze_layers, len(model.roberta.encoder.layer))):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"  [{fold_name}] Frozen {freeze_layers} layers + embeddings: "
            f"{trainable:,}/{total_params:,} params trainable "
            f"({trainable/total_params*100:.1f}%)"
        )

    # Class weights (inverse frequency)
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    total = len(train_labels)
    w_neg = total / (2.0 * n_neg)
    w_pos = total / (2.0 * n_pos)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float).to(device)
    # Use reduction='none' when we have per-sample weights, so we can apply them
    use_sample_weights = train_sample_weights is not None
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        reduction='none' if use_sample_weights else 'mean',
    )

    # Data loaders
    train_ds = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH,
                           sample_weights=train_sample_weights)
    val_ds   = TextDataset(val_texts,   val_labels,   tokenizer, MAX_LENGTH)
    test_ds  = TextDataset(test_texts,  test_labels,  tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # Optimizer + scheduler (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    best_f1 = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            labs  = batch['labels'].to(device)

            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, labs)
            if use_sample_weights:
                sw = batch['sample_weight'].to(device)
                loss = (loss * sw).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # ── Validate ──
        val_metrics = _evaluate(model, val_loader, device)
        logger.info(
            f"  [{fold_name}] Epoch {epoch}/{EPOCHS}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_F1={val_metrics['f1']:.4f}  val_AUC={val_metrics['auc']:.4f}"
        )

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.info(f"  [{fold_name}] Early stopping at epoch {epoch}")
                break

    # Restore best model and evaluate on held-out test project
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = _evaluate(model, test_loader, device)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return test_metrics


def _evaluate(model, loader, device):
    """Run inference and return metrics dict."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in loader:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            labs  = batch['labels']

            out   = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()

            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labs.tolist())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec  = recall_score(all_labels, all_preds, zero_division=0)
    f1   = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    cm   = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        'accuracy': round(acc, 4), 'precision': round(prec, 4),
        'recall': round(rec, 4), 'f1': round(f1, 4), 'auc': round(auc, 4),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'n': len(all_labels),
        'n_design': int(sum(all_labels)),
        'n_nondesign': int(len(all_labels) - sum(all_labels)),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', default=PRETRAINED_MODEL)
    parser.add_argument('--tawos_manual', default=TAWOS_MANUAL)
    parser.add_argument('--out', default='gcp_results/lopo_results.json')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--augment', action='store_true',
                        help='Add heuristic D and N labels to training data')
    parser.add_argument('--freeze_layers', type=int, default=0,
                        help='Freeze bottom N transformer layers + embeddings (0=none)')
    parser.add_argument('--manual_weight', type=float, default=1.0,
                        help='Weight multiplier for manual labels vs heuristic (e.g. 5.0)')
    parser.add_argument('--aug_max', type=int, default=0,
                        help='Max augmented samples per class (0=unlimited). '
                             'Prioritises highest-confidence labels.')
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load manual labels (used for both train and test)
    df = pd.read_csv(args.tawos_manual)
    projects = sorted(df['project'].unique())
    logger.info(f"Projects: {projects}")
    logger.info(f"Manual samples: {len(df)}")

    # Load heuristic augmentation data if requested
    aug_d = pd.DataFrame()
    aug_n = pd.DataFrame()
    if args.augment:
        # D labels — sort by score (highest confidence first)
        d_parts = []
        if Path(D_LABELS_PATH).exists():
            d_df = pd.read_csv(D_LABELS_PATH)
            d_df['text'] = d_df['summary'].fillna('') + ' [SEP] ' + d_df['description'].fillna('')
            d_df['label'] = 'design'
            d_parts.append(d_df[['project', 'issue_key', 'text', 'label', 'score']])
        if Path(D_LABELS_SVR).exists():
            d_df = pd.read_csv(D_LABELS_SVR)
            d_df['project'] = 'SERVER'
            d_df['text'] = d_df['summary'].fillna('') + ' [SEP] ' + d_df['description'].fillna('')
            d_df['label'] = 'design'
            if 'score' not in d_df.columns:
                d_df['score'] = 5  # minimum threshold score
            d_parts.append(d_df[['project', 'issue_key', 'text', 'label', 'score']])
        if d_parts:
            aug_d = pd.concat(d_parts, ignore_index=True)
            aug_d = aug_d.sort_values('score', ascending=False)
            if args.aug_max > 0:
                aug_d = aug_d.head(args.aug_max)
            logger.info(f"Augmentation D labels: {len(aug_d)}")

        # N labels — sort by confidence_tier (1=highest confidence first)
        if Path(N_LABELS_PATH).exists():
            n_df = pd.read_csv(N_LABELS_PATH)
            n_df['text'] = n_df['summary'].fillna('') + ' [SEP] ' + n_df['description'].fillna('')
            n_df['label'] = 'non-design'
            n_df = n_df.sort_values('confidence_tier', ascending=True)
            if args.aug_max > 0:
                n_df = n_df.head(args.aug_max)
            aug_n = n_df[['project', 'issue_key', 'text', 'label']]
            logger.info(f"Augmentation N labels: {len(aug_n)}")

    # Load tokenizer once (shared across folds)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    results = []

    for held_out in projects:
        logger.info(f"\n{'='*60}")
        logger.info(f"LOPO FOLD: held-out project = {held_out}")
        logger.info(f"{'='*60}")

        # Test set: manual labels from held-out project ONLY
        test_df  = df[df['project'] == held_out]
        test_texts  = test_df['text'].fillna('').tolist()
        test_labels = (test_df['label'] == 'design').astype(int).tolist()

        # Train set: manual labels from other 8 projects (excl val project)
        train_df = df[df['project'] != held_out]

        # Use one of the remaining projects as validation (the smallest one)
        remaining_projects = [p for p in projects if p != held_out]
        val_project = min(remaining_projects, key=lambda p: len(df[df['project'] == p]))
        val_df   = train_df[train_df['project'] == val_project]
        train_df = train_df[train_df['project'] != val_project]

        train_texts  = train_df['text'].fillna('').tolist()
        train_labels = (train_df['label'] == 'design').astype(int).tolist()
        val_texts    = val_df['text'].fillna('').tolist()
        val_labels   = (val_df['label'] == 'design').astype(int).tolist()

        # Build sample weights: manual labels get full weight, heuristic get 1.0
        n_manual = len(train_texts)
        train_sample_weights = [args.manual_weight] * n_manual

        # Add augmentation data (excluding held-out AND val projects)
        if args.augment:
            excl_projects = {held_out, val_project}
            if not aug_d.empty:
                aug_d_fold = aug_d[~aug_d['project'].isin(excl_projects)]
                train_texts  += aug_d_fold['text'].fillna('').tolist()
                train_labels += [1] * len(aug_d_fold)
                train_sample_weights += [1.0] * len(aug_d_fold)
            if not aug_n.empty:
                aug_n_fold = aug_n[~aug_n['project'].isin(excl_projects)]
                train_texts  += aug_n_fold['text'].fillna('').tolist()
                train_labels += [0] * len(aug_n_fold)
                train_sample_weights += [1.0] * len(aug_n_fold)

        # Only pass sample weights if they're not all equal
        use_weights = args.manual_weight != 1.0 and args.augment
        if not use_weights:
            train_sample_weights = None

        n_train_d = sum(train_labels)
        n_test_d  = sum(test_labels)
        aug_info = ''
        if args.augment:
            aug_info = ' + augmented'
            if use_weights:
                aug_info += f' (manual_weight={args.manual_weight})'
        logger.info(
            f"  Train: {len(train_texts)} (D={n_train_d}, N={len(train_texts)-n_train_d}) "
            f"from {len(train_df['project'].unique())} projects"
            + aug_info
        )
        logger.info(f"  Val:   {len(val_texts)} ({val_project})")
        logger.info(
            f"  Test:  {len(test_texts)} (D={n_test_d}, N={len(test_texts)-n_test_d}) "
            f"— {held_out} (manual labels only)"
        )

        metrics = train_and_evaluate(
            args.pretrained_model, tokenizer,
            train_texts, train_labels,
            val_texts, val_labels,
            test_texts, test_labels,
            device, held_out,
            freeze_layers=args.freeze_layers,
            train_sample_weights=train_sample_weights,
        )
        metrics['project'] = held_out
        results.append(metrics)

        logger.info(
            f"  [{held_out}] TEST: F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  "
            f"Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  "
            f"Rec={metrics['recall']:.4f}"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("LEAVE-ONE-PROJECT-OUT RESULTS")
    print(f"{'='*70}")
    hdr = f"  {'Project':<15} {'n':>5} {'D%':>5} {'F1':>7} {'AUC':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'Spec':>7}"
    print(hdr)
    print(f"  {'-'*75}")

    f1s, aucs, accs = [], [], []
    for r in results:
        spec = r['tn'] / (r['tn'] + r['fp']) if (r['tn'] + r['fp']) > 0 else 0
        d_pct = r['n_design'] / r['n'] * 100
        print(
            f"  {r['project']:<15} {r['n']:>5} {d_pct:>4.0f}% "
            f"{r['f1']:>7.4f} {r['auc']:>7.4f} {r['accuracy']:>7.4f} "
            f"{r['precision']:>7.4f} {r['recall']:>7.4f} {spec:>7.4f}"
        )
        f1s.append(r['f1'])
        aucs.append(r['auc'])
        accs.append(r['accuracy'])

    print(f"  {'-'*75}")
    print(
        f"  {'MEAN':<15} {'':>5} {'':>5} "
        f"{np.mean(f1s):>7.4f} {np.mean(aucs):>7.4f} {np.mean(accs):>7.4f}"
    )
    print(
        f"  {'STD':<15} {'':>5} {'':>5} "
        f"{np.std(f1s):>7.4f} {np.std(aucs):>7.4f} {np.std(accs):>7.4f}"
    )

    # Save results
    out_path = Path(args.out)
    summary = {
        'method': 'LOPO (Leave-One-Project-Out)',
        'pretrained_model': args.pretrained_model,
        'data': args.tawos_manual,
        'augmented': args.augment,
        'aug_d_count': len(aug_d) if args.augment else 0,
        'aug_n_count': len(aug_n) if args.augment else 0,
        'n_projects': len(projects),
        'n_total_samples': len(df),
        'hyperparameters': {
            'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
            'batch_size': BATCH_SIZE, 'max_length': MAX_LENGTH,
            'dropout': DROPOUT, 'weight_decay': WEIGHT_DECAY,
            'freeze_layers': args.freeze_layers,
            'manual_weight': args.manual_weight,
            'aug_max': args.aug_max,
        },
        'per_project': results,
        'mean_f1': round(float(np.mean(f1s)), 4),
        'std_f1': round(float(np.std(f1s)), 4),
        'mean_auc': round(float(np.mean(aucs)), 4),
        'std_auc': round(float(np.std(aucs)), 4),
        'mean_accuracy': round(float(np.mean(accs)), 4),
        'std_accuracy': round(float(np.std(accs)), 4),
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
