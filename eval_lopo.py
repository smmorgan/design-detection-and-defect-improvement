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
TAWOS_UNLABELLED_PATTERN  = './output/tawos_labeled_{project}.csv'
TAWOS_UNLABELLED_PROJECTS = [
    'CONFSERVER', 'DM', 'DNN', 'FAB', 'JRASERVER',
    'MESOS', 'MULE', 'NEXUS', 'TIMOB',
]
MAX_LENGTH       = 512
BATCH_SIZE       = 16
LR               = 1e-5
EPOCHS           = 5
PATIENCE         = 2
DROPOUT          = 0.1
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.1
RANDOM_SEED      = 42


# ── Metadata enrichment ──────────────────────────────────────────────────────

def build_priority_lookup():
    """Build a lookup dict {issue_key: priority} from tawos_labeled_*.csv files."""
    lookup = {}
    for proj in TAWOS_UNLABELLED_PROJECTS:
        fpath = Path(TAWOS_UNLABELLED_PATTERN.format(project=proj))
        if not fpath.exists():
            continue
        part = pd.read_csv(fpath, usecols=['issue_key', 'priority'])
        part = part.dropna(subset=['priority'])
        for _, row in part.iterrows():
            lookup[row['issue_key']] = row['priority']
    logger.info(f"Built priority lookup: {len(lookup)} entries")
    return lookup


def prepend_metadata(texts, issue_types, priorities=None):
    """Prepend structured metadata fields to each text string.

    Format: [issue_type] Bug [priority] High [SEP] original text...
    Fields with missing values are omitted.
    """
    enriched = []
    for i, text in enumerate(texts):
        prefix_parts = []
        if issue_types is not None and i < len(issue_types):
            it = issue_types[i]
            if pd.notna(it) and str(it).strip():
                prefix_parts.append(f'[issue_type] {it}')
        if priorities is not None and i < len(priorities):
            pr = priorities[i]
            if pd.notna(pr) and str(pr).strip():
                prefix_parts.append(f'[priority] {pr}')
        if prefix_parts:
            text = ' '.join(prefix_parts) + ' [SEP] ' + text
        enriched.append(text)
    return enriched


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

    # Class weights (square-root of inverse frequency — softer than full inverse)
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    total = len(train_labels)
    w_neg = np.sqrt(total / (2.0 * n_neg))
    w_pos = np.sqrt(total / (2.0 * n_pos))
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

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Tune threshold on validation set, apply to test
    val_probs, val_labels = _get_probs(model, val_loader, device)
    opt_threshold, opt_val_f1 = _find_optimal_threshold(val_probs, val_labels)
    logger.info(
        f"  [{fold_name}] Optimal threshold: {opt_threshold:.2f} "
        f"(val F1 @ 0.50={f1_score(val_labels, [1 if p>=0.5 else 0 for p in val_probs], zero_division=0):.4f} "
        f"→ val F1 @ {opt_threshold:.2f}={opt_val_f1:.4f})"
    )

    test_metrics = _evaluate(model, test_loader, device, threshold=opt_threshold)
    test_metrics['threshold'] = round(opt_threshold, 4)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return test_metrics


def _get_probs(model, loader, device):
    """Run inference and return (probs, labels) as plain lists."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labs = batch['labels']
            out  = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labs.tolist())
    return all_probs, all_labels


def _find_optimal_threshold(probs, labels, steps=100):
    """Find the threshold in [0.05, 0.95] that maximises F1 on the given set."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, steps):
        preds = [1 if p >= t else 0 for p in probs]
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    return float(best_t), float(best_f1)


def _evaluate(model, loader, device, threshold=0.5):
    """Run inference and return metrics dict."""
    all_probs, all_labels = _get_probs(model, loader, device)
    all_preds = [1 if p >= threshold else 0 for p in all_probs]

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


# ── Stage 2 pseudo-labelling ─────────────────────────────────────────────────

def pseudo_label_tawos(pretrained_path, tokenizer, device, manual_keys):
    """Run Stage 2: pseudo-label unlabelled TAWOS data using the pretrained model.

    Loads all tawos_labeled_*.csv files, runs inference with the pretrained
    Stage 1 model, and returns a DataFrame with predictions and confidences.
    Rows whose issue_key appears in manual_keys are excluded to avoid
    duplicating manually-labelled data.

    Args:
        pretrained_path: Path to pretrained Stage 1 checkpoint.
        tokenizer: Pre-loaded tokenizer (shared with fold training).
        device: torch device.
        manual_keys: Set of issue_keys already in the manual labels.

    Returns:
        DataFrame with columns [project, issue_key, text, predicted_label, confidence].
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: Pseudo-labelling unlabelled TAWOS data")
    logger.info("=" * 60)

    # Load all unlabelled TAWOS data
    frames = []
    for proj in TAWOS_UNLABELLED_PROJECTS:
        fpath = Path(TAWOS_UNLABELLED_PATTERN.format(project=proj))
        if not fpath.exists():
            logger.warning(f"  {proj}: file not found ({fpath}), skipping")
            continue
        part = pd.read_csv(fpath, usecols=['project', 'issue_key', 'text'])
        part = part.dropna(subset=['text'])
        logger.info(f"  {proj}: {len(part)} samples loaded")
        frames.append(part)

    if not frames:
        logger.warning("No unlabelled TAWOS data found")
        return pd.DataFrame()

    tawos_df = pd.concat(frames, ignore_index=True)
    logger.info(f"Total unlabelled TAWOS samples: {len(tawos_df)}")

    # Remove rows that overlap with manual labels
    before = len(tawos_df)
    tawos_df = tawos_df[~tawos_df['issue_key'].isin(manual_keys)]
    removed = before - len(tawos_df)
    if removed > 0:
        logger.info(f"Removed {removed} samples overlapping with manual labels")
    logger.info(f"Samples for pseudo-labelling: {len(tawos_df)}")

    # Load pretrained model for inference
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_path, num_labels=2,
    )
    model.to(device)
    model.eval()

    texts = tawos_df['text'].fillna('').tolist()
    dummy_labels = [0] * len(texts)
    ds = TextDataset(texts, dummy_labels, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_confs = []

    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1)  # [batch, 2]
            # Predicted class and confidence in that class
            preds = torch.argmax(probs, dim=1)
            confs = probs.max(dim=1).values
            all_preds.extend(preds.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())

    # Free model
    del model
    torch.cuda.empty_cache()

    tawos_df = tawos_df.copy()
    tawos_df['predicted_label'] = all_preds
    tawos_df['confidence'] = all_confs

    n_design = sum(all_preds)
    logger.info(f"Pseudo-labelling complete:")
    logger.info(f"  Design predicted: {n_design} ({n_design/len(all_preds)*100:.1f}%)")
    logger.info(f"  Non-design predicted: {len(all_preds)-n_design} ({(len(all_preds)-n_design)/len(all_preds)*100:.1f}%)")
    logger.info(f"  Mean confidence: {np.mean(all_confs):.4f}")

    return tawos_df


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
    parser.add_argument('--pseudo_label', action='store_true',
                        help='Enable Stage 2 pseudo-labelling: use pretrained model to '
                             'label unlabelled TAWOS data, then add high-confidence '
                             'pseudo-labels to training.')
    parser.add_argument('--pseudo_threshold', type=float, default=0.85,
                        help='Confidence threshold for including pseudo-labelled samples '
                             '(default: 0.85)')
    parser.add_argument('--pseudo_weight', type=float, default=1.0,
                        help='Sample weight for pseudo-labelled samples (default: 1.0)')
    parser.add_argument('--pseudo_max', type=int, default=0,
                        help='Max pseudo-labelled samples per class (0=unlimited). '
                             'Balances D and N by taking top-confidence from each class.')
    parser.add_argument('--metadata', action='store_true',
                        help='Prepend issue metadata (issue_type, priority) to text input. '
                             'Format: [issue_type] X [priority] Y [SEP] text')
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

    # Build priority lookup from tawos_labeled files if metadata requested
    priority_lookup = {}
    if args.metadata:
        priority_lookup = build_priority_lookup()
        df['priority'] = df['issue_key'].map(priority_lookup)
        logger.info(
            f"Metadata enrichment enabled: "
            f"{df['issue_type'].notna().sum()} issue_types, "
            f"{df['priority'].notna().sum()} priorities matched"
        )

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
            d_parts.append(d_df[['project', 'issue_key', 'issue_type', 'text', 'label', 'score']])
        if Path(D_LABELS_SVR).exists():
            d_df = pd.read_csv(D_LABELS_SVR)
            d_df['project'] = 'SERVER'
            d_df['text'] = d_df['summary'].fillna('') + ' [SEP] ' + d_df['description'].fillna('')
            d_df['label'] = 'design'
            if 'score' not in d_df.columns:
                d_df['score'] = 5  # minimum threshold score
            d_parts.append(d_df[['project', 'issue_key', 'issue_type', 'text', 'label', 'score']])
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
            aug_n = n_df[['project', 'issue_key', 'issue_type', 'text', 'label']]
            logger.info(f"Augmentation N labels: {len(aug_n)}")

        # Add priority from lookup for augmentation data
        if args.metadata and priority_lookup:
            if not aug_d.empty:
                aug_d['priority'] = aug_d['issue_key'].map(priority_lookup)
            if not aug_n.empty:
                aug_n['priority'] = aug_n['issue_key'].map(priority_lookup)

    # Load tokenizer once (shared across folds)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    # Stage 2: pseudo-label unlabelled TAWOS data (once, before folds)
    pseudo_df = pd.DataFrame()
    if args.pseudo_label:
        manual_keys = set(df['issue_key'].tolist())
        pseudo_df = pseudo_label_tawos(
            args.pretrained_model, tokenizer, device, manual_keys,
        )
        if not pseudo_df.empty:
            before = len(pseudo_df)
            pseudo_df = pseudo_df[pseudo_df['confidence'] >= args.pseudo_threshold]
            logger.info(
                f"Filtered to {len(pseudo_df)}/{before} samples with "
                f"confidence >= {args.pseudo_threshold}"
            )
            # Balance classes by taking top-confidence from each
            if args.pseudo_max > 0 and not pseudo_df.empty:
                pseudo_df = pseudo_df.sort_values('confidence', ascending=False)
                d_mask = pseudo_df['predicted_label'] == 1
                pseudo_d = pseudo_df[d_mask].head(args.pseudo_max)
                pseudo_n = pseudo_df[~d_mask].head(args.pseudo_max)
                pseudo_df = pd.concat([pseudo_d, pseudo_n], ignore_index=True)
                logger.info(
                    f"Balanced to {args.pseudo_max} per class: "
                    f"{len(pseudo_d)} D + {len(pseudo_n)} N"
                )
            n_d = (pseudo_df['predicted_label'] == 1).sum()
            n_n = (pseudo_df['predicted_label'] == 0).sum()
            logger.info(f"  Design: {n_d}, Non-design: {n_n}")

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

        # Use the project with the most balanced class distribution as validation
        remaining_projects = [p for p in projects if p != held_out]
        def class_balance(p):
            sub = df[df['project'] == p]
            ratio = (sub['label'] == 'design').mean()
            return abs(ratio - 0.5)  # closer to 0 = more balanced
        val_project = min(remaining_projects, key=class_balance)
        val_df   = train_df[train_df['project'] == val_project]
        train_df = train_df[train_df['project'] != val_project]

        train_texts  = train_df['text'].fillna('').tolist()
        train_labels = (train_df['label'] == 'design').astype(int).tolist()
        val_texts    = val_df['text'].fillna('').tolist()
        val_labels   = (val_df['label'] == 'design').astype(int).tolist()

        # Prepend metadata to text inputs if requested
        if args.metadata:
            train_texts = prepend_metadata(
                train_texts,
                train_df['issue_type'].tolist(),
                train_df['priority'].tolist() if 'priority' in train_df.columns else None,
            )
            val_texts = prepend_metadata(
                val_texts,
                val_df['issue_type'].tolist(),
                val_df['priority'].tolist() if 'priority' in val_df.columns else None,
            )
            test_texts = prepend_metadata(
                test_texts,
                test_df['issue_type'].tolist(),
                test_df['priority'].tolist() if 'priority' in test_df.columns else None,
            )

        # Build sample weights: manual labels get full weight, heuristic get 1.0
        n_manual = len(train_texts)
        train_sample_weights = [args.manual_weight] * n_manual

        # Add augmentation data (excluding held-out AND val projects)
        if args.augment:
            excl_projects = {held_out, val_project}
            if not aug_d.empty:
                aug_d_fold = aug_d[~aug_d['project'].isin(excl_projects)]
                aug_d_texts = aug_d_fold['text'].fillna('').tolist()
                if args.metadata:
                    aug_d_texts = prepend_metadata(
                        aug_d_texts,
                        aug_d_fold['issue_type'].tolist(),
                        aug_d_fold['priority'].tolist() if 'priority' in aug_d_fold.columns else None,
                    )
                train_texts  += aug_d_texts
                train_labels += [1] * len(aug_d_fold)
                train_sample_weights += [1.0] * len(aug_d_fold)
            if not aug_n.empty:
                aug_n_fold = aug_n[~aug_n['project'].isin(excl_projects)]
                aug_n_texts = aug_n_fold['text'].fillna('').tolist()
                if args.metadata:
                    aug_n_texts = prepend_metadata(
                        aug_n_texts,
                        aug_n_fold['issue_type'].tolist(),
                        aug_n_fold['priority'].tolist() if 'priority' in aug_n_fold.columns else None,
                    )
                train_texts  += aug_n_texts
                train_labels += [0] * len(aug_n_fold)
                train_sample_weights += [1.0] * len(aug_n_fold)

        # Add pseudo-labelled data (excluding held-out AND val projects)
        if args.pseudo_label and not pseudo_df.empty:
            excl_projects = {held_out, val_project}
            pseudo_fold = pseudo_df[~pseudo_df['project'].isin(excl_projects)]
            train_texts  += pseudo_fold['text'].fillna('').tolist()
            train_labels += pseudo_fold['predicted_label'].tolist()
            train_sample_weights += [args.pseudo_weight] * len(pseudo_fold)

        # Only pass sample weights if they're not all equal
        use_weights = (
            (args.augment and args.manual_weight != 1.0) or
            (args.pseudo_label and args.pseudo_weight != 1.0)
        )
        if not use_weights:
            train_sample_weights = None

        n_train_d = sum(train_labels)
        n_test_d  = sum(test_labels)
        aug_info = ''
        if args.augment:
            aug_info = ' + augmented'
            if use_weights and args.manual_weight != 1.0:
                aug_info += f' (manual_weight={args.manual_weight})'
        if args.pseudo_label and not pseudo_df.empty:
            aug_info += f' + {len(pseudo_fold)} pseudo-labelled'
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
            f"Rec={metrics['recall']:.4f}  Threshold={metrics.get('threshold', 0.5):.2f}"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("LEAVE-ONE-PROJECT-OUT RESULTS")
    print(f"{'='*70}")
    hdr = f"  {'Project':<15} {'n':>5} {'D%':>5} {'F1':>7} {'AUC':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'Spec':>7} {'Thresh':>7}"
    print(hdr)
    print(f"  {'-'*83}")

    f1s, aucs, accs = [], [], []
    for r in results:
        spec = r['tn'] / (r['tn'] + r['fp']) if (r['tn'] + r['fp']) > 0 else 0
        d_pct = r['n_design'] / r['n'] * 100
        print(
            f"  {r['project']:<15} {r['n']:>5} {d_pct:>4.0f}% "
            f"{r['f1']:>7.4f} {r['auc']:>7.4f} {r['accuracy']:>7.4f} "
            f"{r['precision']:>7.4f} {r['recall']:>7.4f} {spec:>7.4f} "
            f"{r.get('threshold', 0.5):>7.2f}"
        )
        f1s.append(r['f1'])
        aucs.append(r['auc'])
        accs.append(r['accuracy'])

    print(f"  {'-'*83}")
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
    method = 'LOPO (Leave-One-Project-Out)'
    if args.metadata:
        method += ' + metadata (issue_type, priority)'
    if args.pseudo_label:
        method += ' + Stage 2 pseudo-labels'
    summary = {
        'method': method,
        'metadata': args.metadata,
        'pretrained_model': args.pretrained_model,
        'data': args.tawos_manual,
        'augmented': args.augment,
        'aug_d_count': len(aug_d) if args.augment else 0,
        'aug_n_count': len(aug_n) if args.augment else 0,
        'pseudo_label': args.pseudo_label,
        'pseudo_threshold': args.pseudo_threshold if args.pseudo_label else None,
        'pseudo_total_above_threshold': len(pseudo_df) if args.pseudo_label else 0,
        'pseudo_design_count': int((pseudo_df['predicted_label'] == 1).sum()) if args.pseudo_label and not pseudo_df.empty else 0,
        'pseudo_nondesign_count': int((pseudo_df['predicted_label'] == 0).sum()) if args.pseudo_label and not pseudo_df.empty else 0,
        'n_projects': len(projects),
        'n_total_samples': len(df),
        'hyperparameters': {
            'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
            'batch_size': BATCH_SIZE, 'max_length': MAX_LENGTH,
            'dropout': DROPOUT, 'weight_decay': WEIGHT_DECAY,
            'freeze_layers': args.freeze_layers,
            'manual_weight': args.manual_weight,
            'aug_max': args.aug_max,
            'pseudo_weight': args.pseudo_weight if args.pseudo_label else None,
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
