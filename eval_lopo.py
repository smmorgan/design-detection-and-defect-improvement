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
import time
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
    set_seed,
)
from torch.optim import AdamW

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = TaskType = get_peft_model = None

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
LLM_D_LABELS_PATH = './output/llm_d_labels.csv'
LLM_N_LABELS_PATH = './output/llm_n_labels.csv'
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
    use_lora=False,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    batch_size=BATCH_SIZE,
    gradient_checkpointing=False,
    class_weighting='sqrt',
    tune_threshold=True,
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
    # Decoder models (Qwen/Llama/Gemma) pool the logit off the last
    # non-pad token and raise ValueError for batch_size != 1 if the config
    # has no pad_token_id -- the tokenizer usually defines one (often ==
    # eos) but AutoConfig doesn't pick it up automatically.
    if getattr(model_config, 'pad_token_id', None) is None and tokenizer.pad_token_id is not None:
        model_config.pad_token_id = tokenizer.pad_token_id
    # ModernBERT self-enables torch.compile (reference_compile=None ->
    # is_triton_available()) by default. With per-batch dynamic padding,
    # this recompiles/re-autotunes for every new sequence-length shape,
    # which blew the 12GB budget well before steady-state runtime memory
    # was even reached -- disable it, LOPO folds are too small to benefit
    # from compilation anyway.
    if hasattr(model_config, 'reference_compile'):
        model_config.reference_compile = False

    if use_lora:
        if LoraConfig is None:
            raise ImportError("peft is required for --lora (pip install peft)")
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_path, config=model_config, torch_dtype=torch.bfloat16,
        )
        model.to(device)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_path, config=model_config,
        )
        model.to(device)

    if gradient_checkpointing:
        # Trades ~20-30% step time for a large activation-memory cut, letting
        # ModernBERT (no KV-cache reuse to fall back on like decoder models)
        # run LOPO at the same batch_size as RoBERTa's best config instead of
        # a memory-constrained smaller one -- keeps the model-family
        # comparison apples-to-apples on batch_size.
        model.gradient_checkpointing_enable()
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
        if use_lora:
            # With LoRA, only the adapter weights require grad -- the input
            # embeddings don't. Reentrant-autograd checkpointing needs at
            # least one requires_grad=True tensor flowing into a checkpointed
            # block or it silently returns None gradients for everything
            # inside it (confirmed via a Qwen2.5+LoRA smoke test: this call
            # is what turns a train_loss that never moves back into real
            # updates). enable_input_require_grads() hooks the embedding
            # output to force this.
            model.enable_input_require_grads()

    # Freeze bottom N transformer layers + embeddings. Layer-list location
    # varies by architecture (BERT/RoBERTa: base.encoder.layer, DistilBERT:
    # base.transformer.layer, ModernBERT: base.layers directly), so look it
    # up generically off model.base_model instead of assuming RoBERTa.
    # Skipped under LoRA -- the adapter wrapping already handles which
    # parameters are trainable.
    if freeze_layers > 0 and not use_lora:
        base = model.base_model
        for param in base.embeddings.parameters():
            param.requires_grad = False
        if hasattr(base, 'encoder') and hasattr(base.encoder, 'layer'):
            encoder_layers = base.encoder.layer
        elif hasattr(base, 'transformer') and hasattr(base.transformer, 'layer'):
            encoder_layers = base.transformer.layer
        elif hasattr(base, 'layers'):
            encoder_layers = base.layers
        else:
            raise AttributeError(
                f"Don't know where to find transformer layers on {type(base).__name__} "
                "for freeze_layers -- add a case above for this architecture."
            )
        for i in range(min(freeze_layers, len(encoder_layers))):
            for param in encoder_layers[i].parameters():
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"  [{fold_name}] Frozen {freeze_layers} layers + embeddings: "
            f"{trainable:,}/{total_params:,} params trainable "
            f"({trainable/total_params*100:.1f}%)"
        )

    # Class weights: inverse frequency, or its square root (softer) by default
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    total = len(train_labels)
    w_neg = total / (2.0 * n_neg)
    w_pos = total / (2.0 * n_pos)
    if class_weighting == 'sqrt':
        w_neg, w_pos = np.sqrt(w_neg), np.sqrt(w_pos)
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

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
    train_t0 = time.monotonic()

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
            # bf16 logits (LoRA models load in torch.bfloat16) vs. the fp32
            # class-weight tensor raise a dtype mismatch in F.cross_entropy --
            # upcast for the loss only, standard mixed-precision practice.
            loss = loss_fn(out.logits.float(), labs)
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

    # Includes per-epoch validation passes -- this is the fold's full fine-tuning cost.
    train_wall_s = time.monotonic() - train_t0

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Tune threshold on validation set, apply to test (or keep 0.5 if disabled)
    opt_threshold = 0.5
    if tune_threshold:
        val_probs, val_labels = _get_probs(model, val_loader, device)
        opt_threshold, opt_val_f1 = _find_optimal_threshold(val_probs, val_labels)
        logger.info(
            f"  [{fold_name}] Optimal threshold: {opt_threshold:.2f} "
            f"(val F1 @ 0.50={f1_score(val_labels, [1 if p>=0.5 else 0 for p in val_probs], zero_division=0):.4f} "
            f"→ val F1 @ {opt_threshold:.2f}={opt_val_f1:.4f})"
        )

    infer_t0 = time.monotonic()
    test_metrics = _evaluate(model, test_loader, device, threshold=opt_threshold,
                             return_probs=True)
    infer_wall_s = time.monotonic() - infer_t0
    test_metrics['threshold'] = round(opt_threshold, 4)
    test_metrics['epochs_run'] = epoch
    test_metrics['train_wall_clock_s'] = round(train_wall_s, 3)
    test_metrics['test_inference_wall_clock_s'] = round(infer_wall_s, 3)
    test_metrics['test_inference_s_per_ticket'] = round(infer_wall_s / max(1, test_metrics['n']), 5)

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


def _evaluate(model, loader, device, threshold=0.5, return_probs=False):
    """Run inference and return metrics dict (plus raw 'probs' if return_probs)."""
    all_probs, all_labels = _get_probs(model, loader, device)
    metrics = _metrics_from_probs(all_probs, all_labels, threshold)
    if return_probs:
        metrics['probs'] = all_probs
    return metrics


def _metrics_from_probs(all_probs, all_labels, threshold=0.5):
    """Compute the fold metrics dict from stored probabilities.

    Split out of _evaluate so --threshold pooled can re-score a fold after the
    fact, once the pooled threshold is known, without re-running inference.
    """
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

    metrics = {
        'accuracy': round(acc, 4), 'precision': round(prec, 4),
        'recall': round(rec, 4), 'f1': round(f1, 4), 'auc': round(auc, 4),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'n': len(all_labels),
        'n_design': int(sum(all_labels)),
        'n_nondesign': int(len(all_labels) - sum(all_labels)),
    }
    return metrics


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
                        help='Add augmentation D and N labels to training data (source set by --aug_source)')
    parser.add_argument('--aug_source', default='heuristic', choices=['heuristic', 'llm', 'both'],
                        help='Augmentation label source: keyword heuristic (default), '
                             'LLM labeler (llm_label_tickets.py --role labeler output), or both '
                             '(each source capped at aug_max/2 to avoid scale mismatch between '
                             'heuristic point scores and LLM confidence)')
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
    parser.add_argument('--lora', action='store_true',
                        help='Fine-tune with LoRA adapters instead of full fine-tuning '
                             '(required for decoder models like Qwen2.5 that don\'t fit '
                             'full fp32 fine-tuning on this GPU)')
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Trade compute for activation memory to allow a larger '
                             'batch_size on memory-constrained models (e.g. ModernBERT)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Seeds python/numpy/torch RNGs. Run several seeds to '
                             'measure run-to-run variance -- GPU kernels are not '
                             'deterministic, so one seed does not pin the result.')
    parser.add_argument('--class_weighting', default='sqrt', choices=['sqrt', 'inverse'],
                        help='sqrt of inverse-frequency class weights (SqrtW configs) '
                             'or plain inverse frequency (original Baseline / Base+Meta)')
    parser.add_argument('--threshold', default='tuned', choices=['tuned', 'fixed', 'pooled'],
                        help='tuned: pick the F1-maximising threshold on the val project '
                             '(~177 tickets -- a noisy optimum); fixed: use 0.5 (original '
                             'Baseline / Base+Meta); pooled: tune on the other folds\' '
                             'out-of-fold predictions (~1.6k tickets, far more stable). '
                             'pooled never sees the held-out project\'s labels and costs '
                             'no extra training.')
    parser.add_argument('--val_selection', default='balance', choices=['balance', 'smallest'],
                        help='Validation project per fold: most class-balanced, or '
                             'smallest (the rule the original Baseline run used)')
    args = parser.parse_args()

    set_seed(args.seed)
    logger.info(f"Seed: {args.seed}")

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

    # Load augmentation data if requested. --aug_source selects the keyword
    # heuristic (LABELLING_METHODOLOGY.md), the LLM labeler (llm_label_tickets.py
    # --role labeler), or both -- each source capped independently at aug_max
    # so combining sources doesn't let one source's scoring scale dominate.
    aug_d = pd.DataFrame()
    aug_n = pd.DataFrame()
    if args.augment:
        per_source_max = args.aug_max // 2 if (args.aug_source == 'both' and args.aug_max > 0) else args.aug_max

        d_parts = []
        n_parts = []

        if args.aug_source in ('heuristic', 'both'):
            # D labels — sort by score (highest confidence first)
            heur_d_parts = []
            if Path(D_LABELS_PATH).exists():
                d_df = pd.read_csv(D_LABELS_PATH)
                d_df['text'] = d_df['summary'].fillna('') + ' [SEP] ' + d_df['description'].fillna('')
                d_df['label'] = 'design'
                heur_d_parts.append(d_df[['project', 'issue_key', 'issue_type', 'text', 'label', 'score']])
            if Path(D_LABELS_SVR).exists():
                d_df = pd.read_csv(D_LABELS_SVR)
                d_df['project'] = 'SERVER'
                d_df['text'] = d_df['summary'].fillna('') + ' [SEP] ' + d_df['description'].fillna('')
                d_df['label'] = 'design'
                if 'score' not in d_df.columns:
                    d_df['score'] = 5  # minimum threshold score
                heur_d_parts.append(d_df[['project', 'issue_key', 'issue_type', 'text', 'label', 'score']])
            if heur_d_parts:
                heur_d = pd.concat(heur_d_parts, ignore_index=True).sort_values('score', ascending=False)
                if per_source_max > 0:
                    heur_d = heur_d.head(per_source_max)
                logger.info(f"Heuristic augmentation D labels: {len(heur_d)}")
                d_parts.append(heur_d)

            # N labels — sort by confidence_tier (1=highest confidence first)
            if Path(N_LABELS_PATH).exists():
                n_df = pd.read_csv(N_LABELS_PATH)
                n_df['text'] = n_df['summary'].fillna('') + ' [SEP] ' + n_df['description'].fillna('')
                n_df['label'] = 'non-design'
                n_df = n_df.sort_values('confidence_tier', ascending=True)
                if per_source_max > 0:
                    n_df = n_df.head(per_source_max)
                logger.info(f"Heuristic augmentation N labels: {len(n_df)}")
                n_parts.append(n_df[['project', 'issue_key', 'issue_type', 'text', 'label']])

        if args.aug_source in ('llm', 'both'):
            # LLM labels from llm_label_tickets.py --role labeler. Schema:
            # project, issue_key, issue_type, summary, description, label, score (0-1 confidence).
            if Path(LLM_D_LABELS_PATH).exists():
                llm_d = pd.read_csv(LLM_D_LABELS_PATH)
                llm_d = llm_d[llm_d['label'] == 'design'].copy()
                llm_d['text'] = llm_d['summary'].fillna('') + ' [SEP] ' + llm_d['description'].fillna('')
                llm_d = llm_d.sort_values('score', ascending=False)
                if per_source_max > 0:
                    llm_d = llm_d.head(per_source_max)
                logger.info(f"LLM augmentation D labels: {len(llm_d)}")
                d_parts.append(llm_d[['project', 'issue_key', 'issue_type', 'text', 'label', 'score']])

            if Path(LLM_N_LABELS_PATH).exists():
                llm_n = pd.read_csv(LLM_N_LABELS_PATH)
                llm_n = llm_n[llm_n['label'] == 'non-design'].copy()
                llm_n['text'] = llm_n['summary'].fillna('') + ' [SEP] ' + llm_n['description'].fillna('')
                llm_n = llm_n.sort_values('score', ascending=False)
                if per_source_max > 0:
                    llm_n = llm_n.head(per_source_max)
                logger.info(f"LLM augmentation N labels: {len(llm_n)}")
                n_parts.append(llm_n[['project', 'issue_key', 'issue_type', 'text', 'label']])

        if d_parts:
            aug_d = pd.concat(d_parts, ignore_index=True)
            logger.info(f"Total augmentation D labels ({args.aug_source}): {len(aug_d)}")
        if n_parts:
            aug_n = pd.concat(n_parts, ignore_index=True)
            logger.info(f"Total augmentation N labels ({args.aug_source}): {len(aug_n)}")

        # Add priority from lookup for augmentation data
        if args.metadata and priority_lookup:
            if not aug_d.empty:
                aug_d['priority'] = aug_d['issue_key'].map(priority_lookup)
            if not aug_n.empty:
                aug_n['priority'] = aug_n['issue_key'].map(priority_lookup)

    # Load tokenizer once (shared across folds)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if args.lora:
        # Convention for decoder-only models under LoRA; the pooling logic
        # in transformers' GenericForSequenceClassification handles either
        # padding side correctly, but left-padding is the standard for
        # decoder generation/classification workflows.
        tokenizer.padding_side = 'left'

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
    prediction_rows = []
    fold_probs = {}   # project -> (test probs, test labels), for --threshold pooled

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
        if args.val_selection == 'smallest':
            val_project = min(remaining_projects, key=lambda p: len(df[df['project'] == p]))
        else:
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
            use_lora=args.lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            batch_size=args.batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            class_weighting=args.class_weighting,
            tune_threshold=args.threshold == 'tuned',
            # 'pooled' scores at 0.5 here and is re-scored after the fold loop,
            # once every fold's out-of-fold probabilities are available.
        )
        metrics['project'] = held_out
        # Per-ticket predictions for qualitative error analysis (R1.3/R2.6);
        # test_df row order matches the unshuffled test DataLoader.
        test_probs = metrics.pop('probs')
        fold_probs[held_out] = (test_probs, test_labels)
        for (_, row), prob, true_label in zip(test_df.iterrows(), test_probs, test_labels):
            prediction_rows.append({
                'project': held_out,
                'issue_key': row['issue_key'],
                'issue_type': row.get('issue_type'),
                'label': true_label,
                'prob_design': round(prob, 6),
                'threshold': None,   # filled in below (pooled needs all folds first)
                'pred': None,
                'val_project': val_project,
            })
        results.append(metrics)

        logger.info(
            f"  [{held_out}] TEST: F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  "
            f"Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  "
            f"Rec={metrics['recall']:.4f}  Threshold={metrics.get('threshold', 0.5):.2f}"
        )

    # ── Pooled threshold (second pass) ───────────────────────────────────────
    # Tuning per fold on a single ~177-ticket validation project is a noisy
    # optimum: the chosen threshold swings widely between otherwise identical
    # runs, and that swing lands directly on test F1. Pooling the other folds'
    # out-of-fold predictions tunes on ~1.6k tickets instead. The held-out
    # project is excluded from its own pool, so no test label informs its
    # threshold -- though the pooled probabilities do come from models that
    # trained on the held-out project, which is why this is a stability fix
    # and not a clean-room nested CV.
    if args.threshold == 'pooled':
        for m in results:
            k = m['project']
            pool_probs, pool_labels = [], []
            for other, (probs_o, labels_o) in fold_probs.items():
                if other != k:
                    pool_probs.extend(probs_o)
                    pool_labels.extend(labels_o)
            t, pool_f1 = _find_optimal_threshold(pool_probs, pool_labels)
            probs_k, labels_k = fold_probs[k]
            m.update(_metrics_from_probs(probs_k, labels_k, t))
            m['threshold'] = round(t, 4)
            logger.info(
                f"  [{k}] Pooled threshold: {t:.2f} "
                f"(tuned on {len(pool_labels)} tickets from 9 projects, "
                f"pooled F1={pool_f1:.4f}) -> test F1={m['f1']:.4f}"
            )

    # Thresholds are final only now -- fill in the per-ticket predictions.
    thresholds = {m['project']: m.get('threshold', 0.5) for m in results}
    for row in prediction_rows:
        t = thresholds[row['project']]
        row['threshold'] = t
        row['pred'] = int(row['prob_design'] >= t)

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
            'batch_size': args.batch_size, 'max_length': MAX_LENGTH,
            'dropout': DROPOUT, 'weight_decay': WEIGHT_DECAY,
            'freeze_layers': args.freeze_layers,
            'manual_weight': args.manual_weight,
            'aug_max': args.aug_max,
            'pseudo_weight': args.pseudo_weight if args.pseudo_label else None,
            'lora': args.lora,
            'lora_r': args.lora_r if args.lora else None,
            'lora_alpha': args.lora_alpha if args.lora else None,
            'lora_dropout': args.lora_dropout if args.lora else None,
            'gradient_checkpointing': args.gradient_checkpointing,
            'seed': args.seed,
            'class_weighting': args.class_weighting,
            'threshold': args.threshold,
            'val_selection': args.val_selection,
        },
        'per_project': results,
        'mean_f1': round(float(np.mean(f1s)), 4),
        'std_f1': round(float(np.std(f1s)), 4),
        'mean_auc': round(float(np.mean(aucs)), 4),
        'std_auc': round(float(np.std(aucs)), 4),
        'mean_accuracy': round(float(np.mean(accs)), 4),
        'std_accuracy': round(float(np.std(accs)), 4),
    }
    summary['total_train_wall_clock_s'] = round(sum(r['train_wall_clock_s'] for r in results), 3)
    summary['total_test_inference_wall_clock_s'] = round(
        sum(r['test_inference_wall_clock_s'] for r in results), 3)
    summary['device'] = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    pred_path = out_path.with_name(out_path.stem + '_predictions.csv')
    pd.DataFrame(prediction_rows).to_csv(pred_path, index=False)
    print(f"Per-ticket predictions saved to {pred_path}")


if __name__ == '__main__':
    main()
