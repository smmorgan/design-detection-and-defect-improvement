"""
BERT-based Design Mining for Software Architecture Issues
==========================================================

Based on the dissertation: "An Algorithmic Approach to Understanding the 
Impact of Design Work in Software Projects" by Steven Morgan

This script implements Research Question 1:
"Can design mining be performed on tickets collected from JIRA using 
transformers to outperform the state-of-the-art?"

The approach fine-tunes BERT for binary classification of software tickets
as either 'design' or 'general' (non-design) related.

Author: Based on methodology from Steven Morgan's dissertation
"""

import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup
)

from tqdm import tqdm

# TAWOS database connectivity
try:
    from tawos_connector import TAWOSConnector, TAWOSConfig, fetch_tawos_data
    TAWOS_AVAILABLE = True
except ImportError:
    TAWOS_AVAILABLE = False
    TAWOSConfig = None  # Type hint placeholder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Configuration class for model training hyperparameters.
    
    Based on dissertation Section 4.2.6 Model Architecture:
    - BERT base model with 512 token limit
    - Learning rate of 2e-5 to minimize effect on pre-trained knowledge
    - Batch size of 16
    - 3-5 epochs for verification
    - Dropout rate 0.1-0.3 to prevent overfitting
    """
    
    # Model parameters
    MODEL_NAME = 'distilbert-base-uncased'  # Override with --model at runtime
    MAX_LENGTH = 512  # As specified in dissertation
    NUM_LABELS = 2    # Binary: 'design' or 'general'
    
    # Training parameters (from dissertation 4.2.6)
    LEARNING_RATE = 2e-5  # "to minimize effect on pre-trained knowledge"
    BATCH_SIZE = 16
    NUM_EPOCHS = 5        # "3-5 epochs are used to verify effectiveness"
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    DROPOUT_RATE = 0.1    # "low rate being used to test"
    
    # Data parameters
    TRAIN_SPLIT = 0.7     # 60-70% for training (from Trust & Mingham 2024)
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    RANDOM_SEED = 42
    
    # Fine-tuning for transfer learning (Section 4.2.2)
    SECOND_FINETUNE_LR = 1e-5  # "smaller learning rate" for domain adaptation
    SECOND_FINETUNE_EPOCHS = 5  # increased from 2; early stopping prevents overfitting
    SECOND_FINETUNE_PATIENCE = 2  # early stopping patience for Stage 3
    
    # Paths
    OUTPUT_DIR = './model_output'
    CHECKPOINT_DIR = './checkpoints'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Dataset Classes
# =============================================================================

class DesignMiningDataset(Dataset):
    """Dataset class for design mining classification.
    
    Handles tokenization with BERT tokenizer including special tokens:
    [CLS] - denoting specific example
    [SEP] - marking separation between different parts
    
    As described in dissertation Section 4.2.6
    """
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int],
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize with model tokenizer
        # This automatically adds special tokens (e.g. [CLS]/[SEP] or <s>/</s>)
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class JIRATicketDataset(Dataset):
    """Dataset class specifically for JIRA tickets with multiple fields.
    
    Combines title, description, and comments with [SEP] tokens as described
    in dissertation Section 4.2.6:
    
    "[CLS]Document reasoning for using either an event-driven or transactional 
    architecture. [SEP] Create an architecture decision record which describes 
    [SEP] overall system flow [SEP] basic assumptions..."
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        title_col: str = 'title',
        description_col: str = 'description',
        comments_col: str = 'comments',
        label_col: str = 'label'
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.title_col = title_col
        self.description_col = description_col
        self.comments_col = comments_col
        self.label_col = label_col
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _combine_fields(self, row: pd.Series) -> str:
        """Combine JIRA fields with [SEP] separators."""
        parts = []
        
        # Add title
        if self.title_col in row and pd.notna(row[self.title_col]):
            parts.append(str(row[self.title_col]))
        
        # Add description
        if self.description_col in row and pd.notna(row[self.description_col]):
            parts.append(str(row[self.description_col]))
        
        # Add comments (if available)
        if self.comments_col in row and pd.notna(row[self.comments_col]):
            comments = str(row[self.comments_col])
            # Split comments if they're concatenated
            if isinstance(comments, str) and len(comments) > 0:
                parts.append(comments)
        
        # Join with [SEP] token
        return ' [SEP] '.join(parts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        text = self._combine_fields(row)
        label = int(row[self.label_col])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# =============================================================================
# Data Preprocessing
# =============================================================================

class DataPreprocessor:
    """Preprocessor for software engineering text data.

    Implements preprocessing steps from dissertation Section 4.2.1:
    - Remove duplicates
    - Remove automatically generated issues
    - Remove issues with < 7 words after stop word removal
    - Normalize whitespace
    - Remove URLs, email addresses, system-generated info
    - Apply software-specific stop words
    """

    def __init__(self, min_words: int = 7, stopwords_file: str = 'software_stopwords.txt', verbose: bool = True):
        self.min_words = min_words
        self.verbose = verbose
        self.SOFTWARE_STOPWORDS = self._load_stopwords(stopwords_file)

        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'duplicates_removed': 0,
            'auto_generated_removed': 0,
            'short_texts_removed': 0,
            'chars_removed_cleaning': 0
        }

    @staticmethod
    def _load_stopwords(filepath: str) -> set:
        """Load software engineering stop words from file.

        Based on Mahadi et al. vocabulary from enhanced_literature.txt
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                stopwords = {line.strip() for line in f if line.strip()}
            logger.info(f"Loaded {len(stopwords)} stop words from {filepath}")
            return stopwords
        except FileNotFoundError:
            logger.warning(f"Stopwords file not found: {filepath}. Using empty set.")
            return set()
        
    @staticmethod
    def clean_jira_markup(text: str) -> str:
        """Strip JIRA/Confluence wiki markup and boilerplate sections.

        Removes structural noise that is present in TAWOS tickets but absent
        from Stack Overflow data, reducing domain mismatch for transfer learning.
        """
        import re

        # Remove JIRA heading markup (h1. through h6.)
        text = re.sub(r'h[1-6]\.\s*', '', text)

        # Remove JIRA formatting: *bold*, _italic_, +underline+, -strikethrough-
        text = re.sub(r'(?<!\w)\*([^*\n]+)\*(?!\w)', r'\1', text)
        text = re.sub(r'(?<!\w)_([^_\n]+)_(?!\w)', r'\1', text)
        text = re.sub(r'(?<!\w)\+([^+\n]+)\+(?!\w)', r'\1', text)
        text = re.sub(r'(?<!\w)-([^-\n]+)-(?!\w)', r'\1', text)

        # Remove {code}, {noformat}, {quote}, {panel} blocks
        text = re.sub(r'\{code(?::[^}]*)?\}[\s\S]*?\{code\}', '', text)
        text = re.sub(r'\{noformat\}[\s\S]*?\{noformat\}', '', text)
        text = re.sub(r'\{quote\}[\s\S]*?\{quote\}', '', text)
        text = re.sub(r'\{panel(?::[^}]*)?\}[\s\S]*?\{panel\}', '', text)

        # Remove {color}, {anchor}, and other inline macros
        text = re.sub(r'\{color(?::[^}]*)?\}', '', text)
        text = re.sub(r'\{anchor:[^}]*\}', '', text)

        # Remove JIRA list markers (# ordered, * unordered, - dash lists)
        # Also handle inline lists (JIRA text often flattened to single line)
        text = re.sub(r'(?m)^[#*\-]+\s+', '', text)
        text = re.sub(r'\s+[#]+\s+', ' ', text)

        # Remove JIRA table markup (||header|| and |cell|)
        text = re.sub(r'\|\|?', ' ', text)

        # Remove JIRA link syntax [text|url] and [url]
        text = re.sub(r'\[([^|\]]*)\|[^\]]+\]', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]', r'\1', text)

        # Remove boilerplate JIRA sections that add noise
        boilerplate_headers = [
            r'Steps\s+to\s+Reproduce',
            r'Expected\s+Results?',
            r'Actual\s+Results?',
            r'Workaround',
            r'Environment',
            r'Affected\s+Versions?',
            r'Fix\s+Versions?',
        ]
        for header in boilerplate_headers:
            # Remove from header to next header or end of text
            text = re.sub(
                rf'(?i){header}\s*[:\-]?\s*',
                ' ',
                text,
            )

        # Remove escaped/triple quotes from CSV encoding artifacts
        text = re.sub(r'"{2,}', '', text)

        return text

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        import re

        if pd.isna(text) or text is None:
            return ""

        original_text = str(text)
        original_len = len(original_text)
        text = original_text

        # Strip JIRA wiki markup and boilerplate (before other cleaning)
        text = self.clean_jira_markup(text)

        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove file paths
        text = re.sub(r'[A-Za-z]:\\[\S]+', '', text)
        text = re.sub(r'/[\S]+/[\S]+', '', text)

        # Remove code blocks (markdown style)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\']', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        cleaned_text = text.strip()

        # Track statistics
        chars_removed = original_len - len(cleaned_text)
        self.stats['chars_removed_cleaning'] += chars_removed

        # Verbose logging for first few examples
        if self.verbose and self.stats['total_processed'] < 3 and chars_removed > 10:
            logger.info(f"\n--- Text Cleaning Example #{self.stats['total_processed'] + 1} ---")
            logger.info(f"Original ({original_len} chars): {original_text[:200]}{'...' if len(original_text) > 200 else ''}")
            logger.info(f"Cleaned ({len(cleaned_text)} chars):  {cleaned_text[:200]}{'...' if len(cleaned_text) > 200 else ''}")
            logger.info(f"Removed: {chars_removed} characters")

        return cleaned_text
    
    def is_auto_generated(self, text: str) -> bool:
        """Check if text appears to be auto-generated."""
        auto_patterns = [
            r'^Merge pull request',
            r'^Merge branch',
            r'^Automatic merge',
            r'^Auto-generated',
            r'^This is an automated',
            r'^\[Bot\]',
            r'^Dependabot',
            r'^renovate\[bot\]'
        ]

        import re
        for pattern in auto_patterns:
            if re.search(pattern, str(text), re.IGNORECASE):
                if self.verbose and self.stats['auto_generated_removed'] < 3:
                    logger.info(f"\n--- Auto-Generated Detection Example #{self.stats['auto_generated_removed'] + 1} ---")
                    logger.info(f"Matched pattern: {pattern}")
                    logger.info(f"Text: {str(text)[:150]}{'...' if len(str(text)) > 150 else ''}")
                return True
        return False
    
    def word_count_after_stopwords(self, text: str) -> int:
        """Count words after removing stop words."""
        words = text.lower().split()
        meaningful_words = [w for w in words if w not in self.SOFTWARE_STOPWORDS]
        removed_words = [w for w in words if w in self.SOFTWARE_STOPWORDS]

        # Verbose logging for examples
        if self.verbose and self.stats['short_texts_removed'] < 3 and len(meaningful_words) < self.min_words:
            logger.info(f"\n--- Short Text Example (will be removed) #{self.stats['short_texts_removed'] + 1} ---")
            logger.info(f"Original text: {text[:200]}{'...' if len(text) > 200 else ''}")
            logger.info(f"Total words: {len(words)}")
            logger.info(f"Meaningful words after stopword removal: {len(meaningful_words)}")
            logger.info(f"Stopwords removed: {removed_words[:10]}{'...' if len(removed_words) > 10 else ''}")
            logger.info(f"Meaningful words: {meaningful_words}")
            logger.info(f"Threshold: {self.min_words} words")

        return len(meaningful_words)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Apply all preprocessing steps to a dataframe."""
        logger.info("\n" + "="*60)
        logger.info("PREPROCESSING PIPELINE")
        logger.info("="*60)
        logger.info(f"Starting preprocessing on {len(df)} records...")

        # Show sample of original data
        if self.verbose and len(df) > 0:
            logger.info(f"\n--- Original Data Sample ---")
            for i in range(min(2, len(df))):
                logger.info(f"Example {i+1}: {df[text_col].iloc[i][:150]}{'...' if len(str(df[text_col].iloc[i])) > 150 else ''}")

        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=[text_col])
        duplicates_removed = initial_count - len(df)
        self.stats['duplicates_removed'] = duplicates_removed
        logger.info(f"\n[Step 1/4] Duplicate Removal: Removed {duplicates_removed} duplicates ({duplicates_removed/initial_count*100:.1f}%)")
        logger.info(f"  Remaining: {len(df)} records")

        # Clean text
        logger.info(f"\n[Step 2/4] Text Cleaning: Removing URLs, emails, code blocks, special characters...")
        self.stats['total_processed'] = 0
        df[text_col] = df[text_col].apply(self.clean_text)
        logger.info(f"  Total characters removed: {self.stats['chars_removed_cleaning']:,}")

        # Remove auto-generated
        logger.info(f"\n[Step 3/4] Auto-Generated Detection: Checking for bot-generated content...")
        self.stats['auto_generated_removed'] = 0
        auto_mask = df[text_col].apply(self.is_auto_generated)
        auto_count = auto_mask.sum()
        df = df[~auto_mask]
        self.stats['auto_generated_removed'] = auto_count
        logger.info(f"  Removed {auto_count} auto-generated entries ({auto_count/initial_count*100:.1f}%)")
        logger.info(f"  Remaining: {len(df)} records")

        # Remove short texts
        logger.info(f"\n[Step 4/4] Short Text Removal: Filtering texts with < {self.min_words} meaningful words...")
        self.stats['short_texts_removed'] = 0
        word_counts = df[text_col].apply(self.word_count_after_stopwords)
        short_mask = word_counts < self.min_words
        short_count = short_mask.sum()
        df = df[word_counts >= self.min_words]
        self.stats['short_texts_removed'] = short_count
        logger.info(f"  Removed {short_count} short entries ({short_count/initial_count*100:.1f}%)")
        logger.info(f"  Remaining: {len(df)} records")

        # Show sample of processed data
        if self.verbose and len(df) > 0:
            logger.info(f"\n--- Processed Data Sample ---")
            for i in range(min(2, len(df))):
                text = df[text_col].iloc[i]
                wc = len([w for w in text.lower().split() if w not in self.SOFTWARE_STOPWORDS])
                logger.info(f"Example {i+1} ({wc} meaningful words): {text[:150]}{'...' if len(text) > 150 else ''}")

        # Summary statistics
        logger.info("\n" + "="*60)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Initial records:        {initial_count:>8,}")
        logger.info(f"Duplicates removed:     {duplicates_removed:>8,} ({duplicates_removed/initial_count*100:>5.1f}%)")
        logger.info(f"Auto-generated removed: {auto_count:>8,} ({auto_count/initial_count*100:>5.1f}%)")
        logger.info(f"Short texts removed:    {short_count:>8,} ({short_count/initial_count*100:>5.1f}%)")
        logger.info(f"Final records:          {len(df):>8,} ({len(df)/initial_count*100:>5.1f}%)")
        logger.info(f"Total reduction:        {initial_count - len(df):>8,} ({(initial_count - len(df))/initial_count*100:>5.1f}%)")
        logger.info("="*60 + "\n")

        return df.reset_index(drop=True)


# =============================================================================
# Evaluation Metrics
# =============================================================================

class MetricsCalculator:
    """Calculate evaluation metrics as defined in dissertation Section 4.2.5.
    
    Metrics:
    - Accuracy (Eq. 1): (TP + TN) / (TP + TN + FP + FN)
    - Recall (Eq. 2): TP / (TP + FN)
    - Precision (Eq. 3): TP / (TP + FP)
    - F1 Score (Eq. 4): 2 * (precision * recall) / (precision + recall)
    - AUC (Eq. 5): (TPR + TNR) / 2
    """
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate all metrics from dissertation."""
        
        metrics = {}
        
        # Accuracy (Eq. 1)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision (Eq. 3)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        
        # Recall (Eq. 2)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        
        # F1 Score (Eq. 4)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC (Eq. 5) - requires probability scores
        if y_prob is not None:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], prefix: str = ""):
        """Pretty print metrics."""
        logger.info(f"\n{prefix} Evaluation Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        if 'auc' in metrics:
            logger.info(f"  AUC:       {metrics['auc']:.4f}")
        logger.info(f"  Confusion Matrix: TP={metrics['true_positives']}, "
                   f"TN={metrics['true_negatives']}, "
                   f"FP={metrics['false_positives']}, "
                   f"FN={metrics['false_negatives']}")


# =============================================================================
# Model Training
# =============================================================================

class DesignMiningTrainer:
    """Trainer class for BERT-based design mining model.
    
    Implements the training procedure from dissertation Section 4.2.6:
    1. Fine-tune BERT with classification layer
    2. Use [CLS] token aggregation for classification
    3. Dropout layer to prevent overfitting
    4. Dense layer with Softmax for final classification
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.DEVICE
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

        # Initialize model with configuration
        model_config = AutoConfig.from_pretrained(
            config.MODEL_NAME,
            num_labels=config.NUM_LABELS,
        )
        # Set dropout where supported by the architecture
        if hasattr(model_config, 'hidden_dropout_prob'):
            model_config.hidden_dropout_prob = config.DROPOUT_RATE
        if hasattr(model_config, 'attention_probs_dropout_prob'):
            model_config.attention_probs_dropout_prob = config.DROPOUT_RATE
        # DistilBERT uses different attribute names
        if hasattr(model_config, 'dropout'):
            model_config.dropout = config.DROPOUT_RATE
        if hasattr(model_config, 'attention_dropout'):
            model_config.attention_dropout = config.DROPOUT_RATE

        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            config=model_config
        )
        self.model.to(self.device)
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator()

        # Optional class-weighted loss (set during train() calls)
        self._loss_fn = None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        logger.info(f"Model initialized on {self.device}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_data_loaders(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        test_texts: Optional[List[str]] = None,
        test_labels: Optional[List[int]] = None
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """Create data loaders for training, validation, and test sets."""
        
        train_dataset = DesignMiningDataset(
            train_texts, train_labels, 
            self.tokenizer, self.config.MAX_LENGTH
        )
        val_dataset = DesignMiningDataset(
            val_texts, val_labels,
            self.tokenizer, self.config.MAX_LENGTH
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        test_loader = None
        if test_texts is not None and test_labels is not None:
            test_dataset = DesignMiningDataset(
                test_texts, test_labels,
                self.tokenizer, self.config.MAX_LENGTH
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
                num_workers=0
            )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Use class-weighted loss when available (Stage 3),
            # otherwise fall back to the model's default loss.
            if self._loss_fn is not None:
                loss = self._loss_fn(outputs.logits, labels)
            else:
                loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluate model on data loader."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                # Get predictions and probabilities (Softmax output)
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of 'design' class
        
        avg_loss = total_loss / len(data_loader)
        metrics = self.metrics_calc.calculate_all_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        return avg_loss, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        class_weights: Optional[torch.Tensor] = None,
        early_stopping_patience: int = 0,
    ) -> Dict:
        """Full training loop.

        Args:
            class_weights: Optional tensor of per-class weights for
                CrossEntropyLoss (e.g. ``torch.tensor([w_neg, w_pos])``).
            early_stopping_patience: Stop after this many epochs without
                improvement in validation F1.  0 = disabled (default,
                preserves original behaviour for Stage 1).
        """

        num_epochs = num_epochs or self.config.NUM_EPOCHS
        learning_rate = learning_rate or self.config.LEARNING_RATE

        # Optional weighted loss function for class-imbalanced training
        self._loss_fn = None
        if class_weights is not None:
            self._loss_fn = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device)
            )
            logger.info(f"Using class-weighted loss: {class_weights.tolist()}")

        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.WEIGHT_DECAY
        )

        # Scheduler with warmup
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * self.config.WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        logger.info(f"\nStarting training for {num_epochs} epochs")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
        if early_stopping_patience:
            logger.info(f"Early stopping patience: {early_stopping_patience} epochs")

        best_f1 = 0
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*50}")

            # Training
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            self.history['train_loss'].append(train_loss)
            logger.info(f"Training Loss: {train_loss:.4f}")

            # Validation
            val_loss, val_metrics = self.evaluate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)

            logger.info(f"Validation Loss: {val_loss:.4f}")
            self.metrics_calc.print_metrics(val_metrics, "Validation")

            # Save best model (based on F1 as per dissertation goal)
            if val_metrics['f1_score'] > best_f1:
                best_f1 = val_metrics['f1_score']
                best_model_state = self.model.state_dict().copy()
                epochs_without_improvement = 0
                logger.info(f"New best F1 score: {best_f1:.4f}")
            else:
                epochs_without_improvement += 1

            # Early stopping
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered: no F1 improvement for "
                    f"{early_stopping_patience} epoch(s). Best F1: {best_f1:.4f}"
                )
                break

            # Legacy warning (kept for Stage 1 backward compat)
            if epoch > 0 and not early_stopping_patience:
                prev_f1 = self.history['val_metrics'][-2]['f1_score']
                if val_metrics['f1_score'] <= prev_f1 - 0.01:
                    logger.info("F1 score declining, consider early stopping")

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"\nRestored best model with F1: {best_f1:.4f}")

        # Clean up weighted loss
        self._loss_fn = None

        return self.history
    
    def predict(
        self, 
        texts: List[str]
    ) -> Tuple[List[int], List[float]]:
        """Generate predictions with confidence scores."""
        self.model.eval()
        
        predictions = []
        confidences = []
        
        dataset = DesignMiningDataset(
            texts, 
            [0] * len(texts),  # Dummy labels
            self.tokenizer, 
            self.config.MAX_LENGTH
        )
        data_loader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE)
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                predictions.extend(preds.cpu().numpy().tolist())
                confidences.extend(probs.max(dim=1).values.cpu().numpy().tolist())
        
        return predictions, confidences
    
    def save_model(self, path: str):
        """Save model and tokenizer."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save training history
        with open(os.path.join(path, 'training_history.json'), 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            history_serializable = {
                'train_loss': [float(x) for x in self.history['train_loss']],
                'val_loss': [float(x) for x in self.history['val_loss']],
                'val_metrics': self.history['val_metrics']
            }
            json.dump(history_serializable, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model and tokenizer."""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")


# =============================================================================
# Transfer Learning for TAWOS Dataset
# =============================================================================

class TransferLearningPipeline:
    """Pipeline for transfer learning from Stack Overflow to TAWOS dataset.

    Implements dissertation Section 4.2.2:
    1. Fine-tune BERT on Stack Overflow dataset (already labeled)
    2. Apply to TAWOS dataset to generate labels
    3. Second fine-tuning with smaller learning rate on TAWOS

    Supports two data sources for TAWOS:
    - File-based: Load from CSV file
    - Database: Connect to TAWOS MySQL database
    """

    def __init__(self, config: Config, tawos_config: Optional['TAWOSConfig'] = None):
        self.config = config
        self.trainer = DesignMiningTrainer(config)
        self.tawos_config = tawos_config
        self.tawos_connector = None

        # Initialize TAWOS connector if config provided
        if tawos_config and TAWOS_AVAILABLE:
            self.tawos_connector = TAWOSConnector(tawos_config)

    def connect_tawos_db(self) -> bool:
        """Connect to TAWOS MySQL database.

        Returns:
            True if connection successful
        """
        if not TAWOS_AVAILABLE:
            logger.error("TAWOS connector not available. Install mysql-connector-python")
            return False

        if not self.tawos_connector:
            logger.error("TAWOS config not provided")
            return False

        return self.tawos_connector.connect()

    def fetch_tawos_from_db(
        self,
        projects: Optional[List[str]] = None,
        include_comments: bool = False,
        max_issues: Optional[int] = None
    ) -> Tuple[List[str], pd.DataFrame]:
        """Fetch TAWOS issues from MySQL database.

        Args:
            projects: Optional list of Apache project keys to filter
            include_comments: Whether to include comment text
            max_issues: Maximum number of issues to fetch

        Returns:
            Tuple of (texts list, full DataFrame)
        """
        if not self.tawos_connector:
            raise RuntimeError("TAWOS connector not initialized")

        df = self.tawos_connector.fetch_issues(
            include_comments=include_comments,
            projects=projects,
            limit=max_issues
        )

        texts = df['text'].tolist() if not df.empty else []
        return texts, df
    
    def stage1_finetune_stackoverflow(
        self,
        so_texts: List[str],
        so_labels: List[int]
    ) -> Dict:
        """Stage 1: Fine-tune on Stack Overflow data."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 1: Fine-tuning on Stack Overflow Dataset")
        logger.info("="*60)
        
        # Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            so_texts, so_labels,
            test_size=(1 - self.config.TRAIN_SPLIT),
            random_state=self.config.RANDOM_SEED,
            stratify=so_labels
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels,
            test_size=0.5,
            random_state=self.config.RANDOM_SEED,
            stratify=temp_labels
        )
        
        logger.info(f"Train size: {len(train_texts)}")
        logger.info(f"Validation size: {len(val_texts)}")
        logger.info(f"Test size: {len(test_texts)}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.trainer.create_data_loaders(
            train_texts, train_labels,
            val_texts, val_labels,
            test_texts, test_labels
        )
        
        # Train
        history = self.trainer.train(train_loader, val_loader)
        
        # Final evaluation on test set
        if test_loader:
            logger.info("\nFinal evaluation on Stack Overflow test set:")
            test_loss, test_metrics = self.trainer.evaluate(test_loader)
            MetricsCalculator.print_metrics(test_metrics, "Test")
            
            # Check if F1 > 0.90 as specified
            if test_metrics['f1_score'] >= 0.90:
                logger.info("✓ Target F1 >= 0.90 achieved!")
            else:
                logger.warning(f"Target F1 not reached: {test_metrics['f1_score']:.4f} < 0.90")
        
        return history
    
    def load_manually_labelled_data(
        self,
        directory: str,
        max_n_labels: int = 400,
        random_seed: int = 42
    ) -> Tuple[List[str], List[int]]:
        """Load human-labelled D/N data from TSV files for Stage 3 injection.

        Reads all TSV files in the directory, maps D->1 and N->0 (also
        handles 'design'/'non-design' from SERVER.tsv), and undersamples
        the N class to ``max_n_labels`` to reduce class imbalance.

        Args:
            directory: Path to folder containing manually-labelled TSV files.
            max_n_labels: Cap on N-labelled samples (undersampling).
            random_seed: Seed for reproducible N undersampling.

        Returns:
            Tuple of (texts, labels) ready for Stage 3 training.
        """
        import glob as _glob
        import random as _random

        logger.info("\n" + "="*60)
        logger.info("Loading Manually-Labelled Data for Stage 3")
        logger.info("="*60)

        _random.seed(random_seed)

        all_D_texts: List[str] = []
        all_N_texts: List[str] = []

        tsv_files = sorted(_glob.glob(os.path.join(directory, '*.tsv')))
        if not tsv_files:
            logger.warning(f"No TSV files found in {directory}")
            return [], []

        for fpath in tsv_files:
            fname = os.path.basename(fpath)
            try:
                import csv as _csv
                _csv.field_size_limit(10 ** 7)
                with open(fpath, encoding='utf-8') as fh:
                    reader = _csv.reader(fh, delimiter='\t')
                    headers = next(reader)
                    try:
                        text_idx = headers.index('text')
                    except ValueError:
                        logger.warning(f"  {fname}: missing 'text' column, skipping")
                        continue

                    # Priority: human 'Label' (D/N) > 'label_name' > 'predicted_label'
                    # This lets all project TSVs contribute via their predicted_label column.
                    label_map = {'design': 'D', 'non-design': 'N', 'D': 'D', 'N': 'N'}
                    if 'Label' in headers:
                        label_idx = headers.index('Label')
                    elif 'label_name' in headers:
                        label_idx = headers.index('label_name')
                    elif 'predicted_label' in headers:
                        label_idx = headers.index('predicted_label')
                    else:
                        logger.warning(f"  {fname}: no label column found, skipping")
                        continue

                    file_D = file_N = 0
                    for row in reader:
                        if len(row) <= max(label_idx, text_idx):
                            continue
                        lname = label_map.get(row[label_idx].strip())
                        text  = row[text_idx].strip()
                        if not text or lname is None:
                            continue
                        if lname == 'D':
                            all_D_texts.append(text)
                            file_D += 1
                        elif lname == 'N':
                            all_N_texts.append(text)
                            file_N += 1

                logger.info(f"  {fname}: D={file_D}, N={file_N}")
            except Exception as e:
                logger.warning(f"  {fname}: error reading file – {e}")

        # Undersample N to reduce imbalance
        if len(all_N_texts) > max_n_labels:
            all_N_texts = _random.sample(all_N_texts, max_n_labels)
            logger.info(f"Undersampled N to {max_n_labels} samples")

        texts  = all_D_texts + all_N_texts
        labels = [1] * len(all_D_texts) + [0] * len(all_N_texts)

        logger.info(f"Total manually-labelled: D={len(all_D_texts)}, N={len(all_N_texts)}")
        logger.info(f"D ratio: {len(all_D_texts) / max(1, len(texts)) * 100:.1f}%")

        return texts, labels

    def stage2_label_tawos(
        self,
        tawos_texts: List[str],
        confidence_threshold: float = 0.8
    ) -> Tuple[List[int], List[float]]:
        """Stage 2: Generate labels for TAWOS dataset."""
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: Labeling TAWOS Dataset")
        logger.info("="*60)
        
        predictions, confidences = self.trainer.predict(tawos_texts)
        
        # Analyze confidence distribution
        high_conf_count = sum(1 for c in confidences if c >= confidence_threshold)
        logger.info(f"Total predictions: {len(predictions)}")
        logger.info(f"High confidence (>= {confidence_threshold}): {high_conf_count} "
                   f"({high_conf_count/len(predictions)*100:.1f}%)")
        logger.info(f"Design tickets predicted: {sum(predictions)} "
                   f"({sum(predictions)/len(predictions)*100:.1f}%)")
        
        return predictions, confidences
    
    def stage3_finetune_tawos(
        self,
        tawos_texts: List[str],
        tawos_labels: List[int],
        confidences: List[float],
        confidence_threshold: float = 0.85,
        extra_texts: Optional[List[str]] = None,
        extra_labels: Optional[List[int]] = None,
    ) -> Dict:
        """Stage 3: Second fine-tuning on high-confidence TAWOS samples.

        Uses smaller learning rate with early stopping as per dissertation.
        Optionally injects pre-labelled data (e.g. from manually-labelled TSV
        files or SERVER.tsv) that bypasses the confidence filter entirely.

        Improvements over the original implementation:
        - Holds out 20% of human-labelled data as an honest validation set
          (not contaminated by pseudo-labels).
        - Computes inverse-frequency class weights for CrossEntropyLoss
          to handle class imbalance without naively undersampling.
        - Uses early stopping on validation F1 (patience=2) with up to 5 epochs.

        Args:
            tawos_texts: Model-predicted TAWOS texts.
            tawos_labels: Model-predicted labels.
            confidences: Prediction confidence scores.
            confidence_threshold: Minimum confidence to include predicted samples.
            extra_texts: Pre-labelled texts to inject unconditionally (confidence=1.0).
            extra_labels: Labels corresponding to extra_texts.
        """
        logger.info("\n" + "="*60)
        logger.info("STAGE 3: Second Fine-tuning on TAWOS Dataset")
        logger.info("="*60)

        # Filter to high-confidence pseudo-labelled samples
        high_conf_mask = [c >= confidence_threshold for c in confidences]
        filtered_texts = [t for t, m in zip(tawos_texts, high_conf_mask) if m]
        filtered_labels = [l for l, m in zip(tawos_labels, high_conf_mask) if m]

        logger.info(f"Using {len(filtered_texts)} high-confidence predicted samples "
                    f"(threshold={confidence_threshold})")

        # ── Hold out human-labelled validation + test sets ────────────
        # Split human-labelled data into train/val/test (60/20/20).
        # Val is used for early stopping; test is fully held-out for
        # honest evaluation after training completes.
        val_texts: List[str] = []
        val_labels: List[int] = []
        test_texts: List[str] = []
        test_labels: List[int] = []
        train_extra_texts: List[str] = []
        train_extra_labels: List[int] = []

        if extra_texts and len(extra_texts) >= 20:
            # First split: 80% trainval / 20% test
            trainval_texts, test_texts, trainval_labels, test_labels = train_test_split(
                extra_texts, extra_labels,
                test_size=0.2,
                random_state=self.config.RANDOM_SEED,
                stratify=extra_labels,
            )
            # Second split: 75% train / 25% val (= 60/20 of original)
            train_extra_texts, val_texts, train_extra_labels, val_labels = train_test_split(
                trainval_texts, trainval_labels,
                test_size=0.25,
                random_state=self.config.RANDOM_SEED,
                stratify=trainval_labels,
            )
            logger.info(f"Human-labelled split: {len(train_extra_texts)} train, "
                        f"{len(val_texts)} val, {len(test_texts)} test (held-out)")
        elif extra_texts:
            # Too few samples to split — use all for training
            train_extra_texts = list(extra_texts)
            train_extra_labels = list(extra_labels)
            logger.info(f"Too few human-labelled samples to split; using all "
                        f"{len(extra_texts)} for training")

        # Combine pseudo-labelled + human-labelled for training
        train_texts = list(train_extra_texts) + filtered_texts
        train_labels_list = list(train_extra_labels) + filtered_labels

        logger.info(f"Stage 3 training set: {len(train_texts)} samples "
                    f"(D={sum(train_labels_list)}, N={len(train_labels_list)-sum(train_labels_list)})")

        # If no held-out human validation, fall back to random split
        if not val_texts:
            logger.info("No held-out human validation; falling back to 80/20 split")
            train_texts, val_texts, train_labels_list, val_labels = train_test_split(
                train_texts, train_labels_list,
                test_size=0.2,
                random_state=self.config.RANDOM_SEED,
                stratify=train_labels_list,
            )

        logger.info(f"Validation set: {len(val_texts)} samples "
                    f"(D={sum(val_labels)}, N={len(val_labels)-sum(val_labels)})")
        if test_texts:
            logger.info(f"Test set (held-out): {len(test_texts)} samples "
                        f"(D={sum(test_labels)}, N={len(test_labels)-sum(test_labels)})")

        # ── Compute class weights (inverse frequency) ────────────────
        n_pos = sum(train_labels_list)
        n_neg = len(train_labels_list) - n_pos
        if n_pos > 0 and n_neg > 0:
            total = len(train_labels_list)
            w_neg = total / (2.0 * n_neg)
            w_pos = total / (2.0 * n_pos)
            class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float)
            logger.info(f"Class weights: non-design={w_neg:.3f}, design={w_pos:.3f}")
        else:
            class_weights = None
            logger.warning("Single-class training set; skipping class weights")

        # Create data loaders
        train_loader, val_loader, _ = self.trainer.create_data_loaders(
            train_texts, train_labels_list,
            val_texts, val_labels
        )

        # Train with smaller learning rate, more epochs, and early stopping
        history = self.trainer.train(
            train_loader,
            val_loader,
            num_epochs=self.config.SECOND_FINETUNE_EPOCHS,
            learning_rate=self.config.SECOND_FINETUNE_LR,
            class_weights=class_weights,
            early_stopping_patience=self.config.SECOND_FINETUNE_PATIENCE,
        )

        # ── Evaluate on held-out test set ────────────────────────────
        if test_texts:
            logger.info("\n" + "="*60)
            logger.info("HELD-OUT TEST EVALUATION (human-labelled, never seen during training)")
            logger.info("="*60)
            test_dataset = DesignMiningDataset(
                test_texts, test_labels,
                self.trainer.tokenizer, self.config.MAX_LENGTH,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=False,
            )
            test_loss, test_metrics = self.trainer.evaluate(test_loader)
            logger.info(f"  Test Loss:      {test_loss:.4f}")
            logger.info(f"  Test Accuracy:  {test_metrics.get('accuracy', 0):.4f}")
            logger.info(f"  Test Precision: {test_metrics.get('precision', 0):.4f}")
            logger.info(f"  Test Recall:    {test_metrics.get('recall', 0):.4f}")
            logger.info(f"  Test F1:        {test_metrics.get('f1_score', 0):.4f}")
            logger.info(f"  Test AUC:       {test_metrics.get('auc', 0):.4f}")
            history['test_metrics'] = test_metrics

        return history

    def export_labeled_tawos(
        self,
        df: pd.DataFrame,
        labels: List[int],
        confidences: List[float],
        output_path: str,
        confidence_threshold: float = 0.0
    ) -> str:
        """Export labeled TAWOS data to CSV.

        Args:
            df: Original DataFrame from TAWOS
            labels: Predicted labels (0=non-design, 1=design)
            confidences: Prediction confidence scores
            output_path: Path to save labeled CSV
            confidence_threshold: Minimum confidence to include

        Returns:
            Path to saved file
        """
        logger.info("\n" + "="*60)
        logger.info("Exporting Labeled TAWOS Dataset")
        logger.info("="*60)

        # Add predictions to DataFrame
        df_labeled = df.copy()
        df_labeled['predicted_label'] = labels
        df_labeled['label_name'] = ['design' if l == 1 else 'non-design' for l in labels]
        df_labeled['confidence'] = confidences

        # Filter by confidence threshold
        if confidence_threshold > 0:
            df_labeled = df_labeled[df_labeled['confidence'] >= confidence_threshold]
            logger.info(f"Filtered to {len(df_labeled)} samples with confidence >= {confidence_threshold}")

        # Save to CSV
        df_labeled.to_csv(output_path, index=False)
        logger.info(f"Saved labeled dataset to: {output_path}")

        # Log statistics
        design_count = sum(1 for l in df_labeled['predicted_label'] if l == 1)
        total = len(df_labeled)
        logger.info(f"Total issues: {total}")
        logger.info(f"Design issues: {design_count} ({design_count/total*100:.1f}%)")
        logger.info(f"Non-design issues: {total - design_count} ({(total-design_count)/total*100:.1f}%)")
        logger.info(f"Average confidence: {df_labeled['confidence'].mean():.4f}")

        return output_path

    def run_full_pipeline_with_db(
        self,
        so_texts: List[str],
        so_labels: List[int],
        tawos_projects: Optional[List[str]] = None,
        include_comments: bool = False,
        max_tawos_issues: Optional[int] = None,
        confidence_threshold: float = 0.8,
        output_path: Optional[str] = None
    ) -> Dict:
        """Run complete transfer learning pipeline with TAWOS database.

        Args:
            so_texts: Stack Overflow texts for initial training
            so_labels: Stack Overflow labels
            tawos_projects: Optional list of Apache projects to include
            include_comments: Whether to include JIRA comments
            max_tawos_issues: Maximum number of TAWOS issues
            confidence_threshold: Threshold for label confidence
            output_path: Path to save labeled TAWOS data

        Returns:
            Dictionary with pipeline results
        """
        logger.info("\n" + "="*60)
        logger.info("FULL TRANSFER LEARNING PIPELINE WITH TAWOS DATABASE")
        logger.info("="*60)

        results = {}

        # Connect to TAWOS
        if not self.connect_tawos_db():
            raise ConnectionError("Failed to connect to TAWOS database")

        try:
            # Stage 1: Fine-tune on Stack Overflow
            results['stage1'] = self.stage1_finetune_stackoverflow(so_texts, so_labels)

            # Fetch TAWOS data from database
            logger.info("\nFetching TAWOS data from database...")
            tawos_texts, tawos_df = self.fetch_tawos_from_db(
                projects=tawos_projects,
                include_comments=include_comments,
                max_issues=max_tawos_issues
            )

            logger.info(f"Fetched {len(tawos_texts)} TAWOS issues")
            results['tawos_fetched'] = len(tawos_texts)

            # Stage 2: Label TAWOS
            tawos_labels, confidences = self.stage2_label_tawos(
                tawos_texts,
                confidence_threshold=confidence_threshold
            )
            results['stage2'] = {
                'total_labeled': len(tawos_labels),
                'high_confidence': sum(1 for c in confidences if c >= confidence_threshold),
                'design_predicted': sum(tawos_labels)
            }

            # Export labeled data
            if output_path:
                self.export_labeled_tawos(
                    tawos_df, tawos_labels, confidences,
                    output_path, confidence_threshold=0.0
                )
                results['labeled_output'] = output_path

            # Stage 3: Second fine-tuning
            results['stage3'] = self.stage3_finetune_tawos(
                tawos_texts, tawos_labels, confidences,
                confidence_threshold=confidence_threshold
            )

        finally:
            if self.tawos_connector:
                self.tawos_connector.disconnect()

        return results


# =============================================================================
# Baseline Comparisons
# =============================================================================

class BaselineComparison:
    """Compare BERT with traditional methods as in dissertation.
    
    Compares against:
    - Logistic Regression
    - Support Vector Machine
    (As done in Mahadi et al. and specified in dissertation)
    """
    
    @staticmethod
    def train_baseline_models(
        train_texts: List[str],
        train_labels: List[int],
        test_texts: List[str],
        test_labels: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """Train and evaluate baseline models."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        logger.info("\n" + "="*60)
        logger.info("Training Baseline Models for Comparison")
        logger.info("="*60)
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        
        results = {}
        
        # Logistic Regression
        logger.info("\nTraining Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, train_labels)
        lr_preds = lr.predict(X_test)
        lr_probs = lr.predict_proba(X_test)[:, 1]
        results['logistic_regression'] = MetricsCalculator.calculate_all_metrics(
            np.array(test_labels), lr_preds, lr_probs
        )
        MetricsCalculator.print_metrics(results['logistic_regression'], "Logistic Regression")
        
        # Support Vector Machine
        logger.info("\nTraining SVM...")
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        svm.fit(X_train, train_labels)
        svm_preds = svm.predict(X_test)
        svm_probs = svm.predict_proba(X_test)[:, 1]
        results['svm'] = MetricsCalculator.calculate_all_metrics(
            np.array(test_labels), svm_preds, svm_probs
        )
        MetricsCalculator.print_metrics(results['svm'], "SVM")
        
        return results


# =============================================================================
# Sample Data Generator (for testing)
# =============================================================================

def generate_sample_data(n_samples: int = 1000) -> Tuple[List[str], List[int]]:
    """Generate sample data for testing the pipeline.
    
    In practice, you would load:
    - Stack Overflow data from: https://zenodo.org/records/4010209
    - TAWOS data from: https://github.com/SOLAR-group/TAWOS
    """
    
    # Design-related keywords
    design_templates = [
        "We need to decide on the architecture pattern for {component}. Should we use {pattern1} or {pattern2}?",
        "Design decision: implementing {feature} using {approach} to improve {quality}",
        "Architecture review needed for the {component} module integration",
        "Proposing a new design for {component} to address scalability concerns",
        "Technical debt: refactoring {component} architecture to use {pattern1}",
        "Design document for {feature} implementation",
        "System design: how should we structure the {component} layer?",
        "Architecture decision record for {feature} implementation approach",
        "Design review: evaluating {pattern1} vs {pattern2} for {component}",
        "Architectural spike: investigating {pattern1} for {feature}"
    ]
    
    # Non-design (general) templates
    general_templates = [
        "Bug: {component} crashes when user clicks {action}",
        "Fix typo in {component} error message",
        "Update {component} dependency to version {version}",
        "Add unit tests for {component} module",
        "Documentation update for {feature}",
        "Performance: optimize {component} query response time",
        "UI: change button color in {component} screen",
        "Logging: add debug logs to {component}",
        "Hotfix: {component} returns null pointer exception",
        "Chore: clean up unused imports in {component}"
    ]
    
    components = ['UserService', 'PaymentGateway', 'AuthModule', 'DataPipeline', 
                  'APIGateway', 'CacheLayer', 'MessageQueue', 'DatabaseConnector']
    patterns = ['microservices', 'monolith', 'event-driven', 'layered', 'CQRS', 'saga']
    features = ['authentication', 'caching', 'logging', 'monitoring', 'notifications']
    qualities = ['scalability', 'maintainability', 'performance', 'security']
    actions = ['submit', 'login', 'logout', 'save', 'delete']
    versions = ['2.0', '3.1', '4.0', '5.2']
    approaches = ['dependency injection', 'factory pattern', 'observer pattern', 'repository pattern']
    
    import random
    random.seed(42)
    
    texts = []
    labels = []
    
    for _ in range(n_samples):
        # 30% design, 70% general (realistic distribution)
        is_design = random.random() < 0.3
        
        if is_design:
            template = random.choice(design_templates)
            labels.append(1)
        else:
            template = random.choice(general_templates)
            labels.append(0)
        
        text = template.format(
            component=random.choice(components),
            pattern1=random.choice(patterns),
            pattern2=random.choice(patterns),
            feature=random.choice(features),
            quality=random.choice(qualities),
            action=random.choice(actions),
            version=random.choice(versions),
            approach=random.choice(approaches)
        )
        texts.append(text)
    
    return texts, labels


# =============================================================================
# Metrics Tracking
# =============================================================================

def save_metrics_to_csv(
    metrics: Dict,
    config: Config,
    output_file: str = 'training_metrics.csv'
):
    """Save training metrics to CSV file for tracking across runs.

    Args:
        metrics: Dictionary containing all metrics to save
        config: Configuration object with hyperparameters
        output_file: Path to CSV file (will append if exists)
    """
    import csv
    from pathlib import Path

    # Check if file exists to determine if we need headers
    file_exists = Path(output_file).exists()

    # Add timestamp as run ID
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    metrics['run_id'] = run_id
    metrics['timestamp'] = datetime.now().isoformat()

    # Flatten the metrics dictionary
    flat_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            # Flatten nested dictionaries
            for sub_key, sub_value in value.items():
                flat_metrics[f"{key}_{sub_key}"] = sub_value
        else:
            flat_metrics[key] = value

    # Write to CSV
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat_metrics.keys()))

        # Write header if new file
        if not file_exists:
            writer.writeheader()

        writer.writerow(flat_metrics)

    logger.info(f"Metrics saved to {output_file} (run_id: {run_id})")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function to run the training pipeline."""
    
    parser = argparse.ArgumentParser(
        description='BERT-based Design Mining for Software Architecture Issues'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        default='demo',
        choices=['demo', 'full', 'transfer'],
        help='Training mode: demo (sample data), full (real data), transfer (full pipeline)'
    )
    parser.add_argument(
        '--stackoverflow_path',
        type=str,
        default=None,
        help='Path to Stack Overflow dataset'
    )
    parser.add_argument(
        '--tawos_path',
        type=str,
        default=None,
        help='Path to a TAWOS TSV file OR a directory of TSV files. '
             'When a directory is given, all .tsv files with a "text" column '
             'are combined and Stage 2 predictions are appended to any row '
             'with a missing/empty predicted_label.'
    )
    parser.add_argument(
        '--val_data',
        type=str,
        default=None,
        help='Path to held-out validation CSV (if not provided, validation split is taken from training data)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Transformer model name (e.g. bert-base-uncased, distilbert-base-uncased, roberta-base). Overrides Config.MODEL_NAME.'
    )
    parser.add_argument(
        '--pretrained_model',
        type=str,
        default=None,
        help='Path to a previously fine-tuned model to use for transfer learning (skips Stage 1)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./model_output',
        help='Directory to save model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Learning rate for training'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate (0.0 to 1.0)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length for BERT'
    )
    parser.add_argument(
        '--min_words',
        type=int,
        default=7,
        help='Minimum meaningful words after stopword removal'
    )
    parser.add_argument(
        '--no_preprocess',
        action='store_true',
        default=False,
        help='Skip DataPreprocessor entirely (recommended for full datasets with BERT)'
    )
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.1,
        help='Warmup ratio for learning rate scheduler'
    )

    # TAWOS Database connection arguments
    # Credentials are read from environment variables for security
    parser.add_argument(
        '--db_host',
        type=str,
        default=os.environ.get('TAWOS_DB_HOST', 'localhost'),
        help='TAWOS MySQL database host (env: TAWOS_DB_HOST)'
    )
    parser.add_argument(
        '--db_port',
        type=int,
        default=int(os.environ.get('TAWOS_DB_PORT', '3306')),
        help='TAWOS MySQL database port (env: TAWOS_DB_PORT)'
    )
    parser.add_argument(
        '--db_name',
        type=str,
        default=os.environ.get('TAWOS_DB_NAME', 'tawos'),
        help='TAWOS database name (env: TAWOS_DB_NAME)'
    )
    parser.add_argument(
        '--db_user',
        type=str,
        default=os.environ.get('TAWOS_DB_USER', 'root'),
        help='TAWOS database user (env: TAWOS_DB_USER)'
    )
    parser.add_argument(
        '--db_password',
        type=str,
        default=os.environ.get('TAWOS_DB_PASSWORD', ''),
        help='TAWOS database password (env: TAWOS_DB_PASSWORD)'
    )
    parser.add_argument(
        '--tawos_projects',
        type=str,
        nargs='+',
        default=None,
        help='List of Apache project keys to include (e.g., HADOOP SPARK)'
    )
    parser.add_argument(
        '--include_comments',
        action='store_true',
        help='Include JIRA comments in text'
    )
    parser.add_argument(
        '--include_issue_type',
        action='store_true',
        help='Prepend issue type token (e.g. [TYPE: Bug]) to the classification text'
    )
    parser.add_argument(
        '--max_tawos_issues',
        type=int,
        default=None,
        help='Maximum number of TAWOS issues to fetch'
    )
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=0.85,
        help='Confidence threshold for labeling (0.0-1.0)'
    )
    parser.add_argument(
        '--labeled_output',
        type=str,
        default=None,
        help='Path to save labeled TAWOS dataset CSV'
    )
    parser.add_argument(
        '--manually_labelled_dir',
        type=str,
        default=None,
        help='Path to directory of manually-labelled TSV files (D/N and design/non-design labels). '
             'These are injected into Stage 3 with confidence=1.0, bypassing the confidence filter.'
    )
    parser.add_argument(
        '--max_n_labels',
        type=int,
        default=400,
        help='Maximum number of N (non-design) labels to use from manually-labelled files '
             'to reduce class imbalance during Stage 3 fine-tuning (default: 400)'
    )
    parser.add_argument(
        '--skip_stage2',
        action='store_true',
        help='Skip Stage 2 TAWOS pseudo-labeling entirely. Stage 3 will train only on '
             '--manually_labelled_dir data. Requires --manually_labelled_dir to be set.'
    )

    args = parser.parse_args()

    # Configuration
    config = Config()
    config.OUTPUT_DIR = args.output_dir
    if args.model:
        config.MODEL_NAME = args.model
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.learning_rate
    config.BATCH_SIZE = args.batch_size
    config.DROPOUT_RATE = args.dropout
    config.MAX_LENGTH = args.max_length
    config.WARMUP_RATIO = args.warmup_ratio
    
    logger.info("="*60)
    logger.info("BERT-based Design Mining Training Pipeline")
    logger.info("Based on: 'An Algorithmic Approach to Understanding the")
    logger.info("Impact of Design Work in Software Projects'")
    logger.info("="*60)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Device: {config.DEVICE}")
    logger.info(f"  Model: {config.MODEL_NAME}")
    logger.info(f"  Max Length: {config.MAX_LENGTH}")
    logger.info(f"  Batch Size: {config.BATCH_SIZE}")
    logger.info(f"  Learning Rate: {config.LEARNING_RATE}")
    logger.info(f"  Epochs: {config.NUM_EPOCHS}")
    
    if args.mode == 'demo':
        # Demo mode with sample data
        logger.info("\n" + "="*60)
        logger.info("Running in DEMO mode with sample data")
        logger.info("="*60)
        
        # Generate sample data
        texts, labels = generate_sample_data(n_samples=500)
        
        # Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels,
            test_size=0.3,
            random_state=config.RANDOM_SEED,
            stratify=labels
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels,
            test_size=0.5,
            random_state=config.RANDOM_SEED,
            stratify=temp_labels
        )
        
        logger.info(f"\nData split:")
        logger.info(f"  Training: {len(train_texts)} samples")
        logger.info(f"  Validation: {len(val_texts)} samples")
        logger.info(f"  Test: {len(test_texts)} samples")
        
        # Initialize trainer
        trainer = DesignMiningTrainer(config)
        
        # Create data loaders
        train_loader, val_loader, test_loader = trainer.create_data_loaders(
            train_texts, train_labels,
            val_texts, val_labels,
            test_texts, test_labels
        )
        
        # Train
        history = trainer.train(train_loader, val_loader)
        
        # Final evaluation
        if test_loader:
            logger.info("\n" + "="*60)
            logger.info("Final Test Set Evaluation")
            logger.info("="*60)
            test_loss, test_metrics = trainer.evaluate(test_loader)
            MetricsCalculator.print_metrics(test_metrics, "Test")
        
        # Train baseline models for comparison
        baseline_results = BaselineComparison.train_baseline_models(
            train_texts, train_labels,
            test_texts, test_labels
        )
        
        # Compare results
        logger.info("\n" + "="*60)
        logger.info("Model Comparison Summary")
        logger.info("="*60)
        logger.info(f"{'Model':<25} {'Accuracy':<12} {'F1 Score':<12} {'AUC':<12}")
        logger.info("-" * 60)
        logger.info(f"{'BERT':<25} {test_metrics['accuracy']:<12.4f} "
                   f"{test_metrics['f1_score']:<12.4f} {test_metrics.get('auc', 0):<12.4f}")
        for model_name, metrics in baseline_results.items():
            logger.info(f"{model_name:<25} {metrics['accuracy']:<12.4f} "
                       f"{metrics['f1_score']:<12.4f} {metrics.get('auc', 0):<12.4f}")
        
        # Save model
        trainer.save_model(config.OUTPUT_DIR)

    elif args.mode == 'full':
        # Full mode with real data from CSV
        logger.info("\n" + "="*60)
        logger.info("Running in FULL mode with real data")
        logger.info("="*60)

        if not args.stackoverflow_path:
            logger.error("Full mode requires --stackoverflow_path")
            return

        # Load data from CSV
        logger.info(f"\nLoading data from: {args.stackoverflow_path}")
        try:
            df = pd.read_csv(args.stackoverflow_path)
            logger.info(f"Loaded {len(df)} records")
            logger.info(f"Columns: {list(df.columns)}")

            # Check for required columns
            # Assuming CSV has 'text' and 'label' columns
            # Adjust these column names based on your actual CSV structure
            text_col = 'text' if 'text' in df.columns else df.columns[0]
            label_col = 'label' if 'label' in df.columns else df.columns[1]

            logger.info(f"Using text column: '{text_col}'")
            logger.info(f"Using label column: '{label_col}'")

            # Display label distribution before encoding
            if label_col in df.columns:
                label_dist = df[label_col].value_counts()
                logger.info(f"\nOriginal label distribution:")
                for label, count in label_dist.items():
                    logger.info(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

            # Encode labels to integers
            logger.info("\nEncoding labels to integers...")
            unique_labels = df[label_col].unique()
            logger.info(f"Unique labels found: {unique_labels}")

            # Create label mapping
            # Try to detect if labels are already numeric
            try:
                # Check if labels can be converted to int
                df[label_col] = df[label_col].astype(int)
                logger.info("Labels are numeric - using as-is")
                label_mapping = {i: i for i in df[label_col].unique()}
            except (ValueError, TypeError):
                # Labels are categorical strings - create mapping
                # Assume 'design' or similar -> 1, 'general' or 'non-design' -> 0
                label_mapping = {}
                for label in unique_labels:
                    label_str = str(label).lower()
                    if 'design' in label_str or label_str in ['1', 'true', 'yes']:
                        label_mapping[label] = 1
                    else:
                        label_mapping[label] = 0

                logger.info(f"Label mapping: {label_mapping}")
                df[label_col] = df[label_col].map(label_mapping)

            # Verify all labels are 0 or 1
            if not set(df[label_col].unique()).issubset({0, 1}):
                logger.error(f"Invalid labels after encoding: {df[label_col].unique()}")
                logger.error("Labels must be 0 (non-design) or 1 (design)")
                return

            logger.info(f"\nEncoded label distribution:")
            label_dist = df[label_col].value_counts()
            for label, count in label_dist.items():
                label_name = "design" if label == 1 else "non-design"
                logger.info(f"  {label} ({label_name}): {count} ({count/len(df)*100:.1f}%)")

            if args.no_preprocess:
                logger.info("\nSkipping preprocessing (--no_preprocess set)")
                df_clean = df
                texts = df_clean[text_col].tolist()
                labels = df_clean[label_col].tolist()
            else:
                # Initialize preprocessor
                logger.info("\nInitializing data preprocessor...")
                preprocessor = DataPreprocessor(min_words=args.min_words, verbose=True)

                # Preprocess data
                df_clean = preprocessor.preprocess_dataframe(df, text_col)

                # Extract texts and labels (labels are now integers)
                texts = df_clean[text_col].tolist()
                labels = df_clean[label_col].tolist()

            # Verify labels are integers
            logger.info(f"Label types after extraction: {type(labels[0]) if labels else 'empty'}")
            logger.info(f"Sample labels: {labels[:5] if len(labels) >= 5 else labels}")

            # Split data
            if args.val_data:
                # Use separate validation file
                logger.info(f"\nLoading validation data from: {args.val_data}")
                df_val = pd.read_csv(args.val_data)
                logger.info(f"Loaded {len(df_val)} validation records")

                # Detect columns in validation data
                val_text_col = 'text' if 'text' in df_val.columns else df_val.columns[0]
                val_label_col = 'label' if 'label' in df_val.columns else df_val.columns[1]

                # Encode validation labels
                try:
                    df_val[val_label_col] = df_val[val_label_col].astype(int)
                except (ValueError, TypeError):
                    for label in df_val[val_label_col].unique():
                        label_str = str(label).lower()
                        if 'design' in label_str or label_str in ['1', 'true', 'yes']:
                            label_mapping[label] = 1
                        else:
                            label_mapping[label] = 0
                    df_val[val_label_col] = df_val[val_label_col].map(label_mapping)

                # Preprocess validation data
                val_preprocessor = DataPreprocessor(min_words=args.min_words, verbose=False)
                df_val_clean = val_preprocessor.preprocess_dataframe(df_val, val_text_col)

                val_texts = df_val_clean[val_text_col].tolist()
                val_labels = df_val_clean[val_label_col].tolist()

                # Split training data into train + test only
                train_texts, test_texts, train_labels, test_labels = train_test_split(
                    texts, labels,
                    test_size=0.15,
                    random_state=config.RANDOM_SEED,
                    stratify=labels
                )
            else:
                # Split training data into train/val/test
                train_texts, temp_texts, train_labels, temp_labels = train_test_split(
                    texts, labels,
                    test_size=0.3,
                    random_state=config.RANDOM_SEED,
                    stratify=labels
                )
                val_texts, test_texts, val_labels, test_labels = train_test_split(
                    temp_texts, temp_labels,
                    test_size=0.5,
                    random_state=config.RANDOM_SEED,
                    stratify=temp_labels
                )

            total_samples = len(train_texts) + len(val_texts) + len(test_texts)
            logger.info(f"\n" + "="*60)
            logger.info("DATA SPLIT")
            logger.info("="*60)
            if args.val_data:
                logger.info(f"  Validation source: {args.val_data}")
            logger.info(f"  Training:   {len(train_texts):>6,} samples")
            logger.info(f"  Validation: {len(val_texts):>6,} samples")
            logger.info(f"  Test:       {len(test_texts):>6,} samples")
            logger.info("="*60)

            # Initialize trainer
            logger.info("\nInitializing BERT model...")
            trainer = DesignMiningTrainer(config)

            # Create data loaders
            logger.info("Creating data loaders...")
            train_loader, val_loader, test_loader = trainer.create_data_loaders(
                train_texts, train_labels,
                val_texts, val_labels,
                test_texts, test_labels
            )

            # Train
            logger.info("\nStarting training...")
            history = trainer.train(train_loader, val_loader)

            # Final evaluation
            test_metrics = None
            if test_loader:
                logger.info("\n" + "="*60)
                logger.info("Final Test Set Evaluation")
                logger.info("="*60)
                test_loss, test_metrics = trainer.evaluate(test_loader)
                MetricsCalculator.print_metrics(test_metrics, "Test")

            # Save model
            trainer.save_model(config.OUTPUT_DIR)

            # Collect and save metrics to CSV
            logger.info("\n" + "="*60)
            logger.info("Saving metrics to CSV")
            logger.info("="*60)

            metrics_to_save = {
                # Configuration
                'mode': args.mode,
                'model_name': config.MODEL_NAME,
                'max_length': config.MAX_LENGTH,
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'num_epochs': config.NUM_EPOCHS,
                'dropout_rate': config.DROPOUT_RATE,
                'random_seed': config.RANDOM_SEED,

                # Data statistics
                'data_source': args.stackoverflow_path,
                'initial_samples': len(df),
                'final_samples': len(df_clean),
                'train_samples': len(train_texts),
                'val_samples': len(val_texts),
                'test_samples': len(test_texts),
                'duplicates_removed': preprocessor.stats['duplicates_removed'] if not args.no_preprocess else 0,
                'auto_generated_removed': preprocessor.stats['auto_generated_removed'] if not args.no_preprocess else 0,
                'short_texts_removed': preprocessor.stats['short_texts_removed'] if not args.no_preprocess else 0,

                # Best validation metrics (from training history)
                'best_val_epoch': len(history['val_metrics']),
                'best_val': history['val_metrics'][-1] if history['val_metrics'] else {},

                # Final test metrics
                'test': test_metrics if test_metrics else {},

                # Training info
                'output_dir': config.OUTPUT_DIR,
                'device': str(config.DEVICE),
            }

            # Save to CSV
            csv_path = os.path.join(config.OUTPUT_DIR, 'training_metrics.csv')
            save_metrics_to_csv(metrics_to_save, config, csv_path)

            # Also save a detailed JSON version
            json_path = os.path.join(config.OUTPUT_DIR, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(json_path, 'w') as f:
                # Add full training history
                metrics_to_save['full_history'] = {
                    'train_loss': [float(x) for x in history['train_loss']],
                    'val_loss': [float(x) for x in history['val_loss']],
                    'val_metrics_per_epoch': history['val_metrics']
                }
                json.dump(metrics_to_save, f, indent=2, default=str)

            logger.info(f"Detailed metrics saved to {json_path}")

        except FileNotFoundError:
            logger.error(f"File not found: {args.stackoverflow_path}")
            return
        except Exception as e:
            logger.error(f"Error loading or processing data: {e}")
            import traceback
            traceback.print_exc()
            return

    elif args.mode == 'transfer':
        # Full transfer learning pipeline
        logger.info("\n" + "="*60)
        logger.info("Running TRANSFER LEARNING pipeline")
        logger.info("="*60)

        use_pretrained = args.pretrained_model is not None

        if not use_pretrained and not args.stackoverflow_path:
            logger.error("Transfer mode requires --stackoverflow_path or --pretrained_model")
            return

        # Check if using database or file for TAWOS
        use_db = args.tawos_path is None

        if use_db:
            if not TAWOS_AVAILABLE:
                logger.error("TAWOS database mode requires mysql-connector-python")
                logger.error("Install with: pip install mysql-connector-python")
                return

            logger.info("Using TAWOS MySQL database connection")
            logger.info(f"  Host: {args.db_host}:{args.db_port}")
            logger.info(f"  Database: {args.db_name}")
            logger.info(f"  User: {args.db_user}")
            if args.tawos_projects:
                logger.info(f"  Projects: {', '.join(args.tawos_projects)}")

        # Initialize pipeline
        tawos_config = None
        if use_db:
            tawos_config = TAWOSConfig(
                host=args.db_host,
                port=args.db_port,
                database=args.db_name,
                user=args.db_user,
                password=args.db_password,
                projects=args.tawos_projects or [],
                max_issues=args.max_tawos_issues,
                include_issue_type=args.include_issue_type
            )

        pipeline = TransferLearningPipeline(config, tawos_config=tawos_config)

        # Load manually-labelled data for Stage 3 injection (if provided)
        manually_labelled_texts: Optional[List[str]] = None
        manually_labelled_labels: Optional[List[int]] = None
        if args.manually_labelled_dir:
            manually_labelled_texts, manually_labelled_labels = \
                pipeline.load_manually_labelled_data(
                    args.manually_labelled_dir,
                    max_n_labels=args.max_n_labels,
                    random_seed=config.RANDOM_SEED,
                )

        # Stage 1: Either load pre-trained model or train from scratch
        if use_pretrained:
            logger.info(f"\nLoading pre-trained model from: {args.pretrained_model}")
            logger.info("Skipping Stage 1 (Stack Overflow fine-tuning)")
            pipeline.trainer.load_model(args.pretrained_model)
        else:
            # Load Stack Overflow data and train
            logger.info(f"\nLoading Stack Overflow data from: {args.stackoverflow_path}")
            try:
                df_so = pd.read_csv(args.stackoverflow_path)
                text_col = 'text' if 'text' in df_so.columns else df_so.columns[0]
                label_col = 'label' if 'label' in df_so.columns else df_so.columns[1]
                try:
                    df_so[label_col] = df_so[label_col].astype(int)
                except (ValueError, TypeError):
                    unique_labels = df_so[label_col].unique()
                    label_mapping = {}
                    for label in unique_labels:
                        label_str = str(label).lower()
                        if 'design' in label_str or label_str in ['1', 'true', 'yes']:
                            label_mapping[label] = 1
                        else:
                            label_mapping[label] = 0
                    df_so[label_col] = df_so[label_col].map(label_mapping)

                so_texts = df_so[text_col].tolist()
                so_labels = df_so[label_col].tolist()
                logger.info(f"Loaded {len(so_texts)} Stack Overflow samples")
                logger.info(f"Design: {sum(so_labels)}, Non-design: {len(so_labels) - sum(so_labels)}")
            except Exception as e:
                logger.error(f"Error loading Stack Overflow data: {e}")
                return

            pipeline.stage1_finetune_stackoverflow(so_texts, so_labels)

        # Stage 2 & 3: Fetch TAWOS data, label, and fine-tune
        if args.skip_stage2:
            logger.info("\nSkipping Stage 2 (--skip_stage2 set)")
            logger.info("Stage 3 will train only on manually-labelled data")
            if not manually_labelled_texts:
                logger.error("--skip_stage2 requires --manually_labelled_dir with valid TSV files")
                return
            pipeline.stage3_finetune_tawos(
                [], [], [],
                confidence_threshold=args.confidence_threshold,
                extra_texts=manually_labelled_texts,
                extra_labels=manually_labelled_labels,
            )
        elif use_db:
            if not pipeline.connect_tawos_db():
                logger.error("Failed to connect to TAWOS database")
                return

            try:
                logger.info("\nFetching TAWOS data from database...")
                tawos_texts, tawos_df = pipeline.fetch_tawos_from_db(
                    projects=args.tawos_projects,
                    include_comments=args.include_comments,
                    max_issues=args.max_tawos_issues
                )
                logger.info(f"Fetched {len(tawos_texts)} TAWOS issues")

                # Stage 2: Label TAWOS
                tawos_labels, confidences = pipeline.stage2_label_tawos(
                    tawos_texts,
                    confidence_threshold=args.confidence_threshold
                )

                # Export labeled data
                output_path = args.labeled_output or os.path.join(
                    config.OUTPUT_DIR,
                    f'tawos_labeled_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                )
                pipeline.export_labeled_tawos(
                    tawos_df, tawos_labels, confidences,
                    output_path, confidence_threshold=0.0
                )

                # Stage 3: Second fine-tuning
                pipeline.stage3_finetune_tawos(
                    tawos_texts, tawos_labels, confidences,
                    confidence_threshold=args.confidence_threshold,
                    extra_texts=manually_labelled_texts,
                    extra_labels=manually_labelled_labels,
                )

                logger.info("\n" + "="*60)
                logger.info("TRANSFER LEARNING RESULTS")
                logger.info("="*60)
                logger.info(f"TAWOS issues fetched: {len(tawos_texts)}")
                logger.info(f"Design issues predicted: {sum(tawos_labels)}")
                logger.info(f"High confidence labels: {sum(1 for c in confidences if c >= args.confidence_threshold)}")
                logger.info(f"Labeled dataset saved to: {output_path}")

            finally:
                if pipeline.tawos_connector:
                    pipeline.tawos_connector.disconnect()
        else:
            # File-based TAWOS mode (accepts a single .tsv/.csv file OR a directory of TSVs)
            import glob as _glob
            import csv as _csv
            from pathlib import Path
            tawos_path = Path(args.tawos_path)

            if tawos_path.is_dir():
                logger.info(f"Loading TAWOS data from directory: {tawos_path}")
                tsv_files = sorted(tawos_path.glob('*.tsv'))
                frames = []
                for fp in tsv_files:
                    try:
                        _csv.field_size_limit(10 ** 7)
                        df_part = pd.read_csv(fp, sep='\t', dtype=str)
                        if 'text' not in df_part.columns:
                            logger.warning(f"  {fp.name}: no 'text' column, skipping")
                            continue
                        df_part = df_part[df_part['text'].notna() & (df_part['text'].str.strip() != '')]
                        logger.info(f"  {fp.name}: {len(df_part)} rows with text")
                        frames.append(df_part)
                    except Exception as e:
                        logger.warning(f"  {fp.name}: error reading – {e}")
                if not frames:
                    logger.error(f"No usable TSV files found in {tawos_path}")
                    return
                df_tawos = pd.concat(frames, ignore_index=True)
                logger.info(f"Combined: {len(df_tawos)} TAWOS samples from {len(frames)} files")
            else:
                logger.info(f"Loading TAWOS data from file: {tawos_path}")
                try:
                    _csv.field_size_limit(10 ** 7)
                    df_tawos = pd.read_csv(tawos_path, sep='\t', dtype=str)
                except Exception as e:
                    logger.error(f"Error loading TAWOS data: {e}")
                    return

            text_col = 'text' if 'text' in df_tawos.columns else df_tawos.columns[0]
            tawos_texts = df_tawos[text_col].fillna('').tolist()
            logger.info(f"Loaded {len(tawos_texts)} TAWOS samples")

            # Stage 2: Label TAWOS
            tawos_labels, confidences = pipeline.stage2_label_tawos(
                tawos_texts,
                confidence_threshold=args.confidence_threshold
            )

            # Export labeled data if output path specified
            if args.labeled_output:
                df_tawos['predicted_label'] = tawos_labels
                df_tawos['confidence'] = confidences
                df_tawos['label_name'] = ['design' if l == 1 else 'non-design' for l in tawos_labels]
                df_tawos.to_csv(args.labeled_output, index=False)
                logger.info(f"Saved labeled dataset to: {args.labeled_output}")

            # Stage 3: Second fine-tuning
            pipeline.stage3_finetune_tawos(
                tawos_texts, tawos_labels, confidences,
                confidence_threshold=args.confidence_threshold,
                extra_texts=manually_labelled_texts,
                extra_labels=manually_labelled_labels,
            )

        # Save final model
        pipeline.trainer.save_model(config.OUTPUT_DIR)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info(f"Model saved to: {config.OUTPUT_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
