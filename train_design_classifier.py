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
    BertTokenizer, 
    BertForSequenceClassification,
    BertConfig,
    get_linear_schedule_with_warmup
)

from tqdm import tqdm

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
    MODEL_NAME = 'bert-base-uncased'
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
    SECOND_FINETUNE_EPOCHS = 2  # "less epochs to ensure modest adaptation"
    
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
        tokenizer: BertTokenizer,
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
        
        # Tokenize with BERT tokenizer
        # This automatically adds [CLS] and [SEP] tokens
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
        tokenizer: BertTokenizer,
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
    
    # Software engineering stop words (based on Mahadi et al. vocabulary)
    SOFTWARE_STOPWORDS = {
        'null', 'none', 'n/a', 'na', 'undefined', 'todo', 'fixme',
        'http', 'https', 'www', 'com', 'org', 'net', 'io',
        'github', 'gitlab', 'bitbucket', 'jira', 'confluence',
        'import', 'export', 'class', 'def', 'function', 'return',
        'true', 'false', 'boolean', 'string', 'int', 'float',
        'public', 'private', 'protected', 'static', 'void',
        'try', 'catch', 'finally', 'throw', 'throws', 'exception'
    }
    
    def __init__(self, min_words: int = 7):
        self.min_words = min_words
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        import re
        
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        
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
        
        return text.strip()
    
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
                return True
        return False
    
    def word_count_after_stopwords(self, text: str) -> int:
        """Count words after removing stop words."""
        words = text.lower().split()
        meaningful_words = [w for w in words if w not in self.SOFTWARE_STOPWORDS]
        return len(meaningful_words)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Apply all preprocessing steps to a dataframe."""
        logger.info(f"Starting preprocessing on {len(df)} records...")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=[text_col])
        logger.info(f"Removed {initial_count - len(df)} duplicates")
        
        # Clean text
        df[text_col] = df[text_col].apply(self.clean_text)
        
        # Remove auto-generated
        auto_mask = df[text_col].apply(self.is_auto_generated)
        df = df[~auto_mask]
        logger.info(f"Removed {auto_mask.sum()} auto-generated entries")
        
        # Remove short texts
        word_counts = df[text_col].apply(self.word_count_after_stopwords)
        df = df[word_counts >= self.min_words]
        logger.info(f"Removed {(word_counts < self.min_words).sum()} short entries")
        
        logger.info(f"Final dataset size: {len(df)} records")
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
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
        
        # Initialize model with configuration
        model_config = BertConfig.from_pretrained(
            config.MODEL_NAME,
            num_labels=config.NUM_LABELS,
            hidden_dropout_prob=config.DROPOUT_RATE,
            attention_probs_dropout_prob=config.DROPOUT_RATE
        )
        
        self.model = BertForSequenceClassification.from_pretrained(
            config.MODEL_NAME,
            config=model_config
        )
        self.model.to(self.device)
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator()
        
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
        learning_rate: Optional[float] = None
    ) -> Dict:
        """Full training loop."""
        
        num_epochs = num_epochs or self.config.NUM_EPOCHS
        learning_rate = learning_rate or self.config.LEARNING_RATE
        
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
        
        best_f1 = 0
        best_model_state = None
        
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
                logger.info(f"New best F1 score: {best_f1:.4f}")
            
            # Early stopping check - if F1 not improving
            if epoch > 0:
                prev_f1 = self.history['val_metrics'][-2]['f1_score']
                if val_metrics['f1_score'] <= prev_f1 - 0.01:
                    logger.info("F1 score declining, consider early stopping")
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"\nRestored best model with F1: {best_f1:.4f}")
        
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
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
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
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.trainer = DesignMiningTrainer(config)
    
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
        confidence_threshold: float = 0.8
    ) -> Dict:
        """Stage 3: Second fine-tuning on high-confidence TAWOS samples.
        
        Uses smaller learning rate and fewer epochs as per dissertation.
        """
        logger.info("\n" + "="*60)
        logger.info("STAGE 3: Second Fine-tuning on TAWOS Dataset")
        logger.info("="*60)
        
        # Filter to high-confidence samples
        high_conf_mask = [c >= confidence_threshold for c in confidences]
        filtered_texts = [t for t, m in zip(tawos_texts, high_conf_mask) if m]
        filtered_labels = [l for l, m in zip(tawos_labels, high_conf_mask) if m]
        
        logger.info(f"Using {len(filtered_texts)} high-confidence samples")
        
        # Split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            filtered_texts, filtered_labels,
            test_size=0.2,
            random_state=self.config.RANDOM_SEED,
            stratify=filtered_labels
        )
        
        # Create data loaders
        train_loader, val_loader, _ = self.trainer.create_data_loaders(
            train_texts, train_labels,
            val_texts, val_labels
        )
        
        # Train with smaller learning rate and fewer epochs
        history = self.trainer.train(
            train_loader, 
            val_loader,
            num_epochs=self.config.SECOND_FINETUNE_EPOCHS,
            learning_rate=self.config.SECOND_FINETUNE_LR
        )
        
        return history


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
        help='Path to TAWOS dataset'
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
    
    args = parser.parse_args()
    
    # Configuration
    config = Config()
    config.OUTPUT_DIR = args.output_dir
    config.NUM_EPOCHS = args.epochs
    
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
        
    elif args.mode == 'transfer':
        # Full transfer learning pipeline
        logger.info("\n" + "="*60)
        logger.info("Running TRANSFER LEARNING pipeline")
        logger.info("="*60)
        
        if not args.stackoverflow_path or not args.tawos_path:
            logger.error("Transfer mode requires --stackoverflow_path and --tawos_path")
            return
        
        # Load datasets (you'll need to implement proper loading)
        # This is a placeholder - replace with actual data loading
        logger.info(f"Loading Stack Overflow data from: {args.stackoverflow_path}")
        logger.info(f"Loading TAWOS data from: {args.tawos_path}")
        
        # For demo, generate sample data
        so_texts, so_labels = generate_sample_data(1000)
        tawos_texts, _ = generate_sample_data(500)
        
        # Initialize transfer learning pipeline
        pipeline = TransferLearningPipeline(config)
        
        # Stage 1: Fine-tune on Stack Overflow
        pipeline.stage1_finetune_stackoverflow(so_texts, so_labels)
        
        # Stage 2: Label TAWOS
        tawos_labels, confidences = pipeline.stage2_label_tawos(tawos_texts)
        
        # Stage 3: Second fine-tuning
        pipeline.stage3_finetune_tawos(tawos_texts, tawos_labels, confidences)
        
        # Save final model
        pipeline.trainer.save_model(config.OUTPUT_DIR)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info(f"Model saved to: {config.OUTPUT_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
