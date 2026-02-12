"""
Hyperparameter Search for Design Mining Model
==============================================

Run experiments with different hyperparameter configurations to find
the best settings for your dataset. Includes validation on held-out data.

Usage:
# Run all suggested configs
python hyperparameter_search.py --train_data data/data/train_data/raw/combined.csv \
--val_data data/data/validation_data/raw/validation.csv \
--output_dir models/design_mining \
--run_all

# Grid search
python hyperparameter_search.py --train_data data/data/train_data/raw/combined.csv \
    --val_data data/data/validation_data/raw/validation.csv \
    --output_dir models/design_mining \
    --grid_search

# Run specific config
python hyperparameter_search.py \
--train_data path/to/train.csv \
--val_data path/to/val.csv \
--config balanced
"""

import os
import sys
import time
import json
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


# Define hyperparameter search space
HYPERPARAMETER_GRID = {
    # Learning rates to try
    'learning_rate': [
        1e-5,   # Lower - more conservative
        2e-5,   # Current default
        3e-5,   # Higher - faster learning
        5e-5    # Even higher
    ],

    # Batch sizes
    'batch_size': [
        8,      # Smaller - better gradients, slower
        16,     # Current default
        32,     # Larger - faster training
    ],

    # Training epochs
    'epochs': [
        3,      # Fewer - prevent overfitting
        5,      # Current default
        7,      # More - better convergence
        10      # Many - risk overfitting
    ],

    # Dropout rates
    'dropout': [
        0.1,    # Current default - low
        0.2,    # Medium
        0.3,    # Higher - more regularization
    ],

    # Min words threshold
    'min_words': [
        5,      # Include shorter texts
        7,      # Current default
        10,     # Stricter filtering
    ],
}


# Suggested configurations based on common patterns
SUGGESTED_CONFIGS = [
    # Conservative: prevent overfitting
    {
        'name': 'conservative',
        'learning_rate': 1e-5,
        'batch_size': 8,
        'epochs': 7,
        'dropout': 0.3,
        'min_words': 7,
        'description': 'Low LR, small batch, high dropout to prevent overfitting'
    },

    # Aggressive: maximize learning
    {
        'name': 'aggressive',
        'learning_rate': 5e-5,
        'batch_size': 32,
        'epochs': 10,
        'dropout': 0.1,
        'min_words': 5,
        'description': 'High LR, large batch, many epochs for maximum learning'
    },

    # Balanced: good starting point
    {
        'name': 'balanced',
        'learning_rate': 2e-5,
        'batch_size': 16,
        'epochs': 10,
        'dropout': 0.2,
        'min_words': 7,
        'description': 'Balanced settings, more epochs'
    },

    # Fine-tuning focused
    {
        'name': 'fine_tune',
        'learning_rate': 1e-5,
        'batch_size': 16,
        'epochs': 10,
        'dropout': 0.2,
        'min_words': 7,
        'description': 'Lower LR for careful fine-tuning, more epochs'
    },

    # Recommended optimized
    {
        'name': 'optimized',
        'learning_rate': 1e-5,
        'batch_size': 32,
        'epochs': 10,
        'dropout': 0.2,
        'min_words': 7,
        'warmup_ratio': 0.15,
        'description': 'Recommended settings for best performance'
    },
]


class ExperimentRunner:
    """Run and track hyperparameter search experiments."""

    def __init__(self, train_data: str, val_data: Optional[str], output_dir: str):
        self.train_data = train_data
        self.val_data = val_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Master tracking file
        self.tracking_file = self.output_dir / 'experiment_tracking.csv'
        self.results = []

    def run_experiment(self, config: Dict, experiment_id: Optional[str] = None) -> Dict:
        """Run a single training experiment with given config."""

        if experiment_id is None:
            experiment_id = f"{config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        experiment_dir = self.output_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {experiment_id}")
        print(f"{'='*80}")
        print(f"Description: {config.get('description', 'N/A')}")
        print(f"Configuration:")
        for key, value in config.items():
            if key not in ['name', 'description']:
                print(f"  {key:20s}: {value}")
        print(f"Output: {experiment_dir}")
        print(f"{'='*80}\n")

        # Build command
        cmd = [
            sys.executable,  # Use current Python interpreter
            'train_design_classifier.py',
            '--mode', 'full',
            '--stackoverflow_path', self.train_data,
            '--output_dir', str(experiment_dir),
            '--epochs', str(config.get('epochs', 5)),
            '--learning_rate', str(config.get('learning_rate', 2e-5)),
            '--batch_size', str(config.get('batch_size', 16)),
            '--dropout', str(config.get('dropout', 0.1)),
            '--min_words', str(config.get('min_words', 7)),
        ]

        # Add optional parameters
        if 'warmup_ratio' in config:
            cmd.extend(['--warmup_ratio', str(config['warmup_ratio'])])
        if 'max_length' in config:
            cmd.extend(['--max_length', str(config['max_length'])])

        # Log command
        cmd_str = ' \\\n  '.join(cmd)
        with open(experiment_dir / 'command.txt', 'w') as f:
            f.write(cmd_str)

        print(f"Command:\n{cmd_str}\n")

        # Run training
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            success = True
            error_msg = None
        except subprocess.CalledProcessError as e:
            success = False
            error_msg = f"Exit code {e.returncode}: {e.stderr}"
            print(f"❌ Experiment failed: {error_msg}")

        elapsed_time = time.time() - start_time

        # Extract metrics
        metrics = self._extract_metrics(experiment_dir)

        # Validate on held-out validation data if provided
        val_metrics = None
        if self.val_data and success:
            val_metrics = self._validate_model(experiment_dir, self.val_data)

        # Record results
        result_record = {
            'experiment_id': experiment_id,
            'config_name': config['name'],
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'duration_seconds': elapsed_time,
            'error_message': error_msg,
            **{f'config_{k}': v for k, v in config.items() if k not in ['name', 'description']},
            **{f'train_{k}': v for k, v in (metrics or {}).items()},
            **{f'val_{k}': v for k, v in (val_metrics or {}).items()},
        }

        self.results.append(result_record)
        self._save_tracking()

        if success:
            print(f"✓ Experiment completed in {elapsed_time:.1f}s")
            if metrics:
                print(f"  Train F1: {metrics.get('test_f1_score', 'N/A'):.4f}")
                print(f"  Train AUC: {metrics.get('test_auc', 'N/A'):.4f}")
            if val_metrics:
                print(f"  Val F1: {val_metrics.get('test_f1_score', 'N/A'):.4f}")
                print(f"  Val AUC: {val_metrics.get('test_auc', 'N/A'):.4f}")

        return result_record

    def _extract_metrics(self, experiment_dir: Path) -> Optional[Dict]:
        """Extract metrics from experiment output."""

        metrics_file = experiment_dir / 'training_metrics.csv'
        if not metrics_file.exists():
            return None

        df = pd.read_csv(metrics_file)
        if len(df) == 0:
            return None

        # Get latest run
        latest = df.iloc[-1].to_dict()
        return latest

    def _validate_model(self, experiment_dir: Path, val_data: str) -> Optional[Dict]:
        """Validate trained model on held-out validation data."""

        print(f"\n  Running validation on: {val_data}")

        # Create a validation script
        val_script = experiment_dir / 'validate.py'
        with open(val_script, 'w') as f:
            f.write(f'''
import sys
import pandas as pd
import torch
from train_design_classifier import (
    DesignMiningTrainer, Config, DataPreprocessor, MetricsCalculator
)

# Load validation data
df = pd.read_csv("{val_data}")
print(f"Loaded {{len(df)}} validation samples")

# Detect columns
text_col = 'text' if 'text' in df.columns else df.columns[0]
label_col = 'label' if 'label' in df.columns else df.columns[1]

# Encode labels
try:
    df[label_col] = df[label_col].astype(int)
except:
    label_mapping = {{}}
    for label in df[label_col].unique():
        label_str = str(label).lower()
        if 'design' in label_str or label_str in ['1', 'true', 'yes']:
            label_mapping[label] = 1
        else:
            label_mapping[label] = 0
    df[label_col] = df[label_col].map(label_mapping)

# Preprocess
preprocessor = DataPreprocessor(min_words=7, verbose=False)
df_clean = preprocessor.preprocess_dataframe(df, text_col)

texts = df_clean[text_col].tolist()
labels = df_clean[label_col].tolist()

print(f"After preprocessing: {{len(texts)}} samples")

# Load model
config = Config()
config.OUTPUT_DIR = "{experiment_dir}"
trainer = DesignMiningTrainer(config)
trainer.load_model("{experiment_dir}")

# Predict
predictions, confidences = trainer.predict(texts)

# Calculate metrics
import numpy as np
metrics = MetricsCalculator.calculate_all_metrics(
    np.array(labels),
    np.array(predictions)
)

# Save metrics
import json
with open("{experiment_dir / 'val_metrics.json'}", 'w') as f:
    json.dump(metrics, f, indent=2)

print("Validation metrics:")
MetricsCalculator.print_metrics(metrics, "Validation")
''')

        # Run validation
        try:
            subprocess.run(
                [sys.executable, str(val_script)],
                check=True,
                capture_output=True,
                text=True
            )

            # Read metrics
            val_metrics_file = experiment_dir / 'val_metrics.json'
            if val_metrics_file.exists():
                with open(val_metrics_file) as f:
                    val_metrics = json.load(f)
                return val_metrics
        except Exception as e:
            print(f"  ⚠️  Validation failed: {e}")

        return None

    def _save_tracking(self):
        """Save experiment tracking to CSV."""

        df = pd.DataFrame(self.results)
        df.to_csv(self.tracking_file, index=False)
        print(f"\n📊 Tracking saved to: {self.tracking_file}")

    def compare_results(self, sort_by: str = 'val_test_f1_score'):
        """Compare and rank all experiments."""

        if not self.results:
            print("No experiments to compare")
            return None

        df = pd.DataFrame(self.results)

        # Filter successful experiments
        df_success = df[df['success'] == True].copy()

        if len(df_success) == 0:
            print("No successful experiments to compare")
            return None

        # Determine sort column
        if sort_by not in df_success.columns:
            # Try alternatives
            alternatives = ['train_test_f1_score', 'train_best_val_f1_score']
            for alt in alternatives:
                if alt in df_success.columns:
                    sort_by = alt
                    break

        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPARISON (sorted by {sort_by})")
        print(f"{'='*80}\n")

        # Sort by performance
        df_sorted = df_success.sort_values(sort_by, ascending=False)

        # Select key columns to display
        display_cols = ['experiment_id', 'config_name']

        # Add config columns
        config_cols = [c for c in df_sorted.columns if c.startswith('config_')]
        display_cols.extend(config_cols[:5])  # Limit to first 5 config params

        # Add performance columns
        perf_cols = [c for c in df_sorted.columns if 'f1_score' in c or 'auc' in c or 'accuracy' in c]
        display_cols.extend(perf_cols[:6])  # Limit to first 6 metrics

        # Filter to existing columns
        display_cols = [c for c in display_cols if c in df_sorted.columns]

        print(df_sorted[display_cols].head(10).to_string(index=False))
        print(f"\n{'='*80}")

        # Show best config
        best_idx = df_sorted[sort_by].idxmax()
        best_experiment = df_sorted.loc[best_idx]

        print(f"\n🏆 BEST CONFIGURATION: {best_experiment['config_name']}")
        print(f"   Experiment ID: {best_experiment['experiment_id']}")
        print(f"   {sort_by}: {best_experiment[sort_by]:.4f}")

        print("\n   Configuration:")
        for col in config_cols:
            if col in best_experiment:
                param_name = col.replace('config_', '')
                print(f"     {param_name:20s}: {best_experiment[col]}")

        # Save comparison
        comparison_file = self.output_dir / 'comparison.csv'
        df_sorted[display_cols].to_csv(comparison_file, index=False)
        print(f"\n📊 Comparison saved to: {comparison_file}")

        return df_sorted


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Run hyperparameter search with validation'
    )
    parser.add_argument(
        '--train_data',
        required=True,
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--val_data',
        default=None,
        help='Path to held-out validation data CSV (optional)'
    )
    parser.add_argument(
        '--output_dir',
        default='./experiments',
        help='Output directory for experiments'
    )
    parser.add_argument(
        '--run_all',
        action='store_true',
        help='Run all suggested configs'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Run specific config by name (e.g., "balanced", "optimized")'
    )
    parser.add_argument(
        '--grid_search',
        action='store_true',
        help='Run full grid search over parameter space (WARNING: many experiments!)'
    )

    args = parser.parse_args()

    # Initialize runner
    runner = ExperimentRunner(args.train_data, args.val_data, args.output_dir)

    if args.grid_search:
        print("⚠️  Grid search will run many experiments!")
        print(f"   Total combinations: {len(HYPERPARAMETER_GRID['learning_rate'])} × "
              f"{len(HYPERPARAMETER_GRID['batch_size'])} × "
              f"{len(HYPERPARAMETER_GRID['epochs'])} × "
              f"{len(HYPERPARAMETER_GRID['dropout'])} = "
              f"{len(HYPERPARAMETER_GRID['learning_rate']) * len(HYPERPARAMETER_GRID['batch_size']) * len(HYPERPARAMETER_GRID['epochs']) * len(HYPERPARAMETER_GRID['dropout'])}")

        response = input("Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return

        # Generate all combinations
        import itertools
        keys = list(HYPERPARAMETER_GRID.keys())
        values = [HYPERPARAMETER_GRID[k] for k in keys]

        for i, combination in enumerate(itertools.product(*values)):
            config = {
                'name': f'grid_{i:03d}',
                'description': 'Grid search configuration',
                **dict(zip(keys, combination))
            }
            runner.run_experiment(config)

    elif args.run_all:
        print(f"Running {len(SUGGESTED_CONFIGS)} suggested configurations...")
        for config in SUGGESTED_CONFIGS:
            runner.run_experiment(config)

    elif args.config:
        # Run specific config
        config = next((c for c in SUGGESTED_CONFIGS if c['name'] == args.config), None)
        if config is None:
            print(f"Config '{args.config}' not found.")
            print("Available configs:")
            for c in SUGGESTED_CONFIGS:
                print(f"  - {c['name']}: {c['description']}")
            return

        runner.run_experiment(config)

    else:
        # Show available configs
        print("Available configurations:")
        for i, config in enumerate(SUGGESTED_CONFIGS, 1):
            print(f"\n{i}. {config['name']}:")
            print(f"   {config['description']}")
            for key, value in config.items():
                if key not in ['name', 'description']:
                    print(f"     {key:20s}: {value}")

        print("\n\nUsage:")
        print(f"  # Run all configs:")
        print(f"  python {sys.argv[0]} --train_data <path> --val_data <path> --run_all")
        print(f"\n  # Run specific config:")
        print(f"  python {sys.argv[0]} --train_data <path> --val_data <path> --config optimized")
        print(f"\n  # Run grid search:")
        print(f"  python {sys.argv[0]} --train_data <path> --val_data <path> --grid_search")
        return

    # Compare results
    if runner.results:
        sort_metric = 'val_test_f1_score' if args.val_data else 'train_test_f1_score'
        runner.compare_results(sort_by=sort_metric)


if __name__ == '__main__':
    main()
