"""
Analyze Training Metrics from CSV
==================================

Visualize and compare training runs to find optimal hyperparameters.

Usage:
    python analyze_metrics.py --csv path/to/training_metrics.csv
"""

import pandas as pd
import argparse
from pathlib import Path


def analyze_metrics(csv_path):
    """Analyze and display metrics from training runs."""

    print(f"\nReading metrics from: {csv_path}")
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        print("No data found in CSV file.")
        return

    print(f"\nTotal runs: {len(df)}")

    # Show column names
    print(f"\nAvailable metrics: {list(df.columns)}")

    # Key metrics to analyze (support both prefixed and unprefixed column names)
    key_metrics = [
        'train_test_accuracy', 'train_test_precision', 'train_test_recall',
        'train_test_f1_score', 'train_test_auc',
        'test_accuracy', 'test_precision', 'test_recall',
        'test_f1_score', 'test_auc'
    ]

    # Check which metrics are available
    available_metrics = [m for m in key_metrics if m in df.columns]

    if not available_metrics:
        print("\nWarning: No test metrics found in CSV.")
        return

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(df[available_metrics].describe())

    # Best runs for each metric
    print("\n" + "="*80)
    print("BEST RUNS PER METRIC")
    print("="*80)

    for metric in available_metrics:
        if metric in df.columns:
            best_idx = df[metric].idxmax()
            best_run = df.loc[best_idx]

            print(f"\nBest {metric}: {best_run[metric]:.4f}")
            run_id = best_run.get('experiment_id', best_run.get('run_id', 'N/A'))
            print(f"  Run ID: {run_id}")
            print(f"  Timestamp: {best_run.get('timestamp', 'N/A')}")

            # Show hyperparameters (support both prefixed and unprefixed)
            hyperparam_cols = [
                'config_learning_rate', 'config_batch_size', 'config_epochs',
                'config_dropout',
                'learning_rate', 'batch_size', 'num_epochs',
                'dropout_rate', 'max_length'
            ]
            for col in hyperparam_cols:
                if col in df.columns:
                    print(f"  {col}: {best_run[col]}")

    # Resolve the f1 score column name (prefixed or unprefixed)
    f1_col = 'train_test_f1_score' if 'train_test_f1_score' in df.columns else 'test_f1_score'

    # Correlation analysis
    print("\n" + "="*80)
    print("HYPERPARAMETER CORRELATIONS WITH TEST F1 SCORE")
    print("="*80)

    if f1_col in df.columns:
        hyperparam_cols = [
            'config_learning_rate', 'config_batch_size', 'config_epochs',
            'config_dropout',
            'learning_rate', 'batch_size', 'num_epochs',
            'dropout_rate', 'max_length', 'warmup_ratio'
        ]

        correlations = {}
        for col in hyperparam_cols:
            if col in df.columns and df[col].nunique() > 1:
                corr = df[col].corr(df[f1_col])
                correlations[col] = corr

        if correlations:
            sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

            for param, corr in sorted_corrs:
                direction = "↑" if corr > 0 else "↓"
                strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
                print(f"  {param:20s}: {corr:+.3f} {direction} ({strength})")

    # Top 5 configurations
    print("\n" + "="*80)
    print("TOP 5 CONFIGURATIONS (by Test F1 Score)")
    print("="*80)

    if f1_col in df.columns:
        top5 = df.nlargest(5, f1_col)

        display_cols = ['experiment_id', f1_col,
                       'train_test_accuracy', 'train_test_auc',
                       'config_learning_rate', 'config_batch_size', 'config_epochs',
                       'run_id', 'test_f1_score', 'test_accuracy', 'test_auc',
                       'learning_rate', 'batch_size', 'num_epochs']
        display_cols = [c for c in display_cols if c in df.columns]

        print(top5[display_cols].to_string(index=False))

    # Performance trends
    print("\n" + "="*80)
    print("PERFORMANCE TRENDS")
    print("="*80)

    # Check if performance is improving over time
    if 'timestamp' in df.columns and f1_col in df.columns:
        df_sorted = df.sort_values('timestamp')

        # Calculate running best
        df_sorted['best_so_far'] = df_sorted[f1_col].cummax()

        first_f1 = df_sorted[f1_col].iloc[0]
        last_f1 = df_sorted[f1_col].iloc[-1]
        best_f1 = df_sorted[f1_col].max()

        print(f"  First run F1:  {first_f1:.4f}")
        print(f"  Latest run F1: {last_f1:.4f}")
        print(f"  Best F1 ever:  {best_f1:.4f}")

        improvement = best_f1 - first_f1
        print(f"  Total improvement: {improvement:+.4f} ({improvement/first_f1*100:+.1f}%)")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if f1_col in df.columns:
        best_f1 = df[f1_col].max()
        mean_f1 = df[f1_col].mean()
        std_f1 = df[f1_col].std()

        print(f"  Best F1: {best_f1:.4f}")
        print(f"  Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")

        if best_f1 < 0.85:
            print("\n  ⚠️  Performance below 0.85. Consider:")
            print("     - Increasing training epochs (try 10-15)")
            print("     - Lowering learning rate (try 1e-5)")
            print("     - Adding more training data")
            print("     - Checking data quality")
        elif best_f1 < 0.90:
            print("\n  ✓ Good performance! To reach 0.90+, try:")
            print("     - Fine-tuning learning rate (1e-5 to 3e-5)")
            print("     - Adjusting dropout (0.2-0.3)")
            print("     - Training longer (10-12 epochs)")
        elif best_f1 < 0.95:
            print("\n  ✓✓ Very good performance! To reach 0.95+, try:")
            print("     - Ensemble methods (train multiple models)")
            print("     - Domain-specific BERT (e.g., CodeBERT)")
            print("     - Data augmentation")
        else:
            print("\n  ✓✓✓ Excellent performance! Consider:")
            print("     - Testing on additional datasets")
            print("     - Publishing results")

    # Data quality check
    if 'final_samples' in df.columns and 'initial_samples' in df.columns:
        avg_retention = (df['final_samples'] / df['initial_samples']).mean()
        print(f"\n  Average data retention: {avg_retention*100:.1f}%")

        if avg_retention < 0.5:
            print("  ⚠️  Losing >50% of data in preprocessing!")
            print("     - Check min_words threshold (currently removing short texts)")
            print("     - Review stopword list")
            print("     - Examine auto-generated detection patterns")


def export_best_config(csv_path, output_file='best_config.txt'):
    """Export the configuration of the best run."""

    df = pd.read_csv(csv_path)

    if 'test_f1_score' not in df.columns:
        print("Cannot export: no test_f1_score found")
        return

    best_idx = df['test_f1_score'].idxmax()
    best_run = df.loc[best_idx]

    config_params = [
        'learning_rate', 'batch_size', 'num_epochs',
        'dropout_rate', 'max_length', 'warmup_ratio', 'min_words'
    ]

    with open(output_file, 'w') as f:
        f.write("# Best Configuration\n")
        f.write(f"# Test F1 Score: {best_run['test_f1_score']:.4f}\n")
        f.write(f"# Test Accuracy: {best_run.get('test_accuracy', 'N/A'):.4f}\n")
        f.write(f"# Test AUC: {best_run.get('test_auc', 'N/A'):.4f}\n")
        f.write(f"# Run ID: {best_run.get('run_id', 'N/A')}\n\n")

        f.write("# Command to reproduce:\n")
        f.write("python train_design_classifier.py --mode full \\\n")
        f.write(f"  --stackoverflow_path <your_data_path> \\\n")

        for param in config_params:
            if param in df.columns:
                value = best_run[param]
                # Format the parameter for command line
                param_name = param.replace('_', '-')
                f.write(f"  --{param_name} {value} \\\n")

        f.write(f"  --output_dir ./models/best_model\n")

    print(f"\nBest configuration exported to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze training metrics')
    parser.add_argument(
        '--csv',
        required=True,
        help='Path to training_metrics.csv'
    )
    parser.add_argument(
        '--export-best',
        action='store_true',
        help='Export best configuration to file'
    )

    args = parser.parse_args()

    analyze_metrics(args.csv)

    if args.export_best:
        export_best_config(args.csv)
