"""
Comprehensive Quantitative Analysis of Design Mining Model Training
====================================================================

Generates detailed visualizations and statistical analysis for all grid search
experiments across bert-base-uncased, distilbert-base-uncased, and roberta-base.

Usage:
    python generate_analysis_report.py --output_dir ./analysis_output
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import argparse
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

FIGURE_DPI = 150
FIGURE_SIZE = (12, 8)


def aggregate_all_results(base_dir='models/design_mining'):
    """Aggregate results from all grid directories."""
    models = {
        'bert-base-uncased': f'{base_dir}/bert-base-grid',
        'distilbert-base-uncased': f'{base_dir}/distilbert-base-grid',
        'roberta-base': f'{base_dir}/roberta-base-grid'
    }

    all_results = []
    for model_name, grid_dir in models.items():
        csv_files = glob.glob(f"{grid_dir}/*/training_metrics.csv")

        for f in csv_files:
            try:
                df = pd.read_csv(f)
                df['model_type'] = model_name
                df['experiment_dir'] = Path(f).parent.name
                all_results.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")

    if not all_results:
        raise ValueError("No results found!")

    combined = pd.concat(all_results, ignore_index=True)
    print(f"Loaded {len(combined)} total experiments")
    return combined


def plot_model_comparison_boxplot(df, output_dir):
    """Create boxplot comparing model performance distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['test_f1_score', 'test_accuracy', 'test_auc']
    titles = ['F1 Score Distribution', 'Accuracy Distribution', 'AUC Distribution']

    for ax, metric, title in zip(axes, metrics, titles):
        # Order by median performance
        order = df.groupby('model_type')[metric].median().sort_values(ascending=False).index

        sns.boxplot(data=df, x='model_type', y=metric, ax=ax, order=order)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.tick_params(axis='x', rotation=15)

        # Add mean markers
        means = df.groupby('model_type')[metric].mean()
        for i, model in enumerate(order):
            ax.scatter(i, means[model], color='red', s=50, zorder=5, marker='D', label='Mean' if i == 0 else '')

    axes[0].legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_boxplot.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison_boxplot.png")


def plot_model_comparison_violin(df, output_dir):
    """Create violin plot showing full distribution shapes."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Melt for easier plotting
    metrics_df = df[['model_type', 'test_f1_score', 'test_accuracy', 'test_auc']].melt(
        id_vars='model_type', var_name='Metric', value_name='Score'
    )
    metrics_df['Metric'] = metrics_df['Metric'].map({
        'test_f1_score': 'F1 Score',
        'test_accuracy': 'Accuracy',
        'test_auc': 'AUC'
    })

    sns.violinplot(data=metrics_df, x='model_type', y='Score', hue='Metric', ax=ax, split=False)
    ax.set_title('Performance Distribution by Model and Metric', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(title='Metric', loc='lower right')
    ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_violin.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: model_comparison_violin.png")


def plot_learning_rate_analysis(df, output_dir):
    """Analyze impact of learning rate across models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, model in zip(axes, df['model_type'].unique()):
        model_df = df[df['model_type'] == model]

        # Group by learning rate
        lr_groups = model_df.groupby('learning_rate').agg({
            'test_f1_score': ['mean', 'std', 'max', 'count']
        }).round(4)
        lr_groups.columns = ['mean', 'std', 'max', 'count']
        lr_groups = lr_groups.reset_index()

        x = range(len(lr_groups))
        ax.bar(x, lr_groups['mean'], yerr=lr_groups['std'], capsize=5, alpha=0.7, label='Mean ± Std')
        ax.scatter(x, lr_groups['max'], color='red', s=100, zorder=5, marker='*', label='Max')

        ax.set_xticks(x)
        ax.set_xticklabels([f"{lr:.0e}" for lr in lr_groups['learning_rate']], rotation=45)
        ax.set_xlabel('Learning Rate', fontsize=10)
        ax.set_ylabel('F1 Score', fontsize=10)
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0.5, 1.0)

        # Add count labels
        for i, (_, row) in enumerate(lr_groups.iterrows()):
            ax.annotate(f'n={int(row["count"])}', (i, row['mean'] + row['std'] + 0.02),
                       ha='center', fontsize=8)

    plt.suptitle('Learning Rate Impact on F1 Score', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate_analysis.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: learning_rate_analysis.png")


def plot_batch_size_analysis(df, output_dir):
    """Analyze impact of batch size across models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, model in zip(axes, df['model_type'].unique()):
        model_df = df[df['model_type'] == model]

        bs_groups = model_df.groupby('batch_size').agg({
            'test_f1_score': ['mean', 'std', 'max']
        }).round(4)
        bs_groups.columns = ['mean', 'std', 'max']
        bs_groups = bs_groups.reset_index()

        x = range(len(bs_groups))
        ax.bar(x, bs_groups['mean'], yerr=bs_groups['std'], capsize=5, alpha=0.7)
        ax.scatter(x, bs_groups['max'], color='red', s=100, zorder=5, marker='*')

        ax.set_xticks(x)
        ax.set_xticklabels(bs_groups['batch_size'])
        ax.set_xlabel('Batch Size', fontsize=10)
        ax.set_ylabel('F1 Score', fontsize=10)
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.set_ylim(0.5, 1.0)

    plt.suptitle('Batch Size Impact on F1 Score', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'batch_size_analysis.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: batch_size_analysis.png")


def plot_epochs_analysis(df, output_dir):
    """Analyze impact of training epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, model in zip(axes, df['model_type'].unique()):
        model_df = df[df['model_type'] == model]

        epoch_groups = model_df.groupby('num_epochs').agg({
            'test_f1_score': ['mean', 'std', 'max']
        }).round(4)
        epoch_groups.columns = ['mean', 'std', 'max']
        epoch_groups = epoch_groups.reset_index()

        ax.plot(epoch_groups['num_epochs'], epoch_groups['mean'], 'b-o', label='Mean', markersize=8)
        ax.fill_between(epoch_groups['num_epochs'],
                        epoch_groups['mean'] - epoch_groups['std'],
                        epoch_groups['mean'] + epoch_groups['std'],
                        alpha=0.3)
        ax.plot(epoch_groups['num_epochs'], epoch_groups['max'], 'r--*', label='Max', markersize=10)

        ax.set_xlabel('Epochs', fontsize=10)
        ax.set_ylabel('F1 Score', fontsize=10)
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_ylim(0.5, 1.0)

    plt.suptitle('Training Epochs Impact on F1 Score', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'epochs_analysis.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: epochs_analysis.png")


def plot_dropout_analysis(df, output_dir):
    """Analyze impact of dropout rate."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, model in zip(axes, df['model_type'].unique()):
        model_df = df[df['model_type'] == model]

        dropout_groups = model_df.groupby('dropout_rate').agg({
            'test_f1_score': ['mean', 'std', 'max']
        }).round(4)
        dropout_groups.columns = ['mean', 'std', 'max']
        dropout_groups = dropout_groups.reset_index()

        x = range(len(dropout_groups))
        ax.bar(x, dropout_groups['mean'], yerr=dropout_groups['std'], capsize=5, alpha=0.7)
        ax.scatter(x, dropout_groups['max'], color='red', s=100, zorder=5, marker='*')

        ax.set_xticks(x)
        ax.set_xticklabels(dropout_groups['dropout_rate'])
        ax.set_xlabel('Dropout Rate', fontsize=10)
        ax.set_ylabel('F1 Score', fontsize=10)
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.set_ylim(0.5, 1.0)

    plt.suptitle('Dropout Rate Impact on F1 Score', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'dropout_analysis.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: dropout_analysis.png")


def plot_heatmap_hyperparameters(df, output_dir):
    """Create heatmaps showing hyperparameter interactions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, model in zip(axes, df['model_type'].unique()):
        model_df = df[df['model_type'] == model]

        # Create pivot table for learning_rate vs batch_size
        pivot = model_df.pivot_table(
            values='test_f1_score',
            index='learning_rate',
            columns='batch_size',
            aggfunc='mean'
        )

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                    vmin=0.6, vmax=0.95, cbar_kws={'label': 'Mean F1'})
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Batch Size', fontsize=10)
        ax.set_ylabel('Learning Rate', fontsize=10)

    plt.suptitle('Learning Rate × Batch Size Interaction (Mean F1 Score)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_lr_batchsize.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: heatmap_lr_batchsize.png")


def plot_heatmap_epochs_dropout(df, output_dir):
    """Create heatmaps for epochs vs dropout interaction."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, model in zip(axes, df['model_type'].unique()):
        model_df = df[df['model_type'] == model]

        pivot = model_df.pivot_table(
            values='test_f1_score',
            index='dropout_rate',
            columns='num_epochs',
            aggfunc='mean'
        )

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                    vmin=0.6, vmax=0.95, cbar_kws={'label': 'Mean F1'})
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=10)
        ax.set_ylabel('Dropout Rate', fontsize=10)

    plt.suptitle('Dropout × Epochs Interaction (Mean F1 Score)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_dropout_epochs.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: heatmap_dropout_epochs.png")


def plot_correlation_matrix(df, output_dir):
    """Create correlation matrix for hyperparameters vs metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    hp_cols = ['learning_rate', 'batch_size', 'num_epochs', 'dropout_rate']
    metric_cols = ['test_f1_score', 'test_accuracy', 'test_auc']

    for ax, model in zip(axes, df['model_type'].unique()):
        model_df = df[df['model_type'] == model]

        # Calculate correlations
        corr_data = []
        for hp in hp_cols:
            row = []
            for metric in metric_cols:
                corr = model_df[hp].corr(model_df[metric])
                row.append(corr)
            corr_data.append(row)

        corr_df = pd.DataFrame(corr_data, index=hp_cols, columns=['F1', 'Accuracy', 'AUC'])

        sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax,
                    vmin=-0.5, vmax=0.5, cbar_kws={'label': 'Correlation'})
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')

    plt.suptitle('Hyperparameter-Metric Correlations', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: correlation_matrix.png")


def plot_variance_analysis(df, output_dir):
    """Analyze variance/stability across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate variance metrics
    variance_data = []
    for model in df['model_type'].unique():
        model_df = df[df['model_type'] == model]
        variance_data.append({
            'Model': model,
            'Std Dev': model_df['test_f1_score'].std(),
            'IQR': model_df['test_f1_score'].quantile(0.75) - model_df['test_f1_score'].quantile(0.25),
            'Range': model_df['test_f1_score'].max() - model_df['test_f1_score'].min(),
            'CV': model_df['test_f1_score'].std() / model_df['test_f1_score'].mean()
        })

    variance_df = pd.DataFrame(variance_data)

    x = np.arange(len(variance_df))
    width = 0.2

    ax.bar(x - width*1.5, variance_df['Std Dev'], width, label='Std Dev', alpha=0.8)
    ax.bar(x - width*0.5, variance_df['IQR'], width, label='IQR', alpha=0.8)
    ax.bar(x + width*0.5, variance_df['Range'], width, label='Range', alpha=0.8)
    ax.bar(x + width*1.5, variance_df['CV'], width, label='CV', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(variance_df['Model'], rotation=15)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Model Stability Analysis (F1 Score Variance Metrics)', fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'variance_analysis.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: variance_analysis.png")


def plot_top_configurations(df, output_dir, n=20):
    """Visualize top N configurations."""
    top_n = df.nlargest(n, 'test_f1_score').copy()
    top_n['config_label'] = top_n.apply(
        lambda r: f"lr={r['learning_rate']:.0e}, bs={int(r['batch_size'])}, ep={int(r['num_epochs'])}, do={r['dropout_rate']}",
        axis=1
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {'bert-base-uncased': '#1f77b4',
              'distilbert-base-uncased': '#ff7f0e',
              'roberta-base': '#2ca02c'}
    bar_colors = [colors[m] for m in top_n['model_type']]

    bars = ax.barh(range(n), top_n['test_f1_score'], color=bar_colors, alpha=0.8)

    ax.set_yticks(range(n))
    ax.set_yticklabels(top_n['config_label'], fontsize=9)
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title(f'Top {n} Configurations by F1 Score', fontsize=14, fontweight='bold')
    ax.set_xlim(0.85, 0.95)
    ax.invert_yaxis()

    # Add model labels
    for i, (bar, model) in enumerate(zip(bars, top_n['model_type'])):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                model.split('-')[0], va='center', fontsize=8)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[m], label=m) for m in colors]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'top_configurations.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: top_configurations.png")


def plot_performance_by_dataset_size(df, output_dir):
    """Analyze how performance varies with dataset size (min_words threshold)."""
    if 'final_samples' not in df.columns:
        print("Skipping dataset size analysis - no final_samples column")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, model in zip(axes, df['model_type'].unique()):
        model_df = df[df['model_type'] == model]

        ax.scatter(model_df['final_samples'], model_df['test_f1_score'], alpha=0.5, s=30)

        # Add trend line
        z = np.polyfit(model_df['final_samples'], model_df['test_f1_score'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(model_df['final_samples'].min(), model_df['final_samples'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', label=f'Trend (slope={z[0]:.2e})')

        ax.set_xlabel('Dataset Size (samples)', fontsize=10)
        ax.set_ylabel('F1 Score', fontsize=10)
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')

    plt.suptitle('F1 Score vs Dataset Size', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_dataset_size.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: performance_vs_dataset_size.png")


def plot_metrics_scatter_matrix(df, output_dir):
    """Create scatter matrix of all metrics."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    metrics = ['test_f1_score', 'test_accuracy', 'test_auc']
    metric_labels = ['F1 Score', 'Accuracy', 'AUC']

    colors = {'bert-base-uncased': '#1f77b4',
              'distilbert-base-uncased': '#ff7f0e',
              'roberta-base': '#2ca02c'}

    for i, (m1, l1) in enumerate(zip(metrics, metric_labels)):
        for j, (m2, l2) in enumerate(zip(metrics, metric_labels)):
            ax = axes[i, j]

            if i == j:
                # Diagonal: histogram
                for model in df['model_type'].unique():
                    model_df = df[df['model_type'] == model]
                    ax.hist(model_df[m1], bins=20, alpha=0.5, color=colors[model], label=model)
                ax.set_xlabel(l1)
                if i == 0:
                    ax.legend(fontsize=6)
            else:
                # Off-diagonal: scatter
                for model in df['model_type'].unique():
                    model_df = df[df['model_type'] == model]
                    ax.scatter(model_df[m2], model_df[m1], alpha=0.3, s=10, color=colors[model])
                ax.set_xlabel(l2)
                ax.set_ylabel(l1)

    plt.suptitle('Metrics Scatter Matrix', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_scatter_matrix.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: metrics_scatter_matrix.png")


def perform_statistical_tests(df, output_dir):
    """Perform statistical significance tests between models."""
    results = []

    models = df['model_type'].unique()
    metrics = ['test_f1_score', 'test_accuracy', 'test_auc']

    for metric in metrics:
        for m1, m2 in combinations(models, 2):
            data1 = df[df['model_type'] == m1][metric].dropna()
            data2 = df[df['model_type'] == m2][metric].dropna()

            # Independent samples t-test
            t_stat, t_pval = stats.ttest_ind(data1, data2)

            # Mann-Whitney U test (non-parametric)
            u_stat, u_pval = stats.mannwhitneyu(data1, data2, alternative='two-sided')

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((data1.std()**2 + data2.std()**2) / 2)
            cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0

            results.append({
                'Metric': metric,
                'Model 1': m1,
                'Model 2': m2,
                'Mean 1': data1.mean(),
                'Mean 2': data2.mean(),
                'Diff': data1.mean() - data2.mean(),
                't-statistic': t_stat,
                't p-value': t_pval,
                'U-statistic': u_stat,
                'U p-value': u_pval,
                "Cohen's d": cohens_d,
                'Significant (p<0.05)': t_pval < 0.05
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'statistical_tests.csv', index=False)
    print("Saved: statistical_tests.csv")

    return results_df


def plot_statistical_significance(stats_df, output_dir):
    """Visualize statistical significance results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['test_f1_score', 'test_accuracy', 'test_auc']
    titles = ['F1 Score', 'Accuracy', 'AUC']

    for ax, metric, title in zip(axes, metrics, titles):
        metric_df = stats_df[stats_df['Metric'] == metric]

        comparisons = [f"{r['Model 1'].split('-')[0]} vs\n{r['Model 2'].split('-')[0]}"
                       for _, r in metric_df.iterrows()]
        diffs = metric_df['Diff'].values
        pvals = metric_df['t p-value'].values

        colors = ['green' if p < 0.05 else 'gray' for p in pvals]
        bars = ax.bar(comparisons, diffs, color=colors, alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel('Mean Difference', fontsize=10)
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')

        # Add p-value labels
        for bar, pval in zip(bars, pvals):
            height = bar.get_height()
            ax.annotate(f'p={pval:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3 if height > 0 else -10),
                       textcoords='offset points',
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=8)

    plt.suptitle('Statistical Significance of Model Differences\n(Green = p < 0.05)',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'statistical_significance.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: statistical_significance.png")


def plot_best_vs_mean_comparison(df, output_dir):
    """Compare best achievable vs mean performance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    summary_data = []
    for model in df['model_type'].unique():
        model_df = df[df['model_type'] == model]
        summary_data.append({
            'Model': model,
            'Mean F1': model_df['test_f1_score'].mean(),
            'Max F1': model_df['test_f1_score'].max(),
            'Std F1': model_df['test_f1_score'].std()
        })

    summary_df = pd.DataFrame(summary_data)
    x = np.arange(len(summary_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, summary_df['Mean F1'], width, yerr=summary_df['Std F1'],
                   label='Mean ± Std', capsize=5, alpha=0.8)
    bars2 = ax.bar(x + width/2, summary_df['Max F1'], width, label='Best', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Model'], rotation=15)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Mean vs Best F1 Score by Model', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0.6, 1.0)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'best_vs_mean_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: best_vs_mean_comparison.png")


def plot_hyperparameter_importance(df, output_dir):
    """Estimate hyperparameter importance using variance analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    hyperparams = ['learning_rate', 'batch_size', 'num_epochs', 'dropout_rate']

    for ax, model in zip(axes, df['model_type'].unique()):
        model_df = df[df['model_type'] == model]

        # Calculate variance explained by each hyperparameter
        importances = []
        for hp in hyperparams:
            groups = model_df.groupby(hp)['test_f1_score']
            between_var = groups.mean().var()
            total_var = model_df['test_f1_score'].var()
            importance = between_var / total_var if total_var > 0 else 0
            importances.append(importance)

        # Normalize
        total = sum(importances)
        if total > 0:
            importances = [i/total for i in importances]

        ax.barh(hyperparams, importances, alpha=0.8)
        ax.set_xlabel('Relative Importance', fontsize=10)
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)

    plt.suptitle('Hyperparameter Importance (Variance-Based)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'hyperparameter_importance.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("Saved: hyperparameter_importance.png")


def generate_summary_table(df, output_dir):
    """Generate comprehensive summary statistics table."""
    summary = []

    for model in df['model_type'].unique():
        model_df = df[df['model_type'] == model]

        # Find best configuration
        best_idx = model_df['test_f1_score'].idxmax()
        best = model_df.loc[best_idx]

        summary.append({
            'Model': model,
            'Experiments': len(model_df),
            'Mean F1': f"{model_df['test_f1_score'].mean():.4f}",
            'Std F1': f"{model_df['test_f1_score'].std():.4f}",
            'Max F1': f"{model_df['test_f1_score'].max():.4f}",
            'Min F1': f"{model_df['test_f1_score'].min():.4f}",
            'Mean Accuracy': f"{model_df['test_accuracy'].mean():.4f}",
            'Max Accuracy': f"{model_df['test_accuracy'].max():.4f}",
            'Mean AUC': f"{model_df['test_auc'].mean():.4f}",
            'Max AUC': f"{model_df['test_auc'].max():.4f}",
            'Best LR': best['learning_rate'],
            'Best Batch': int(best['batch_size']),
            'Best Epochs': int(best['num_epochs']),
            'Best Dropout': best['dropout_rate']
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
    print("Saved: summary_statistics.csv")

    return summary_df


def generate_optimal_configs_table(df, output_dir, n=10):
    """Generate table of top N optimal configurations."""
    top_n = df.nlargest(n, 'test_f1_score')[
        ['model_type', 'test_f1_score', 'test_accuracy', 'test_auc',
         'learning_rate', 'batch_size', 'num_epochs', 'dropout_rate']
    ].copy()

    top_n.columns = ['Model', 'F1', 'Accuracy', 'AUC', 'LR', 'Batch', 'Epochs', 'Dropout']
    top_n['Rank'] = range(1, n+1)
    top_n = top_n[['Rank', 'Model', 'F1', 'Accuracy', 'AUC', 'LR', 'Batch', 'Epochs', 'Dropout']]

    top_n.to_csv(output_dir / 'top_configurations.csv', index=False)
    print("Saved: top_configurations.csv")

    return top_n


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive analysis report')
    parser.add_argument('--output_dir', default='./analysis_output', help='Output directory')
    parser.add_argument('--base_dir', default='models/design_mining', help='Base models directory')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Design Mining Model Analysis Report Generator")
    print("="*60)

    # Load data
    print("\nLoading experiment data...")
    df = aggregate_all_results(args.base_dir)

    print(f"\nGenerating visualizations in {output_dir}/")
    print("-"*60)

    # Generate all visualizations
    plot_model_comparison_boxplot(df, output_dir)
    plot_model_comparison_violin(df, output_dir)
    plot_learning_rate_analysis(df, output_dir)
    plot_batch_size_analysis(df, output_dir)
    plot_epochs_analysis(df, output_dir)
    plot_dropout_analysis(df, output_dir)
    plot_heatmap_hyperparameters(df, output_dir)
    plot_heatmap_epochs_dropout(df, output_dir)
    plot_correlation_matrix(df, output_dir)
    plot_variance_analysis(df, output_dir)
    plot_top_configurations(df, output_dir)
    plot_performance_by_dataset_size(df, output_dir)
    plot_metrics_scatter_matrix(df, output_dir)
    plot_best_vs_mean_comparison(df, output_dir)
    plot_hyperparameter_importance(df, output_dir)

    # Statistical tests
    print("\nPerforming statistical tests...")
    stats_df = perform_statistical_tests(df, output_dir)
    plot_statistical_significance(stats_df, output_dir)

    # Summary tables
    print("\nGenerating summary tables...")
    summary_df = generate_summary_table(df, output_dir)
    top_configs = generate_optimal_configs_table(df, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total visualizations: 15")
    print(f"Total data tables: 4")

    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
