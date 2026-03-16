"""
Traditional ML Models for Design Issue Classification
======================================================

Trains Gradient Boosting and SVM classifiers on manually labelled
software issue tickets, classifying them as 'design' or 'non-design'.

Evaluation includes:
  - Holdout test set metrics
  - Stratified K-Fold cross-validation
  - Leave-One-Project-Out (LOPO) cross-validation
  - Hyperparameter tuning via GridSearchCV

Usage:
    python traditional_ml/train_traditional_models.py
    python traditional_ml/train_traditional_models.py --include_issue_type
    python traditional_ml/train_traditional_models.py --skip_lopo --skip_gridsearch
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traditional_ml/traditional_training.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class TraditionalModelConfig:
    RANDOM_SEED = 42
    TFIDF_MAX_FEATURES = 10000
    TFIDF_NGRAM_RANGE = (1, 2)
    CV_FOLDS = 5
    TEST_SPLIT = 0.15
    OUTPUT_DIR = 'traditional_ml/results'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load the manually labelled CSV and encode labels."""
    df = pd.read_csv(csv_path)
    df['text'] = df['text'].fillna('')
    df['label_encoded'] = (df['label'] == 'design').astype(int)

    logger.info(f"Loaded {len(df)} samples from {csv_path}")
    logger.info(f"  Design:     {df['label_encoded'].sum()}")
    logger.info(f"  Non-design: {(df['label_encoded'] == 0).sum()}")
    logger.info(f"  Projects:   {sorted(df['project'].unique())}")
    return df


def load_augmented_data(
    design_path: str,
    nondesign_path: str,
    n_augment: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Load augmented design/non-design data and return a balanced sample.

    Samples n_augment/2 design and n_augment/2 non-design examples,
    constructs a 'text' column from summary + description to match
    the manually labelled format.
    """
    n_per_class = n_augment // 2

    # Design augmentation
    d_df = pd.read_csv(design_path)
    d_df['text'] = (
        d_df['summary'].fillna('').str.strip('"')
        + ' [SEP] '
        + d_df['description'].fillna('').str.strip('"')
    )
    d_df['label'] = 'design'
    d_df['label_encoded'] = 1
    d_df['label_source'] = 'heuristic'
    d_sample = d_df.sample(n=min(n_per_class, len(d_df)), random_state=seed)

    # Non-design augmentation
    n_df = pd.read_csv(nondesign_path)
    n_df['text'] = (
        n_df['summary'].fillna('').str.strip('"')
        + ' [SEP] '
        + n_df['description'].fillna('').str.strip('"')
    )
    n_df['label'] = 'non-design'
    n_df['label_encoded'] = 0
    n_df['label_source'] = 'heuristic'
    n_sample = n_df.sample(n=min(n_per_class, len(n_df)), random_state=seed)

    # Keep only the columns needed
    cols = ['project', 'issue_key', 'issue_type', 'text', 'label',
            'label_encoded', 'label_source']
    aug = pd.concat([d_sample[cols], n_sample[cols]], ignore_index=True)

    logger.info(f"Augmented data: {len(aug)} samples "
                f"(D={aug['label_encoded'].sum()}, "
                f"N={(aug['label_encoded'] == 0).sum()})")
    return aug


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class TraditionalModelTrainer:
    def __init__(self, config: TraditionalModelConfig):
        self.config = config
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.type_encoder: Optional[OneHotEncoder] = None
        self.models: Dict[str, object] = {}
        self.best_params: Dict[str, dict] = {}

    # -- Feature building ---------------------------------------------------
    def build_features(
        self,
        texts: List[str],
        issue_types: Optional[List[str]] = None,
        fit: bool = True,
    ) -> sparse.csr_matrix:
        if fit:
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.TFIDF_MAX_FEATURES,
                ngram_range=self.config.TFIDF_NGRAM_RANGE,
            )
            X = self.vectorizer.fit_transform(texts)
        else:
            X = self.vectorizer.transform(texts)

        if issue_types is not None:
            types_arr = np.array(issue_types).reshape(-1, 1)
            if fit:
                self.type_encoder = OneHotEncoder(
                    sparse_output=True, handle_unknown='ignore',
                )
                type_feats = self.type_encoder.fit_transform(types_arr)
            else:
                type_feats = self.type_encoder.transform(types_arr)
            X = sparse.hstack([X, type_feats], format='csr')

        return X

    # -- Metrics helper -----------------------------------------------------
    @staticmethod
    def _calc_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        }
        if y_prob is not None:
            try:
                metrics['auc'] = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                metrics['auc'] = 0.0
        return metrics

    @staticmethod
    def _print_metrics(metrics: Dict[str, float], label: str = '') -> None:
        logger.info(f"\n{label} Evaluation Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        if 'auc' in metrics:
            logger.info(f"  AUC:       {metrics['auc']:.4f}")
        logger.info(
            f"  Confusion: TP={metrics['tp']} TN={metrics['tn']} "
            f"FP={metrics['fp']} FN={metrics['fn']}"
        )

    # -- Model factories ----------------------------------------------------
    @staticmethod
    def _make_models(params: Optional[Dict[str, dict]] = None) -> Dict[str, object]:
        svm_kw = params.get('svm', {}) if params else {}
        gb_kw = params.get('gradient_boosting', {}) if params else {}
        return {
            'svm': SVC(
                probability=True, class_weight='balanced', random_state=42, **svm_kw,
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=42, **gb_kw,
            ),
        }

    # -- Hyperparameter search ----------------------------------------------
    def hyperparameter_search(
        self, X: sparse.csr_matrix, y: np.ndarray,
    ) -> Dict[str, dict]:
        logger.info("\n" + "=" * 60)
        logger.info("Hyperparameter Search (GridSearchCV)")
        logger.info("=" * 60)

        cv = StratifiedKFold(
            n_splits=self.config.CV_FOLDS, shuffle=True,
            random_state=self.config.RANDOM_SEED,
        )
        sample_w = compute_sample_weight('balanced', y)

        grids = {
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto'],
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
            },
        }

        best = {}
        for name, param_grid in grids.items():
            logger.info(f"\n  Searching {name} ...")
            model = self._make_models()[name]
            fit_params = {}
            if name == 'gradient_boosting':
                fit_params['sample_weight'] = sample_w

            gs = GridSearchCV(
                model, param_grid, cv=cv, scoring='f1',
                n_jobs=-1, verbose=0, refit=False,
            )
            gs.fit(X, y, **fit_params)
            best[name] = gs.best_params_
            logger.info(f"    Best params: {gs.best_params_}")
            logger.info(f"    Best CV F1:  {gs.best_score_:.4f}")

        self.best_params = best
        return best

    # -- Train & evaluate on holdout ----------------------------------------
    def train_and_evaluate(
        self,
        X_train: sparse.csr_matrix, y_train: np.ndarray,
        X_test: sparse.csr_matrix, y_test: np.ndarray,
    ) -> Dict[str, Dict]:
        logger.info("\n" + "=" * 60)
        logger.info("Training Final Models on Holdout Split")
        logger.info("=" * 60)

        results = {}
        models = self._make_models(self.best_params)
        sample_w = compute_sample_weight('balanced', y_train)

        for name, model in models.items():
            logger.info(f"\n  Training {name} ...")
            if name == 'gradient_boosting':
                model.fit(X_train, y_train, sample_weight=sample_w)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = self._calc_metrics(y_test, y_pred, y_prob)
            self._print_metrics(metrics, name.upper())
            results[name] = {
                'metrics': metrics,
                'y_pred': y_pred.tolist(),
                'y_prob': y_prob.tolist(),
                'best_params': self.best_params.get(name, {}),
            }
            self.models[name] = model

        return results

    # -- Stratified K-Fold CV -----------------------------------------------
    def cross_validate(
        self, X: sparse.csr_matrix, y: np.ndarray,
    ) -> Dict[str, Dict]:
        logger.info("\n" + "=" * 60)
        logger.info(f"Stratified {self.config.CV_FOLDS}-Fold Cross-Validation")
        logger.info("=" * 60)

        cv = StratifiedKFold(
            n_splits=self.config.CV_FOLDS, shuffle=True,
            random_state=self.config.RANDOM_SEED,
        )

        cv_results: Dict[str, List[Dict]] = {'svm': [], 'gradient_boosting': []}

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            sample_w = compute_sample_weight('balanced', y_tr)
            models = self._make_models(self.best_params)

            for name, model in models.items():
                if name == 'gradient_boosting':
                    model.fit(X_tr, y_tr, sample_weight=sample_w)
                else:
                    model.fit(X_tr, y_tr)

                y_pred = model.predict(X_te)
                y_prob = model.predict_proba(X_te)[:, 1]
                cv_results[name].append(self._calc_metrics(y_te, y_pred, y_prob))

            logger.info(
                f"  Fold {fold}: SVM F1={cv_results['svm'][-1]['f1']:.4f}  "
                f"GB F1={cv_results['gradient_boosting'][-1]['f1']:.4f}"
            )

        summary = {}
        for name, folds in cv_results.items():
            metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            agg = {}
            for k in metric_keys:
                vals = [f[k] for f in folds]
                agg[f'mean_{k}'] = float(np.mean(vals))
                agg[f'std_{k}'] = float(np.std(vals))
            summary[name] = agg
            logger.info(
                f"\n  {name}: Mean F1={agg['mean_f1']:.4f} ± {agg['std_f1']:.4f}  "
                f"Mean AUC={agg['mean_auc']:.4f} ± {agg['std_auc']:.4f}"
            )

        return summary

    # -- LOPO cross-validation ----------------------------------------------
    def lopo_evaluate(
        self, df: pd.DataFrame, include_issue_type: bool = False,
        aug_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        logger.info("\n" + "=" * 60)
        logger.info("Leave-One-Project-Out (LOPO) Cross-Validation")
        if aug_df is not None:
            logger.info(f"  (with {len(aug_df)} augmented samples added to training)")
        logger.info("=" * 60)

        # Only use manual labels for test folds
        manual_df = df[df.get('label_source', pd.Series(['human'] * len(df))) == 'human']
        projects = sorted(manual_df['project'].unique())
        lopo_results: Dict[str, List[Dict]] = {'svm': [], 'gradient_boosting': []}

        for held_out in projects:
            test_df = manual_df[manual_df['project'] == held_out]
            train_df = manual_df[manual_df['project'] != held_out]

            # Add augmented data to training (excluding held-out project)
            if aug_df is not None:
                aug_fold = aug_df[aug_df['project'] != held_out]
                train_df = pd.concat([train_df, aug_fold], ignore_index=True)

            train_texts = train_df['text'].tolist()
            test_texts = test_df['text'].tolist()
            y_train = train_df['label_encoded'].values
            y_test = test_df['label_encoded'].values

            train_types = train_df['issue_type'].tolist() if include_issue_type else None
            test_types = test_df['issue_type'].tolist() if include_issue_type else None

            X_train = self.build_features(train_texts, train_types, fit=True)
            X_test = self.build_features(test_texts, test_types, fit=False)

            n_design = int(y_test.sum())
            if n_design < 10:
                logger.warning(
                    f"  [{held_out}] Only {n_design} design samples in test fold"
                )

            sample_w = compute_sample_weight('balanced', y_train)
            models = self._make_models(self.best_params)

            for name, model in models.items():
                if name == 'gradient_boosting':
                    model.fit(X_train, y_train, sample_weight=sample_w)
                else:
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics = self._calc_metrics(y_test, y_pred, y_prob)
                metrics['project'] = held_out
                metrics['n'] = len(y_test)
                metrics['n_design'] = n_design
                metrics['n_nondesign'] = len(y_test) - n_design
                lopo_results[name].append(metrics)

            logger.info(
                f"  [{held_out}] n={len(y_test)} D={n_design}  "
                f"SVM F1={lopo_results['svm'][-1]['f1']:.4f}  "
                f"GB F1={lopo_results['gradient_boosting'][-1]['f1']:.4f}"
            )

        # Summary table
        print(f"\n{'='*80}")
        print("LOPO RESULTS")
        print(f"{'='*80}")
        for name in ['svm', 'gradient_boosting']:
            print(f"\n  --- {name.upper()} ---")
            hdr = (
                f"  {'Project':<15} {'n':>5} {'D%':>5} {'F1':>7} "
                f"{'AUC':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7}"
            )
            print(hdr)
            print(f"  {'-'*70}")

            f1s, aucs, accs = [], [], []
            for r in lopo_results[name]:
                d_pct = r['n_design'] / r['n'] * 100
                print(
                    f"  {r['project']:<15} {r['n']:>5} {d_pct:>4.0f}% "
                    f"{r['f1']:>7.4f} {r['auc']:>7.4f} {r['accuracy']:>7.4f} "
                    f"{r['precision']:>7.4f} {r['recall']:>7.4f}"
                )
                f1s.append(r['f1'])
                aucs.append(r['auc'])
                accs.append(r['accuracy'])

            print(f"  {'-'*70}")
            print(
                f"  {'MEAN':<15} {'':>5} {'':>5} "
                f"{np.mean(f1s):>7.4f} {np.mean(aucs):>7.4f} {np.mean(accs):>7.4f}"
            )
            print(
                f"  {'STD':<15} {'':>5} {'':>5} "
                f"{np.std(f1s):>7.4f} {np.std(aucs):>7.4f} {np.std(accs):>7.4f}"
            )

        return {
            name: {
                'per_project': results,
                'mean_f1': float(np.mean([r['f1'] for r in results])),
                'std_f1': float(np.std([r['f1'] for r in results])),
                'mean_auc': float(np.mean([r['auc'] for r in results])),
                'std_auc': float(np.std([r['auc'] for r in results])),
                'mean_accuracy': float(np.mean([r['accuracy'] for r in results])),
            }
            for name, results in lopo_results.items()
        }

    # -- Full pipeline ------------------------------------------------------
    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        include_issue_type: bool = False,
        skip_gridsearch: bool = False,
        skip_lopo: bool = False,
        output_dir: str = 'traditional_ml/results',
        aug_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        all_results: Dict = {
            'method': 'Traditional ML (TF-IDF + SVM / Gradient Boosting)',
            'timestamp': datetime.now().isoformat(),
            'n_manual_samples': len(df),
            'n_augmented_samples': len(aug_df) if aug_df is not None else 0,
            'n_total_samples': len(df) + (len(aug_df) if aug_df is not None else 0),
            'include_issue_type': include_issue_type,
            'tfidf_params': {
                'max_features': self.config.TFIDF_MAX_FEATURES,
                'ngram_range': list(self.config.TFIDF_NGRAM_RANGE),
            },
        }

        # --- Combine manual + augmented for holdout/CV ---
        if aug_df is not None:
            combined = pd.concat([df, aug_df], ignore_index=True)
            logger.info(f"\nCombined: {len(df)} manual + {len(aug_df)} augmented "
                        f"= {len(combined)} total")
        else:
            combined = df

        # --- Train / test split (on combined data) ---
        texts = combined['text'].tolist()
        y = combined['label_encoded'].values
        issue_types = combined['issue_type'].tolist() if include_issue_type else None

        (
            texts_train, texts_test,
            y_train, y_test,
        ) = train_test_split(
            texts, y,
            test_size=self.config.TEST_SPLIT,
            stratify=y,
            random_state=self.config.RANDOM_SEED,
        )

        if include_issue_type:
            types_train, types_test = train_test_split(
                issue_types,
                test_size=self.config.TEST_SPLIT,
                stratify=y,
                random_state=self.config.RANDOM_SEED,
            )
        else:
            types_train = types_test = None

        X_train = self.build_features(texts_train, types_train, fit=True)
        X_test = self.build_features(texts_test, types_test, fit=False)

        logger.info(f"\nTrain: {X_train.shape[0]}  Test: {X_test.shape[0]}  "
                     f"Features: {X_train.shape[1]}")

        # --- Hyperparameter search ---
        if not skip_gridsearch:
            self.hyperparameter_search(X_train, y_train)
        all_results['best_params'] = self.best_params

        # --- Holdout evaluation ---
        holdout = self.train_and_evaluate(X_train, y_train, X_test, y_test)
        all_results['holdout'] = {
            name: res['metrics'] for name, res in holdout.items()
        }
        all_results['holdout_best_params'] = {
            name: res['best_params'] for name, res in holdout.items()
        }

        # --- Cross-validation ---
        X_full = self.build_features(texts, issue_types, fit=True)
        cv_summary = self.cross_validate(X_full, y)
        all_results['cross_validation'] = cv_summary

        # --- LOPO (manual labels only for test, aug added to training) ---
        if not skip_lopo:
            lopo = self.lopo_evaluate(df, include_issue_type, aug_df=aug_df)
            all_results['lopo'] = lopo

        # --- Visualizations ---
        plot_confusion_matrices(holdout, y_test, output_dir)
        plot_roc_curves(holdout, y_test, output_dir)
        plot_model_comparison(all_results, output_dir)
        if not skip_lopo:
            plot_lopo_comparison(all_results['lopo'], output_dir)

        # --- Save JSON ---
        json_path = os.path.join(output_dir, 'traditional_models_results.json')
        # Remove non-serialisable arrays before saving
        save_results = {k: v for k, v in all_results.items()}
        with open(json_path, 'w') as f:
            json.dump(save_results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {json_path}")

        return all_results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_confusion_matrices(
    holdout: Dict, y_test: np.ndarray, output_dir: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = ['svm', 'gradient_boosting']
    titles = ['SVM', 'Gradient Boosting']

    for ax, name, title in zip(axes, names, titles):
        m = holdout[name]['metrics']
        cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Non-Design', 'Design'],
            yticklabels=['Non-Design', 'Design'],
            cbar=False,
        )
        ax.set_title(f"{title}\nF1={m['f1']:.3f}  AUC={m.get('auc', 0):.3f}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    fig.suptitle('Holdout Test Set — Confusion Matrices', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'traditional_confusion_matrices.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved {path}")


def plot_roc_curves(
    holdout: Dict, y_test: np.ndarray, output_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'svm': '#1f77b4', 'gradient_boosting': '#ff7f0e'}
    labels = {'svm': 'SVM', 'gradient_boosting': 'Gradient Boosting'}

    for name in ['svm', 'gradient_boosting']:
        y_prob = np.array(holdout[name]['y_prob'])
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[name],
                label=f"{labels[name]} (AUC = {roc_auc:.3f})", lw=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Holdout Test Set')
    ax.legend(loc='lower right')
    plt.tight_layout()
    path = os.path.join(output_dir, 'traditional_roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved {path}")


def plot_model_comparison(results: Dict, output_dir: str) -> None:
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    model_names = ['SVM', 'Gradient Boosting']
    model_keys = ['svm', 'gradient_boosting']

    # Holdout metrics
    holdout_vals = {
        mk: [results['holdout'][k].get(m, 0) for k in model_keys]
        for m, mk in zip(metrics_keys, metrics_keys)
    }

    x = np.arange(len(metrics_keys))
    width = 0.3
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (model_name, model_key) in enumerate(zip(model_names, model_keys)):
        vals = [results['holdout'][model_key].get(m, 0) for m in metrics_keys]
        bars = ax.bar(x + i * width, vals, width, label=model_name)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — Holdout Test Set')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([m.upper() for m in metrics_keys])
    ax.set_ylim(0, 1.15)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, 'traditional_model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved {path}")


def plot_lopo_comparison(lopo: Dict, output_dir: str) -> None:
    projects = [r['project'] for r in lopo['svm']['per_project']]
    svm_f1 = [r['f1'] for r in lopo['svm']['per_project']]
    gb_f1 = [r['f1'] for r in lopo['gradient_boosting']['per_project']]

    x = np.arange(len(projects))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width / 2, svm_f1, width, label='SVM', color='#1f77b4')
    ax.bar(x + width / 2, gb_f1, width, label='Gradient Boosting', color='#ff7f0e')

    ax.set_ylabel('F1 Score')
    ax.set_title('LOPO Cross-Validation — Per-Project F1 Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(projects, rotation=45, ha='right')
    ax.legend()

    # Add mean lines
    ax.axhline(np.mean(svm_f1), color='#1f77b4', linestyle='--', alpha=0.5,
               label=f'SVM mean={np.mean(svm_f1):.3f}')
    ax.axhline(np.mean(gb_f1), color='#ff7f0e', linestyle='--', alpha=0.5,
               label=f'GB mean={np.mean(gb_f1):.3f}')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, 'traditional_lopo_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Train traditional ML models (SVM + Gradient Boosting) '
                    'for design issue classification.',
    )
    parser.add_argument(
        '--data', default='./output/all_manually_labelled.csv',
        help='Path to labelled CSV file',
    )
    parser.add_argument(
        '--out_dir', default='traditional_ml/results',
        help='Directory for output files',
    )
    parser.add_argument(
        '--include_issue_type', action='store_true',
        help='Include issue_type as a feature (one-hot encoded)',
    )
    parser.add_argument(
        '--skip_lopo', action='store_true',
        help='Skip LOPO cross-validation (faster)',
    )
    parser.add_argument(
        '--skip_gridsearch', action='store_true',
        help='Skip hyperparameter search, use defaults',
    )
    parser.add_argument(
        '--cv_folds', type=int, default=5,
        help='Number of CV folds',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--aug_design', default=None,
        help='Path to augmented design labels CSV',
    )
    parser.add_argument(
        '--aug_nondesign', default=None,
        help='Path to augmented non-design labels CSV',
    )
    parser.add_argument(
        '--n_augment', type=int, default=0,
        help='Total number of augmented samples to add (balanced 50/50)',
    )
    args = parser.parse_args()

    config = TraditionalModelConfig()
    config.CV_FOLDS = args.cv_folds
    config.RANDOM_SEED = args.seed
    config.OUTPUT_DIR = args.out_dir

    df = load_and_prepare_data(args.data)

    aug_df = None
    if args.n_augment > 0 and args.aug_design and args.aug_nondesign:
        aug_df = load_augmented_data(
            args.aug_design, args.aug_nondesign,
            args.n_augment, seed=args.seed,
        )

    trainer = TraditionalModelTrainer(config)
    trainer.run_full_pipeline(
        df,
        include_issue_type=args.include_issue_type,
        skip_gridsearch=args.skip_gridsearch,
        skip_lopo=args.skip_lopo,
        output_dir=args.out_dir,
        aug_df=aug_df,
    )


if __name__ == '__main__':
    main()
