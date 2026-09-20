#!/usr/bin/env python3
"""
Cross-project variance and qualitative error analysis for LOPO results.

Addresses R2.6 (LOPO variance), R1.3 (qualitative error analysis), and feeds
R2.4 (project-specific vocabulary). See REVIEWER_FEEDBACK.md.

Subcommands:
  variance       Per-project F1/AUC matrix across every model family, project
                 characteristics (size, design rate, ticket length, issue-type
                 mix, vocabulary shift vs. the other nine projects, keyword
                 heuristic separability), Spearman correlations of difficulty
                 against those characteristics, and cross-model agreement on
                 which projects are hard (Kendall's W). Needs only existing
                 gcp_results/*.json -- no GPU.

  sample-errors  Draw false positives / false negatives from a per-ticket
                 predictions CSV (written by eval_lopo.py or eval_llm_baseline.py)
                 into a coding sheet for manual error categorisation. Optional
                 --compare CSVs (e.g. Claude few-shot predictions) are joined so
                 each error shows whether other models made the same mistake.

Usage:
    python analyze_lopo_errors.py variance
    python analyze_lopo_errors.py sample-errors \\
        --predictions gcp_results/lopo_metadata_baseline_sqrtw_rerun_results_predictions.csv \\
        --compare claude_fewshot=gcp_results/llm_fewshot_anthropic_results_predictions.csv \\
        --per_cell 10 --out gcp_results/error_analysis/coding_sheet.csv

Error-category codebook for the coding sheet (one primary category per ticket):
  LABEL_AMBIGUOUS   Ticket is genuinely borderline; a second rater could
                    reasonably disagree with the gold label.
  LABEL_ERROR       Gold label looks wrong on re-reading.
  DESIGN_VOCAB_IMPL Implementation/bug work described with design-sounding
                    words (architecture, refactor, API, ...) -> FP.
  IMPLICIT_DESIGN   Design work described without design vocabulary
                    (e.g. terse summary, domain jargon only) -> FN.
  PROJECT_JARGON    Decision hinges on project-specific terms (docker, helm,
                    plugin, mesos, ...).
  SPARSE_TEXT       Too little text to decide (empty/one-line description).
  ISSUE_TYPE        Issue-type metadata pulled the prediction the wrong way.
  OTHER             Anything else -- describe in notes.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from generate_n_labels import compute_design_score
from significance_tests import load_results

RESULTS_DIR = Path('gcp_results')
MANUAL_LABELS = Path('ieee_dataport/manual_labels/manually_labelled_issues.csv')
TRAD_PATH = Path('traditional_ml/results/traditional_models_results.json')
OUT_DIR = RESULTS_DIR / 'error_analysis'

# Same model-family lineup as significance_tests.run_model_family_tests(), plus
# the pre-selection RoBERTa baseline so config-selection effects are visible.
MODELS = {
    'roberta_best': (RESULTS_DIR / 'lopo_metadata_baseline_sqrtw_results.json', None),
    'roberta_baseline': (RESULTS_DIR / 'lopo_results.json', None),
    'modernbert': (RESULTS_DIR / 'lopo_modernbert_results.json', None),
    'qwen_lora_pretrained': (RESULTS_DIR / 'lopo_qwen25_1.5b_lora_pretrained_results.json', None),
    'qwen_lora_cold': (RESULTS_DIR / 'lopo_qwen25_1.5b_lora_results.json', None),
    'svm': (TRAD_PATH, ('lopo', 'svm')),
    'gradient_boosting': (TRAD_PATH, ('lopo', 'gradient_boosting')),
    'claude_zeroshot': (RESULTS_DIR / 'llm_zeroshot_anthropic_results.json', None),
    'claude_fewshot': (RESULTS_DIR / 'llm_fewshot_anthropic_results.json', None),
    'gpt4o_zeroshot': (RESULTS_DIR / 'llm_zeroshot_openai_results.json', None),
    'gpt4o_fewshot': (RESULTS_DIR / 'llm_fewshot_openai_results.json', None),
    'llama_zeroshot': (RESULTS_DIR / 'llm_zeroshot_local_results.json', None),
    'llama_fewshot': (RESULTS_DIR / 'llm_fewshot_local_results.json', None),
}

TOKEN_RE = re.compile(r'[a-z][a-z0-9_]+')


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def js_divergence(p_counts, q_counts):
    """Jensen-Shannon divergence (base 2, in [0, 1]) between two unigram Counters."""
    vocab = sorted(set(p_counts) | set(q_counts))
    p = np.array([p_counts.get(w, 0) for w in vocab], dtype=float)
    q = np.array([q_counts.get(w, 0) for w in vocab], dtype=float)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def kendalls_w(scores):
    """Kendall's coefficient of concordance for a (raters x items) score matrix.

    Here raters = models, items = projects. W=1 means every model ranks project
    difficulty identically. No tie correction (ties are rare in 4-dp F1s).
    """
    m, n = scores.shape
    ranks = np.vstack([stats.rankdata(row) for row in scores])
    rank_sums = ranks.sum(axis=0)
    s = np.sum((rank_sums - rank_sums.mean()) ** 2)
    w = 12 * s / (m ** 2 * (n ** 3 - n))
    chi2 = m * (n - 1) * w
    p = stats.chi2.sf(chi2, n - 1)
    return float(w), float(p)


def project_characteristics(df):
    """One row per project, each measured against the pooled other nine projects."""
    df = df.copy()
    df['text'] = df['summary'].fillna('') + ' ' + df['description'].fillna('')
    df['tokens'] = df['text'].map(tokenize)
    df['is_design'] = (df['label'] == 'design').astype(int)
    df['keyword_score'] = [
        compute_design_score(s, d)
        for s, d in zip(df['summary'].fillna(''), df['description'].fillna(''))
    ]

    rows = []
    for project, sub in df.groupby('project'):
        rest = df[df['project'] != project]
        sub_counts = Counter(t for toks in sub['tokens'] for t in toks)
        rest_counts = Counter(t for toks in rest['tokens'] for t in toks)
        rest_vocab = set(rest_counts)
        n_tokens = sum(sub_counts.values())
        oov = sum(c for t, c in sub_counts.items() if t not in rest_vocab) / n_tokens

        design_counts = Counter(t for toks in sub.loc[sub['is_design'] == 1, 'tokens'] for t in toks)
        rest_design_counts = Counter(t for toks in rest.loc[rest['is_design'] == 1, 'tokens'] for t in toks)

        rest_types = set(rest['issue_type'].dropna())
        type_share = sub['issue_type'].value_counts(normalize=True)

        rows.append({
            'project': project,
            'n': len(sub),
            'design_rate': sub['is_design'].mean(),
            'median_tokens': float(np.median(sub['tokens'].map(len))),
            'empty_description_rate': (sub['description'].fillna('').str.strip().str.len() < 20).mean(),
            'bug_share': float(type_share.get('Bug', 0.0)),
            'unseen_issue_type_rate': (~sub['issue_type'].isin(rest_types)).mean(),
            'oov_token_rate': oov,
            'jsd_all_vs_rest': js_divergence(sub_counts, rest_counts),
            'jsd_design_vs_rest_design': js_divergence(design_counts, rest_design_counts),
            # Keyword-only score (not the full cherry-picking selection rule): how
            # well canonical architecture vocabulary separates human D from N labels.
            'keyword_score_auc': roc_auc_score(sub['is_design'], sub['keyword_score']),
            'design_with_zero_keywords_rate': (sub.loc[sub['is_design'] == 1, 'keyword_score'] == 0).mean(),
        })
    return pd.DataFrame(rows).set_index('project')


def run_variance(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    f1, auc = {}, {}
    for name, (path, keys) in MODELS.items():
        if not Path(path).exists():
            print(f"  [skip] {name}: {path} not found")
            continue
        r = load_results(path, keys)
        f1[name] = dict(zip(r['projects'], r['f1']))
        auc[name] = dict(zip(r['projects'], r['auc']))
    f1 = pd.DataFrame(f1)
    auc = pd.DataFrame(auc)

    chars = project_characteristics(pd.read_csv(MANUAL_LABELS))

    # F1 is prevalence-dependent: an all-design classifier scores 2p/(1+p).
    # Report lift over that floor so low-design-rate projects aren't penalised twice.
    chars['f1_all_design_floor'] = 2 * chars['design_rate'] / (1 + chars['design_rate'])

    difficulty = pd.DataFrame({
        'roberta_best_f1': f1['roberta_best'],
        'roberta_best_auc': auc['roberta_best'],
        'mean_f1_all_models': f1.mean(axis=1),
        'mean_auc_all_models': auc.mean(axis=1),
        'roberta_best_f1_lift_over_floor': f1['roberta_best'] - chars['f1_all_design_floor'],
    })

    correlations = []
    for target in difficulty.columns:
        for feat in chars.columns:
            rho, p = stats.spearmanr(difficulty[target], chars[feat])
            correlations.append({'target': target, 'feature': feat,
                                 'spearman_rho': float(rho), 'p_value': float(p)})
    correlations = pd.DataFrame(correlations)

    w_f1, p_w_f1 = kendalls_w(f1.T.values)
    w_auc, p_w_auc = kendalls_w(auc.T.values)
    # Is the fine-tuned model hard on the same projects as the training-free LLMs?
    llm_cols = [c for c in f1.columns if c.startswith(('claude', 'gpt4o', 'llama'))]
    rho_llm, p_llm = stats.spearmanr(f1['roberta_best'], f1[llm_cols].mean(axis=1))

    # Worst project per model -- is it always the same one?
    worst = {m: f1[m].idxmin() for m in f1.columns}

    f1.round(4).to_csv(OUT_DIR / 'per_project_f1_matrix.csv')
    auc.round(4).to_csv(OUT_DIR / 'per_project_auc_matrix.csv')
    chars.join(difficulty).round(4).to_csv(OUT_DIR / 'project_characteristics.csv')
    correlations.round(4).to_csv(OUT_DIR / 'difficulty_correlations.csv', index=False)

    summary = {
        'n_projects': len(f1),
        'n_models': f1.shape[1],
        'kendalls_w_f1': w_f1, 'kendalls_w_f1_p': p_w_f1,
        'kendalls_w_auc': w_auc, 'kendalls_w_auc_p': p_w_auc,
        'spearman_roberta_best_vs_mean_llm_f1': float(rho_llm),
        'spearman_roberta_best_vs_mean_llm_f1_p': float(p_llm),
        'worst_project_per_model': worst,
        'per_project_f1_range': {m: [float(f1[m].min()), float(f1[m].max())] for m in f1.columns},
        'caveat': 'n=10 projects: correlations are exploratory, uncorrected, and low-powered.',
    }
    with open(OUT_DIR / 'variance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    pd.set_option('display.width', 200)
    print("\nPer-project F1 (models x projects):")
    print(f1.round(3).to_string())
    print("\nProject characteristics:")
    print(chars.join(difficulty).round(3).to_string())
    print("\nSpearman correlations with |rho| >= 0.5:")
    print(correlations[correlations['spearman_rho'].abs() >= 0.5].round(3).to_string(index=False))
    print(f"\nKendall's W across {f1.shape[1]} models: F1 W={w_f1:.3f} (p={p_w_f1:.4f}), "
          f"AUC W={w_auc:.3f} (p={p_w_auc:.4f})")
    print(f"RoBERTa-best vs mean-LLM per-project F1: rho={rho_llm:.3f} (p={p_llm:.4f})")
    print(f"Worst project per model: {worst}")
    print(f"\nOutputs written to {OUT_DIR}/")


def run_sample_errors(args):
    preds = pd.read_csv(args.predictions)
    labels = pd.read_csv(MANUAL_LABELS)[['issue_key', 'summary', 'description', 'label']]
    labels = labels.rename(columns={'label': 'gold_label'})
    sheet = preds.merge(labels, on='issue_key', how='left')

    for spec in args.compare or []:
        name, path = spec.split('=', 1)
        other = pd.read_csv(path)
        keep = ['issue_key', 'pred'] + (['rationale'] if 'rationale' in other.columns else [])
        other = other[keep].rename(columns={c: f'{name}_{c}' for c in keep if c != 'issue_key'})
        sheet = sheet.merge(other, on='issue_key', how='left')

    sheet['error_type'] = np.select(
        [(sheet['pred'] == 1) & (sheet['label'] == 0), (sheet['pred'] == 0) & (sheet['label'] == 1)],
        ['FP', 'FN'], default='correct',
    )
    errors = sheet[sheet['error_type'] != 'correct']
    if args.projects:
        errors = errors[errors['project'].isin(args.projects)]

    # Distance from the fold's decision threshold -- confidently-wrong errors are
    # the most informative, borderline ones the most likely to be label noise.
    if 'threshold' in errors.columns:
        errors = errors.assign(margin=(errors['prob_design'] - errors['threshold']).abs())

    # Shuffle once, then keep the first per_cell rows of each project x error_type
    # cell (groupby.apply drops the grouping columns on newer pandas).
    sampled = (
        errors.sample(frac=1, random_state=args.seed)
        .groupby(['project', 'error_type']).head(args.per_cell)
        .sort_values(['project', 'error_type'])
    )
    sampled = sampled.assign(error_category='', notes='')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out, index=False)

    counts = sheet.groupby('project')['error_type'].value_counts().unstack(fill_value=0)
    print(counts.to_string())
    print(f"\nSampled {len(sampled)} errors ({args.per_cell} per project x error type) -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('variance', help='Per-project variance analysis from existing results')

    se = sub.add_parser('sample-errors', help='Build an error-coding sheet from per-ticket predictions')
    se.add_argument('--predictions', required=True)
    se.add_argument('--compare', nargs='*', help='name=path CSVs of other models\' predictions to join')
    se.add_argument('--projects', nargs='*', help='Restrict to these projects (default: all)')
    se.add_argument('--per_cell', type=int, default=10, help='Errors sampled per project x FP/FN')
    se.add_argument('--seed', type=int, default=42)
    se.add_argument('--out', default=str(OUT_DIR / 'coding_sheet.csv'))

    args = parser.parse_args()
    if args.command == 'variance':
        run_variance(args)
    else:
        run_sample_errors(args)


if __name__ == '__main__':
    main()
