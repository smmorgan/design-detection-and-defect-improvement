#!/usr/bin/env python3
"""
Statistical significance tests for LOPO experiment comparisons.

Runs Wilcoxon signed-rank tests (non-parametric, paired) across the 10 LOPO folds
for each pair of configurations. Also reports effect sizes (Cohen's d) and
confidence intervals for the mean difference.

Usage:
    python significance_tests.py
    python significance_tests.py --seed_avg   # model-family tests on seed-averaged
                                               # per-fold F1 (aggregate_seed_sweep.py)
"""

import argparse
import json
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats


def load_results(filepath, keys=None):
    """Load LOPO results and extract per-project F1 and AUC scores.

    `keys` navigates to a nested per_project list, e.g. keys=("lopo", "svm")
    for traditional_models_results.json where results sit under data["lopo"]["svm"]["per_project"].
    """
    with open(filepath) as f:
        data = json.load(f)
    for k in (keys or ()):
        data = data[k]
    per_project = sorted(data["per_project"], key=lambda p: p["project"])
    return {
        "projects": [p["project"] for p in per_project],
        "f1": np.array([p["f1"] for p in per_project]),
        "auc": np.array([p["auc"] for p in per_project]),
    }


def cohens_d(a, b):
    """Paired Cohen's d (mean difference / pooled SD)."""
    diff = a - b
    return diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0


def mean_diff_ci(a, b, confidence=0.95):
    """Bootstrap-free CI for mean difference using t-distribution."""
    diff = a - b
    n = len(diff)
    mean = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean - t_crit * se, mean + t_crit * se


def compare_pair(name_a, res_a, name_b, res_b, metric="f1"):
    """Run Wilcoxon signed-rank test and compute effect size for one pair."""
    a = res_a[metric]
    b = res_b[metric]
    diff = a - b

    # Wilcoxon signed-rank (two-sided)
    try:
        stat, p_value = stats.wilcoxon(a, b, alternative="two-sided")
    except ValueError:
        # All differences are zero
        stat, p_value = 0.0, 1.0

    d = cohens_d(a, b)
    ci_lo, ci_hi = mean_diff_ci(a, b)

    return {
        "comparison": f"{name_a} vs {name_b}",
        "metric": metric,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "ci_95_lower": float(ci_lo),
        "ci_95_upper": float(ci_hi),
        "cohens_d": float(d),
        "wilcoxon_stat": float(stat),
        "p_value": float(p_value),
        "significant_005": bool(p_value < 0.05),
        "significant_001": bool(p_value < 0.01),
    }


def holm_adjust(p_values):
    """Holm-Bonferroni step-down adjusted p-values (same order as input)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def interpret_effect(d):
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def run_model_family_comparisons(results_dir, seed_avg=False):
    """Compare RoBERTa (best LOPO config) against SVM, GB, and the LLM baselines.

    Answers R1#4 ("statistical tests ... between BERTs and classical MLs") and
    extends it to the LLM baseline added for R1#1/R2#3. Kept separate from the
    augmentation-ablation comparisons above since these are different models,
    not different configs of the same model.

    With seed_avg, the fine-tuned models use per-fold F1 averaged over the
    multi-seed sweep, since a single run's per-fold F1 carries GPU training
    noise large enough to flip these comparisons.
    """
    trad_path = Path("traditional_ml/results/traditional_models_results.json")

    configs = {
        "roberta_best": (results_dir / "lopo_metadata_baseline_sqrtw_results.json", None),
        "modernbert": (results_dir / "lopo_modernbert_results.json", None),
        "qwen25_1.5b_lora": (results_dir / "lopo_qwen25_1.5b_lora_results.json", None),
        "qwen25_1.5b_lora_pretrained": (results_dir / "lopo_qwen25_1.5b_lora_pretrained_results.json", None),
        "svm": (trad_path, ("lopo", "svm")),
        "gradient_boosting": (trad_path, ("lopo", "gradient_boosting")),
        "llm_zeroshot_claude": (results_dir / "llm_zeroshot_anthropic_results.json", None),
        "llm_fewshot_claude": (results_dir / "llm_fewshot_anthropic_results.json", None),
        "llm_zeroshot_gpt4o": (results_dir / "llm_zeroshot_openai_results.json", None),
        "llm_fewshot_gpt4o": (results_dir / "llm_fewshot_openai_results.json", None),
        "llm_zeroshot_local": (results_dir / "llm_zeroshot_local_results.json", None),
        "llm_fewshot_local": (results_dir / "llm_fewshot_local_results.json", None),
    }
    if seed_avg:
        sweep_dir = results_dir / "seed_sweep"
        trad_sweep_dir = Path("traditional_ml/seed_sweep")
        configs["roberta_best"] = (sweep_dir / "base_meta_seedavg_results.json", None)
        configs["modernbert"] = (sweep_dir / "modernbert_meta_sqrtw_seedavg_results.json", None)
        configs["svm"] = (trad_sweep_dir / "baseline_svm_seedavg_results.json", None)
        configs["gradient_boosting"] = (trad_sweep_dir / "baseline_gradient_boosting_seedavg_results.json", None)

    loaded = {}
    for name, (path, keys) in configs.items():
        if path.exists():
            loaded[name] = load_results(path, keys)
            print(f"Loaded {name}: mean F1={loaded[name]['f1'].mean():.4f}, "
                  f"mean AUC={loaded[name]['auc'].mean():.4f}")
        else:
            print(f"WARNING: {path} not found, skipping {name}")

    if "roberta_best" not in loaded:
        print("roberta_best missing -- skipping model-family comparisons")
        return

    print(f"\n{'='*80}")
    print("MODEL-FAMILY SIGNIFICANCE TESTS (Wilcoxon signed-rank, two-sided)")
    print(f"{'='*80}\n")

    comparisons = [("roberta_best", other) for other in
                   ["modernbert", "qwen25_1.5b_lora", "qwen25_1.5b_lora_pretrained", "svm", "gradient_boosting", "llm_zeroshot_claude", "llm_fewshot_claude",
                    "llm_zeroshot_gpt4o", "llm_fewshot_gpt4o",
                    "llm_zeroshot_local", "llm_fewshot_local"]
                   if other in loaded]

    all_results = []
    for metric in ["f1", "auc"]:
        print(f"\n--- {metric.upper()} ---\n")
        metric_results = []
        for name_a, name_b in comparisons:
            res_a, res_b = loaded[name_a], loaded[name_b]
            if res_a["projects"] != res_b["projects"]:
                print(f"SKIPPING {name_a} vs {name_b}: project sets don't match "
                      f"({res_a['projects']} vs {res_b['projects']})")
                continue
            metric_results.append(compare_pair(name_a, res_a, name_b, res_b, metric))
        # Holm correction across this metric's family of comparisons
        for result, p_holm in zip(metric_results,
                                  holm_adjust([r["p_value"] for r in metric_results])):
            result["p_holm"] = float(p_holm)
            result["significant_holm_005"] = bool(p_holm < 0.05)
            effect = interpret_effect(result["cohens_d"])
            sig = "**" if result["significant_001"] else ("*" if result["significant_005"] else "")
            print(f"{result['comparison']:<35} diff={result['mean_diff']:+.4f} "
                  f"d={result['cohens_d']:+.3f} ({effect}) p={result['p_value']:.4f} "
                  f"p_holm={p_holm:.4f} {sig}")
        all_results.extend(metric_results)

    suffix = "_seedavg" if seed_avg else ""
    out_path = results_dir / f"significance_tests_model_family{suffix}.json"
    with open(out_path, "w") as f:
        json.dump({"method": "Wilcoxon signed-rank (model family), Holm-corrected per metric",
                   "seed_averaged": seed_avg,
                   "roberta_source": str(configs["roberta_best"][0]),
                   "comparisons": all_results}, f, indent=2)
    print(f"\nModel-family results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_avg', action='store_true',
                        help='Only run model-family tests, using seed-averaged per-fold '
                             'F1 for the fine-tuned models (run aggregate_seed_sweep.py first)')
    args = parser.parse_args()
    results_dir = Path("gcp_results")
    if args.seed_avg:
        run_model_family_comparisons(results_dir, seed_avg=True)
        return

    # Define all configurations to compare
    configs = {
        "baseline": results_dir / "lopo_results.json",
        "augmented": results_dir / "lopo_augmented_results.json",
        "aug_top250": results_dir / "lopo_augmented_top250_results.json",
        "aug_top500": results_dir / "lopo_augmented_top500_results.json",
        "aug_top750": results_dir / "lopo_augmented_top750_results.json",
        "aug_mw5": results_dir / "lopo_augmented_mw5_results.json",
        "freeze10": results_dir / "lopo_freeze10_results.json",
        "freeze10_aug": results_dir / "lopo_freeze10_augmented_results.json",
    }

    # Load all results
    loaded = {}
    for name, path in configs.items():
        if path.exists():
            loaded[name] = load_results(path)
            print(f"Loaded {name}: mean F1={loaded[name]['f1'].mean():.4f}, mean AUC={loaded[name]['auc'].mean():.4f}")
        else:
            print(f"WARNING: {path} not found, skipping {name}")

    print(f"\n{'='*80}")
    print("PAIRWISE SIGNIFICANCE TESTS (Wilcoxon signed-rank, two-sided)")
    print(f"{'='*80}\n")

    # Key comparisons (not all pairs — focus on meaningful ones)
    key_comparisons = [
        ("baseline", "augmented"),
        ("baseline", "aug_top250"),
        ("baseline", "aug_top500"),
        ("baseline", "aug_top750"),
        ("baseline", "aug_mw5"),
        ("baseline", "freeze10"),
        ("augmented", "freeze10_aug"),
        ("aug_top250", "augmented"),
        ("aug_top250", "aug_top500"),
    ]

    all_results = []

    for metric in ["f1", "auc"]:
        print(f"\n--- {metric.upper()} ---\n")
        print(f"{'Comparison':<35} {'Mean Diff':>10} {'95% CI':>20} {'Cohen d':>9} {'Effect':>12} {'p-value':>10} {'Sig':>5}")
        print("-" * 105)

        for name_a, name_b in key_comparisons:
            if name_a not in loaded or name_b not in loaded:
                continue

            result = compare_pair(name_a, loaded[name_a], name_b, loaded[name_b], metric)
            all_results.append(result)

            effect = interpret_effect(result["cohens_d"])
            sig = "**" if result["significant_001"] else ("*" if result["significant_005"] else "")
            ci_str = f"[{result['ci_95_lower']:+.4f}, {result['ci_95_upper']:+.4f}]"

            print(f"{result['comparison']:<35} {result['mean_diff']:>+10.4f} {ci_str:>20} {result['cohens_d']:>+9.3f} {effect:>12} {result['p_value']:>10.4f} {sig:>5}")

    # Bonferroni correction note
    n_tests = len([r for r in all_results if r["metric"] == "f1"])
    bonferroni_threshold = 0.05 / n_tests if n_tests > 0 else 0.05
    print(f"\n{'='*80}")
    print(f"Note: {n_tests} comparisons per metric.")
    print(f"Bonferroni-corrected threshold (alpha=0.05): p < {bonferroni_threshold:.4f}")
    print(f"* = p < 0.05, ** = p < 0.01")
    print(f"n = 10 projects per comparison (small sample — interpret with caution)")
    print(f"{'='*80}")

    # Save results to JSON
    output = {
        "method": "Wilcoxon signed-rank test (two-sided, paired)",
        "n_folds": 10,
        "bonferroni_threshold": bonferroni_threshold,
        "comparisons": all_results,
    }
    out_path = results_dir / "significance_tests.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    run_model_family_comparisons(results_dir)


if __name__ == "__main__":
    main()
