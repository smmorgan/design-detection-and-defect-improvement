#!/usr/bin/env python3
"""
Statistical significance tests for LOPO experiment comparisons.

Runs Wilcoxon signed-rank tests (non-parametric, paired) across the 10 LOPO folds
for each pair of configurations. Also reports effect sizes (Cohen's d) and
confidence intervals for the mean difference.

Usage:
    python significance_tests.py
"""

import json
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats


def load_results(filepath):
    """Load LOPO results and extract per-project F1 and AUC scores."""
    with open(filepath) as f:
        data = json.load(f)
    projects = [p["project"] for p in data["per_project"]]
    f1_scores = [p["f1"] for p in data["per_project"]]
    auc_scores = [p["auc"] for p in data["per_project"]]
    return {
        "projects": projects,
        "f1": np.array(f1_scores),
        "auc": np.array(auc_scores),
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


def main():
    results_dir = Path("gcp_results")

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


if __name__ == "__main__":
    main()
