#!/usr/bin/env python3
"""
Re-score finished LOPO runs with a pooled decision threshold.

Per-fold tuning picks the threshold that maximises F1 on one ~177-ticket
validation project. That optimum is noisy: across seeds the chosen threshold
for a fold can swing from 0.13 to 0.75, and the swing lands on test F1. This
re-scores each fold using a threshold tuned on the *other* folds' out-of-fold
predictions (~1.6k tickets), which is the same procedure as
`eval_lopo.py --threshold pooled` but applied to predictions already on disk,
so it costs no GPU time.

The held-out project is excluded from its own pool, so no test label informs
its threshold. (The pooled probabilities do come from models that trained on
the held-out project, so this is a stability fix, not a clean-room nested CV.)

Usage:
    python pooled_threshold.py                      # re-score the seed sweep
    python pooled_threshold.py path/to/*_predictions.csv
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SWEEP_DIR = Path("gcp_results/seed_sweep")
PATTERN = re.compile(r"^(?P<name>.+)_seed(?P<seed>\d+)_results_predictions\.csv$")


def tune(probs, labels, steps=100):
    """Threshold in [0.05, 0.95] maximising F1 (same grid as eval_lopo.py)."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, steps):
        score = f1_score(labels, (np.asarray(probs) >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t, best_f1


def rescore(csv_path):
    """Return {project: {'f1_pooled', 'f1_orig', 'threshold_pooled', 'threshold_orig'}}."""
    df = pd.read_csv(csv_path)
    out = {}
    for proj, fold in df.groupby("project"):
        pool = df[df["project"] != proj]
        t, _ = tune(pool["prob_design"].values, pool["label"].values)
        preds = (fold["prob_design"].values >= t).astype(int)
        out[proj] = {
            "f1_pooled": float(f1_score(fold["label"].values, preds, zero_division=0)),
            "f1_orig": float(f1_score(fold["label"].values,
                                      fold["pred"].values, zero_division=0)),
            "threshold_pooled": round(t, 4),
            "threshold_orig": float(fold["threshold"].iloc[0]),
        }
    return out


def main():
    paths = [Path(p) for p in sys.argv[1:]] or sorted(SWEEP_DIR.glob("*_seed*_predictions.csv"))
    if not paths:
        sys.exit("No predictions CSVs found.")

    by_config = defaultdict(dict)  # config -> seed -> per-project results
    for path in paths:
        m = PATTERN.match(path.name)
        key, seed = (m["name"], int(m["seed"])) if m else (path.stem, 0)
        by_config[key][seed] = rescore(path)

    summary = {}
    print(f"{'Config':<24} {'n':>2} {'per-fold tuned':>18} {'pooled':>18} {'Δ SD':>8}")
    print("-" * 74)
    for name, by_seed in sorted(by_config.items()):
        seeds = sorted(by_seed)
        orig = [float(np.mean([p["f1_orig"] for p in by_seed[s].values()])) for s in seeds]
        pooled = [float(np.mean([p["f1_pooled"] for p in by_seed[s].values()])) for s in seeds]
        ddof = 1 if len(seeds) > 1 else 0
        # Per-fold SD across seeds, averaged over folds: does pooling steady the folds?
        projects = sorted(by_seed[seeds[0]])
        fold_sd = {
            kind: float(np.mean([
                np.std([by_seed[s][proj][kind] for s in seeds], ddof=ddof)
                for proj in projects
            ]))
            for kind in ("f1_orig", "f1_pooled")
        }
        thr_range = float(np.mean([
            np.ptp([by_seed[s][proj]["threshold_pooled"] for s in seeds]) for proj in projects
        ]))
        thr_range_orig = float(np.mean([
            np.ptp([by_seed[s][proj]["threshold_orig"] for s in seeds]) for proj in projects
        ]))
        summary[name] = {
            "seeds": seeds,
            "mean_f1_per_fold_tuned": round(float(np.mean(orig)), 4),
            "sd_f1_per_fold_tuned": round(float(np.std(orig, ddof=ddof)), 4),
            "mean_f1_pooled": round(float(np.mean(pooled)), 4),
            "sd_f1_pooled": round(float(np.std(pooled, ddof=ddof)), 4),
            "mean_per_fold_sd_tuned": round(fold_sd["f1_orig"], 4),
            "mean_per_fold_sd_pooled": round(fold_sd["f1_pooled"], 4),
            "mean_threshold_range_tuned": round(thr_range_orig, 4),
            "mean_threshold_range_pooled": round(thr_range, 4),
            "per_seed": {s: {"per_fold_tuned": round(o, 4), "pooled": round(p, 4)}
                         for s, o, p in zip(seeds, orig, pooled)},
        }
        d = summary[name]
        print(f"{name:<24} {len(seeds):>2} "
              f"{d['mean_f1_per_fold_tuned']:>10.4f} ± {d['sd_f1_per_fold_tuned']:.4f} "
              f"{d['mean_f1_pooled']:>10.4f} ± {d['sd_f1_pooled']:.4f} "
              f"{d['sd_f1_pooled'] - d['sd_f1_per_fold_tuned']:>+8.4f}")

    print(f"\n{'Config':<24} {'mean per-fold SD':>20} {'mean threshold range':>26}")
    print(f"{'':<24} {'tuned':>9} {'pooled':>10} {'tuned':>12} {'pooled':>13}")
    print("-" * 74)
    for name, d in summary.items():
        print(f"{name:<24} {d['mean_per_fold_sd_tuned']:>9.4f} {d['mean_per_fold_sd_pooled']:>10.4f} "
              f"{d['mean_threshold_range_tuned']:>12.3f} {d['mean_threshold_range_pooled']:>13.3f}")

    # Seed-averaged per-fold F1 under pooled thresholds, for significance_tests.py
    for name, by_seed in by_config.items():
        seeds = sorted(by_seed)
        projects = sorted(by_seed[seeds[0]])
        per_project = [
            {"project": proj,
             "f1": float(np.mean([by_seed[s][proj]["f1_pooled"] for s in seeds])),
             "auc": None}
            for proj in projects
        ]
        (SWEEP_DIR / f"{name}_pooled_seedavg_results.json").write_text(json.dumps({
            "method": f"Seed-averaged LOPO, pooled threshold ({len(seeds)} seeds: {seeds})",
            "per_project": per_project,
            "mean_f1": round(float(np.mean([p["f1"] for p in per_project])), 4),
        }, indent=2))

    out = SWEEP_DIR / "pooled_threshold_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
