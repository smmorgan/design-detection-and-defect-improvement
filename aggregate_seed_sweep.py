#!/usr/bin/env python3
"""
Aggregate the multi-seed LOPO sweep (run_seed_sweep.sh) into mean ± SD.

For each config, reports the across-seed mean and SD of mean F1/AUC/accuracy,
plus per-fold F1 mean/SD/range across seeds and the spread of tuned thresholds.
Also writes a seed-averaged results file per config ({name}_seedavg_results.json)
in the same per_project format eval_lopo.py emits, so significance_tests.py
--seed_avg can test on per-fold F1 that isn't a single noisy draw.

Usage:
    python aggregate_seed_sweep.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SWEEP_DIR = Path("gcp_results/seed_sweep")
PATTERN = re.compile(r"^(?P<name>.+)_seed(?P<seed>\d+)_results\.json$")
METRICS = ["f1", "auc", "accuracy", "precision", "recall"]


def main():
    runs = defaultdict(dict)  # name -> seed -> data
    for path in sorted(SWEEP_DIR.glob("*_seed*_results.json")):
        m = PATTERN.match(path.name)
        if m:
            runs[m["name"]][int(m["seed"])] = json.loads(path.read_text())

    summary = {}
    for name, by_seed in sorted(runs.items()):
        seeds = sorted(by_seed)
        projects = sorted(p["project"] for p in by_seed[seeds[0]]["per_project"])
        # fold_vals[metric][project] -> list over seeds
        fold_vals = {m: defaultdict(list) for m in METRICS + ["threshold"]}
        for seed in seeds:
            for p in by_seed[seed]["per_project"]:
                for m in fold_vals:
                    if p.get(m) is not None:
                        fold_vals[m][p["project"]].append(p[m])

        run_means = {m: [float(np.mean([p[m] for p in by_seed[s]["per_project"]]))
                         for s in seeds] for m in METRICS}
        per_fold = {
            proj: {
                "f1_mean": round(float(np.mean(fold_vals["f1"][proj])), 4),
                "f1_sd": round(float(np.std(fold_vals["f1"][proj], ddof=1)), 4) if len(seeds) > 1 else None,
                "f1_range": round(float(np.ptp(fold_vals["f1"][proj])), 4),
                "threshold_range": ([round(min(fold_vals["threshold"][proj]), 3),
                                     round(max(fold_vals["threshold"][proj]), 3)]
                                    if fold_vals["threshold"][proj] else None),
            }
            for proj in projects
        }
        summary[name] = {
            "seeds": seeds,
            "n_seeds": len(seeds),
            **{f"mean_{m}_across_seeds": round(float(np.mean(v)), 4) for m, v in run_means.items()},
            **{f"sd_{m}_across_seeds": (round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None)
               for m, v in run_means.items()},
            "per_seed_mean_f1": {s: round(v, 4) for s, v in zip(seeds, run_means["f1"])},
            "per_fold": per_fold,
        }

        # Seed-averaged per-fold results, same shape as an eval_lopo.py output
        seedavg = {
            "method": f"Seed-averaged LOPO ({len(seeds)} seeds: {seeds})",
            "source_config": by_seed[seeds[0]].get("hyperparameters"),
            "pretrained_model": by_seed[seeds[0]].get("pretrained_model"),
            "per_project": [
                {"project": proj,
                 **{m: float(np.mean(fold_vals[m][proj])) for m in METRICS}}
                for proj in projects
            ],
        }
        seedavg["mean_f1"] = round(float(np.mean([p["f1"] for p in seedavg["per_project"]])), 4)
        (SWEEP_DIR / f"{name}_seedavg_results.json").write_text(json.dumps(seedavg, indent=2))

    out = SWEEP_DIR / "seed_sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"{'Config':<24} {'n':>2} {'F1 mean ± SD':>16} {'min':>7} {'max':>7} {'AUC mean ± SD':>16}")
    print("-" * 78)
    for name, s in summary.items():
        f1s = list(s["per_seed_mean_f1"].values())
        sd_f1 = s["sd_f1_across_seeds"] or 0.0
        sd_auc = s["sd_auc_across_seeds"] or 0.0
        print(f"{name:<24} {s['n_seeds']:>2} "
              f"{s['mean_f1_across_seeds']:>8.4f} ± {sd_f1:.4f} "
              f"{min(f1s):>7.4f} {max(f1s):>7.4f} "
              f"{s['mean_auc_across_seeds']:>8.4f} ± {sd_auc:.4f}")
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
