#!/usr/bin/env python3
"""
Aggregate the multi-seed traditional-ML (SVM/GB) sweep into mean +/- SD.

Mirrors aggregate_seed_sweep.py's approach but reads
traditional_ml/seed_sweep/{config}_seed{N}/traditional_models_results.json,
which nests per-project LOPO results under ["lopo"][model_name]["per_project"]
rather than at the top level.

Usage:
    python aggregate_traditional_sweep.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SWEEP_DIR = Path("traditional_ml/seed_sweep")
PATTERN = re.compile(r"^(?P<name>.+)_seed(?P<seed>\d+)$")
MODELS = ["svm", "gradient_boosting"]
METRICS = ["f1", "auc", "accuracy"]


def main():
    runs = defaultdict(dict)  # name -> seed -> data
    for d in sorted(SWEEP_DIR.iterdir()):
        if not d.is_dir():
            continue
        m = PATTERN.match(d.name)
        result_file = d / "traditional_models_results.json"
        if m and result_file.exists():
            runs[m["name"]][int(m["seed"])] = json.loads(result_file.read_text())

    summary = {}
    for name, by_seed in sorted(runs.items()):
        seeds = sorted(by_seed)
        summary[name] = {"seeds": seeds, "n_seeds": len(seeds), "models": {}}
        for model in MODELS:
            projects = sorted(p["project"] for p in by_seed[seeds[0]]["lopo"][model]["per_project"])
            fold_vals = {m: defaultdict(list) for m in METRICS}
            for seed in seeds:
                for p in by_seed[seed]["lopo"][model]["per_project"]:
                    for m in fold_vals:
                        if p.get(m) is not None:
                            fold_vals[m][p["project"]].append(p[m])

            run_means = {m: [float(np.mean([p[m] for p in by_seed[s]["lopo"][model]["per_project"]]))
                             for s in seeds] for m in METRICS}
            per_fold = {
                proj: {
                    "f1_mean": round(float(np.mean(fold_vals["f1"][proj])), 4),
                    "f1_sd": round(float(np.std(fold_vals["f1"][proj], ddof=1)), 4) if len(seeds) > 1 else None,
                }
                for proj in projects
            }
            summary[name]["models"][model] = {
                **{f"mean_{m}_across_seeds": round(float(np.mean(v)), 4) for m, v in run_means.items()},
                **{f"sd_{m}_across_seeds": (round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None)
                   for m, v in run_means.items()},
                "per_seed_mean_f1": {s: round(v, 4) for s, v in zip(seeds, run_means["f1"])},
                "per_fold": per_fold,
            }

            # Seed-averaged per-fold results, same shape eval scripts/significance_tests.py expect
            seedavg = {
                "method": f"{name} {model} seed-averaged LOPO ({len(seeds)} seeds: {seeds})",
                "per_project": [
                    {"project": proj,
                     **{m: float(np.mean(fold_vals[m][proj])) for m in METRICS}}
                    for proj in projects
                ],
            }
            seedavg["mean_f1"] = round(float(np.mean([p["f1"] for p in seedavg["per_project"]])), 4)
            (SWEEP_DIR / f"{name}_{model}_seedavg_results.json").write_text(json.dumps(seedavg, indent=2))

    out = SWEEP_DIR / "traditional_seed_sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"{'Config':<12} {'Model':<18} {'n':>2} {'F1 mean +/- SD':>18} {'min':>7} {'max':>7} {'AUC mean +/- SD':>18}")
    print("-" * 92)
    for name, s in summary.items():
        for model, ms in s["models"].items():
            f1s = list(ms["per_seed_mean_f1"].values())
            sd_f1 = ms["sd_f1_across_seeds"] or 0.0
            sd_auc = ms["sd_auc_across_seeds"] or 0.0
            print(f"{name:<12} {model:<18} {s['n_seeds']:>2} "
                  f"{ms['mean_f1_across_seeds']:>10.4f} +/- {sd_f1:.4f} "
                  f"{min(f1s):>7.4f} {max(f1s):>7.4f} "
                  f"{ms['mean_auc_across_seeds']:>10.4f} +/- {sd_auc:.4f}")
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
