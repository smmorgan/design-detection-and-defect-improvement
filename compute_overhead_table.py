#!/usr/bin/env python3
"""
Computational overhead comparison across every model family evaluated in the
paper: LLM zero-/few-shot baselines, RoBERTa/DistilBERT/BERT Stage-1
pretraining, and the traditional-ML (SVM/GB) LOPO pipeline.

Addresses the reviewer request for a computational-cost comparison alongside
the accuracy comparison in significance_tests.py's model-family tests.

Sources of timing, all derived from existing artifacts rather than re-run:
  - LLM baselines: `total_wall_clock_s` / `estimated_cost_usd` already recorded
    per-run by eval_llm_baseline.py.
  - RoBERTa/DistilBERT: first-to-last timestamp span of their Stage-1
    pretraining `training.log` (the one-time SO-pretraining run whose
    checkpoint LOPO fine-tuning starts from -- NOT per-LOPO-fold timing,
    which was never logged separately; see caveat in the printed output).
  - BERT-base: never completed a successful run (CUDA OOM on the same GPU
    that trained RoBERTa/DistilBERT successfully) across 3 attempts --
    reported as a failure, not a number.
  - Traditional ML (SVM/GB): phase timestamps parsed out of the single
    training.log run that produced traditional_models_results.json (grid
    search, holdout fit, LOPO), before that log's second run for the
    augmented-data ablation begins.

Usage:
    python compute_overhead_table.py
"""

import json
import re
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("gcp_results")
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def parse_ts(line):
    m = TS_RE.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")


def llm_overhead():
    runs = []
    for provider, label in [("anthropic", "claude-sonnet-5"), ("openai", "gpt-4o-2024-08-06"),
                             ("local", "llama3.1:8b (local)")]:
        for mode in ["zeroshot", "fewshot"]:
            path = RESULTS_DIR / f"llm_{mode}_{provider}_results.json"
            if not path.exists():
                continue
            d = json.load(open(path))
            n = sum(p["n"] for p in d["per_project"])
            wall_s = d["total_wall_clock_s"]
            cost = d["estimated_cost_usd"]
            runs.append({
                "config": f"{label} {mode}",
                "n_tickets": n,
                "wall_clock_s": wall_s,
                "wall_clock_per_ticket_s": wall_s / n,
                "cost_usd": cost,
                "cost_per_1k_tickets_usd": (cost / n) * 1000 if cost == cost else None,  # NaN check
                "input_tokens": d["total_input_tokens"],
                "output_tokens": d["total_output_tokens"],
            })
    return runs


def training_log_span(log_path):
    first = last = None
    with open(log_path) as f:
        for line in f:
            ts = parse_ts(line)
            if ts is None:
                continue
            if first is None:
                first = ts
            last = ts
    if first is None:
        return None
    return (last - first).total_seconds()


def transformer_pretraining_overhead():
    configs = [
        ("roberta-base", RESULTS_DIR / "roberta_standard_0309_0738", False),
        ("distilbert-base", RESULTS_DIR / "distilbert_standard_0309_0738", False),
        ("modernbert-base", RESULTS_DIR / "modernbert_standard_0831_1935", False),
        ("qwen2.5-1.5b+lora", RESULTS_DIR / "qwen25_1.5b_lora_standard_0903", True),
    ]
    runs = []
    for name, d, is_lora in configs:
        log_path = d / "training.log"
        if not log_path.exists():
            continue
        span_s = training_log_span(log_path)
        text = log_path.read_text()
        n_params = None
        m = re.search(r"Total parameters: ([\d,]+)", text)
        if m:
            n_params = int(m.group(1).replace(",", ""))
        n_trainable = None
        if is_lora:
            m = re.search(r"trainable params: ([\d,]+) \|\| all params: ([\d,]+)", text)
            if m:
                n_trainable = int(m.group(1).replace(",", ""))
        runs.append({
            "config": name,
            "phase": "Stage-1 SO pretraining (single run, not per-LOPO-fold)",
            "wall_clock_s": span_s,
            "n_parameters": n_params,
            "n_parameters_trainable": n_trainable,
            "is_lora": is_lora,
        })
    return runs


def decoder_lora_lopo_overhead():
    """Full 10-fold LOPO fine-tuning wall-clock for Qwen2.5-1.5B+LoRA, both
    from the public checkpoint directly (cold-start) and from the Stage-1
    SO-pretrained checkpoint. Unlike RoBERTa/DistilBERT/ModernBERT, whose
    LOPO fine-tuning time was never logged as a separate span, both Qwen
    LOPO runs have a dedicated training.log, so this is directly reportable.
    """
    configs = [
        ("qwen2.5-1.5b+lora (cold-start)", RESULTS_DIR / "qwen25_1.5b_lora_lopo",
         "Full 10-fold LOPO fine-tuning from public checkpoint (no Stage-1 pretraining)"),
        ("qwen2.5-1.5b+lora (pretrained)", RESULTS_DIR / "qwen25_1.5b_lora_pretrained_lopo",
         "Full 10-fold LOPO fine-tuning from Stage-1 SO-pretrained checkpoint"),
    ]
    runs = []
    for name, d, phase in configs:
        log_path = d / "training.log"
        if not log_path.exists():
            continue
        span_s = training_log_span(log_path)
        n_trainable = n_total = None
        m = re.search(r"trainable params: ([\d,]+) \|\| all params: ([\d,]+)", log_path.read_text())
        if m:
            n_trainable = int(m.group(1).replace(",", ""))
            n_total = int(m.group(2).replace(",", ""))
        runs.append({
            "config": name,
            "phase": phase,
            "wall_clock_s": span_s,
            "n_parameters_total": n_total,
            "n_parameters_trainable": n_trainable,
        })
    return runs


def bert_failure_summary():
    dirs = sorted(RESULTS_DIR.glob("bert_baseline_*"))
    attempts = []
    gpu_capacity = None
    for d in dirs:
        log_path = d / "training.log"
        if not log_path.exists():
            continue
        text = log_path.read_text()
        oom = "torch.OutOfMemoryError: CUDA out of memory" in text
        if oom and gpu_capacity is None:
            m = re.search(r"GPU 0 has a total capacity of ([\d.]+ GiB)", text)
            if m:
                gpu_capacity = m.group(1)
        attempts.append({"dir": d.name, "result": "CUDA OOM" if oom else "unknown"})
    return {
        "config": "bert-base",
        "attempts": attempts,
        "n_successful_runs": 0,
        "gpu_capacity": gpu_capacity,
        "note": "The batch_size=32 sweep never completed a successful run across "
                f"{len(attempts)} attempts -- same GPU that successfully trained "
                "RoBERTa-base and DistilBERT-base at batch_size=32. A separate, "
                "successful bert-base-uncased Stage-1 run does exist at "
                "batch_size=16 (gcp_results/bert_very_conservative_0308_2027, "
                "F1=0.9045/AUC=0.9634); BERT was excluded from Stage 2/LOPO "
                "because RoBERTa scored higher on Stage 1, not because no BERT "
                "checkpoint was available.",
    }


def traditional_ml_overhead():
    log_path = Path("traditional_ml/traditional_training.log")
    if not log_path.exists():
        return None
    lines = log_path.read_text().splitlines()

    # Only the first pipeline run in the log corresponds to
    # traditional_models_results.json; a second run later in the same log
    # file (starting from its second "Loaded 1786 samples" line) is a
    # separate augmented-data ablation and is out of scope here.
    second_run_idx = None
    seen_first = False
    for i, line in enumerate(lines):
        if "Loaded 1786 samples" in line:
            if seen_first:
                second_run_idx = i
                break
            seen_first = True
    block = lines[:second_run_idx] if second_run_idx else lines

    markers = {}
    patterns = {
        "start": r"Loaded 1786 samples",
        "svm_grid_search_done": r"Searching gradient_boosting",  # timestamp of svm's own line precedes; captured via next
        "gb_grid_search_done": r"Training Final Models on Holdout Split",
        "svm_fit_done": r"Training gradient_boosting \.\.\.",
        "gb_fit_done": r"Stratified 5-Fold Cross-Validation",
        "cv_done": r"Leave-One-Project-Out|LOPO",
        "end": r"Results saved to traditional_ml/results/traditional_models_results.json",
    }
    # Several logger.info(f"\n...") calls emit the timestamp on one line and
    # the actual message on the next, untimestamped line -- so carry the last
    # seen timestamp forward rather than requiring both on the same line.
    ts_by_marker = {}
    last_ts = None
    for line in block:
        ts = parse_ts(line)
        if ts is not None:
            last_ts = ts
        if last_ts is None:
            continue
        for key, pat in patterns.items():
            if key not in ts_by_marker and re.search(pat, line):
                ts_by_marker[key] = last_ts

    if "start" not in ts_by_marker or "end" not in ts_by_marker:
        return None

    total_s = (ts_by_marker["end"] - ts_by_marker["start"]).total_seconds()
    phases = {}
    if "svm_grid_search_done" in ts_by_marker:
        phases["svm_grid_search_s"] = (ts_by_marker["svm_grid_search_done"] - ts_by_marker["start"]).total_seconds()
    if "gb_grid_search_done" in ts_by_marker and "svm_grid_search_done" in ts_by_marker:
        phases["gb_grid_search_s"] = (ts_by_marker["gb_grid_search_done"] - ts_by_marker["svm_grid_search_done"]).total_seconds()
    if "cv_done" in ts_by_marker and "gb_fit_done" in ts_by_marker:
        phases["five_fold_cv_s"] = (ts_by_marker["cv_done"] - ts_by_marker["gb_fit_done"]).total_seconds()
    if "end" in ts_by_marker and "cv_done" in ts_by_marker:
        phases["lopo_10fold_s"] = (ts_by_marker["end"] - ts_by_marker["cv_done"]).total_seconds()

    return {
        "config": "SVM + GradientBoosting (combined, TF-IDF features)",
        "total_wall_clock_s": total_s,
        "phases": phases,
        "note": "Grid search, holdout fit, 5-fold CV, and 10-fold LOPO for both "
                "models combined -- CPU-only, no GPU required.",
    }


def main():
    overhead = {
        "llm_baselines": llm_overhead(),
        "transformer_pretraining": transformer_pretraining_overhead(),
        "decoder_lora_lopo": decoder_lora_lopo_overhead(),
        "bert_base": bert_failure_summary(),
        "traditional_ml": traditional_ml_overhead(),
        "caveats": [
            "RoBERTa/DistilBERT timings are for the single Stage-1 SO-pretraining "
            "run each model was fine-tuned from, not per-LOPO-fold fine-tuning time "
            "-- that was never logged as a separate span, so per-fold RoBERTa "
            "training cost cannot be reconstructed from existing artifacts.",
            "BERT-base has zero successful runs (3/3 attempts hit CUDA OOM on an "
            "11.69 GiB GPU), so it has no LOPO results and cannot appear in the "
            "model-family significance tests or an apples-to-apples overhead row.",
            "Qwen2.5-1.5B+LoRA now has both a Stage-1 SO-pretraining run (in "
            "transformer_pretraining, ~16.6h) and two full 10-fold LOPO fine-tuning "
            "spans (in decoder_lora_lopo: cold-start from the public checkpoint, and "
            "from the Stage-1-pretrained checkpoint) -- unlike RoBERTa/DistilBERT/"
            "ModernBERT, whose LOPO fine-tuning time was never logged separately, so "
            "the Qwen LOPO wall-clock figures are not apples-to-apples with the "
            "'single Stage-1 run only' rows for the encoder models.",
            "Only 4,361,216 of Qwen2.5-1.5B's 1,548,078,592 parameters (0.28%) are "
            "trainable under LoRA, in both the Stage-1 pretraining run and the LOPO "
            "fine-tuning runs -- vs. 100% trainable for RoBERTa/DistilBERT/ModernBERT's "
            "full fine-tuning.",
            "LLM baseline wall-clock times include exponential-backoff retry delays "
            "and are affected by shared-account rate limits (see llm_clients.py); "
            "they reflect this run's conditions, not a hard per-ticket API floor.",
        ],
    }

    out_path = RESULTS_DIR / "computational_overhead.json"
    with open(out_path, "w") as f:
        json.dump(overhead, f, indent=2, default=str)

    print(f"{'='*90}")
    print("LLM BASELINE OVERHEAD")
    print(f"{'='*90}")
    print(f"{'Config':<28} {'n':>6} {'wall (s)':>10} {'s/ticket':>9} {'cost ($)':>9} {'$/1k tickets':>13}")
    for r in overhead["llm_baselines"]:
        cost_1k = f"{r['cost_per_1k_tickets_usd']:.2f}" if r["cost_per_1k_tickets_usd"] is not None else "n/a"
        print(f"{r['config']:<28} {r['n_tickets']:>6} {r['wall_clock_s']:>10.0f} "
              f"{r['wall_clock_per_ticket_s']:>9.2f} {r['cost_usd']:>9.2f} {cost_1k:>13}")

    print(f"\n{'='*90}")
    print("TRANSFORMER STAGE-1 PRETRAINING (single run each, GPU)")
    print(f"{'='*90}")
    for r in overhead["transformer_pretraining"]:
        hrs = r["wall_clock_s"] / 3600
        if r["is_lora"]:
            params_str = f"{r['n_parameters_trainable']:,}/{r['n_parameters']:,} trainable"
        else:
            params_str = f"{r['n_parameters']:>12,} params"
        print(f"{r['config']:<20} {params_str:<28} {r['wall_clock_s']:>8.0f}s ({hrs:.2f}h)  [{r['phase']}]")

    if overhead["decoder_lora_lopo"]:
        print(f"\n{'='*90}")
        print("DECODER + LoRA (full 10-fold LOPO fine-tune, GPU -- not pretraining)")
        print(f"{'='*90}")
        for r in overhead["decoder_lora_lopo"]:
            hrs = r["wall_clock_s"] / 3600
            print(f"{r['config']:<32} {r['n_parameters_trainable']:>12,}/{r['n_parameters_total']:,} "
                  f"trainable  {r['wall_clock_s']:>8.0f}s ({hrs:.2f}h)  [{r['phase']}]")

    print(f"\n{'='*90}")
    print("BERT-BASE")
    print(f"{'='*90}")
    b = overhead["bert_base"]
    print(f"Attempts: {len(b['attempts'])}, successful: {b['n_successful_runs']}, "
          f"GPU capacity: {b['gpu_capacity']}")
    print(b["note"])

    print(f"\n{'='*90}")
    print("TRADITIONAL ML (SVM + GradientBoosting, CPU)")
    print(f"{'='*90}")
    t = overhead["traditional_ml"]
    if t:
        print(f"Total: {t['total_wall_clock_s']:.0f}s ({t['total_wall_clock_s']/60:.1f} min)")
        for phase, s in t["phases"].items():
            print(f"  {phase:<22} {s:>8.1f}s")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
