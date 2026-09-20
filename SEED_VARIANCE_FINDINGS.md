# Seed Variance Findings (2026-09-20)

What the multi-seed sweep changed, and what the manuscript needs as a result.
Companion to `RESUBMISSION_GUIDE.md`. Per the standing instruction, this
describes *where* the manuscript needs edits — it does not edit
`paper/manuscript.tex`.

Artifacts: `gcp_results/seed_sweep/` (20 runs), `seed_sweep_summary.json`,
`pooled_threshold_summary.json`, `significance_tests_model_family_seedavg.json`.
Branch `seed-variance-analysis`, commits `0f945bc`, `c004d93`, `07db6a9`.

## 1. What triggered this

Re-running the paper's best configuration with the same seed, data and
checkpoint gave mean F1 **0.7106** against the reported **0.7325**. The cause is
ordinary GPU nondeterminism: cuDNN autotuning and non-associative float
reductions. `RANDOM_SEED = 42` fixes data order and dropout masks, not kernels;
`torch.use_deterministic_algorithms` was never set.

This matters because several reported comparisons are smaller than that noise.

## 2. Sweep results (4 configs x 5 seeds, seeds 42-46)

| Config | Reported (single run) | Sweep mean ± SD | Range |
|---|---|---|---|
| Base+Meta | 0.729 | **0.7425 ± 0.0134** | 0.728–0.760 |
| Base+Meta+SqrtW | 0.733 | 0.7266 ± 0.0116 | 0.709–0.737 |
| Baseline | 0.711 | 0.6965 ± 0.0129 | 0.682–0.712 |
| ModernBERT | 0.669 | 0.6880 ± 0.0190 | 0.667–0.713 |

Run-to-run SD is **0.012–0.019** on mean F1; individual folds move twice that.
AUC is far steadier (SD 0.0015–0.0054).

### 2.1 The reported "best" configuration is not the best

Base+Meta beats Base+Meta+SqrtW by **+0.0159 (p=0.014, d=0.75)** on
seed-averaged folds — the opposite of the reported ranking. The best config was
picked as the maximum over ~14 noisy runs scored on the same test folds, so the
winner was a lucky draw and regressed on re-run. Classic winner's curse.

`tab:meta` spans 0.702–0.733 across six configurations, a range smaller than
two standard deviations of a single configuration. **That table cannot rank
configurations**, and the "+0.0136 over Aug-250" claim does not survive.

## 3. Statistical re-tests (seed-averaged per-fold F1, Holm-corrected)

| Comparison | Reported | Seed-averaged |
|---|---|---|
| vs SVM | +0.066, p_holm=0.041 ✓ | +0.060, **p_holm=0.082 ✗** |
| vs Gradient Boosting | +0.129, p_holm=0.035 ✓ | +0.123, p_holm=0.023 ✓ |
| vs ModernBERT | +0.064, p_holm=0.082 ✗ | +0.039, **p_holm=0.022 ✓** |
| vs Claude zero-shot | +0.060, p_holm=0.098 ✗ | +0.054, p_holm=0.023 ✓ |
| vs Claude few-shot | +0.039, p=0.106 ✗ | +0.033, p=0.131 ✗ |

Two directions here, and both matter:

- **SVM is the casualty.** It is the comparison the abstract's "10–15%" and the
  Conclusion's "outperforming SVM and gradient boosting by 10–15%" rest on, and
  it no longer survives correction. The *gap* is still there and still sizeable;
  it is the significance claim that fails.
- **Seed-averaging is not uniformly deflationary.** ModernBERT and Claude
  zero-shot became *significant* because averaging five runs per fold tightens
  the paired differences. This is the honest case for the method: it is noise
  reduction, not a penalty.

**All AUC comparisons survive Holm**, including SVM (p_holm=0.029). AUC is
threshold-free and has ~8x lower seed variance than F1.

## 4. Threshold tuning: the fix that failed

Per-fold tuning maximises F1 on one ~177-ticket validation project. For SERVER,
the chosen threshold ranges **0.132–0.750** across seeds, with fold F1 SD 0.130
and range 0.322 — one fold swinging a third of an F1 point on seed alone.

Pooled tuning (threshold from the other folds' ~1.6k out-of-fold predictions)
was implemented and tested. **It made things worse:**

| Config | As run | Pooled | Fixed 0.5 |
|---|---|---|---|
| Base+Meta+SqrtW | 0.7266 ± 0.0116 | 0.7205 ± 0.0147 | **0.7286 ± 0.0088** |
| ModernBERT | **0.6880 ± 0.0190** | 0.6673 ± 0.0213 | 0.6691 ± 0.0206 |

The pooled threshold still moved ~0.35 across seeds, so the instability is not
mainly sampling noise in the tuning set — each run's probability calibration
shifts and the optimum moves with it. A larger tuning set cannot fix that.

For RoBERTa+SqrtW, **per-fold tuning is pure added noise**: fixed 0.5 gives both
the better mean and the lower SD. ModernBERT is the exception (+0.019 from
tuning, presumably worse calibration) — but picking the rule per model would
itself be a selection-on-test decision, the same trap that produced 0.7325.

Separately: the paper never discloses that the best configuration tunes a
threshold per fold at all. A reviewer reading the code will find it.

## 5. What is unaffected

The variance analysis in `gcp_results/error_analysis/` holds: it concerns which
projects are hard *relative to each other*. The re-runs keep SERVER worst.
Kendall's W=0.70 across 13 models, the OOV (rho=-0.77) and Bug-share (rho=0.73)
correlations, and the SERVER data-quality findings do not depend on any single
run's F1.

## 6. Recommended next steps

**In priority order.** Items 1–4 are manuscript edits needing no new compute.

1. **Reword the SVM claim** (abstract, Conclusion). The gap is descriptive, not
   significant after correction. Either report it as an effect size with a CI,
   or lead with AUC, where it does survive. Do not repeat "10–15%" as a
   significance claim.
2. **Rebuild `tab:meta` as mean ± SD over 5 seeds** and drop the "softened
   weighting is best" narrative. If a single headline configuration is still
   wanted, Base+Meta (0.743 ± 0.013) is the defensible one — but say plainly
   that the configurations are within noise of each other.
3. **Add a reproducibility/variance paragraph** to Methods and Limitations:
   seeds, hardware, nondeterminism, and the ±0.02 figure. This answers R1.4 and
   R2.6 directly and is the strongest available response to both.
4. **Decide the threshold policy and disclose it.** Recommended: drop per-fold
   tuning, report at 0.5, and state that tuning was tested and rejected as
   unstable. `--threshold fixed` already does this.
5. **Consider promoting AUC to the primary metric** for model comparison. It is
   threshold-free, ~8x more stable across seeds, and every comparison survives
   Holm. F1 stays as the operational metric.
6. **Optional: extend the sweep to the remaining `tab:meta` rows** (Aug250+Meta,
   Aug250+Meta+SqrtW, and the two Thresh variants) so the whole table is
   mean ± SD. ~4 configs x 5 seeds; the Aug runs are slower than baseline, so
   budget 12–15 h. `run_seed_sweep.sh` is resumable — add the configs and
   re-run. Only needed if the table stays in its current form.
7. **Traditional-ML baselines are cheap to re-seed** (CPU, minutes). Doing so
   would put SVM and GB on the same mean ± SD footing as the transformers and
   make the headline comparison symmetric.

### Still open from before, unchanged by this work

- **Blocking:** the `tab:stage1` batch_size=32 rows with no matching run
  (see `RESUBMISSION_GUIDE.md` item 0).
- Second-annotator / inter-rater study (R2.5); the 250-issue stratified sample
  is ready at `ieee_dataport/manual_labels/manually_labelled_sample_250.csv`.
- Manual/heuristic label counts disagree across prose, `tab:labels` and the CSVs.
