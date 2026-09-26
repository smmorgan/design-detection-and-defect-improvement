# Seed Variance Findings (2026-09-20, extended 2026-09-23, 2026-09-26)

What the multi-seed sweep changed, and what the manuscript needs as a result.
Companion to `RESUBMISSION_GUIDE.md`. Per the standing instruction, this
describes *where* the manuscript needs edits — it does not edit
`paper/manuscript.tex`.

Artifacts: `gcp_results/seed_sweep/` (75 runs total: the original 20, 20 more
added 2026-09-21 covering the remaining `tab:meta` rows, and 35 more added
2026-09-26 covering all `tab:lopo` augmentation/freezing rows),
`seed_sweep_summary.json`, `pooled_threshold_summary.json`,
`significance_tests_model_family_seedavg.json`.
Branch `seed-variance-analysis`, commits `0f945bc`, `c004d93`, `07db6a9`, `c0858a0`.

**2026-09-23 update:** the sweep now covers all six `tab:meta` configurations
(item 6 of the original recommendations, below, is done). See §2.2 for the
complete table and a new finding: the augmentation benefit `tab:lopo` reports
does not hold once metadata is added — see §2.3.

**2026-09-26 update:** the sweep now also covers all seven non-baseline
`tab:lopo` rows (the RoBERTa text-only augmentation/freezing ablation that
item 6's earlier note flagged as "out of scope, would require GPU re-runs").
See §2.4: the single-run "Aug-250 best, monotonically decreasing beyond
that" story does not survive seed-averaging — none of the six augmentation
levels (including baseline) are statistically distinguishable from each
other. The freeze-10 finding (freezing hurts, augmentation partially
recovers it) does survive.

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

### 2.2 All six `tab:meta` rows, seed-averaged (added 2026-09-23)

The sweep now covers the remaining four configs (`aug250_meta`,
`aug250_meta_sqrtw`, `base_meta_thresh`, `aug250_meta_thresh`; 20 more runs,
seeds 42–46, finished 2026-09-21 00:35, no failures). Full picture:

| Config (paper's `tab:meta` name) | Reported (single run) | Sweep mean ± SD | Range |
|---|---|---|---|
| Base+Meta | 0.729 | **0.7425 ± 0.0134** | 0.728–0.760 |
| Base+Meta+SqrtW | 0.733 | 0.7266 ± 0.0116 | 0.709–0.737 |
| Base+Meta+Thresh | 0.702 | 0.7230 ± 0.0141 | 0.707–0.736 |
| Aug250+Meta | 0.729 | 0.7178 ± 0.0092 | 0.703–0.728 |
| Aug250+Meta+SqrtW | 0.706 | 0.7089 ± 0.0171 | 0.682–0.726 |
| Aug250+Meta+Thresh | 0.701 | 0.7087 ± 0.0171 | 0.679–0.721 |

(For reference, the two configs outside `tab:meta`: Baseline 0.6965 ± 0.0129,
ModernBERT 0.6880 ± 0.0190.)

This confirms and sharpens the §2.1 finding: **Base+Meta is not just
better than Base+Meta+SqrtW, it beats every other `tab:meta` row**, by a
margin (+0.0159 over the next-best, SqrtW) that is comparable to or larger
than each config's own seed SD. Every row involving `Aug250` now sits
*below* every row without it — the opposite of what the single-run table
showed (Aug250+Meta tied Base+Meta at 0.729). The winner's-curse pattern
from §2.1 held across the extension: five of six sweep means came in below
their single-run reported value; only Base+Meta+Thresh moved up (0.702 →
0.723).

### 2.3 New finding: the `tab:lopo` augmentation benefit does not survive adding metadata

The Results section (`tab:lopo`, manuscript.tex:264) reports Aug-250 (no
metadata) beating the manual-only baseline (0.7189 vs 0.7111, single run),
and the Conclusion (manuscript.tex:493) generalizes this to "(2) moderate
data augmentation (250 samples) improves cross-project performance, but
excessive augmentation degrades it." That claim was never re-tested with
metadata enrichment added, because `tab:meta`'s Aug250+Meta row was only a
single run (0.729, effectively tied with Base+Meta's 0.729).

The sweep breaks that tie decisively: Aug250+Meta (0.7178 ± 0.0092) is
**below** Base+Meta (0.7425 ± 0.0134) by 0.0247, roughly 2 SD, once both are
averaged over 5 seeds — and this holds for every Aug250 variant against
every non-Aug250 variant in §2.2's table. The augmentation benefit that
`tab:lopo` documents is specific to the no-metadata setting; once metadata
fields are added, the 250 cherry-picked samples add noise rather than
signal. This is a second, independent case of the winner's-curse pattern in
§2.1, and it directly touches the Conclusion's finding (2), which currently
states the augmentation benefit without that qualification.

### 2.4 `tab:lopo` (RoBERTa, text-only) fully re-swept: the augmentation ranking does not survive either (added 2026-09-26)

`run_lopo_aug_sweep.sh` re-ran the seven non-baseline `tab:lopo` configs over
5 seeds each (42–46, 35 runs, finished 2026-09-26, no failures). Baseline
already had sweep data from §2 (0.6965 ± 0.0129).

| Config (paper's `tab:lopo` name) | Reported (single run) | Sweep mean ± SD | Range |
|---|---|---|---|
| Aug-250 | 0.7189 | **0.7072 ± 0.0136** | 0.689–0.724 |
| Aug-500 | 0.713 | 0.7005 ± 0.0132 | 0.679–0.714 |
| Baseline | 0.7111 | 0.6965 ± 0.0129 | 0.682–0.712 |
| Aug-750 | 0.700 | 0.7019 ± 0.0084 | 0.689–0.715 |
| Aug all | 0.695 | 0.6946 ± 0.0132 | 0.680–0.711 |
| Aug-1000 | 0.689 | 0.7036 ± 0.0078 | 0.693–0.716 |
| Pseudo+Aug-250 | 0.688 | *not re-swept* | — |
| Pseudo (SO-transfer) | 0.670 | *not re-swept* | — |
| Freeze 10+aug | 0.6535 | 0.6566 ± 0.0047 | 0.650–0.662 |
| Freeze 10 | 0.6188 | 0.6300 ± 0.0084 | 0.619–0.643 |

**The manuscript's own headline claim about this table is false under
seed-averaging.** manuscript.tex:265 says augmentation "showed monotonically
decreasing performance" beyond Aug-250. The seed-averaged ordering is
Aug-250 (0.7072) > Aug-1000 (0.7036) > Aug-750 (0.7019) > Aug-500 (0.7005) >
Baseline (0.6965) > Aug-all (0.6946) — Aug-1000 jumps from *worst* of the
five augmented configs (single run: 0.689, lowest) to *second-best*
(seed-avg: 0.7036), passing both Aug-500 and Aug-750 on the way. There is no
monotonic trend in either direction.

More importantly, **the six augmentation-level configs span only
0.6946–0.7072 (range 0.0126) once seed-averaged — smaller than any one
config's own seed SD (0.0078–0.0136)**, exactly the pattern found for
`tab:meta` in §2.1. Paired Wilcoxon tests on seed-averaged per-project F1
(10 projects) confirm this directly — none of the pairwise comparisons
among augmentation levels reach significance:

| Comparison | Mean diff | Wilcoxon $p$ | Cohen's $d$ |
|---|---|---|---|
| Aug-250 vs Baseline | +0.0107 | 0.232 | 0.46 |
| Aug-1000 vs Baseline | +0.0071 | 0.770 | 0.23 |
| Aug-250 vs Aug-1000 | +0.0035 | 0.770 | 0.10 |
| Aug-250 vs Aug-500 | +0.0067 | 0.432 | 0.33 |
| Aug-1000 vs Aug-500 | +0.0032 | 1.000 | 0.14 |

**`tab:lopo` cannot rank augmentation levels**, in the text-only setting just
as in the metadata-enriched setting (§2.3). This is a third, independent
instance of the winner's-curse pattern from §2.1: "Aug-250 is best" was
picked as a maximum over six noisy single runs, and five of the six sweep
means came in below their single-run reported value (only Aug-750 and
Aug-1000 moved up).

**The freeze-10 finding does survive**, and with a real statistical test
behind it now: Baseline beats Freeze-10 by +0.0665 F1 (Wilcoxon $p$=0.027,
$d$=1.0) — a gap roughly 5–8x the size of any augmentation-level comparison
above. Freeze-10+aug vs. Freeze-10 (+0.0267, $p$=0.084, $d$=0.36) trends the
same direction as the single-run story (augmentation partially recovers the
damage from freezing) but does not itself reach significance at this sample
size — soften "partially recovered" to reflect that if precision matters
here.

The two Pseudo-label rows were not part of this sweep (self-training reruns
were out of scope for this pass) and remain single-run figures; treat them
with the same caution as the un-swept Qwen+LoRA rows in `tab:llm_baselines`
— i.e., don't lean on their exact values in a close comparison.

#### What this means for the manuscript

Everywhere `tab:lopo`'s single-run numbers or "Aug-250 is best, more
augmentation hurts" narrative are used, the claim needs to be walked back to
"all six no-metadata configurations (baseline through Aug-1000/all) are
statistically indistinguishable at ~0.70 F1; only freezing layer 10
produces a reliable, sizeable drop." Locations that currently make the
stronger claim:

- **manuscript.tex:150** (Results intro): "LOPO performance improves as up
  to 250 cherry-picked design examples are added... but degrades
  monotonically beyond that (falling to 0.695 when all ~10K... are used)."
  The monotonic-degradation part is not supported once seed-averaged (Aug-
  1000 is *higher* than Aug-500 and Aug-750, not lower).
- **manuscript.tex:265** (Results, Table~\ref{tab:lopo} discussion): "The
  `Aug top-250` configuration... achieved the best mean F1 of 0.7189...
  Additional augmentation beyond 250 samples showed monotonically
  decreasing performance, contradicting prior studies..." — the sentence
  already carries a hedge citing Section~\ref{sec:metadata}'s ±0.01–0.02
  variance finding, but now that `tab:lopo` itself has been re-swept, the
  hedge should be replaced with the direct result: this table's ranking
  does not survive seed-averaging at all, in either setting.
- **manuscript.tex:267**: the freeze-10 numbers (0.6188 / 0.6535) should be
  updated to the seed-averaged figures (0.6300 ± 0.0084 / 0.6566 ± 0.0047),
  and "partially recovered" can stay but is only trend-level (p=0.084), not
  a confirmed effect the way the freeze-vs-baseline gap is (p=0.027).
- **Table `tab:lopo` (manuscript.tex:269–293)**: rebuild as mean ± SD over 5
  seeds for the eight re-swept rows (Baseline, Aug-250/500/750/1000/all,
  Freeze10, Freeze10+aug), matching `tab:meta`'s and `tab:lopo_trad`'s
  format; leave the two Pseudo rows as single-run values with a note that
  they were not re-swept (same caveat style as the Qwen+LoRA footnote in
  `tab:llm_baselines`).
- **manuscript.tex:301** (SVM/GB comparison): "RoBERTa's best text-only LOPO
  result (F1=0.719, single run, Table~\ref{tab:lopo})" should cite the
  seed-averaged Aug-250 figure (0.7072) or, more defensibly given §2.4,
  the whole indistinguishable cluster's approximate value (~0.70) rather
  than singling out one "best" row.
- **manuscript.tex:521** (Discussion): "additional automated data
  augmentation in this study showed diminishing and eventually negative
  returns beyond 250 samples in the text-only setting" needs the same
  walk-back — replace with the no-ranking-signal finding.
- **manuscript.tex:550** (Conclusion, finding (2)): "moderate data
  augmentation (250 samples) improves cross-project performance in the
  text-only setting, but this benefit does not survive once JIRA metadata
  fields are added" — the first half no longer holds either. Finding (2)
  should be rewritten to state that augmentation showed no reliable effect
  at any level in either setting (text-only or metadata-enriched), and that
  layer freezing (not augmentation) is the one training-configuration
  choice in this ablation with a robust, statistically supported effect.

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
2. **Rebuild `tab:meta` as mean ± SD over 5 seeds**, using the complete
   six-row table in §2.2 (this can now be done fully — all six rows have
   sweep data as of 2026-09-23), and drop the "softened weighting is best"
   narrative. Base+Meta (0.7425 ± 0.0134) is the defensible headline
   configuration: it now leads all six rows, not just SqrtW, by a margin
   larger than any pair of within-`tab:meta` rows is separated by — but
   still say plainly that the bottom four rows (0.709–0.723) are close
   enough to call roughly tied.
3. **Add a reproducibility/variance paragraph** to Methods and Limitations:
   seeds, hardware, nondeterminism, and the ±0.02 figure. This answers R1.4 and
   R2.6 directly and is the strongest available response to both.
4. **Decide the threshold policy and disclose it.** Recommended: drop per-fold
   tuning, report at 0.5, and state that tuning was tested and rejected as
   unstable. `--threshold fixed` already does this. Note this now also
   resolves the `Thresh` rows in `tab:meta`: both `Base+Meta+Thresh` and
   `Aug250+Meta+Thresh` sit below their non-Thresh counterparts in §2.2, so
   dropping per-fold tuning is consistent with the full table, not just the
   earlier partial one.
5. **Consider promoting AUC to the primary metric** for model comparison. It is
   threshold-free, ~8x more stable across seeds, and every comparison survives
   Holm. F1 stays as the operational metric.
6. ~~Optional: extend the sweep to the remaining `tab:meta` rows~~ **Done
   2026-09-21.** All four remaining configs (Aug250+Meta, Aug250+Meta+SqrtW,
   Base+Meta+Thresh, Aug250+Meta+Thresh) finished, 20/20 runs, no failures.
   See §2.2 for the complete table.
7. ~~Traditional-ML baselines are cheap to re-seed~~ **Done 2026-09-23.**
   `traditional_ml/train_traditional_models.py` was re-run for SVM and GB
   across all three augmentation levels (baseline/Aug-250/Aug-500) x 5 seeds
   (42-46), 15 runs, CPU-only, ~65 min total. Aggregated with the new
   `aggregate_traditional_sweep.py` into `traditional_ml/seed_sweep/`. This
   resolved the two-unreconciled-numbers problem in `RESUBMISSION_GUIDE.md`
   residual issue #1 (single-run GB was F1=0.604 unaugmented / F1=0.636
   Aug-500; seed-averaged it is F1=0.630 ± 0.015 unaugmented / F1=0.641 ±
   0.015 at Aug-500 — both numbers moved but the qualitative story, "GB
   trails SVM and benefits only marginally from augmentation," is unchanged).
   SVM's baseline (no-augmentation) LOPO F1 turned out to be **exactly
   seed-invariant** (0.666 ± 0.000): with no augmentation there is no
   per-seed randomness left in the TF-IDF + linear-SVM pipeline. GB does vary
   by seed even without augmentation (SD ≈ 0.015), because
   `GradientBoostingClassifier`'s `subsample < 1.0` in the best grid-search
   config introduces its own seeded stochasticity independent of data
   augmentation. `tab:lopo_trad` (manuscript.tex) was rebuilt as mean ± SD
   over 5 seeds for all 6 rows, matching `tab:meta`'s protocol, closing
   `RESUBMISSION_GUIDE.md` residual issue #2 for the traditional-ML table
   specifically (`tab:lopo`, the RoBERTa text-only augmentation ablation,
   remains un-swept — that would require GPU re-runs, out of scope for this
   CPU-only pass). `significance_tests.py --seed_avg` was repointed at the
   new `traditional_ml/seed_sweep/baseline_{svm,gradient_boosting}_seedavg_results.json`
   files (still the no-augmentation config, matching the encoder/decoder/LLM
   baselines' protocol) and re-run; GB's significance result is unchanged
   qualitatively (still significant after Holm on both F1 and AUC) but the
   effect size grew (d=1.50 → 2.15) since seed-averaging shrank GB's paired-
   difference variance more than it shrank the mean difference.
8. **Qualify the Conclusion's augmentation finding** (manuscript.tex:493,
   "(2) moderate data augmentation (250 samples) improves cross-project
   performance, but excessive augmentation degrades it"). Per §2.3 and
   §2.4, this doesn't hold with metadata added (Aug250 uniformly *hurts*
   relative to no augmentation) **and it no longer holds in the text-only
   setting either** now that `tab:lopo` itself has been seed-averaged — see
   item 9. Drop finding (2) as a general claim; augmentation shows no
   reliable effect at any level in either setting.
9. ~~`tab:lopo`, the RoBERTa text-only augmentation ablation, remains
   un-swept — that would require GPU re-runs, out of scope for [the
   CPU-only] pass~~ **Done 2026-09-26.** All seven non-baseline rows
   (Aug-250/500/750/1000/all, Freeze10, Freeze10+aug) re-run over 5 seeds
   (42–46), 35 runs, no failures. See §2.4 for the complete table, the
   pairwise significance tests, and the specific manuscript locations that
   need rewording as a result (the "Aug-250 best, monotonic decline"
   narrative does not survive — see §2.4's "What this means for the
   manuscript" list for the full set of edits, covering manuscript.tex
   lines 150, 265, 267, the `tab:lopo` table itself, 301, 521, and 550).
   The two Pseudo-label rows were not part of this pass and remain
   single-run.

### Still open from before, unchanged by this work

- **Blocking:** the `tab:stage1` batch_size=32 rows with no matching run
  (see `RESUBMISSION_GUIDE.md` item 0).
- Second-annotator / inter-rater study (R2.5); the 250-issue stratified sample
  is ready at `ieee_dataport/manual_labels/manually_labelled_sample_250.csv`.
- Manual/heuristic label counts disagree across prose, `tab:labels` and the CSVs.
