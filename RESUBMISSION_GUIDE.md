# Resubmission Guide — Status

All 11 reviewer points (`REVIEWER_FEEDBACK.md`) are now written directly into
`paper/manuscript.tex`, along with the seed-variance/reproducibility
correction described below. This guide previously described *where* to make
edits without touching the manuscript directly; as of 2026-09-23 the user
asked for the edits to be made directly, so this file now tracks status and
residual issues instead of pending guidance.

## What changed in this pass (2026-09-23)

- **Table 1 (`tab:stage1`) blocking issue — actually fixed.** The three
  unsupported batch\_size=32 rows (roberta/bert/distilbert, none of which had
  a completed training run) are dropped; the table now reports the 7 verified
  batch\_size=16 rows only, with roberta (lr=1e-5) bolded as both best F1 and
  best AUC, matching the prose. (A previous pass had marked this "DONE,
  option 2" in this file without the table actually being edited — that
  status line was wrong; verify against the manuscript, not this file, going
  forward.)
- **Label counts fixed.** `tab:labels` and its prose previously disagreed
  with the underlying CSVs. Corrected to 496 manual-design / 1,290
  manual-non-design / 4,989 cherry-picked-design / 4,995
  cherry-picked-non-design, matching `ieee_dataport/manual_labels/` and
  `ieee_dataport/heuristic_labels/`.
- **R1.3 (qualitative error analysis) and R2.6 (LOPO variance) written in**,
  combined into §5.2 "Cross-Project Generalization Challenges" as recommended.
- **R2.5 (inter-rater reliability) written in**, §5.4 "Threats to Validity"
  (Construct validity paragraph), cross-referenced with R1.3.
- **Seed-variance / winner's-curse correction applied throughout.** The
  previously-reported best configuration, `Base+Meta+SqrtW` (F1=0.7325,
  single run), was a winner's-curse pick — re-running it gave F1=0.7106.
  `Base+Meta` (no additional weighting) is the actual best configuration
  under 5-seed-averaged evaluation (F1=0.743 $\pm$0.013). This propagated to:
  the abstract, `tab:meta` (rebuilt as mean$\pm$SD over 5 seeds, all 6 rows),
  `tab:llm_baselines`, the significance-testing section (`tab:significance`
  and `significance_tests_model_family_seedavg.json` were regenerated against
  the corrected baseline — `significance_tests.py`'s `--seed_avg` path was
  edited to point at `base_meta_seedavg_results.json` instead of the SqrtW
  variant), §5.1, §5.2, §5.4 (new Internal-validity paragraph), and the
  Conclusion. Under the corrected baseline, all 11 model-family comparisons
  are significant after Holm correction on both F1 and AUC (previously only
  5/11 were significant on F1).

## Traditional-ML seed sweep (2026-09-23, later same day)

Residual issues #1 and #2 below (as originally written) are now resolved for
SVM/GB. `traditional_ml/train_traditional_models.py` was re-run for both
models across all three augmentation levels (baseline/Aug-250/Aug-500) x 5
seeds (42-46) — 15 CPU-only runs, ~65 min total — and aggregated with the new
`aggregate_traditional_sweep.py` into `traditional_ml/seed_sweep/`. Changes
made directly to `paper/manuscript.tex`:

- `tab:lopo_trad` rebuilt as mean $\pm$ SD over 5 seeds for all 6 rows (same
  protocol as `tab:meta`), with an added paragraph explaining the
  seed-invariance finding for SVM (see below) and stating the two
  previously-unreconciled single-run GB numbers explicitly, rather than
  silently picking one.
- The grid-search hyperparameter sentence (previously citing untraceable
  F1=0.753/0.721) corrected to the actual cross-validated F1 ranges observed
  across the 5 baseline seeds (SVM 0.718-0.750, GB 0.672-0.692).
- Abstract, §5.1 (Discussion), and Conclusion's GB comparison numbers updated
  from the single unaugmented run (F1=0.604) to the seed-averaged baseline
  (F1=0.630 $\pm$ 0.015); relative-improvement figures recomputed (23.0% ->
  17.9% relative F1, 8.3% -> 6.8% relative accuracy for RoBERTa vs. GB).
- `significance_tests.py --seed_avg` repointed at the new
  `traditional_ml/seed_sweep/baseline_{svm,gradient_boosting}_seedavg_results.json`
  files and re-run; `tab:significance`'s GB row updated (diff 0.139->0.113,
  d=1.50->2.15, still significant after Holm on both F1 and AUC — the
  qualitative conclusion is unchanged, only the numbers).

**New finding surfaced by the sweep:** SVM's no-augmentation LOPO F1 is
*exactly* seed-invariant (0.666 $\pm$ 0.000) — with no augmentation there is
no per-seed randomness left in the deterministic TF-IDF + linear-SVM
pipeline. GB still varies by seed even unaugmented (SD $\approx$0.015)
because its best grid-search config uses `subsample<1.0`, which is its own
source of seeded stochasticity independent of data augmentation. Full
writeup in `SEED_VARIANCE_FINDINGS.md` recommendation 7.

**Still not swept:** `tab:lopo` (§4.3, RoBERTa's own text-only augmentation
ablation) remains single-run — closing that gap needs GPU re-runs, out of
scope for this CPU-only traditional-ML pass.

## Known residual issues (not fixed in this pass)

1. ~~GB baseline inconsistency between §4.5 and §5.1/§4.7/Conclusion.~~
   **Resolved 2026-09-23** — see "Traditional-ML seed sweep" above.
2. ~~`tab:lopo` and `tab:lopo_trad` (§4.3/§4.5) were not re-swept.~~
   **Partially resolved 2026-09-23** — `tab:lopo_trad` (SVM/GB) is now
   seed-averaged; `tab:lopo` (RoBERTa text-only ablation) still is not. See
   "Traditional-ML seed sweep" above.
3. **Two near-duplicate "Future work" paragraphs in the Conclusion**
   (predates this pass) — one on defect-rate causal analysis in general terms,
   one restating it with more methodological detail (instrumental variables,
   confounders). Consider merging.
4. **Per-source min/max range for cherry-picked label counts was removed
   rather than corrected** (§3.4, "yielding 415--4,890 design labels per
   source") since it couldn't be verified against the wrong totals it was
   originally paired with. If that per-source breakdown is wanted back, it
   needs to be recomputed from the heuristic-labeling script's per-source
   output.
5. **INTER_RATER_FINDINGS.md §7's optional adjudication pass** on the 68
   annotator disagreements was not performed.

## Source documents (kept for provenance)

- `ERROR_ANALYSIS_FINDINGS.md` — R1.3 source analysis.
- `INTER_RATER_FINDINGS.md` — R2.5 source analysis.
- `SEED_VARIANCE_FINDINGS.md` — seed-variance/winner's-curse source analysis.
- `REVIEWER_FEEDBACK.md` — point-by-point status, now all "Done."
