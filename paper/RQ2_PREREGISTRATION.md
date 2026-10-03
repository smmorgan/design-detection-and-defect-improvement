# RQ2 Pre-Registration: Design Work, Its Timing, and Defect Rate

**Status:** drafted 2026-09-27, before any outcome-data modeling. Written per
`paper/RQ2_COLLAPSED_PLAN.md` §5 step 4 ("write the pre-registration ... before
touching the outcome data"). This document fixes the specification grid,
inclusion rules, and analysis plan so that later results can't be
selectively reported — the project has already been burned once by this
failure mode (`SEED_VARIANCE_FINDINGS.md`: the published 0.7325 F1 was a
winner's-curse pick across ~14 unregistered configurations).

No bug-count / defect-rate data has been aggregated or modeled as of this
writing. `data/tawos_full/*.csv` (full per-project issue pulls) and the
design-classifier scores (`gcp_results/rq2_labels/`, in progress) exist, but
the panel itself has not been built.

## 1. Research question and hypotheses (from RQ2_COLLAPSED_PLAN.md §2)

> Does design work detected in JIRA tickets predict subsequent change in a
> project's defect rate, and does the temporal distribution of that work
> improve prediction beyond its volume alone?

- **H1** — design-work volume in window *t* predicts defect rate at *t+k*,
  net of activity level and past defects.
- **H2** — temporal-distribution features add predictive power over volume
  alone.
- **H3** (rival) — defects at *t* predict design work at *t+k* (crisis-driven
  pattern); must be ruled out because it produces the same lag-1 correlation
  as H1, in reverse.

## 2. Scope decision (plan step 3)

**Restrict to the 10 hand-labelled projects**: CONFSERVER, DM, DNN, FAB,
JRASERVER, MESOS, MULE, NEXUS, SERVER, TIMOB. Not the full 39-project TAWOS
set.

Reason: per-project sensitivity/specificity (needed for the Rogan-Gladen
correction in §4) can only be estimated where hand labels exist
(`gcp_results/rq2_sensitivity_specificity.json`, computed from the frozen
Base+Meta 5-seed LOPO sweep). Applying the classifier to any of the other 29
projects would be out-of-distribution with unknown, uncorrectable error. The
plan itself flags this as leaving n=10 clusters, thin for panel inference —
accepted as a stated limitation rather than solved by expanding scope, since
solving it properly means hand-labelling ~100 tickets in each new project,
which is new primary data collection, not something to decide unilaterally
mid-analysis.

Sensitivity ranges from 0.531 (SERVER) to 0.942 (TIMOB); specificity from
0.713 (FAB) to 0.950 (DNN). No project is dropped for poor classifier
performance — all 10 get a Rogan-Gladen correction using their own
sensitivity/specificity, and SERVER's low sensitivity is reported as a
per-project caveat rather than an exclusion criterion (dropping the
worst-classified project would itself be a researcher degree of freedom).

## 3. Inclusion rule (plan blocker 6)

A project-quarter is included if the project has **≥12 quarters of history
and ≥50 total bug reports** over its full pulled history
(`data/tawos_full/<PROJECT>.csv`, uncapped — see
`COMPONENT_SCHEMA_FINDINGS.md` for why the IEEE DataPort 3,000-issue export
cannot be used for this). All 10 projects clear this comfortably:

| Project | Quarters spanned | Total bug reports |
|---|---|---|
| FAB | 16.9 | 3,562 |
| DNN | 25.8 | 7,319 |
| DM | 26.6 | 2,551 |
| MESOS | 38.7 | 4,891 |
| TIMOB | 38.3 | 15,742 |
| SERVER | 46.2 | 22,342 |
| NEXUS | 50.2 | 5,975 |
| MULE | 64.9 | 5,421 |
| CONFSERVER | 68.4 | 25,477 |
| JRASERVER | 74.9 | 20,630 |

No project is excluded by this rule. **t=0** for each project is its first
issue's creation date (JIRA adoption date), not the codebase's first commit
— per plan blocker 6, front-loading is measured relative to observed JIRA
history, and this limitation is stated explicitly in the discussion rather
than left implicit.

## 4. Variables

- **Predictor — design ratio.** `(design tickets resolved in window) /
  (all tickets resolved in window)`, dated by `Resolution_Date`. Ticket-level
  design/non-design comes from the frozen Base+Meta classifier
  (`generate_versioned_labels.py`; 5-seed ensemble, mean probability,
  threshold 0.5 — see §6). **Rogan-Gladen corrected** per project using that
  project's sensitivity/specificity from
  `gcp_results/rq2_sensitivity_specificity.json`:
  `corrected_prevalence = (apparent_prevalence + specificity - 1) /
  (sensitivity + specificity - 1)`, clipped to `[0, 1]`. Both corrected and
  uncorrected design ratios are carried through the full analysis and
  reported side by side (plan blocker 2) — the correction is not applied
  silently.
- **Outcome — bug count.** Count of `issue_type == 'Bug'` tickets *created*
  in window *t+k*, by project (and by project×component where the component
  join succeeds — see §5).
- **Exposure offset.** `log(non-bug issues resolved in window t)`, as a GLM
  offset (coefficient fixed at 1, not estimated) — without this, a busy
  quarter mechanically inflates both sides of the regression (plan §3).
- **Controls.** Lagged bug count (`t-1`), project fixed effects, a linear
  calendar-time trend (quarter index), and project age in quarters at
  window *t*.
- **Temporal features (H2).** Gini coefficient of cumulative design-ticket
  resolution across the project's history to date; front-loadedness index
  (share of design tickets resolved in the first third of observed
  project-quarters); quarters-since-nearest-design-change-point (a
  change-point detector on the design-ratio series, method fixed before
  fitting — see §5.3).

## 5. Specification grid (plan blocker 5 — pre-registered to prevent
   winner's-curse selection)

All cells below are run and reported; no cell is chosen after seeing
results. Final significance uses Holm correction across every hypothesis
test in the grid (not just the "preferred" cell).

1. **Unit of analysis**: project×quarter (primary); project×component×quarter
   as a secondary/robustness unit, restricted to component-linked issues
   only, and only reported if ≥6 of the 10 projects retain enough
   component-quarter cells to fit (component coverage ranges 46-95%,
   `COMPONENT_SCHEMA_FINDINGS.md`).
2. **Lag k**: 1, 2, 3, 4 quarters, estimated jointly as a distributed lag in
   one specification (not selected by best fit, per plan §3).
3. **Design-ratio correction**: uncorrected vs. Rogan-Gladen corrected
   (both reported; corrected is the primary interpretation).
4. **Missing-component handling** (project×component unit only): (a) drop
   issues with no component, (b) bucket them as an explicit "Unclassified"
   component. Both run; (a) is primary given the uneven coverage
   (`COMPONENT_SCHEMA_FINDINGS.md`).
5. **Missing/sparse-text handling**: issues with an empty `description`
   (title-only) are classified as-is (the classifier's normal operating
   condition — R1.3 error analysis already characterizes `SPARSE_TEXT` as a
   known failure mode, not grounds for exclusion here).
6. **Change-point method** (temporal features only): PELT with an
   RBF cost function (`ruptures` package), fixed penalty selected by
   BIC — chosen before fitting, not tuned against H2's p-value.

Total pre-registered hypothesis tests: 2 corrections × (1 or 2 unit levels) ×
2 component-handling rules (unit-2 only), each cell contributing one joint H1
test and one joint H2 test — enumerated fully in the analysis script's config
list before any model is fit on outcome data, and Holm-corrected as one
family.

**Correction (2026-09-27, before any cell was run):** this sentence
previously read "4 lags × 2 corrections × ...", which double-counts against
§5.2 above — lags 1-4 are estimated jointly as a single 4-df block in one
specification, not selected or tested individually, so they are not a
separate multiplying axis in the Holm family; H1 is one joint LR/permutation
test per cell, not four. The "4 lags ×" phrasing was a leftover from an
earlier draft where each lag might have been tested separately and was never
reconciled with §5.2's later, more specific design. With 1 project×quarter
cell (uncorrected + corrected, since the unit itself doesn't vary) plus 4
project×component×quarter cells (2 corrections × 2 missing-component rules),
that's **6 cells × 2 hypotheses (H1, H2) = 12 tests** in the Holm family. H3
(the reverse/rival model) is assessed per cell via its own lagged
coefficients, as in `RQ2_INITIAL_RESULTS.md`, rather than folded into this
family — it has no single pre-registered joint-test formulation the way H1/H2
do (M0-vs-M1 and M1-vs-M2 are natural nested comparisons; H3 has no
pre-registered restricted/full pair), and inventing one post hoc to fold it
into Holm would itself be an unregistered decision made after seeing the
data. If a joint H3 test is wanted later, it should be pre-specified before
looking at H3's own results, not retrofitted now.

## 6. Label generator freeze (plan step 2)

Frozen config, matching `run_seed_sweep.sh`'s `base_meta` entry exactly:
RoBERTa Stage-1 pretrained (`gcp_results/roberta_conservative_0309_0738`),
`--metadata --class_weighting inverse --threshold fixed --val_selection
balance`, seeds 42-46. Per SEED_VARIANCE_FINDINGS.md, this beats the
published Base+Meta+SqrtW and per-fold threshold tuning adds noise for no
gain, so neither is used here. Ensemble = mean of the 5 seeds' predicted
probability per ticket, thresholded at 0.5. Generated by
`generate_versioned_labels.py` + `ensemble_versioned_labels.py`. This label
set, once generated, is **versioned and frozen** — not regenerated mid-analysis
even if a later step suggests a different config would fit better.

## 7. Model and test (plan §3)

Negative binomial GLM, log link, `log(non-bug issues)` offset, project fixed
effects, cluster-robust (by project) standard errors.

- **M0**: controls only (lagged bugs, calendar trend, project age, project FE).
- **M1**: M0 + design ratio (H1).
- **M2**: M1 + temporal features (H2 = the M1→M2 increment).
- **Standalone reverse model**: design ratio at *t+k* ~ bug count at *t* +
  controls (H3), fit independently so a null H1 doesn't strand H2 or make H3
  untestable.

Comparison: likelihood-ratio test (nested), out-of-sample MAE under
forward-chaining (blocked-by-time) cross-validation, and a permutation null
(shuffle design ratio within project, refit, 1,000 permutations) as a
non-parametric check on the LR test's p-value under the panel's actual
autocorrelation structure.

## 8. Measurement-error sensitivity (plan blocker 2)

Alongside the Rogan-Gladen point correction, run **SIMEX** (simulation-
extrapolation) on M1/M2 using each project's classifier error rate as the
known measurement-error variance, to check whether the corrected point
estimate is stable under a different correction method. Both corrected
estimates (Rogan-Gladen and SIMEX) reported alongside the naive uncorrected
one.

## 9. Power / equivalence, and what a null result means (plan blocker 8)

Before fitting M1, run a power analysis for the negative binomial GLM at the
assembled panel's actual n (10 projects × quarters retained after the
inclusion rule — to be computed once the panel is built, expected on the
order of 350-450 project-quarters given §3's quarter counts) to determine
the minimum detectable effect size at alpha=0.05 (Holm-adjusted) power=0.80.
Report TOST equivalence bounds for H1 and H2 so that a non-significant
result can be reported as a bounded null (a meaningful finding against the
unmeasured industry claim the dissertation opens with) rather than as an
uninformative failure to reject.

## 9b. Known small-cluster caveat, corrected (found while building
   `fit_rq2_models.py`; corrected 2026-09-27 after diagnosing real-data runs)

**Original (2026-09-27, pre-real-data) description, since found incomplete:**
with only 10 project clusters and 10 project fixed effects, individual
coefficients collinear with project identity get near-degenerate SEs on
synthetic panels, attributed at the time to clusters ≈ number of FE levels.

**What real-data diagnosis found:** the calendar-time trend and project-age
control (§4/§7's `calendar_index` and `quarter_index`) are not merely
correlated with project identity given only 10 clusters — they are
**exactly** collinear with `C(project)`, for any number of clusters:
`quarter_index = calendar_index - (project's own first-issue quarter)`, and
that per-project constant is already absorbed by the project fixed effects.
Confirmed via `np.linalg.matrix_rank` (12 design-matrix columns, rank 11)
and by fitting all three combinations of {both terms, drop calendar_index,
drop quarter_index} on the real panel — identical log-likelihood and
identical design-ratio-lag coefficients to 1e-6 regardless. This is a
textbook age-period-cohort non-identification problem baked into this
document's own model design (§4/§7), not a coding bug or a small-N
artifact — it would occur with 10,000 clusters just as with 10. A second,
separate instance of the same mechanism was found in M2:
`gini_design_effort` and `front_loaded_index`, as originally implemented,
were per-project constants (broadcast to every quarter), also exactly
collinear with `C(project)`, silently inflating H2's likelihood-ratio test
degrees of freedom from the true value to 3 (see `RQ2_INITIAL_RESULTS.md`
§2d for the fix: `gini_design_effort` is now a genuine expanding-window
statistic per this document's own "cumulative ... to date" wording;
`front_loaded_index` remains a whole-history constant by its own definition
and is excluded from the LR test's degrees of freedom via a general
rank-based df computation rather than a hardcoded term count).

**Resolution:** `calendar_index` was dropped as a regressor (kept only as
the time axis for blocked-time CV splitting) in favor of `quarter_index`,
which is a pure reparametrization with zero effect on any reported
hypothesis test — proven above, not assumed. **Practical implication
unchanged from the original text: don't report per-coefficient Wald
significance for the project fixed effects or any term that turns out to be
collinear with them** — H1 and H2 are joint likelihood-ratio tests on nested
models with degrees of freedom computed from actual design-matrix rank
(`fit_rq2_models.py`'s `nested_lr_test`), which is unaffected by which
reparametrization of the nuisance terms is used.

## 10. What is explicitly out of scope for this analysis

- Extending beyond the 10 hand-labelled projects (would require new manual
  labelling — a separate, explicit decision, not a default).
- Release/version-based time windows (Fix_Version coverage is 0% for DM and
  inconsistent elsewhere — `COMPONENT_SCHEMA_FINDINGS.md`).
- Treating "bug reports" as equivalent to "defects" — TAWOS has no commit/LOC
  data; this is stated as a construct-validity limitation (plan blocker 7),
  not corrected for statistically.
