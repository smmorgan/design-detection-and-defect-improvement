# RQ2 Initial Model Results (first real-data run)

**Status:** 2026-09-27/29. The full pre-registered 12-test spec grid has now
run, with Holm correction applied (§6, **read §6 first for current
numbers** -- §3/§4 describe the primary unit only, and their permutation
p-values are superseded by §2e and §2f). Eight bugs found and fixed so far:
NB dispersion (§2), H3's freq_weights misuse (§2b), an exact
calendar_index/quarter_index/project-FE collinearity (§2c), H2 df inflation
(§2d), a permutation null validating the wrong LR statistic (§2e), and three
found 2026-09-29 (§2f): a Holm step-down implemented as Hochberg step-up, a
permutation p-value formula that could reach 0 and was too coarse at 200
reps, and a component-unit CV that silently failed every fold. The
Rogan-Gladen degeneracy (§1) is still unresolved and SIMEX hasn't been run.

## 1. Blocking issue found: Rogan-Gladen correction is currently degenerate

The corrected design ratio (`design_ratio_corrected` in `data/rq2_panel.csv`)
clips to exactly 0 for a large, project-varying share of quarters:

| Project | Sensitivity | Specificity | 1-specificity (floor) | Mean apparent (raw) rate | % quarters clipped to 0 |
|---|---|---|---|---|---|
| TIMOB | 0.942 | 0.763 | 0.237 | 0.126 | **95%** |
| FAB | 0.919 | 0.713 | 0.287 | 0.278 | **61%** |
| NEXUS | 0.915 | 0.875 | 0.125 | 0.141 | 49% |
| CONFSERVER | 0.793 | 0.929 | 0.071 | 0.107 | 34% |
| MULE | 0.918 | 0.781 | 0.219 | 0.307 | 21% |
| SERVER | 0.531 | 0.932 | 0.068 | 0.085 | 21% |
| JRASERVER | 0.888 | 0.867 | 0.133 | 0.202 | 20% |
| DNN | 0.764 | 0.950 | 0.050 | 0.066 | 19% |
| MESOS | 0.934 | 0.831 | 0.169 | 0.269 | 15% |
| DM | 0.685 | 0.813 | 0.187 | 0.419 | 0% |

**Root cause, confirmed empirically for TIMOB:** the ~180-ticket-per-project
manual sample used to estimate sensitivity/specificity (`output/all_manually_
labelled.csv`) is drawn entirely from the most recent slice of each
project's history (TIMOB manual sample issue numbers: 24549-28207, out of a
full range of 3-28207) — matching the IEEE DataPort export's "most recent
3,000 issues" cap. Even restricted to that same recent-3000 window, the
classifier predicts design on only 18.6% of all tickets, vs. 41.2% on the
specific ~180 tickets selected for manual labelling. That gap means the
manually-labelled calibration sample is systematically enriched for
design-adjacent tickets relative to the general population it was drawn
from — plausibly because whoever selected candidates for manual annotation
(before this repo's history begins; no selection script is checked in)
pre-filtered for plausible design candidates rather than drawing a simple
random sample, which is a completely reasonable thing to do when curating a
class-balanced-enough set for training/eval, but it breaks the assumption
Rogan-Gladen needs: that sensitivity/specificity estimated on the
calibration sample transfer unchanged to the target population. When the
full population's true apparent rate falls below the specificity-implied
floor `(1 - specificity)`, the corrected estimate is undefined and clips to
zero.

**Consequence:** the primary corrected model (H1 with `design_ratio_
corrected`) gives a *weaker*, non-significant result (LR p=0.105) than the
uncorrected one (LR p=0.021) — precisely because clipping destroys real
variation for the hardest-hit projects (TIMOB most of all). Reporting only
the corrected number, as pre-registered, would have silently thrown away a
real signal because of a broken correction, not because the true effect is
null. **This is why the pre-registration's "report both side by side" rule
matters** — it caught this rather than hiding it.

**Not yet resolved.** Options, none implemented yet:
1. Re-estimate sensitivity/specificity from a random (not manually-curated)
   sample of the full population — requires new hand-labelling.
2. Use a full ROC/calibration curve instead of a single threshold-0.5
   operating point, and pick an operating point appropriate to the
   population's actual score distribution rather than the labelled sample's.
3. Report the uncorrected ratio as primary for this round, with the
   correction attempt documented as an open problem rather than a result.

**Recommendation for now: report `design_ratio_raw`, not `design_ratio_
corrected`, as primary**, with this section as the explanation, until one of
the above is done.

## 2. Bug found and fixed: NB dispersion was silently fixed at alpha=1.0

`fit_rq2_models.py`'s original `fit_nb_glm` used `smf.glm(family=sm.families.
NegativeBinomial())`, which requires `alpha` to be supplied and otherwise
defaults it to a fixed 1.0 (confirmed via a `ValueWarning` at fit time) —
not a fitted dispersion parameter. This dataset's actual MLE-estimated alpha
is ~0.56-0.58, nowhere near 1. Fixed by switching to `smf.negativebinomial`
(the discrete MLE model), which estimates alpha jointly with the mean-
structure coefficients. This changed the log-likelihoods (and therefore the
LR-test statistics) materially — the H1 LR stat moved from 7.06 (p=0.133,
wrong model) to 11.55 (p=0.021, correct model) on the same data. **Do not
trust any RQ2 NB regression result computed before this fix.**

## 2b. Bug found and fixed: H3's `freq_weights` inflated N to ~195,000

`fit_h3_reverse`'s binomial GLM used `freq_weights=resolved_total` to weight
the aggregated design-ratio proportion by the number of tickets it's based
on. `freq_weights` tells statsmodels each row was independently observed
that many times, which inflated `Df Residuals` from ~401 to 195,540 (the sum
of `resolved_total` across rows) and triggered a `SpecificationWarning:
cov_type not fully supported with freq_weights`. The correct weighting for
an aggregated successes/trials proportion is `var_weights` (precision
weighting, doesn't multiply the effective sample size). Switched to
`var_weights`; `Df Residuals` is now the correct ~401 and the freq_weights
warning is gone (a different, narrower `SpecificationWarning: cov_type not
fully supported with var_weights` remains — a known statsmodels limitation
for this exact combination, not something with a fix available, and it
doesn't affect the coefficients or SEs actually reported). The originally-
reported H3 numbers (e.g. "only bug_count_lag3 significant, p=0.005") were
computed under the broken weighting and should not be trusted — see §3 for
the re-run numbers.

## 2c. Bug found and fixed: exact collinearity between calendar_index, quarter_index, and project FE

While cleaning up the `HessianInversionWarning`/`ConvergenceWarning` noise
flagged as a caveat in §3 of the original write-up, found that
`calendar_index` (quarters since 2000Q1), `quarter_index` (project age), and
`C(project)` are **exactly** collinear, not just "small-cluster noisy" as
`RQ2_PREREGISTRATION.md` §9b originally described: `quarter_index =
calendar_index - (project's own start quarter)`, and that per-project
constant is already absorbed by the project fixed effects. Confirmed via
`np.linalg.matrix_rank` (12 columns, rank 11) and by fitting all three
combinations of {both, drop calendar_index, drop quarter_index} — identical
log-likelihood and identical `design_ratio_lag` coefficients to 1e-6
regardless. This is a classic age-period-cohort non-identification problem
baked into the pre-registered model design (§4/§7), not a bug in translating
it to code. Fixed by dropping `calendar_index` as a regressor everywhere
(kept as the time axis for blocked-time CV splitting, which is a legitimate,
different use). This is a pure reparametrization — H1/H2's LR stats and the
`design_ratio_lag` coefficients are unaffected — but it removes almost all
of the `HessianInversionWarning`/`SingularMatrixWarning` noise (H3's GLM,
which does an explicit WLS solve each IRLS iteration, surfaced the same
rank deficiency as an explicit `SingularMatrixWarning` that's now gone).
`RQ2_PREREGISTRATION.md` §9b has been corrected to describe this properly.

## 2d. Bug found and fixed: H2's degrees of freedom were inflated from 1 (later 2) to 3

`gini_design_effort` and `front_loaded_index` were both computed once over
each project's *entire* history and broadcast as a constant to every quarter
of that project (see `build_rq2_panel.py`, pre-fix). A per-project constant
is exactly collinear with `C(project)`, so — same mechanism as §2c —
including them in M2 alongside the project FE adds **zero** real degrees of
freedom: confirmed by showing M2's log-likelihood is bit-for-bit identical
whether both terms are included or only `quarters_since_changepoint` is.
The H2 likelihood-ratio test was nonetheless run with `df=3`, which uses an
overly generous chi-square reference distribution and biases the test
toward *appearing* less significant than it is.

Fixed two ways:
1. **General fix**: replaced the hardcoded `df=3` with a `nested_lr_test`
   helper that computes degrees of freedom from the actual rank difference
   between the two models' design matrices (`np.linalg.matrix_rank`), so any
   future collinearity (known or not) is caught automatically instead of
   silently inflating a p-value. Applied to both H1 and H2's LR tests.
2. **Root-cause fix for `gini_design_effort` specifically**: the
   pre-registration (§4) defines it as "cumulative design-ticket resolution
   *to date*" — i.e. an expanding-window statistic that should vary by
   quarter — but the code computed it once over the whole series. Fixed to
   recompute the Gini coefficient at every quarter from only that project's
   history through the current quarter. This both matches the pre-registered
   definition and removes a look-ahead leak (early quarters were previously
   informed by that project's entire future history). `front_loaded_index`
   is, by its own pre-registered definition ("share of design tickets
   resolved in the first third of *observed* project-quarters"), genuinely a
   whole-history summary — not a bug, just inherently unidentifiable
   alongside project FE. It's kept in the model for descriptive completeness;
   `nested_lr_test`'s rank-based df computation already discounts it
   automatically.

After both fixes, H2 has `df=2` (gini_design_effort, now genuinely
time-varying, plus quarters_since_changepoint; front_loaded_index
contributes nothing). See §3 for updated numbers — the correct df changes
H2's p-value materially (e.g. raw ratio: p=0.510 -> p=0.313; corrected
ratio: p=0.333 -> p=0.173), though the conclusion (not significant at
alpha=0.05) doesn't flip either way.

## 2e. Bug found and fixed: permutation null tested a different sample than the LR test it validates

Found 2026-09-28 while preparing the component-unit spec-grid runs.
`permutation_test_h1` (and its component-unit twin
`permutation_test_h1_component`) recomputed its own "observed" LR statistic
from scratch on `panel` — dropping NA only on the M0/M1 columns (405 rows,
all 10 projects) — instead of reusing `clean`, the M2-column-dropna sample
(367 rows, 8 projects — DM and FAB are dropped because neither has a
detected changepoint, §5.3) that the headline `h1_stat`/`h1_p` printed just
above it were actually fit on. The permutation p-value was therefore
answering "how often does a *differently-sampled* LR stat exceed the
headline one" rather than "how often does *this* LR stat exceed itself under
random reassignment" — not a fatal error (both samples are legitimate
90%+-overlapping subsets of the same panel), but the two numbers being
compared were never the same statistic, which is not a valid permutation
test regardless of how small the gap happens to be.

The component-unit case showed how large this gap can get: on the
project×component×quarter/drop/raw spec, the headline chi-square LR stat is
13.897 (on the M2-clean, ~14–17k-row sample), but the permutation function's
own "observed" stat on the larger M1-only sample was only 6.111 — over 2x
different, because the M1-only sample includes hundreds of short
(project, component) series that the M2 columns (which need enough history
for `quarters_since_changepoint`) exclude. That gap is what surfaced the bug
before any of the real 200-permutation component runs were launched.

**Fixed** by passing `clean` into both permutation functions instead of
`panel` (`clean` already satisfies the functions' own internal `dropna`, so
this is a no-op restriction, not a new exclusion). Re-ran the primary unit's
permutation null with the fix:

| Ratio | Old (bug) permutation p | New (fixed) permutation p | Observed LR stat (now consistent) |
|---|---|---|---|
| raw (uncorrected) | 0.116 | **0.070** | 11.553 (matches headline h1_stat) |
| corrected (Rogan-Gladen) | 0.378 | **0.206** | 7.651 (matches headline h1_stat) |

**This changes the substantive read of §3/§4 below**: the uncorrected
ratio's permutation p moved from a comfortable "not significant" (0.116) to
a borderline 0.070 — still above the conventional 0.05 threshold and still
the honest conclusion is "H1 not supported at alpha=0.05," but the margin is
much thinner than the original write-up suggested, and this is *before* the
pre-registered Holm correction is applied across the full 12-test family
(§5 below), which will only push it further from significance, not closer.
The corrected ratio's p also dropped (0.378 → 0.206) but the qualitative
conclusion (not significant, and still the less-trustworthy of the two given
§1's Rogan-Gladen problem) is unchanged. §3/§4's numbers below are left as
originally reported for the historical record of what was written up before
this fix; the corrected numbers are the ones in this section and should be
used going forward, including in `apply_holm_correction.py`'s TESTS list
(updated 2026-09-28).

## 2f. Bugs found and fixed 2026-09-29: Holm direction, permutation resolution, component CV

Found while transcribing the finished component grid into
`apply_holm_correction.py`.

1. **Holm correction was anti-conservative.** `holm_correction` enforced
   monotonicity with a reverse running *minimum*
   (`np.minimum.accumulate(raw_adj[::-1])[::-1]`), which is Hochberg's step-up
   form, not Holm's step-down. It pulled most adjusted p-values down to the
   largest raw p in the family (e.g. primary-unit raw H1: 0.39 instead of the
   correct 0.71). Fixed to `np.maximum.accumulate(raw_adj)`. No verdict had
   been published with the wrong version, and no test crosses 0.05 either
   way, but any adjusted p quoted before this date is wrong.
2. **Permutation p-values: formula and resolution.** Both permutation
   functions returned `exceed / n_perms`. That can be exactly 0, and failed
   refits counted as non-exceedances instead of shrinking the denominator.
   Changed to `(exceed + 1) / (n_ok + 1)`, and the number of successful
   refits is now logged (it was 100% in every run). Separately, the
   component unit ran only 200 permutations, which can't resolve p below
   about 0.005. That's above the strictest Holm threshold for a 12-test
   family (0.05/12 = 0.0042), so the two "bucket" cells (both 1/200
   exceedances) couldn't be decided. Reran all 4 component cells at 2000
   permutations (min attainable p about 0.0005), more than the pre-registered
   1000 (§7), so this is a deviation in the conservative direction. The
   primary unit was rerun at its pre-registered 1000 with the new formula
   (0.070 -> 0.0709, 0.206 -> 0.2068).
3. **Component-unit blocked-time CV failed in every fold** (0 folds used, 17
   reported "non-converged"). Root cause: `blocked_time_cv_mae` filtered test
   rows to *projects* seen in training, but the component models' fixed effect
   is `C(project_component)`. Any test row from a component not yet seen in
   training made `predict()` raise a PatsyError, and the `except` counted it
   as non-convergence. Every fold had at least one such row. Fixed by adding
   an `fe_col` argument (default `'project'`, `'project_component'` at the
   component unit). All 17 eligible folds are now used in every cell. The
   primary unit is unaffected, since its FE and filter were already both
   `project`.

Superseded logs are kept in `logs/superseded_2026-09-28/`.

## 3. Results after all five bug fixes (§1 still open; §2/2b/2c/2d fixed; §2e's permutation-sample fix supersedes the two permutation p-values below — see §2e)

Panel: 463 project-quarter rows -> 367 after dropping rows with missing
lags/controls (the 4-quarter distributed lag costs the first 4 quarters of
each project's history). M0 now controls for `bug_count_lag1`,
`quarter_index` (project age; `calendar_index` dropped per §2c — a pure
reparametrization, not a scope change), and project fixed effects.

**`design_ratio_raw` (uncorrected, recommended primary reading per §1):**

- **H1 (M0 vs M1, design ratio lags 1-4):** LR stat=11.55, df=4, **p=0.021**.
  `design_ratio_lag1` = -1.23 (z=-2.93, p=0.003), `design_ratio_lag2` = -0.75
  (z=-3.39, p=0.001). Lags 3-4 not significant. Point estimates essentially
  unchanged from the pre-fix run (as expected — none of §2c/§2d touch the
  design-ratio columns). **Direction: more design work now is associated
  with fewer bugs 1-2 quarters later**, net of activity-level exposure,
  lagged bugs, project fixed effects, and project age.
- **Permutation null for H1: p=0.116 — NOT significant.** Unchanged from the
  pre-fix run (expected: none of the fixes touch the design-ratio lag
  coefficients or the permutation procedure itself). **This overturns the
  naive chi-square-based p=0.021** — the permutation null is the more
  trustworthy number given only 10 clusters and strong within-project
  autocorrelation, and it says **H1 is not supported**.
- **H2 (M1 vs M2, temporal features): LR stat=2.33, df=2 (corrected from the
  original df=3 — see §2d), p=0.313.** Not significant either way, but
  notably different from the originally-reported p=0.510 — the corrected
  test is a real, materially different number, not just a cosmetic fix.
- **Blocked-time CV MAE: M0=390.0 (17 folds), M1=403.4 (16 folds, 1 dropped
  for non-convergence), M2=393.1 (16 folds, 1 dropped) — M0 best.** Now
  computed with the §2c fix (removes most convergence warnings) and with
  folds explicitly dropped rather than silently averaged in when either
  fewer than 6 projects have entered the panel yet or the NB MLE fit
  reports non-convergence (`blocked_time_cv_mae`'s new diagnostics). This
  agrees with the permutation null's "H1 not supported" reading. The MAE
  still exceeds the outcome's own mean (246 bugs/quarter) — plausibly
  because pooled MAE across 10 projects spanning ~2,500 to ~25,000 total
  bugs is dominated by the highest-volume projects (CONFSERVER, JRASERVER,
  SERVER); not investigated further here, flagged as a residual limitation
  of raw pooled MAE as the CV metric rather than a bug.
- **H3 (reverse: bug_count -> design_ratio):** binomial GLM, now correctly
  weighted (§2b) with well-behaved coefficients and SEs throughout (the
  earlier degenerate Df Residuals/pseudo-R² are fixed). Only
  `bug_count_lag3` is significant (p=0.005) out of 4 lags, no clean pattern
  — same qualitative read as before the fix, but now on trustworthy
  numbers rather than ones computed with an inflated N.

**`design_ratio_corrected` (Rogan-Gladen; still flagged unreliable, §1):**

- **H1:** LR stat=7.65, df=4, p=0.105. **Permutation null: p=0.378** (newly
  computed this round — wasn't run against the corrected ratio before).
  Even less significant than the uncorrected version, consistent with §1's
  diagnosis that clipping destroys real variation for the hardest-hit
  projects.
- **H2:** LR stat=3.50, df=2 (corrected from df=3), p=0.173 — versus the
  originally-reported p=0.333 under the (wrong) df=3 test.
- **Blocked-time CV MAE: M0=390.0, M1=369.3, M2=368.4 (17 folds each, no
  non-convergence)** — M1/M2 *beat* M0 here, the opposite ordering from the
  uncorrected ratio's CV result. This is a real tension worth flagging, not
  resolved by any of this round's fixes: out-of-sample error favors keeping
  the design ratio under the corrected specification even though its own
  LR test and permutation null don't support it. Another reason §1's
  correction needs to be fixed or replaced before leaning on the corrected
  numbers either way.

## 4. Bottom line

**H1 is not supported by this first pass**, on either the uncorrected or
corrected design ratio — **using the §2e-corrected permutation numbers,
not the ones originally reported in this section** (0.070/0.206, not
0.116/0.378; see §2e for the fix). The uncorrected chi-square LR test looked
significant (p=0.021), but the pre-registered permutation null — the more
trustworthy check given only 10 project clusters and strong within-project
autocorrelation — puts it at **p=0.070**, a genuinely borderline number, not
the comfortable non-significance originally reported; the corrected ratio's
permutation null is weaker still (p=0.206). Both are before Holm correction
across the full 12-test family (§5), which only pushes them further from
significance. The point estimates (`design_ratio_lag1`/`lag2` both negative,
plausible short-lag direction) are not nonsense, but the evidence they
reflect a real population effect rather than autocorrelation-inflated noise
is weak, and now less clearly weak than originally written up. **H2 is not
significant under either ratio** (p=0.313 uncorrected, p=0.173 corrected,
both using the now-corrected df=2 test; H2 has no permutation-null step, so
§2e's fix doesn't touch it). Report this as a null-leaning result, not a
positive one, pending the rest of the spec grid.

This is exactly the scenario `RQ2_PREREGISTRATION.md` §9 planned for
("plan for a null result... a null is a publishable finding against the
unmeasured industry claim... without [pre-registration], a null is
unpublishable and indistinguishable from attenuation"). The classifier
measurement-error problem (§1 above) means this null cannot yet be
distinguished from attenuation-toward-the-null caused by the still-broken
Rogan-Gladen correction — so the honest current status is "no significant
effect detected on the uncorrected measure; the corrected measure isn't
trustworthy yet," not "no effect exists." The CV tension noted above (§3)
adds a second, independent reason not to treat this as a fully clean null
yet.

## 5. What this initial pass does and doesn't establish

**Does:** confirms the pipeline (data pull -> label ensemble -> panel ->
NB model -> permutation null) runs end to end on real data, and surfaces
five real bugs (§1, §2, §2b, §2c, §2d) that would have silently produced
wrong p-values if uncaught. The CV and H3 diagnostics flagged in the
original write-up are now resolved (§2b fixed H3's degenerate weighting and
pseudo-R²; §2c removed the exact collinearity that was driving most of the
`HessianInversionWarning`/`ConvergenceWarning` noise; `blocked_time_cv_mae`
now filters out folds with too few projects present or that fail to
converge, and reports those counts instead of silently averaging them in).

**Doesn't:** this is one cell of the pre-registered specification grid
(project x quarter unit, lags 1-4 jointly). Per `RQ2_PREREGISTRATION.md`
§5, the full grid (component unit, missing-component handling variants,
SIMEX) still needs to run, and Holm correction applies across the *whole*
family once it does. The Rogan-Gladen correction (§1) still needs to be
fixed or replaced before the corrected estimate can be trusted at all — none
of this round's fixes touch that problem, which is a calibration-sample
selection-bias issue, not a modeling bug. The CV-vs-LR-test tension noted
in §3 (corrected ratio's CV favors M1/M2 while its own hypothesis tests
don't) is also unresolved and may or may not be related to the same
correction problem.

**Still open, in priority order:**
1. **Fix or replace the Rogan-Gladen correction** (§1) — needs a decision
   among: (a) a new randomly-sampled calibration set (new hand-labelling,
   out of scope to decide unilaterally per the pre-registration's own §10),
   (b) a full calibration-curve approach, (c) continuing to report only the
   uncorrected ratio as primary with the correction documented as an open
   problem (current status), or (d) implementing SIMEX (§8) as a stability
   check alongside Rogan-Gladen — note SIMEX uses the same
   sensitivity/specificity inputs and so likely inherits the same
   calibration-sample bias rather than fixing it; useful as a second opinion
   on the correction *method*, not a fix for the underlying transfer
   assumption.
2. **Run the remaining pre-registered spec-grid axes** (project x component
   unit x {drop, bucket} missing-component handling x {raw, corrected}) and
   apply Holm correction across the whole 12-test family. **In progress as
   of 2026-09-28**: `run_component_grid.sh` launched in the background
   (nohup, PID captured in the session, not persisted anywhere else — check
   `logs/component_grid_driver.log` for the running/ALL DONE marker and
   `logs/component_{drop,bucket}_{raw,corrected}.log` for each cell's full
   output once done, expected ~1-1.5h/cell, ~5-6h total). This includes the
   §2e permutation-sample fix (applied before launching, not after — the
   background job was not started until both scripts were fixed). Once all
   4 logs are complete, transcribe each cell's H1 permutation p and H2
   chi-square p into `apply_holm_correction.py`'s `TESTS` list (currently
   4/12 filled) and re-run it for the final Holm-adjusted family.
3. Write up the final results section once the above lands.

## 6. Full spec-grid results with Holm correction (2026-09-29)

All 12 pre-registered tests (`RQ2_PREREGISTRATION.md` §5): 3 cells
({project} + {component x drop, component x bucket}) x 2 ratios (raw,
Rogan-Gladen corrected) x 2 hypotheses (H1, H2). Produced by
`run_component_grid.sh` (all six model runs in parallel) and
`apply_holm_correction.py`. Output is saved in `logs/holm_final.txt`, and
per-cell detail is in `logs/primary_{raw,corrected}.log` and
`logs/component_{drop,bucket}_{raw,corrected}.log`.

| Unit | Missing component | Ratio | H1 perm p | H1 chi-sq p | H2 chi-sq p |
|---|---|---|---|---|---|
| project x quarter | -- | raw | 0.0709 | 0.021 | 0.313 |
| project x quarter | -- | corrected | 0.2068 | 0.105 | 0.173 |
| component x quarter | drop | raw | 0.0595 | 0.0076 | 0.377 |
| component x quarter | drop | corrected | 0.0690 | 0.0127 | 0.395 |
| component x quarter | bucket | raw | **0.0045** | 0.0002 | 0.207 |
| component x quarter | bucket | corrected | **0.0065** | 0.0002 | 0.229 |

H1 permutation nulls: 1000 reps (primary), 2000 reps (component), with all
refits succeeding.

**Holm-adjusted (family of 12, alpha = 0.05): no test is significant.** The
closest are the two bucket-cell H1 tests at adjusted p = **0.054** (raw)
and **0.0715** (corrected). The drop-cell H1 tests adjust to 0.595/0.621,
primary-unit raw H1 to 0.621, and everything else to 1.0.

**Reading:**

- **H1 is not supported after the pre-registered multiplicity correction,
  but it is not a clean null either.** The direction is the same in all 6
  cells: `design_ratio_lag1` is negative everywhere (component unit -0.21 to
  -0.30, primary unit about -1.2), so more design work this quarter goes with
  fewer bugs next quarter. Unadjusted permutation p-values are 0.0045-0.07 in
  5 of 6 cells. The bucket cells miss the Holm threshold (0.0042) narrowly.
  This is a consistently signed, individually borderline effect that the
  pre-registered design does not let us call significant.
- **The only individually strong evidence depends on bucketing.** It
  appears only when issues with no component are pooled into a per-project
  "Unclassified" series. When those issues are dropped, permutation p rises
  to 0.06-0.07, which matches the primary unit. So the extra signal comes from
  the Unclassified series. Those issues may differ systematically from
  labelled ones (e.g. triage-stage or cross-cutting tickets), so this is not
  evidence about design work *within real subsystems*. This hasn't been
  investigated yet (§7 item 2).
- **H2 is not supported anywhere** (unadjusted p = 0.17-0.39, all cells).
- **Raw vs. corrected ratio barely matters at the component unit**
  (permutation p within 0.01 of each other in both handling variants). The
  Rogan-Gladen problem (§1) mainly affects the primary unit's corrected
  cell.
- **Blocked-time CV at the component unit (now working, §2f):** M2 < M1 <
  M0 in all 4 cells, but by small margins: drop MAE 16.6 -> 16.5 -> 16.1,
  bucket 25.6 -> 25.4 -> 24.1 (17 folds each). Most of the gain comes from
  M1 -> M2 (the temporal features), which is the opposite of what the H2 LR
  tests say. The M0 -> M1 gain from the design-ratio lags is under 1%.
  This is weak out-of-sample support at best. It goes into the same "CV and
  in-sample tests disagree" bucket as the primary unit's corrected cell (§3).

**Bottom line:** per the pre-registration, RQ2's answer is **no
significant association after correction for multiple testing, with a
consistently negative (beneficial) point estimate for short-lag design
work**. §4's caveat still applies: with the Rogan-Gladen correction still
degenerate, this null cannot yet be separated from attenuation caused by
classifier error.

## 7. Still open (supersedes §5's list)

1. **Rogan-Gladen correction (§1)**: still needs a decision. The audit
   inputs don't fix it (§8b). Recommendation in §8b: raw tests as the
   measurement-error-robust tests, SIMEX (§9b, now run) for corrected effect
   sizes, the 12-test Holm family kept as the confirmatory verdict.
2. ~~**Drop vs. bucket divergence (§6)**~~: done (§9a). The bucket signal is
   carried by the 7 Unclassified series, which are confounded by triage
   practice.
3. **Commit the RQ2 work**: all of it is still uncommitted on
   `rq2-collapsed-plan`.


## 8. Calibration audit: time-drift check (2026-10-03)

300 random resolved tickets (10 per project x era, drawn and frozen before
labelling: `draw_calibration_audit_sample.py`, pre-registration §11) were
hand-labelled blind to the classifier output, all 300 (stopping rank k=10).
`rq2_calibration.py` -> `gcp_results/rq2_calibration_drift.json`, log in
`logs/rq2_calibration_drift.log`. 27/300 are design.

**No detectable drift in calibration over project history.** P(design | classifier
score) does not differ by era: intercept-shift LR test p=0.54 (df=2); with the slope
also allowed to vary, p=0.15 (df=4). Specificity is flat across eras
(78/89, 81/93, 77/91 = 0.88, 0.87, 0.85 early/middle/late). Sensitivity has no
monotone trend (8/11, 7/7, 5/9). Caveat: with 300 tickets this test only reliably
catches large drift; in simulation it detected a moderate drift (early-era odds
ratio ~0.37) only 34% of the time. Read this as "no evidence of drift, large drift
ruled out", not "calibration proven constant".

**The bigger finding: the classifier over-counts design work by about 2x in the
general population.** Population-weighted:

| | early | middle | late | all |
|---|---|---|---|---|
| True design rate (audit labels) | 0.115 | 0.073 | 0.065 | **0.084** (95% CI 0.052-0.118) |
| Classifier-flagged rate | 0.153 | 0.205 | 0.173 | **0.177** (0.134-0.223) |
| Predicted by calibration fit on curated labels | 0.123 | 0.188 | 0.152 | **0.154** |

Overall (unweighted) sensitivity 20/27 = 0.74 (Wilson 0.55-0.87) and specificity
236/273 = 0.86 (0.82-0.90). With ~92% of tickets non-design, the 14% false-positive
rate produces more flagged tickets than the true positives do (37 FP vs 20 TP).

**Consequences:**
- **Option 1 (score-based correction calibrated on the curated labels) fails its
  own transfer check.** The curated-fit calibration predicts 15.4% design; the true
  rate is 8.4%, outside the 95% CI (no bootstrap replicate reached 0.154). The
  curated sample's selection is *not* explained by the classifier score, so it
  can't be reused through calibration either. Any correction has to be estimated
  from the random audit.
- **Rogan-Gladen with audit-based inputs becomes feasible at the pooled level.**
  The apparent rate (0.177) is now above the false-positive floor (1 - 0.86 =
  0.14), unlike with the curated-sample specificities (§1). Per-project
  sensitivity/specificity can't be estimated from 30 tickets each, so a correction
  would have to use pooled audit values. Since drift wasn't detected, those values
  can be pooled across eras too. Individual project-quarters may still fall
  below the floor and clip; that needs checking before relying on the corrected
  cells.
- **For the main (uncorrected) result:** the measured design ratio is roughly
  half noise. That's consistent with attenuation toward the null (pre-registration
  §9). With no detected drift, there is no sign that the error is time-varying in
  a way that would create or reverse the lag effects. Large-drift scenarios are
  ruled out; smaller drift remains possible.

### 8b. Rogan-Gladen with audit-based inputs: still not viable at the panel level

Pooled population-weighted audit values are sensitivity 0.669 and specificity
0.868, so sens + spec - 1 = 0.537 (95% CI 0.33-0.75). Applied to every
project-quarter, they still clip heavily: **44% of project-quarters and ~60% of
component-quarters** fall below the false-positive floor (0.132) and correct to 0.
This ranges from 100% of quarters for SERVER to 0% for DM and FAB.

The reason is that **the classifier's false-positive rate differs by project**
(audit negatives flagged: DM 9/27, MULE 6/25 ... TIMOB 0/28; chi-square
heterogeneity test p=0.004, permutation p=0.003). A single pooled specificity
can't fit all projects. Per-project values from 30 tickets each are far too
imprecise to correct per project-quarter. So **neither the curated-sample inputs
(§1) nor the audit inputs give a usable per-row correction.** I did not re-run the
corrected cells with these inputs; that would swap one degenerate correction for
another.

**Why this matters less than it looks:**
- **Hypothesis tests:** with constant sensitivity/specificity and no clipping,
  Rogan-Gladen is the same linear rescaling of the raw ratio for every row. The
  H1/H2 LR tests, the permutation null and the PELT change points (L2 cost,
  variance-scaled penalty) are all invariant to that rescaling. A "correct"
  pooled correction therefore gives exactly the raw-ratio tests. The raw cells
  *are* the corrected tests, minus clipping artefacts.
- **Project differences in false-positive rate:** these mostly shift a project's
  apparent ratio up or down by a constant amount. That enters the model as a
  per-project constant, which the `C(project)` fixed effects absorb. What they
  don't absorb is any difference across projects in how strongly the apparent
  ratio tracks the true one (sens + spec - 1). This is a residual limitation.
- **Effect sizes:** correct the *coefficient* instead of the data. *(Corrected
  2026-10-03: an earlier draft of this bullet multiplied by 1/(sens + spec - 1)
  = 1.86. That is backwards.)* In expectation the apparent ratio is
  (1 - spec) + a x true ratio, with a = sens + spec - 1 = 0.537. A one-point move
  in the true ratio therefore shows up as only a 0.537-point move in the apparent
  ratio, so a coefficient per unit of *apparent* ratio is *larger* in magnitude
  than the coefficient per unit of *true* ratio. The scale correction is to
  multiply by a = **0.537 (95% CI 0.33-0.75)**. Example: primary-unit
  `design_ratio_lag1` = -1.29 raw becomes about -0.69, i.e. +10 percentage
  points of true design share goes with ~7% fewer bugs the next quarter
  (component unit: -0.23 to -0.30 -> -0.12 to -0.16, ~1-2%). This is only the
  scale part. The per-ticket noise added by misclassification attenuates the
  raw coefficient toward zero, which pushes the other way. SIMEX (§9) handles
  both at once and is the better corrected estimate. These are point estimates
  of an effect that is not significant after Holm, so they are illustrative only.

**Recommendation (needs your decision, since it changes pre-registered cells):**
report the raw-ratio tests as the measurement-error-robust tests. Report the
SIMEX coefficients (§9) as the corrected effect sizes. Report the original
corrected cells as a documented, failed correction rather than as evidence. This
would also remove 6 near-duplicate tests from the Holm family; doing that is a
deviation, so record it in pre-registration §11 if adopted.

**Warning on that last point:** dropping the corrected cells changes the headline
result. With only the 6 raw tests, Holm makes the bucket-cell H1 (p=0.0045)
significant (adjusted 0.027), where it was 0.054 in the 12-test family. Because
this deviation was proposed *after* the results were seen, it must not be adopted
as the confirmatory answer. Keep the 12-test family as the pre-registered verdict
(not significant). If the 6-test family is reported at all, label it exploratory
and note that the bucket signal comes from the "Unclassified" series (§6, §7 item 2).

## 9. Exploratory follow-ups: the Unclassified series and SIMEX (2026-10-03)

### 9a. Does the "Unclassified" series carry the bucket-cell signal? (§7 item 2)

`explore_unclassified_series.py` -> `gcp_results/rq2_unclassified_check.json`,
log `logs/unclassified_check.log`. Exploratory, outside the Holm family. Raw
ratio, on the bucket cell's own M2-complete sample (4,185 rows, 209 series).

- **Sanity check passes:** the bucket panel minus its Unclassified series is
  identical to the drop panel. So the drop/bucket difference comes entirely
  from those series. Only 7 Unclassified series reach the fitted sample
  (FAB, NEXUS and TIMOB have none), totalling 266 rows.
- **The Unclassified series have a much stronger lag effect than real
  components:**

  | Sample | Rows / series | lag1 | lag2 | H1 chi-sq p | H1 perm p |
  |---|---|---|---|---|---|
  | Bucket, all series | 4185 / 209 | -0.30 | -0.23 | 0.0002 | 0.0045 (§6) |
  | Unclassified only | 266 / 7 | **-1.91** | **-2.06** | 1.5e-5 | **0.018** (2000 perms) |
  | Bucket minus Unclassified (= drop) | 3919 / 202 | -0.23 | -0.16 | 0.0076 | 0.0595 (§6) |

  Lag x Unclassified interaction: LR = 42.0, df = 4, chi-square p = 1.7e-8
  (lag1/lag2 interaction terms about -2.05 each). Treat that chi-square p as
  optimistic. The §2e/§6 permutation nulls show the chi-square p-values for
  this model family are about an order of magnitude too small.
- **Why the Unclassified series are suspect as evidence about design work:**
  - They are large: no-component issues are 61% of CONFSERVER's resolved
    issues, 42-45% of JRASERVER, MESOS and DM, and under 10% only for MULE and
    TIMOB. Each Unclassified series is therefore one of the largest series in
    its project.
  - Their composition changes over time much more than any real component's.
    The no-component share falls sharply over project history in CONFSERVER
    (77% -> 39%, early -> late era) and NEXUS (62% -> 8%), and rises in DM, MULE
    and SERVER. That reflects changes in triage practice (when teams started
    assigning components), not changes in the work itself. A series whose
    membership is driven by triage practice can produce a lag association
    between its own flagged-design share and its own bug count without any
    causal link to design work.
  - They differ in content: in all 10 projects they have a *lower* bug share
    than component-linked issues (e.g. DM 3% vs 15%, MULE 21% vs 50%) and a
    higher classifier-flagged design share in 7 of 10.
- **Reading:** the drop-cell result (permutation p = 0.06, lag1 -0.23) is the
  better estimate of the within-subsystem association. The bucket cell's
  stronger result is driven by 7 heterogeneous catch-all series whose
  membership changes with triage practice. It should not be cited as
  support for H1. Since the bucket cell is the only cell that comes near
  Holm significance (§6), this strengthens the "not supported" reading.

### 9b. MC-SIMEX (pre-registration §8)

`simex_rq2.py` -> `gcp_results/rq2_simex/simex_{unit}_{pi}.json`, logs
`logs/simex_*.log`. It uses MC-SIMEX for misclassified binary labels
(Küchenhoff et al. 2006). Each ticket's observed label is re-misclassified
through Pi^lambda, using audit sensitivity/specificity (§8), for lambda in
{0.5, 1, 1.5, 2}. The ratios are rebuilt from tickets, M1 is refit on the same
rows as the reported fits, and the result is extrapolated to lambda = -1. The
ticket-level rebuild reproduces the panel's `resolved_total` and
`design_ratio_raw` exactly (asserted). Two error models:
- `pooled`: sens 0.669 / spec 0.868 for every project.
- `project_spec`: per-project audit specificity with pooled sensitivity.

B = 200 replicates per lambda at the primary unit (3-4 of 800 refits did not
converge; they are included in the means) and B = 50 at the component unit
(all converged). Raw ratio.

`design_ratio_lag1` (naive cluster SE in brackets):

| Unit | Naive | SIMEX quadratic (pooled / project spec) | SIMEX linear (pooled / project spec) |
|---|---|---|---|
| project x quarter | -1.29 [0.33] | -0.61 / -0.91 | -1.30 / -1.42 |
| component, drop | -0.23 [0.09] | -0.47 / -0.46 | -0.29 / -0.29 |
| component, bucket | -0.30 [0.09] | -0.56 / -0.57 | -0.37 / -0.38 |

`design_ratio_lag2` behaves the same way: primary -0.78 -> -0.41 to -0.72;
drop -0.16 -> -0.20 to -0.32; bucket -0.23 -> -0.29 to -0.42. Lags 3-4 stay
small and unstable in sign, as in the naive fits.

**Reading:**
- **The sign of the short-lag effect is stable under every correction.**
  lag1 and lag2 stay negative in all 12 extrapolations (3 units x 2 error
  models x 2 extrapolants). The measurement error is not creating or reversing
  the direction of the association.
- **The magnitude is not stable, and the two units move in opposite
  directions.** At the component unit, adding misclassification shrinks the
  coefficient toward zero (classic attenuation: component-quarters are small,
  so per-ticket noise dominates), and SIMEX roughly doubles it. At the project
  unit, adding misclassification *grows* the coefficient at first (the scale
  effect from §8b dominates, because project-quarters are large and averaging
  removes most per-ticket noise), and SIMEX shrinks it. The two effects
  partly converge: about -0.6 to -0.9 (project) vs -0.5 to -0.6 (component,
  quadratic).
- **The project-unit extrapolation is unreliable.** Its lambda curve is
  non-monotone (it peaks at lambda 0.5-1), so quadratic and linear disagree by
  about 2x. The quadratic estimate (-0.61 to -0.91) agrees with §8b's
  scale-only correction (-0.69). The linear estimate barely moves from the
  naive value.
- **Effect size, illustrative only (H1 is not significant after Holm):** a
  corrected lag1 of about -0.5 to -0.9 means +10 percentage points of true
  design share goes with roughly 5-9% fewer bugs the next quarter.
- **Not done (deviation from pre-registration §8, recorded in §11):** SIMEX
  covers M1 only, not M2. M2's change-point feature would have to be
  re-detected for every replicate, and H2 is null in every cell. There are no
  SIMEX standard errors. "Each project's error rate" is implemented as
  per-project specificity only, because the audit has too few positives per
  project to estimate per-project sensitivity.
