# Collapsing RQ2 and RQ3 into a Single Research Question

**Status:** proposal, 2026-09-20. Supersedes the separate RQ2 / RQ3 framing in the
March 2026 dissertation proposal colloquium deck.

## 1. Current wording (colloquium, March 2026)

- **RQ2** — "Based on the ability to detect design issues, can a prediction be made
  as to a change in defect rate?"
- **RQ3** — "Does the temporal distribution of architectural work within a project's
  life cycle predict changes in defect rates?"

These already share a dependent variable: *change in defect rate*. As written they
are not two studies. They are one study run with and without temporal features.
Testing them separately means two hypothesis tests on the same panel, the same
outcome, and the same labels — double-dipping that inflates false positives while
leaving each test underpowered.

Note that the earlier manuscript framing of RQ3 was broader ("whether the temporal
distribution of architectural effort influences project quality"). The colloquium
version had already narrowed it to defect rate. The merge below keeps that narrower
outcome.

## 2. Proposed single question

> **RQ2: Does design work detected in JIRA tickets predict subsequent change in a
> project's defect rate, and does the temporal distribution of that work improve
> prediction beyond its volume alone?**

Short form for slides and abstract:

> *Does design work predict defect rates — and does* when *it happens matter beyond*
> how much*?*

### Hypotheses

One outcome, three nested hypotheses:

| | Hypothesis | Replaces |
|---|---|---|
| **H1** | Design-work volume in window *t* predicts defect rate at *t+k*, net of activity level and past defects | old RQ2 |
| **H2** | Temporal-distribution features add predictive power over volume alone | old RQ3 |
| **H3** (rival) | Defects at *t* predict design work at *t+k* — the "crisis-driven" pattern | new |

H3 is the addition worth making. The colloquium deck lists crisis-driven as a third
temporal archetype to cluster. It is more useful as the rival explanation that has to
be ruled out, because it produces exactly the lag-1 correlation H1 predicts, only in
reverse. Converting it from an archetype into a falsification test is what makes the
merged question stronger than the two it replaces.

### What the merge costs

The Waterfall-vs-Agile claim becomes an interpretation in the discussion rather than
a directly tested result, because "project quality" is now operationalized solely as
defect rate. State this explicitly in the chapter rather than leaving the committee
to notice it. The dissertation still delivers all four contributions listed on the
colloquium's contributions slide.

## 3. Approach

### Unit of analysis

Project × time window (calendar quarter), and project × component × window if
component data exists (see blocker 1 below). Within-project comparison is what
controls for team, domain, and process.

### Variables

- **Predictor** — design ratio = design tickets / all tickets in window, dated by
  *resolution* date (when the work happened), not creation date.
- **Outcome** — bug count in window *t+k*, dated by creation date (discovery).
- **Exposure** — `log(non-bug issues in window)` as a model offset. This is
  essential. Without it, a busy quarter drives both sides of the regression and
  manufactures a positive result.
- **Controls** — lagged bug count, project fixed effects, calendar-time trend,
  project age.
- **Temporal features** — Gini / concentration of cumulative design effort to date,
  front-loadedness index (share of design work in the first third of observed life),
  distance to nearest design change-point.

### Model

Negative binomial GLM with log-exposure offset, project fixed effects, and
cluster-robust standard errors by project. Distributed lag (1–4 quarters) estimated
in a single specification rather than selecting a winning lag.

### Test

Nested model comparison on the same outcome:

- **M0** — controls only
- **M1** — M0 + design volume
- **M2** — M1 + temporal features

Compare by likelihood-ratio test and by out-of-sample MAE under forward-chaining
(blocked-by-time) cross-validation, plus a permutation null. H2 is exactly the
M1 → M2 increment. Also fit a standalone timing model so that a null H1 does not
strand H2.

## 4. Critical review — where this fails if unaddressed

### 1. Component / application-area availability is a blocker, not a detail

`tawos_connector.py:14-18` documents the Issue table as ID, Issue_Key, Title,
Description, Description_Text, Type, Priority, Status, Resolution, Creation_Date,
Project_ID — no component field. The exported CSVs in `ieee_dataport/` have no
component column either. Both the localized-impact argument and the within-project
control depend on it. **Verify against the live TAWOS MySQL schema before anything
else.** If it is absent, the fallbacks (issue-key prefix, labels, text clustering)
stack a second layer of measurement error on top of the classifier's.

### 2. Attenuation from classifier error

The predictor is a model output at LOPO F1 ≈ 0.73, with a per-project range of
0.45–0.81 and a known recall bias. That is not classical noise — it is error that
varies systematically by project, so it biases toward the null unevenly. A null
result will be dismissed as measurement error unless it is pre-empted: estimate
per-project sensitivity and specificity from the LOPO folds, apply a Rogan–Gladen
prevalence correction to the design ratio, and run SIMEX as a sensitivity check.
Report corrected and uncorrected estimates side by side.

### 3. Freeze the label generator first

The seed sweep found Base+Meta (0.7425 ± 0.0134) beats the published
Base+Meta+SqrtW, and the published 0.7325 was a winner's-curse pick across ~14
configurations. Do not inherit that instability. Pick one configuration, ensemble
the five seeds' predictions (which measurably reduces label noise and directly buys
statistical power), version the label set, and do not regenerate it mid-analysis.

### 4. Calibration outside the 10 labeled projects

Per-project error estimates exist for 10 projects only. Applying the classifier to
all 39 TAWOS projects is out-of-distribution with unknown sensitivity. Either
restrict to the 10, or hand-label ~100 tickets in each of several new projects to
estimate sensitivity and specificity there. Restricting to 10 is clean but leaves
n = 10 clusters, which is thin for panel inference.

### 5. Researcher degrees of freedom

Window length, lag, change-point sensitivity, exposure definition, and inclusion
rules generate dozens of defensible specifications — and this project has already
been burned once by selecting on results. Pre-register the specification grid,
report a specification curve over all of it, and apply Holm correction. Cheap to do,
and it is what makes the result credible.

### 6. Front-loading may be unobservable

JIRA adoption usually postdates initial architecture work, so "early in the data" is
not "early in the project's life." The exported project files span from 7 months
(DM) to 5 years (NEXUS) and are capped at 3,000 issues each, so full database pulls
are required, along with a pre-specified inclusion rule (for example ≥ 12 quarters
and ≥ 50 bugs). Define t = 0 explicitly and test sensitivity to that definition.

### 7. Construct validity of "defect rate"

TAWOS provides bug *reports*, not defects, and carries no commit or LOC data. A rate
per unit of issue activity is a proxy for reporting intensity as much as for quality.
State this as a limitation, and consider linking the subset of projects whose GitHub
repositories can be resolved.

### 8. Plan for a null result

The honest prior is a small or null effect. With a pre-registered design, an exposure
offset, measurement-error correction, and a power analysis with equivalence bounds, a
null is a publishable finding against the unmeasured industry claim the proposal opens
with. Without those, a null is unpublishable and indistinguishable from attenuation.

## 5. Next steps, in order

1. Verify the TAWOS component / version schema — this gates the whole design.
2. Freeze the label generator (Base+Meta, 5-seed ensemble) and emit a versioned label
   set for all in-scope projects.
3. Compute per-project sensitivity and specificity from the existing LOPO folds;
   decide 10-project vs. 39-project scope on that basis.
4. Write the pre-registration (specification grid, inclusion rules, power analysis)
   before touching the outcome data.
5. Build the panel; run M0 / M1 / M2 plus the reverse-direction H3 model.
