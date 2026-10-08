# Qualitative Error Analysis Findings (2026-09-23)

Qualitative classification error analysis for R1.3 (`REVIEWER_FEEDBACK.md`).
Companion to `RESUBMISSION_GUIDE.md`, `SEED_VARIANCE_FINDINGS.md`, and
`INTER_RATER_FINDINGS.md`. Per the standing instruction, this describes
*where* the manuscript needs edits — it does not edit `paper/manuscript.tex`.

Artifacts: `analyze_lopo_errors.py` (`sample-errors` subcommand),
`gcp_results/error_analysis/coding_sheet_base_meta_r13.csv` (65 rows, hand-coded).
Cross-referenced against `gcp_results/error_analysis/project_characteristics.csv`
and `variance_summary.json`, the outputs of the same script's `variance`
subcommand already used for R2.6.

**Note on scope:** an earlier, uncoded sample (`coding_sheet_roberta_best.csv`,
98 rows, all `error_category` blank) was drawn from the previously-reported
"best" config (Base+Meta+SqrtW) on 2026-09-20, before the seed sweep
(`SEED_VARIANCE_FINDINGS.md`) established that plain **Base+Meta** is actually
the stronger config. That sheet was superseded rather than coded — the
analysis below uses the newer, fully-coded sample instead. `coding_sheet_roberta_best.csv`
can be deleted; it contributes nothing not superseded by `coding_sheet_base_meta_r13.csv`.

## 0. Sample construction

Errors were pulled from `gcp_results/seed_sweep/base_meta_seed42_results_predictions.csv`
(Base+Meta, seed 42) for the three projects already singled out in prior
analysis as the LOPO-difficulty extremes: **CONFSERVER** (best, F1=0.811),
**DM** (middle, F1=0.680), and **SERVER** (worst, F1=0.449).

Coverage is exhaustive for two of the three projects:

| Project | Total FP | Total FN | FP coded | FN coded | Coverage |
|---|---|---|---|---|---|
| CONFSERVER | 11 | 7 | 11 | 7 | **100%** |
| SERVER | 7 | 12 | 7 | 12 | **100%** |
| DM | 29 | 13 | 15 | 13 | FN 100%, FP 52% (random sample) |

Every CONFSERVER and SERVER misclassification in this fold was reviewed, not
a subsample — the counts below for those two projects are a census, not an
estimate. DM's 29 false positives were subsampled to 15; DM's false-negative
column is exhaustive.

## 1. Headline: over half of the model's "errors" are label-quality issues, not model failures

| Category | n | FN | FP |
|---|---|---|---|
| LABEL_AMBIGUOUS | 25 | 13 | 12 |
| DESIGN_VOCAB_IMPL | 10 | 0 | 10 |
| LABEL_ERROR | 9 | 7 | 2 |
| SPARSE_TEXT | 7 | 5 | 2 |
| PROJECT_JARGON | 7 | 1 | 6 |
| IMPLICIT_DESIGN | 6 | 6 | 0 |
| OTHER | 1 | 0 | 1 |

**LABEL_AMBIGUOUS + LABEL_ERROR = 34/65 (52.3%)** of sampled errors — a
majority of what looks like model failure is really a ground-truth problem:
either the ticket is genuinely borderline (a second rater could reasonably
disagree, `LABEL_AMBIGUOUS`) or the gold label looks outright wrong on
re-reading (`LABEL_ERROR`). This is the same pattern R2.5's inter-rater study
found from a completely different angle (κ=0.436, moderate agreement) — two
independent analyses now point at label quality, not model quality, as the
dominant limitation. Worth stating explicitly as a converging finding in the
manuscript rather than two disconnected limitations.

`LABEL_ERROR` breaks down into two concrete, checkable sub-patterns:

- **Boilerplate/deployment tickets mislabeled design** (DM-24183, DM-24182,
  DM-24185: near-identical auto-generated "Deploy for X" tickets). These
  directly contradict `LABELLING_METHODOLOGY.md`'s own hard-exclusion rule
  for deployment/version-upgrade tickets — i.e., a **pipeline bug**, not
  annotator disagreement: 3 of the 9 `LABEL_ERROR` rows are the same
  mechanical mistake, and the fix (re-audit gold labels against the
  documented hard-exclusion rules) is well-defined.
- **Genuine design work mislabeled non-design** (DM-27167 "Stop using
  filter alias names in DM stack" — references an RFC and a stack-wide
  name-standardization plan; DM-27253 — explicit schema and API changes to
  the query system). The model's "false positive" here is arguably correct;
  the gold label is the error.

## 2. Category vs. error direction: each failure mode has a distinct signature

- **`DESIGN_VOCAB_IMPL` is 100% false positives (10/10).** Implementation or
  bug-fix tickets that happen to use design-sounding words (API, schema,
  refactor) get pulled toward "design" — exactly the failure mode the
  attribution analysis (R2.4) already flags at the vocabulary level; this is
  its ticket-level signature.
- **`IMPLICIT_DESIGN` is 100% false negatives (6/6).** The mirror image:
  genuine design work described in terse, jargon-free, or domain-specific
  language that lacks the keyword surface the model (and the keyword
  heuristic used for augmentation labels) relies on.
- **`PROJECT_JARGON` skews false positive (6/7)**, and only in CONFSERVER/DM
  (0 in SERVER) — project-specific terms push predictions toward "design"
  when a project uses them heavily, consistent with `project_characteristics.csv`
  showing CONFSERVER/DM's higher `oov_token_rate` and `jsd_all_vs_rest` than
  the pooled corpus.
- **`SPARSE_TEXT` skews false negative (5/7)** — under-described tickets
  default toward "non-design" rather than triggering false alarms.

## 3. Confidence pattern: the model's most confident errors are disproportionately label errors, not model errors

Mean `margin` (distance from the fold's decision threshold) by category:

| Category | mean margin | n |
|---|---|---|
| OTHER | 0.404 | 1 |
| LABEL_ERROR | **0.380** | 9 |
| LABEL_AMBIGUOUS | 0.335 | 25 |
| PROJECT_JARGON | 0.311 | 7 |
| DESIGN_VOCAB_IMPL | 0.309 | 10 |
| IMPLICIT_DESIGN | 0.246 | 6 |
| SPARSE_TEXT | **0.168** | 7 |

`LABEL_ERROR` has the *highest* mean confidence margin of any category — when
the model is most confidently "wrong," it is disproportionately because the
gold label is wrong, not because the model misjudged the ticket. `SPARSE_TEXT`
sits right at the threshold (lowest margin) — under-described tickets produce
genuinely uncertain predictions, which is the expected, benign behavior.
This is a useful, non-obvious framing for the manuscript: **confidently-wrong
predictions are a good place to look for label noise**, not necessarily model
weakness.

## 4. Per-project pattern ties directly into R2.6 (LOPO variance)

| Project | F1 (Base+Meta) | Dominant error categories | Vocabulary signal (`project_characteristics.csv`) |
|---|---|---|---|
| SERVER | 0.449 (worst) | `LABEL_AMBIGUOUS` (9), `LABEL_ERROR` (3), `SPARSE_TEXT` (3) | Highest `empty_description_rate` (21.6%, next-highest project is 2.3%) and highest `design_with_zero_keywords_rate` (40.6%) of all 10 projects; lowest `keyword_score_auc` (0.680) among the three sampled |
| DM | 0.680 (middle) | `LABEL_AMBIGUOUS` (10), `LABEL_ERROR` (6), `PROJECT_JARGON` (4) | Second-highest `oov_token_rate` (0.155) and `jsd_all_vs_rest` (0.304) among all 10 projects — vocabulary shift, not sparse text, is the driver here |
| CONFSERVER | 0.811 (best) | `LABEL_AMBIGUOUS` (6), `DESIGN_VOCAB_IMPL` (5) | Best `keyword_score_auc` (0.774) of the three; errors are less about missing signal than about legitimately hard cases |

This gives R2.6's "project-specific language is the primary barrier"
narrative a concrete mechanism per project instead of one blanket claim:
SERVER's problem is sparse/under-described tickets (a data problem, largely
unfixable by modeling), DM's is genuine vocabulary shift (the classic
domain-adaptation story), and CONFSERVER's residual errors are mostly
legitimate ambiguity rather than either. **Recommend writing R1.3 and R2.6 as
one combined subsection**, exactly as `RESUBMISSION_GUIDE.md` §4 already
suggested before this analysis existed.

## 5. What this means for the manuscript

1. **New subsection**, either in Results (after 4.6 "Attribution Analysis")
   or folded into 5.2 "Cross-Project Generalization Challenges" — the guide's
   original placement suggestion holds. Lead with the category breakdown
   table (§1) and the 52% label-quality finding, since it's the strongest and
   most citable number.
2. **State the sampling method precisely**: exhaustive review of every
   CONFSERVER and SERVER error (18 and 19 respectively) plus DM's 13 false
   negatives and a 15-of-29 random sample of DM's false positives, all from
   one seed (42) of the Base+Meta config. Don't imply this is a
   uniform-random sample across all 10 projects — it's a targeted census of
   the difficulty extremes, which is the right design for an error analysis
   but should be labeled as such, not generalized.
3. **Report the three category signatures in §2** as the mechanism, not just
   the tally — reviewers respond better to "vocabulary-driven FPs vs.
   sparse-text-driven FNs are distinct, identifiable failure modes" than to a
   bare category histogram.
4. **Cross-reference R2.5 explicitly**: two independent methods (a second
   human annotator, and re-reading the model's own errors) both surfaced
   label-quality problems as a major limitation. This is worth one sentence
   tying the two together in 5.4 "Threats to Validity," right next to the
   `INTER_RATER_FINDINGS.md` write-up.
5. **Flag the DM boilerplate mislabeling as an actionable data-quality bug**,
   separate from the "ambiguity is inherent" framing used elsewhere — 3 of
   65 sampled errors are DM "Deploy for X" tickets that violate the project's
   own documented hard-exclusion rule. This is fixable before resubmission
   (re-audit gold labels against `LABELLING_METHODOLOGY.md`'s exclusion
   rules) and is a stronger claim than generic "labels are noisy."
6. **Caveat to state explicitly**: n=65 errors across 3 (of 10) projects,
   one seed, one config. Consistent with `SEED_VARIANCE_FINDINGS.md`'s
   broader caution about single-seed results, treat category proportions as
   illustrative of failure *modes*, not as precise population estimates —
   don't report e.g. "52%" as if it would replicate exactly on a different
   seed or the other 7 projects.

## 6. Recommended next step (optional, not done)

If time allows: re-run `analyze_lopo_errors.py sample-errors` against one or
two of the seven un-sampled projects (e.g. NEXUS or TIMOB, mid-difficulty) to
check whether the three category signatures in §2 hold outside the two
difficulty extremes, or are specific to SERVER/DM/CONFSERVER. Lower priority
than the write-up itself — the current sample already supports every claim
in §5.
