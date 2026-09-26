# Inter-Rater Reliability Findings (2026-09-23)

Second-annotator study for R2.5 (`REVIEWER_FEEDBACK.md`). Companion to
`RESUBMISSION_GUIDE.md` and `SEED_VARIANCE_FINDINGS.md`. Per the standing
instruction, this describes *where* the manuscript needs edits — it does not
edit `paper/manuscript.tex`.

Artifacts: `analyze_inter_rater.py`, `gcp_results/inter_rater/agreement_summary.json`,
`gcp_results/inter_rater/disagreements.csv` (68 rows).

Inputs: `ieee_dataport/manual_labels/manually_labelled_sample_250.csv` (primary
annotator, the project-stratified 250-ticket sample already flagged in
`RESUBMISSION_GUIDE.md` §5 as ready for this) vs.
`ieee_dataport/manual_labels/Jeff's_manually_labelled_sample_250.tsv` (second
annotator, same 250 tickets, labels `Design`/`Not Design`). All 250 keys
matched 1:1, no missing or duplicate tickets on either side.

## 1. Headline numbers

- **Raw agreement: 72.8%** (182/250)
- **Cohen's $\kappa$ = 0.436** — "moderate" agreement (Landis & Koch), not the
  "substantial"/"almost perfect" range a reviewer would want for
  ground-truth-quality labels.
- **PABAK = 0.456** (prevalence-and-bias-adjusted kappa). PABAK > $\kappa$
  here because the two raters' design-class base rates differ sharply (see
  §2), which mechanically depresses raw $\kappa$. Report both: $\kappa$ alone
  understates how often the two raters land on the same ticket by chance-corrected
  standards, but PABAK alone would hide the prevalence problem, which is
  itself a finding.

## 2. This is not random noise — it's a systematic prevalence shift

| | Design | Non-design |
|---|---|---|
| Primary annotator | 69 (27.6%) | 181 (72.4%) |
| Second annotator | 115 (46.0%) | 135 (54.0%) |

The confusion matrix (rows = primary, cols = second annotator):

|  | Peer: Design | Peer: Not Design |
|---|---|---|
| **Mine: design** | 58 | 11 |
| **Mine: non-design** | 57 | 124 |

The disagreement is heavily one-directional: **57 tickets I labeled
non-design, the second annotator called design; only 11 went the other way.**
An exact McNemar test on this 57-vs-11 split rejects symmetry decisively
($p = 1.3\times10^{-8}$) — this is not sampling noise around a shared
threshold, it's a systematic difference in what each rater calls "design."

## 3. The disagreement concentrates in specific issue types, not randomly across the sample

Disagreement rate by JIRA issue type (of 250 total):

| Issue type | n | Disagreement rate | Mine design% | Peer design% |
|---|---|---|---|---|
| Improvement | 38 | **63.2%** | 36.8% | **100%** |
| Suggestion | 24 | **50.0%** | 50.0% | **100%** |
| Story | 32 | 31.3% | 40.6% | 65.6% |
| Task | 42 | 28.6% | 26.2% | 16.7% |
| New Feature | 10 | 20.0% | 80.0% | 100% |
| Bug | 91 | **4.4%** | 2.2% | 2.2% |

Two things stand out:

- **Bug tickets agree almost perfectly** (95.6%, both raters near-unanimously
  call them non-design) — consistent with the primary heuristic's hard
  exclusion of `Bug` type (`LABELLING_METHODOLOGY.md`), and reassuring: the
  clearest cases are genuinely clear to both raters.
- **The second annotator labeled *every single* `Improvement` and
  `Suggestion` ticket as design** (100% each, n=38 and n=24). Spot-checking
  the disagreement set (`gcp_results/inter_rater/disagreements.csv`) turns up
  tickets like "Expansion of available emojis," "Bring back issue count and
  refresh on top of screen," and "Add the ability to remove/hide the Avatar
  for Toolkit Plugin" — minor UI/feature-request tickets that don't read as
  *architectural* design work under the primary annotator's definition (the
  one `LABELLING_METHODOLOGY.md` encodes: architecture, service boundaries,
  data/schema/API design, auth, messaging, etc., with UI-only changes
  explicitly excluded).

## 4. Likely cause: no shared labeling definition was given to the second annotator

There is no annotator guide in this repo — `LABELLING_METHODOLOGY.md`
documents the *heuristic* (automated) labeling rules used for the
cherry-picked augmentation data, not instructions given to human labelers.
Nothing in the repo indicates the second annotator received the primary
annotator's working definition of "design" (architectural/structural
decisions) versus a broader everyday reading of "design" (any feature
work, including UI/UX). The 100% design-rate on Improvement/Suggestion
tickets, several of which are small UI changes, is consistent with that
broader reading. This is a **definitional/instrument problem, not
necessarily a difficulty-of-the-task problem** — it should be reported as
such rather than folded into a single "labels are somewhat subjective"
sentence.

## 5. Per-project pattern

| Project | n | Agreement | Mine design% | Peer design% |
|---|---|---|---|---|
| MESOS | 22 | 90.9% | 27% | 36% |
| CONFSERVER | 29 | 89.7% | 31% | 41% |
| MULE | 18 | 77.8% | 33% | 33% |
| TIMOB | 24 | 75.0% | 29% | 54% |
| DM | 30 | 73.3% | 40% | 67% |
| SERVER | 28 | 71.4% | 29% | 43% |
| DNN | 23 | 65.2% | 13% | 39% |
| NEXUS | 23 | 65.2% | 26% | 61% |
| FAB | 30 | 63.3% | 27% | 30% |
| JRASERVER | 23 | 56.5% | 17% | 52% |

No obvious relationship to the classifier's own per-project difficulty
ranking (SERVER is worst for the *model* but only middling here; JRASERVER
is worst for *inter-rater* agreement but not flagged as especially hard for
the model). Worth a one-line note in the writeup that these are separate
phenomena — annotator disagreement and model difficulty aren't the same
axis — rather than implying one explains the other.

## 6. What this means for the manuscript

`5.4 "Threats to Validity"` (manuscript.tex:356) currently has one sentence
flagging that "manually labeled data would be performed by committee" as an
unaddressed limitation. That sentence can now be replaced with an actual
inter-rater result, but the result is a caveat, not reassurance:

1. **Report $\kappa=0.436$ (moderate) and PABAK=0.456**, both defined, with
   the raw confusion matrix. Do not report only raw agreement (72.8% reads
   deceptively strong on its own for a binary task with imbalanced classes).
2. **State the asymmetry explicitly**: the second annotator's design rate
   (46%) is 1.7x the primary annotator's (27.6%), concentrated in
   Improvement/Suggestion tickets, and this is very unlikely to be chance
   ($p<10^{-7}$, McNemar).
3. **Name the likely cause**: absence of a shared written definition of
   "design" given to the second annotator, distinct from ordinary
   ticket-level ambiguity. This is an actionable limitation (fixable with an
   annotator guide + adjudication round) rather than an inherent property of
   the task — worth distinguishing from R2.6's project-specific-language
   argument, which is a different, harder-to-fix source of difficulty.
4. **Do not claim the label quality is validated** by this study — a $\kappa$
   of 0.436 with a systematic, identifiable-cause disagreement is a genuine
   limitation finding, and should be presented as motivation for future work
   (an adjudicated third pass, or a written annotator guide before any
   further human labeling), not as a check that passed.
5. **Where to place it**: `5.4 Threats to Validity` (line 356) for the
   limitation statement; consider also a short methods note in `3.4 Labeling
   the TAWOS Dataset` (line 119) since this bears directly on how much trust
   to place in the 1,786-ticket manual label set that the whole pipeline is
   built on.
6. **The disagreement set itself is a resource**: `gcp_results/inter_rater/disagreements.csv`
   (68 rows) is a ready-made adjudication sheet if a third pass or a
   tie-breaking rule is wanted before resubmission.

## 7. Recommended next step (not yet done)

If time allows before resubmission: pick a small sample of the 68
disagreements (or all of them, they're not large) and adjudicate with an
explicit rule (e.g., "does the primary annotator's written definition, once
made explicit, resolve this case, or is it genuinely ambiguous even under a
shared definition?"). That converts this from "two raters disagree, unclear
why" into "X% of disagreement was resolved by clarifying the definition, Y%
remained genuinely ambiguous" — a stronger and more specific claim for
`5.4`.
