# Design Mining Dataset for JIRA Issue Classification

Labeled JIRA issues for detecting architectural design work in open-source
software projects. The dataset accompanies the article *Design Mining in
Software Projects Using Transformer Models: A Cross-Project Evaluation on JIRA
Tickets* (Morgan and Daniels, submitted to *Information and Software
Technology*).

It contains:

1. **Manual labels:** 1,786 JIRA issues from ten TAWOS projects, each labeled
   `design` or `non-design` by the primary annotator. These are the ground
   truth for every metric reported in the article.
2. **Inter-rater study:** a 250-issue stratified subsample labeled
   independently by a second annotator, with a blinded third-party
   adjudication of all 68 disagreements.
3. **Heuristic labels:** 9,984 issues labeled by a keyword-scoring heuristic,
   used only as training augmentation.

All issue text comes from the TAWOS dataset (see *Source data and license*).

## Structure

```
design_mining_jira_dataset/
├── README.md
├── manual_labels/
│   └── manually_labelled_issues.csv        # 1,786 primary labels
├── inter_rater/
│   ├── inter_rater_sample_250.csv          # all 250 issues: primary, second and final labels
│   ├── adjudication_results.csv            # the 68 adjudicated disagreements
│   ├── agreement_summary.json              # agreement and adjudication statistics
│   └── adjudication_rubric.md              # definition and procedure used to adjudicate
└── heuristic_labels/
    ├── heuristic_design_labels.csv         # 4,989 heuristic design labels
    └── heuristic_nondesign_labels.csv      # 4,995 heuristic non-design labels
```

## 1. `manual_labels/manually_labelled_issues.csv`

| Column | Description |
|---|---|
| `project` | JIRA project key |
| `issue_key` | Issue identifier (e.g. `CONFSERVER-12345`) |
| `issue_type` | JIRA issue type (Bug, Story, Task, Improvement, ...) |
| `summary` | Issue title |
| `description` | Issue description (JIRA markup cleaned) |
| `label` | `design` or `non-design` |
| `label_source` | Always `human` |

**Label distribution:** 496 design (27.8%), 1,290 non-design (72.2%).

| Project | n | Design | Non-design |
|---|--:|--:|--:|
| CONFSERVER | 179 | 60 | 119 |
| DM | 181 | 47 | 134 |
| DNN | 180 | 45 | 135 |
| FAB | 178 | 52 | 126 |
| JRASERVER | 183 | 52 | 131 |
| MESOS | 177 | 55 | 122 |
| MULE | 182 | 56 | 126 |
| NEXUS | 183 | 52 | 131 |
| SERVER | 162 | 32 | 130 |
| TIMOB | 181 | 45 | 136 |

The labels in this file are the original, unadjudicated primary labels used to
train and evaluate every model in the article. They are left unchanged so that
the reported results can be reproduced. The adjudicated corrections for the
250-issue subsample are in `inter_rater/`.

**Labeling definition.** Design issues involve the documentation or creation
of architecture, refactoring for project-level structural improvement, or the
design of new features. Non-design issues include bug fixes, code cleanup,
UI-only changes, documentation, operational work, and implementation tasks
with clear scope.

## 2. Inter-rater study (`inter_rater/`)

### `inter_rater_sample_250.csv`

A project-stratified subsample of 250 issues from the manual label set. Each
issue was labeled independently by a second annotator who did not see the
primary labels. Every disagreement was then resolved by a third-party
adjudicator who had no other role in the study, was blinded to which annotator
gave which label, and applied the written definition in
`adjudication_rubric.md`.

| Column | Description |
|---|---|
| `project`, `issue_key`, `issue_type`, `summary`, `description` | As above |
| `primary_label` | Primary annotator's label (identical to `manual_labels/`) |
| `second_label` | Second annotator's label |
| `adjudicated_label` | Adjudicator's label; filled only where the two annotators disagreed (68 rows) |
| `adjudication_note` | Adjudicator's short reason for the decision |
| `final_label` | Consensus label: the shared label where the annotators agreed, otherwise the adjudicated label |

### `adjudication_results.csv`

The 68 disagreements on their own, one row per adjudicated issue, with the
same columns as `inter_rater_sample_250.csv` minus `final_label` (for these
rows the final label is the `adjudicated_label`).

| Adjudicated label | Issues | Most common reasons recorded |
|---|--:|---|
| design | 17 | data or schema design (8), data contract (4), new API (2), authentication/authorization, component boundary, architectural contract (1 each) |
| non-design | 51 | documentation, bug fix, UI only, operational (CI/CD, deployment, version upgrade), test changes, small API change |

### Key results (`agreement_summary.json`)

| Measure | Value |
|---|---|
| Raw agreement, primary vs. second | 72.8% (182/250) |
| Cohen's κ, primary vs. second | 0.436 |
| Design rate: primary / second / final | 27.6% / 46.0% / 30.0% |
| Disagreements where only the second annotator said design | 57 of 68 |
| Adjudication sided with primary / second | 50 / 18 of 68 |
| Agreement with final labels: primary | 92.8% (κ = 0.825) |
| Agreement with final labels: second | 80.0% (κ = 0.587) |
| Primary labels overturned | 18 of 250 (12 to design, 6 to non-design; binomial p = 0.24) |

## 3. Heuristic labels (`heuristic_labels/`)

Issues drawn from curated project exports and an unfiltered TAWOS pull
(27 projects in total), labeled by a keyword-scoring heuristic. They were used
only to augment training data; none appear in any evaluation set. The article
reports that this augmentation gave no reliable improvement. A manual audit of
100 sampled high-confidence heuristic labels found 96% agreement with a human
reviewer.

### `heuristic_design_labels.csv` (4,989 issues)

| Column | Description |
|---|---|
| `project`, `issue_key`, `issue_type`, `summary`, `description` | As above |
| `score` | Keyword design score (minimum 5 for inclusion) |
| `score_reason` | Matched keywords and their weights (e.g. `+3:architecture; +2:refactor`) |
| `review_action` | Curation outcome; all rows in this release are `KEEP` |
| `review_note` | Curation note |

Issues were first filtered by hard exclusions (Bug, Sub-task, Test and
Technical task types; version upgrade or migration language;
documentation-only changes; Java stack traces or logging requests; deployment;
UI-dominated summaries). Remaining issues were scored with 37 strong
architectural phrases (+3), 31 medium phrases (+2) and 10 weak single-word
signals (+1), plus +2 or +3 for feature, enhancement and story issue types,
and −2 each for narrowly scoped API or UI changes.

### `heuristic_nondesign_labels.csv` (4,995 issues)

| Column | Description |
|---|---|
| `project`, `issue_key`, `issue_type`, `summary`, `description` | As above |
| `design_score` | Keyword design score (0 for every row) |
| `confidence_tier` | 1 = highest-confidence non-design (4,953 rows); 2 and 3 are lower confidence |
| `n_reason` | Reason for the non-design label (e.g. Bug type, UI signal, stack trace) |

## Preprocessing

Applied to all text fields:

- JIRA markup removed (headings, emphasis, code blocks, tables, link syntax)
- URLs and email addresses removed
- Boilerplate sections removed (Steps to Reproduce, Expected Results, Actual Results, Workaround, Environment)
- Whitespace normalized
- Issues with fewer than 7 meaningful words (after software-specific stop-word removal) excluded
- Automatically generated issues excluded

## Source data and license

All issue text is derived from the TAWOS dataset, which is released under the
Apache License 2.0:

> V. Tawosi, A. Al-Subaihin, R. Moussa, F. Sarro. A Versatile Dataset of Agile
> Open Source Software Projects. *Proceedings of the 19th International
> Conference on Mining Software Repositories (MSR '22)*, 2022, pp. 707–711.
> Dataset: https://doi.org/10.5522/04/21308124 ·
> https://github.com/SOLAR-group/TAWOS

The labels, adjudication data, heuristic scores and documentation added in
this release are licensed under **CC BY 4.0**. Use of the underlying issue
text remains subject to the TAWOS license.

The Stack Overflow data used to pre-train the article's models is not
included here; it is available from:

> A. Mahadi, N. A. Ernst, K. Tongay. Conclusion stability for natural language
> based mining of design discussions. *Empirical Software Engineering* 27(1),
> 2021. https://doi.org/10.1007/s10664-021-10009-1

## Citation

If you use this dataset, please cite it and the accompanying article
(article citation to be added upon publication):

> S. Morgan, J. Daniels. Manually Labelled and Adjudicated JIRA Tickets for Design Mining
> [dataset]. Zenodo, 2026. https://doi.org/10.5281/zenodo.23283811
