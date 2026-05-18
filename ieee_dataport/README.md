# Design Mining Dataset for JIRA Issue Classification

## Overview

This dataset supports the study of automated design issue detection in open-source software projects. It contains manually labelled JIRA issues, heuristically labelled issues used for training augmentation, and raw JIRA issues from the TAWOS corpus used as the unlabelled pool during Leave-One-Project-Out (LOPO) cross-validation experiments.

**Source projects:** CONFSERVER, DM, DNN, FAB, JRASERVER, MESOS, MULE, NEXUS, SERVER, TIMOB  
**TAWOS source:** https://github.com/SOLAR-group/TAWOS

---

## Dataset Structure

```
├── manual_labels/
│   └── manually_labelled_issues.csv        # 1,786 human-labelled JIRA issues
├── heuristic_labels/
│   ├── heuristic_design_labels.csv         # 4,989 heuristically labelled design issues
│   └── heuristic_nondesign_labels.csv      # 4,995 heuristically labelled non-design issues
└── tawos_project_issues/
    ├── CONFSERVER.csv                       # 3,000 issues
    ├── DM.csv                              # 3,000 issues
    ├── DNN.csv                             # 3,000 issues
    ├── FAB.csv                             # 3,000 issues
    ├── JRASERVER.csv                       # 3,000 issues
    ├── MESOS.csv                           # 3,000 issues
    ├── MULE.csv                            # 3,000 issues
    ├── NEXUS.csv                           # 3,000 issues
    └── TIMOB.csv                           # 3,000 issues
```

---

## File Descriptions

### 1. `manual_labels/manually_labelled_issues.csv`

The primary labelled dataset. All 1,786 issues were manually reviewed and assigned a binary label by the authors.

| Column | Description |
|--------|-------------|
| `project` | JIRA project key |
| `issue_key` | Unique issue identifier (e.g. CONFSERVER-12345) |
| `issue_type` | JIRA issue type (Bug, Story, Task, Epic, etc.) |
| `summary` | Issue title |
| `description` | Issue description (JIRA markup cleaned) |
| `label` | Binary label: `design` or `non-design` |
| `label_source` | Always `human` for this file |

**Label distribution:**

| Label | Count | % |
|-------|------:|--:|
| design | 496 | 27.8% |
| non-design | 1,290 | 72.2% |
| **Total** | **1,786** | |

**Per-project breakdown:**

| Project | n | Design | Non-design |
|---------|--:|-------:|-----------:|
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

---

### 2. `heuristic_labels/heuristic_design_labels.csv`

4,989 JIRA issues heuristically identified as design-related using a keyword scoring scheme. All issues were reviewed and assigned a review action before use in augmentation. Issues marked `REMOVE` were excluded from training.

| Column | Description |
|--------|-------------|
| `project` | JIRA project key |
| `issue_key` | Unique issue identifier |
| `issue_type` | JIRA issue type |
| `score` | Keyword-based design score (strong keywords +3, medium +2, weak +1) |
| `score_reason` | Breakdown of matched keywords (e.g. `+3:architecture; +2:refactor`) |
| `review_action` | Manual review decision: `KEEP`, `REMOVE`, or `REASSESS` |
| `review_note` | Reviewer annotation |
| `summary` | Issue title |
| `description` | Issue description |

**Keyword scoring tiers:**

- **Strong (+3):** architecture, microservices, api gateway, event-driven, schema design, domain model, design pattern, oauth, grpc, graphql, kafka, distributed, scalability, ...
- **Medium (+2):** refactor, redesign, sdk, public api, rest api, authentication, authorization, migration, caching, rate limit, circuit breaker, ...
- **Weak (+1):** api, endpoint, service, database, interface, plugin, module, component, framework

---

### 3. `heuristic_labels/heuristic_nondesign_labels.csv`

4,995 JIRA issues heuristically identified as non-design, stratified by confidence tier.

| Column | Description |
|--------|-------------|
| `project` | JIRA project key |
| `issue_key` | Unique issue identifier |
| `issue_type` | JIRA issue type |
| `design_score` | Keyword-based design score (0 for all issues in this file) |
| `confidence_tier` | Confidence level: 1 = highest confidence non-design, higher = lower confidence |
| `n_reason` | Reason for non-design classification (e.g. Bug, UI signal, stack trace present) |
| `summary` | Issue title |
| `description` | Issue description |

---

### 4. `tawos_project_issues/{PROJECT}.csv`

Nine files, one per TAWOS project, each containing 3,000 JIRA issues. These were used as the unlabelled pool during Leave-One-Project-Out cross-validation. Issues are sourced directly from the TAWOS dataset with JIRA markup cleaned.

| Column | Description |
|--------|-------------|
| `issue_key` | Unique issue identifier |
| `project` | JIRA project key |
| `issue_type` | JIRA issue type |
| `summary` | Issue title |
| `description` | Issue description |
| `status` | Issue status (Open, In Progress, Resolved, Closed, etc.) |
| `priority` | Priority level (High, Medium, Low, etc.) |
| `created` | Issue creation timestamp |
| `updated` | Last update timestamp |
| `comments` | Concatenated comment text |

---

## Labelling Methodology

### Manual Labels

Issues were selected from the ten TAWOS projects and manually reviewed by the authors. A design issue is defined as one primarily concerned with software architecture, API design, system structure, or design decision-making, rather than bug fixes, UI changes, documentation, or implementation tasks.

### Heuristic Design Labels

A keyword scoring scheme was applied across the TAWOS corpus. Issues exceeding a minimum score threshold were flagged as candidate design issues, then manually reviewed. Exclusion patterns were applied to remove version upgrades, documentation tasks, deployment issues, UI/cosmetic issues, and issues containing stack traces.

### Heuristic Non-Design Labels

Issues with a design score of zero were stratified into confidence tiers based on negative signals: presence of stack traces, bug-indicative issue types, and UI/cosmetic keywords. Tier 1 represents the highest-confidence non-design issues.

---

## Preprocessing

Applied to all text fields:
- JIRA markup removed (headings, bold, italic, code blocks, tables, link syntax)
- URLs and email addresses removed
- Boilerplate sections removed (Steps to Reproduce, Expected Results, Actual Results, Workaround, Environment)
- Whitespace normalised
- Issues with fewer than 7 meaningful words (after software-specific stop word removal) excluded
- Automatically generated issues excluded

---

## Source Data Attribution

The JIRA issues in this dataset are derived from the TAWOS dataset:

> Oriol, M., Borges, H., Tuya, J., & Harman, M. (2022). TAWOS: A Dataset of Agile Open-Source Software Projects. *IEEE/ACM International Conference on Mining Software Repositories (MSR)*.  
> https://github.com/SOLAR-group/TAWOS

The Stack Overflow pre-training data used in the associated models is from:

> Mahadi, N., Ernst, N. A., & Tongay, Y. (2021). An Empirical Study of Design-Related Issues in Open Source Software Projects. *IEEE Transactions on Software Engineering*.

---

## License

The manually assigned labels and heuristic label annotations are released under **CC BY 4.0**. The underlying issue text is derived from publicly available open-source project issue trackers via the TAWOS dataset; please refer to the TAWOS repository for its licensing terms.

---

## Citation

If you use this dataset, please cite:

```
[Citation to be added upon publication]
```
