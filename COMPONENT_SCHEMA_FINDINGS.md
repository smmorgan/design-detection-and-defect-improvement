# TAWOS Component/Version Schema Verification (RQ2 blocker 1)

**Status:** resolved, 2026-09-27. Answers `paper/RQ2_COLLAPSED_PLAN.md` §4 blocker 1 and
next-step 1.

## Summary

Component data **is available** in the live TAWOS MySQL database and does not require
a text-extraction fallback. `tawos_connector.py`'s docstring schema (Issue table only)
is incomplete, not wrong — components live in two separate tables the connector never
queries.

## What exists

Local MySQL instance (`tawos` database, credentials via `TAWOS_DB_*` env vars) has 14
tables, including three the connector code doesn't touch:

- `Component` (ID, Jira_ID, Name, Description, Project_ID) — 2,001 rows across 39 projects
- `Issue_Component` (Issue_ID, Component_ID) — many-to-many join table, 366,922 rows
- `Version` / `Affected_Version` / `Fix_Version` — release-version data with `Release_Date`

Join path: `Issue.ID -> Issue_Component.Issue_ID -> Issue_Component.Component_ID ->
Component.ID`. An issue can have 0, 1, or several components (avg 1.04–1.26 per linked
issue across the 10 in-scope projects; max 19 for one DM outlier).

## Coverage, the 10 in-scope projects (CONFSERVER, DM, DNN, FAB, JRASERVER, MESOS, MULE,
NEXUS, SERVER, TIMOB)

| Project | n_issues | % with ≥1 component | n distinct components |
|---|---|---|---|
| CONFSERVER | 42,324 | 45.9% | 104 |
| DM | 26,506 | 57.4% | 259 |
| DNN | 10,060 | 86.6% | 143 |
| FAB | 13,682 | 74.6% | 26 |
| JRASERVER | 44,165 | 62.7% | 115 |
| MESOS | 10,157 | 56.1% | 42 |
| MULE | 11,816 | 90.6% | 129 |
| NEXUS | 9,912 | 65.3% | 91 |
| SERVER | 48,663 | 82.6% | 37 |
| TIMOB | 22,059 | 94.9% | 25 |

Database-wide: 39 projects, 458,232 issues, 320,991 (70.0%) linked to at least one
component.

## Quality checks

- **No null/empty names**: 0 rows in `Component` with NULL or blank `Name`.
- **Near-zero duplication**: only 6 case-insensitive duplicate name collisions across
  the whole database (e.g. two rows both named "Ecosystem" in CONFCLOUD), all outside
  the 10 in-scope projects.
- **Names are real architectural subsystems**, not junk labels — e.g. SERVER:
  Sharding/Replication/Querying/Storage; TIMOB: iOS/Android/Windows/MobileWeb; MULE:
  Core/Transport/Modules/Extensions API; FAB: fabric-peer/fabric-orderer/fabric-ledger.
  This is exactly the granularity the plan's "within-project control" and
  "localized-impact" arguments need.
- **Missingness is not strongly concentrated in design-relevant issue types.** Checked
  CONFSERVER (lowest overall coverage, 45.9%) by `Type`: Bug 47.7% vs. Suggestion 43.3%
  — close enough that component presence isn't obviously confounded with the
  design/non-design label. DM is the exception: Epic 36.7% vs. Milestone 95.5%, but
  Epics/Milestones are a small share of DM's issues and not the categories driving the
  design classifier's positive/negative split (Story/Bug/Improvement all sit at
  54–87%).

## What's weaker: Fix_Version coverage

Checked as a secondary temporal-anchor candidate (release-based windows instead of
calendar quarters). This is much less reliable than Component:

| Project | % issues with a Fix_Version | Versions with a Release_Date |
|---|---|---|
| CONFSERVER | 26.8% | 612 |
| DM | 0.0% | 3 |
| DNN | 57.3% | 28 |
| FAB | 51.9% | 30 |
| JRASERVER | 23.3% | 469 |
| MESOS | 47.6% | 78 |
| MULE | 76.8% | 212 |
| NEXUS | 53.2% | 160 |
| SERVER | 64.8% | 402 |
| TIMOB | 58.4% | 529 |

DM effectively has no usable Fix_Version/release-date data. Don't build release-based
time windows into the pre-registration; stick with calendar quarters as the plan
already proposes.

## Recommendation for RQ2 next steps

1. Blocker 1 is cleared — component data supports the project × component × window
   unit of analysis for 9 of the 10 projects at reasonable coverage (56–95%); CONFSERVER
   is the weak case at 46%.
2. No text-extraction fallback (issue-key prefix, labels, clustering) is needed as a
   primary source. It may still be worth a light validation pass — pick ~20 issues per
   project, confirm the assigned `Component.Name` matches what a human would infer from
   the title/description — before trusting it as ground truth, but this is a spot-check,
   not a rebuild.
3. Missing-component issues (5–54% depending on project) need an explicit handling rule
   in the pre-registration: drop them, bucket them as "unclassified," or impute via
   text similarity to labeled components. This should go in the specification grid
   (plan §4 item 5) rather than being decided ad hoc.
4. Do not use Fix_Version/release dates as the time axis; coverage is too inconsistent
   (0% for DM) to support it across all 10 projects.
