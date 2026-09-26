# Adjudication Rubric — Inter-Rater Disagreement Resolution

For use with `adjudication_sheet.csv` (68 rows). Read this in full before
adjudicating the first row. Do not open `adjudication_key.csv` until every
row in the sheet has an `adjudicated_label` — it reveals which label each
original annotator gave, which would defeat the blinding.

## Why this exists

`INTER_RATER_FINDINGS.md` found 72.8% raw agreement (κ=0.436, moderate)
between two annotators on the same 250 tickets, with disagreement
concentrated almost entirely in `Improvement`/`Suggestion` tickets and
one-directional (primary annotator said non-design, second annotator said
design, 57 of 68 cases). The likely cause: no shared written definition of
"design" was ever given to the second annotator. This adjudication tests
that diagnosis directly — if an explicit definition resolves most of the 68
cases, the disagreement was definitional; if it doesn't, the ambiguity is
real and inherent to the task.

## The definition to apply

A ticket is **design** if it describes an **architectural or structural**
decision: service/component boundaries, data or schema design, API design
(new endpoints, versioning, contracts), authentication/authorization
schemes, messaging/eventing architecture, or a comparable structural
decision about how the system is built.

A ticket is **not design** if it is:
- A **UI-only** change (visual/layout/copy — buttons, styling, colors,
  wording, icons) with no structural implication.
- A **bug fix**, even a complex one, that doesn't change the system's
  structure (fixing broken behavior within an existing design, not changing
  the design).
- A **small, additive API change** (add a parameter, expose a single
  existing field, adjust an error code) that doesn't establish or change an
  architectural contract.
- Purely **operational**: deployment, CI/CD, environment configuration,
  version bumps/upgrades, documentation.

This mirrors the hard-exclusion and signal logic in
`LABELLING_METHODOLOGY.md` (written for the automated cherry-picking
heuristic, not human labeling) — apply it as a human judgment call, not a
keyword count. When in doubt, ask: *"does this ticket decide how the system
is structured, or does it operate within/adjust the surface of an existing
structure?"* The former is design; the latter is not.

## Process

1. Work through `adjudication_sheet.csv` in order (already shuffled — don't
   re-sort by issue type or project, that reintroduces the pattern-cueing
   this blinding is meant to avoid).
2. For each row, read the summary + description only. You will not be shown
   which label either original annotator gave.
3. Fill in three columns:
   - `adjudicated_label`: `design` or `non-design`, applying the definition
     above.
   - `resolved_by_definition`: `yes` if the definition above makes the call
     clear-cut once stated explicitly; `no` if the ticket is genuinely
     ambiguous even under a shared definition (state why in `notes`).
   - `notes`: one short phrase, especially for `resolved_by_definition=no`
     rows — what's ambiguous (e.g., "refactor with unclear architectural
     scope," "new API but trivial/single-field").
4. Do not look up or discuss individual tickets with the original annotators
   before finishing the sheet.
5. When done, merge with `adjudication_key.csv` on `adjudication_id` to
   compute: (a) % of the 68 where `adjudicated_label` matches `mine`, (b) %
   matching `peer`, (c) % of `resolved_by_definition=yes` vs `no`. Report
   that split in manuscript §5.4 per `INTER_RATER_FINDINGS.md` §6.
