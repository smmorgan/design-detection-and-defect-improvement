# Adjudication Rubric

This rubric was used to resolve the 68 tickets on which the primary and second
annotators disagreed. The adjudicator had not taken part in either original
labeling round and had no other role in the study. Tickets were shown in a
shuffled order (fixed seed), with only the project, issue type, summary and
description visible. The adjudicator did not see which label either annotator
had given.

## Definition applied

A ticket is **design** if it describes an **architectural or structural**
decision: service or component boundaries, data or schema design, API design
(new endpoints, versioning, contracts), authentication or authorization
schemes, messaging or eventing architecture, or a comparable structural
decision about how the system is built.

A ticket is **not design** if it is:

- A **UI-only** change (visual, layout or copy: buttons, styling, colors,
  wording, icons) with no structural implication.
- A **bug fix**, even a complex one, that fixes broken behavior within the
  existing design without changing the system's structure.
- A **small, additive API change** (adding a parameter, exposing a single
  existing field, adjusting an error code) that does not establish or change
  an architectural contract.
- Purely **operational**: deployment, CI/CD, environment configuration,
  version bumps or upgrades, documentation.

Guiding question: *does this ticket decide how the system is structured, or
does it operate within, or adjust the surface of, an existing structure?* The
former is design; the latter is not.

## Recorded fields

- `adjudicated_label`: `design` or `non-design` under the definition above.
- `adjudication_note`: a short phrase giving the deciding reason (for example
  "data or schema design", "UI only", "small API change (add a parameter)").
