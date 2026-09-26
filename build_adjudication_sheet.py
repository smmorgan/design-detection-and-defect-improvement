"""
Builds the adjudication sheet for the 68 inter-rater disagreements (R2.5,
see INTER_RATER_FINDINGS.md §7). Joins gcp_results/inter_rater/disagreements.csv
against the full labeled samples to pull ticket text, then writes:

  gcp_results/inter_rater/adjudication_sheet.csv  - blind working copy for the
      adjudicator: ticket text + the written definition, no existing labels,
      empty columns to fill in.
  gcp_results/inter_rater/adjudication_key.csv     - same rows with `mine`/
      `peer` labels attached, for scoring after adjudication is done. Not to
      be opened until the sheet is filled in.

Row order is shuffled (fixed seed) so adjacent tickets aren't grouped by
issue_type, which could otherwise cue the adjudicator into a pattern
(e.g. "every Improvement ticket so far has been design").
"""
import csv
import random

DISAGREEMENTS = "gcp_results/inter_rater/disagreements.csv"
PRIMARY_LABELS = "ieee_dataport/manual_labels/manually_labelled_sample_250.csv"
SHEET_OUT = "gcp_results/inter_rater/adjudication_sheet.csv"
KEY_OUT = "gcp_results/inter_rater/adjudication_key.csv"
SEED = 42

with open(PRIMARY_LABELS, newline="", encoding="utf-8") as f:
    descriptions = {row["issue_key"]: row["description"] for row in csv.DictReader(f)}

with open(DISAGREEMENTS, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    row["description"] = descriptions.get(row["issue_key"], "")

random.Random(SEED).shuffle(rows)

sheet_fields = [
    "adjudication_id", "project", "issue_type", "summary", "description",
    "adjudicated_label", "resolved_by_definition", "notes",
]
with open(SHEET_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=sheet_fields)
    w.writeheader()
    for i, row in enumerate(rows, start=1):
        w.writerow({
            "adjudication_id": i,
            "project": row["project"],
            "issue_type": row["issue_type"],
            "summary": row["summary"],
            "description": row["description"],
            "adjudicated_label": "",
            "resolved_by_definition": "",
            "notes": "",
        })

key_fields = ["adjudication_id", "issue_key", "project", "issue_type", "mine", "peer"]
with open(KEY_OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=key_fields)
    w.writeheader()
    for i, row in enumerate(rows, start=1):
        w.writerow({
            "adjudication_id": i,
            "issue_key": row["issue_key"],
            "project": row["project"],
            "issue_type": row["issue_type"],
            "mine": row["mine"],
            "peer": row["peer"],
        })

print(f"Wrote {len(rows)} rows to {SHEET_OUT} and {KEY_OUT}")
