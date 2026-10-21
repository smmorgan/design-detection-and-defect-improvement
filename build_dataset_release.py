"""
Builds the public dataset release for the RQ1 design-mining paper (manual
labels, heuristic augmentation labels, and the inter-rater study with its
blinded adjudication). RQ2 artifacts (classifier-predicted labels, calibration
audit labels, defect panels) are deliberately excluded.

Inputs:
  ieee_dataport/manual_labels/manually_labelled_issues.csv       - 1,786 primary labels
  ieee_dataport/manual_labels/manually_labelled_sample_250.csv   - primary labels, 250 sample
  ieee_dataport/manual_labels/Jeff's_manually_labelled_sample_250.tsv - second annotator
  ieee_dataport/heuristic_labels/*.csv                           - augmentation labels
  gcp_results/inter_rater/adjudication_key.csv                   - adjudication_id -> issue_key
  data/data/adjudication_sheet - adjudication_sheet.tsv          - adjudicator's completed sheet

Writes dataset_release/design_mining_jira_dataset/ and a zip of it.
"""
import csv
import json
import shutil
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score

SRC = Path("ieee_dataport")
ADJ_KEY = Path("gcp_results/inter_rater/adjudication_key.csv")
ADJ_SHEET = Path("data/data/adjudication_sheet - adjudication_sheet.tsv")
OUT_ROOT = Path("dataset_release")
OUT = OUT_ROOT / "design_mining_jira_dataset"
TEMPLATES = Path("dataset_release_templates")


def norm(label):
    return "design" if str(label).strip().lower() == "design" else "non-design"


if OUT.exists():
    shutil.rmtree(OUT)
for sub in ("manual_labels", "heuristic_labels", "inter_rater"):
    (OUT / sub).mkdir(parents=True)

shutil.copy(SRC / "manual_labels/manually_labelled_issues.csv", OUT / "manual_labels")
for name in ("heuristic_design_labels.csv", "heuristic_nondesign_labels.csv"):
    shutil.copy(SRC / "heuristic_labels" / name, OUT / "heuristic_labels")

# Inter-rater sample: primary + second annotator + adjudication in one file.
primary = pd.read_csv(SRC / "manual_labels/manually_labelled_sample_250.csv")
second = pd.read_csv(SRC / "manual_labels/Jeff's_manually_labelled_sample_250.tsv", sep="\t")
sample = primary[["project", "issue_key", "issue_type", "summary", "description"]].copy()
sample["primary_label"] = primary["label"].map(norm)
sample = sample.merge(
    second[["issue_key", "label"]].rename(columns={"label": "second_label"}),
    on="issue_key", how="left", validate="one_to_one",
)
sample["second_label"] = sample["second_label"].map(norm)

key = pd.read_csv(ADJ_KEY, dtype={"adjudication_id": str})
sheet = pd.read_csv(ADJ_SHEET, sep="\t", dtype={"adjudication_id": str}, quoting=csv.QUOTE_MINIMAL)
adj = key.merge(sheet[["adjudication_id", "adjudicated_label", "notes"]], on="adjudication_id", validate="one_to_one")
adj["adjudicated_label"] = adj["adjudicated_label"].map(norm)
adj = adj[["issue_key", "adjudicated_label", "notes"]].rename(columns={"notes": "adjudication_note"})

sample = sample.merge(adj, on="issue_key", how="left", validate="one_to_one")
disagree = sample["primary_label"] != sample["second_label"]
assert disagree.sum() == len(adj) == sample["adjudicated_label"].notna().sum()
sample["final_label"] = sample["adjudicated_label"].where(disagree, sample["primary_label"])
sample.to_csv(OUT / "inter_rater/inter_rater_sample_250.csv", index=False)
sample[disagree][[
    "project", "issue_key", "issue_type", "summary", "description",
    "primary_label", "second_label", "adjudicated_label", "adjudication_note",
]].to_csv(OUT / "inter_rater/adjudication_results.csv", index=False)

# Summary statistics, matching those reported in the paper.
d = sample[disagree]
flip_to_design = int(((sample.primary_label == "non-design") & (sample.final_label == "design")).sum())
flip_to_nondesign = int(((sample.primary_label == "design") & (sample.final_label == "non-design")).sum())
summary = {
    "n": len(sample),
    "primary_design_rate": round((sample.primary_label == "design").mean(), 3),
    "second_design_rate": round((sample.second_label == "design").mean(), 3),
    "raw_agreement": round((~disagree).mean(), 3),
    "cohens_kappa_primary_vs_second": round(cohen_kappa_score(sample.primary_label, sample.second_label), 3),
    "n_disagreements": int(disagree.sum()),
    "disagreements_primary_nondesign_second_design": int(((d.primary_label == "non-design")).sum()),
    "disagreements_primary_design_second_nondesign": int(((d.primary_label == "design")).sum()),
    "adjudication_matches_primary": int((d.adjudicated_label == d.primary_label).sum()),
    "adjudication_matches_second": int((d.adjudicated_label == d.second_label).sum()),
    "final_design_rate": round((sample.final_label == "design").mean(), 3),
    "primary_vs_final_agreement": round((sample.primary_label == sample.final_label).mean(), 3),
    "primary_vs_final_kappa": round(cohen_kappa_score(sample.primary_label, sample.final_label), 3),
    "second_vs_final_agreement": round((sample.second_label == sample.final_label).mean(), 3),
    "second_vs_final_kappa": round(cohen_kappa_score(sample.second_label, sample.final_label), 3),
    "primary_labels_overturned": flip_to_design + flip_to_nondesign,
    "overturned_nondesign_to_design": flip_to_design,
    "overturned_design_to_nondesign": flip_to_nondesign,
    "overturn_direction_binomial_p": round(
        binomtest(min(flip_to_design, flip_to_nondesign), flip_to_design + flip_to_nondesign, 0.5).pvalue, 3),
}
(OUT / "inter_rater/agreement_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

for name in ("README.md", "adjudication_rubric.md"):
    dest = OUT / ("inter_rater" if name == "adjudication_rubric.md" else "") / name
    shutil.copy(TEMPLATES / name, dest)

zip_path = shutil.make_archive(str(OUT), "zip", root_dir=OUT_ROOT, base_dir=OUT.name)
print(json.dumps(summary, indent=2))
print(f"Wrote {OUT} and {zip_path}")
