"""Inter-rater reliability between the primary annotator's labels for the
250-ticket stratified sample (ieee_dataport/manual_labels/manually_labelled_sample_250.csv)
and a second annotator's independent labels of the same tickets.

Usage:
    python analyze_inter_rater.py <peer_labels.tsv>

Writes gcp_results/inter_rater/agreement_summary.json and
gcp_results/inter_rater/disagreements.csv.
"""
import sys
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.stats import binomtest

MINE_PATH = Path("ieee_dataport/manual_labels/manually_labelled_sample_250.csv")
OUT_DIR = Path("gcp_results/inter_rater")

LABEL_MAP = {"design": "design", "non-design": "non-design", "not design": "non-design"}


def load_peer(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    df = pd.read_csv(path, sep=sep)
    df["label_norm"] = df["label"].str.strip().str.lower().map(LABEL_MAP)
    if df["label_norm"].isna().any():
        bad = df.loc[df["label_norm"].isna(), "label"].unique()
        raise ValueError(f"Unrecognized peer label values: {bad}")
    return df


def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {sys.argv[0]} <peer_labels.tsv>")
    peer_path = Path(sys.argv[1])

    mine = pd.read_csv(MINE_PATH)
    mine["label_norm"] = mine["label"].str.strip().str.lower().map(LABEL_MAP)
    peer = load_peer(peer_path)

    m = mine[["issue_key", "project", "issue_type", "label_norm"]].merge(
        peer[["issue_key", "label_norm"]], on="issue_key", suffixes=("_mine", "_peer")
    )
    if len(m) != len(mine):
        missing = set(mine["issue_key"]) - set(peer["issue_key"])
        raise ValueError(f"{len(missing)} tickets in the primary sample have no peer label: {sorted(missing)[:10]}")

    n = len(m)
    agree_mask = m["label_norm_mine"] == m["label_norm_peer"]
    po = agree_mask.mean()
    kappa = cohen_kappa_score(m["label_norm_mine"], m["label_norm_peer"])
    pabak = 2 * po - 1

    cm = confusion_matrix(m["label_norm_mine"], m["label_norm_peer"], labels=["design", "non-design"])

    b = int(((m["label_norm_mine"] == "design") & (m["label_norm_peer"] == "non-design")).sum())
    c = int(((m["label_norm_mine"] == "non-design") & (m["label_norm_peer"] == "design")).sum())
    mcnemar_p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else float("nan")

    per_project = (
        m.groupby("project")
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "agreement": (g["label_norm_mine"] == g["label_norm_peer"]).mean(),
                    "mine_design_rate": (g["label_norm_mine"] == "design").mean(),
                    "peer_design_rate": (g["label_norm_peer"] == "design").mean(),
                }
            ),
            include_groups=False,
        )
        .round(4)
    )

    per_type = (
        m.groupby("issue_type")
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "disagreement_rate": (g["label_norm_mine"] != g["label_norm_peer"]).mean(),
                    "mine_design_rate": (g["label_norm_mine"] == "design").mean(),
                    "peer_design_rate": (g["label_norm_peer"] == "design").mean(),
                }
            ),
            include_groups=False,
        )
        .round(4)
    )

    summary = {
        "n": n,
        "raw_agreement": round(po, 4),
        "cohens_kappa": round(kappa, 4),
        "pabak": round(pabak, 4),
        "mine_design_prevalence": round((m["label_norm_mine"] == "design").mean(), 4),
        "peer_design_prevalence": round((m["label_norm_peer"] == "design").mean(), 4),
        "confusion_matrix_labels": ["design", "non-design"],
        "confusion_matrix_rows_mine_cols_peer": cm.tolist(),
        "mcnemar_b_mine_design_peer_nondesign": b,
        "mcnemar_c_mine_nondesign_peer_design": c,
        "mcnemar_exact_p": mcnemar_p,
        "per_project": per_project.to_dict(orient="index"),
        "per_issue_type": per_type.to_dict(orient="index"),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "agreement_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    disagreements = mine.merge(peer[["issue_key", "label_norm"]], on="issue_key", suffixes=("", "_peer"))
    disagreements = disagreements[disagreements["label_norm"] != disagreements["label_norm_peer"]]
    disagreements = disagreements[
        ["project", "issue_key", "issue_type", "summary", "label_norm", "label_norm_peer"]
    ].rename(columns={"label_norm": "mine", "label_norm_peer": "peer"})
    disagreements.to_csv(OUT_DIR / "disagreements.csv", index=False)

    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_project", "per_issue_type")}, indent=2))
    print(f"\n{len(disagreements)} disagreements written to {OUT_DIR / 'disagreements.csv'}")


if __name__ == "__main__":
    main()
