#!/usr/bin/env python3
"""
Visualization of Integrated Gradients attribution analysis.
Shows top design and non-design word ranks as horizontal bar charts.
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 11

# Load data
with open("gcp_results/feature_importance.json") as f:
    data = json.load(f)

top_design = data["top_design_words"]
top_nondesign = data["top_nondesign_words"]

# Take top 20 for each
n = 20
design_words = [w["word"] for w in top_design[:n]][::-1]
design_scores = [w["mean_attribution"] for w in top_design[:n]][::-1]
design_stds = [w["std_attribution"] for w in top_design[:n]][::-1]
design_counts = [w["count"] for w in top_design[:n]][::-1]

nondesign_words = [w["word"] for w in top_nondesign[:n]][::-1]
nondesign_scores = [w["mean_attribution"] for w in top_nondesign[:n]][::-1]
nondesign_stds = [w["std_attribution"] for w in top_nondesign[:n]][::-1]
nondesign_counts = [w["count"] for w in top_nondesign[:n]][::-1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# Design words (left panel)
y_pos = np.arange(n)
bars1 = ax1.barh(y_pos, design_scores, xerr=design_stds,
                  color="#2171b5", edgecolor="#08519c", alpha=0.85,
                  capsize=3, error_kw={"elinewidth": 1, "alpha": 0.6})
ax1.set_yticks(y_pos)
ax1.set_yticklabels(design_words, fontsize=11, fontweight="bold")
ax1.set_xlabel("Mean Attribution Score", fontsize=12)
ax1.set_title("Design Issue Words", fontsize=14, fontweight="bold", pad=12)
ax1.axvline(x=0, color="gray", linewidth=0.5)

# Add count annotations
for i, (score, count) in enumerate(zip(design_scores, design_counts)):
    ax1.text(score + design_stds[i] + 0.003, i, f"n={count}",
             va="center", fontsize=8, color="#666")

# Non-design words (right panel)
bars2 = ax2.barh(y_pos, nondesign_scores, xerr=nondesign_stds,
                  color="#cb181d", edgecolor="#a50f15", alpha=0.85,
                  capsize=3, error_kw={"elinewidth": 1, "alpha": 0.6})
ax2.set_yticks(y_pos)
ax2.set_yticklabels(nondesign_words, fontsize=11, fontweight="bold")
ax2.set_xlabel("Mean Attribution Score", fontsize=12)
ax2.set_title("Non-Design Issue Words", fontsize=14, fontweight="bold", pad=12)
ax2.axvline(x=0, color="gray", linewidth=0.5)

# Add count annotations
for i, (score, count) in enumerate(zip(nondesign_scores, nondesign_counts)):
    ax2.text(score + nondesign_stds[i] + 0.003, i, f"n={count}",
             va="center", fontsize=8, color="#666")

# Match x-axis range
max_x = max(
    max(s + e for s, e in zip(design_scores, design_stds)),
    max(s + e for s, e in zip(nondesign_scores, nondesign_stds)),
) * 1.25
ax1.set_xlim(0, max_x)
ax2.set_xlim(0, max_x)

fig.suptitle(
    "Integrated Gradients Attribution Analysis\n"
    "Top-20 Most Influential Words by Class",
    fontsize=15, fontweight="bold", y=0.98,
)

fig.text(
    0.5, 0.01,
    f"Method: Layer Integrated Gradients (Sundararajan et al., 2017)  |  "
    f"n={data['n_design_samples']} design + {data['n_nondesign_samples']} non-design samples  |  "
    f"min word count ≥ {data['min_word_count']}  |  Error bars = ±1 std",
    ha="center", fontsize=9, color="#888",
)

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.savefig("gcp_results/attribution_analysis.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.savefig("gcp_results/attribution_analysis.pdf", bbox_inches="tight",
            facecolor="white")
print("Saved to gcp_results/attribution_analysis.png and .pdf")
