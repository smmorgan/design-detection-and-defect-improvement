"""Generate confusion matrices from LOPO cross-validation results."""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

with open("gcp_results/lopo_results.json") as f:
    data = json.load(f)

projects = data["per_project"]

# --- Per-project confusion matrices in a grid ---
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for i, proj in enumerate(projects):
    cm = np.array([[proj["tn"], proj["fp"]],
                    [proj["fn"], proj["tp"]]])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                xticklabels=["Non-Design", "Design"],
                yticklabels=["Non-Design", "Design"],
                cbar=False)
    axes[i].set_title(f"{proj['project']}\nF1={proj['f1']:.3f}  AUC={proj['auc']:.3f}",
                      fontsize=10)
    axes[i].set_ylabel("Actual" if i % 5 == 0 else "")
    axes[i].set_xlabel("Predicted")

fig.suptitle("LOPO Cross-Validation — Per-Project Confusion Matrices", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("gcp_results/lopo_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Aggregate confusion matrix ---
tp = sum(p["tp"] for p in projects)
tn = sum(p["tn"] for p in projects)
fp = sum(p["fp"] for p in projects)
fn = sum(p["fn"] for p in projects)

cm_agg = np.array([[tn, fp], [fn, tp]])

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_agg, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Non-Design", "Design"],
            yticklabels=["Non-Design", "Design"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"LOPO Aggregate Confusion Matrix (n={sum(p['n'] for p in projects)})\n"
             f"Mean F1={data['mean_f1']:.4f}  Mean AUC={data['mean_auc']:.4f}")
plt.tight_layout()
plt.savefig("gcp_results/lopo_confusion_aggregate.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Aggregate: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"Accuracy: {(tp+tn)/(tp+tn+fp+fn):.4f}")
print(f"Precision: {tp/(tp+fp):.4f}")
print(f"Recall: {tp/(tp+fn):.4f}")
print("Saved: gcp_results/lopo_confusion_matrices.png")
print("Saved: gcp_results/lopo_confusion_aggregate.png")
