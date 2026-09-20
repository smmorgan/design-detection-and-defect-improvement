# Reviewer Feedback — Round 1

Verbatim reviewer comments from the first review round, with point IDs added (`R1.n` / `R2.n`) so they can be cross-referenced from code and commit messages. A short **Status** line is included under each point to track resubmission progress; update these as work lands.

---

## Reviewer #1

> The paper presents an empirical study on using three pre-trained variants of BERT to classify JIRA tickets as design or non-design. The models were fine-tuned on a Stack Overflow dataset and used via transfer learning.

**Strengths**

> It appears to be one of the first works to evaluate BERT for JIRA ticket classification to identify design-related tickets.

**Weaknesses**

### R1.1 — Model lineup is limited to traditional BERT; no LLM comparison
> The evaluation is limited to 3 variants of traditional BERT. It does not use modern BERT variants. Moreover, as LLMs have been in use for around two years, and they could serve as a natural baseline or alternative for comparison.

**Status:** Addressed. Added ModernBERT as a modern encoder variant, and Qwen2.5-1.5B+LoRA as a decoder baseline (cold-start and Stage-1-pretrained). Added zero-/few-shot LLM baselines (Claude, GPT-4o, local Llama3.1:8b) — see [R2.3](#r23--where-are-the-llm-baselines) below, same underlying work. Files: `train_design_classifier.py`, `eval_lopo.py`, `eval_llm_baseline.py`, `llm_clients.py`.

### R1.2 — No computational overhead analysis
> The analysis primarily focuses on model classification accuracy. However, the paper does not study computational overhead, which is important when comparing language models to classical ML.

**Status:** Addressed. `compute_overhead_table.py` reports LLM cost ($/1k tickets, s/ticket), transformer Stage-1 pretraining wall-clock, decoder+LoRA fine-tuning wall-clock, and traditional-ML (SVM/GB) CPU timing, plus BERT-base's reproducible OOM failure. Output: `gcp_results/computational_overhead.json`.

### R1.3 — No qualitative classification error analysis
> The study also does not provide a qualitative classification error analysis.

**Status:** Not started. Per-project LOPO F1 varies widely (e.g. RoBERTa: SERVER=0.449 vs CONFSERVER=0.81) — need to dig into *why*: false-positive/false-negative examples, project-specific vocabulary, ticket-type confusion. Overlaps with [R2.4](#r24--downstream-applications-need-fleshing-out) and [R2.6](#r26--generalizability-looks-weak).

### R1.4 — Lack of statistical significance testing
> Another issue is the lack of statistical tests. It is unclear whether there are statistically significant differences between models (among BERT variants as well as between BERTs and classical MLs).

**Status:** Addressed. `significance_tests.py` runs paired Wilcoxon signed-rank tests + Cohen's d across all 10 LOPO folds, both within BERT variants and against SVM/Gradient Boosting/LLM baselines. Output: `gcp_results/significance_tests_model_family.json` (and `significance_tests.json` for the augmentation-ablation comparisons).

### R1.5 — "Design Mining" naming sets the wrong expectation
> "Design Mining" - I expected it to extract architectural designs, design decisions, etc. However, the focus is on classifying texts.

**Status:** Not started — a framing/writing fix (title, abstract, intro scoping), not an analysis gap. Needs a decision: rename/rescope the term, or explicitly define "design mining" as ticket-level classification early and justify the term.

---

## Reviewer #2

**Summary:** This paper investigates whether transformer-based models (like BERT, RoBERTa, and DistilBERT) can beat out traditional machine learning techniques for "design mining". Specifically, the authors fine-tune these models on a Stack Overflow dataset and then use transfer learning to automatically classify design-related discussions within open-source JIRA issues from the TAWOS dataset.

> Applying transformers to scale design mining across operational software engineering artifacts is a really relevant direction, and demonstrating a 10-15% performance improvement over traditional baselines (like SVM and Gradient Boosting) is definitely a promising result. It is an interesting problem space that tackles a real need in understanding architectural effort in software projects.

However, there are several major issues that need to be addressed before it is ready for publication:

### R2.1 — Missing a workflow diagram
> The overall study approach lacks a clear overview. You outline the sequential methodology in the text—fine-tuning on Stack Overflow, applying transfer learning to TAWOS, and running cross-project validation—but burying this in text makes it hard to follow. Adding a visual flowchart of the pipeline will massively improve the clarity of the paper.

**Status:** Not started.

### R2.2 — Cherry-picking heuristic is too vague
> You mention using this to fix the self-affirmation bias and address the class imbalance, where only 28% of the manual data was design-related. You briefly describe filtering out bugs and using threshold based on architectural phrases to grab likely design tickets. However, you end up generating over 9,000 labels this way and admit the heuristic introduces noise and potentially systematic biases. Why exactly is this needed, how are the phrases collected and weighted, and how much is this noise driving the final results? More details are absolutely needed here.

**Status:** Partially addressed. `LABELLING_METHODOLOGY.md` documents the heuristic. Still needed: quantify how much the heuristic-labeled data (vs. the 1,786 manually labeled tickets) drives final results — e.g. an ablation comparing models trained with/without the heuristic-labeled augmentation, and a description of phrase collection/weighting.

### R2.3 — Where are the LLM baselines?
> Given the current landscape, how do modern LLMs handle this design mining task? You justify using BERT because it can be fine-tuned with small datasets. However, completely ignoring zero-shot or few-shot generative LLMs is a massive omission. Why not just directly use LLMs as a baseline?

**Status:** Addressed — same work as [R1.1](#r11--model-lineup-is-limited-to-traditional-bert-no-llm-comparison). Zero-shot and few-shot baselines added for Claude, GPT-4o, and a local open model (Llama3.1:8b) across all 10 LOPO projects, with significance tests against the best fine-tuned model (RoBERTa). Notable finding: RoBERTa is *not* statistically significantly better than Claude few-shot (p=0.11) despite a positive mean F1 gap — worth reporting directly rather than glossing over.

### R2.4 — Downstream applications need fleshing out
> What is the actual downstream application of this design mining? You briefly mention using it to screen JIRA repositories or as a tool to measure if defect rates change after a redesign, but you relegate the actual analysis to future work. This section needs much more discussion. Please provide a qualitative review of the classification results and connect it with related work in downstream applications. The attribution analysis highlights that the model over-indexes on project-specific terms, like "docker", "helm", or "plugin". This deserves a deeper dive.

**Status:** Partially addressed. `gcp_results/attribution_analysis.pdf` contains the underlying attribution analysis already referenced here — needs to be confirmed as cited/discussed in the manuscript text, and expanded into the deeper dive the reviewer is asking for (project-specific term over-indexing, downstream application discussion). Overlaps with [R1.3](#r13--no-qualitative-classification-error-analysis).

### R2.5 — Single-annotator ground truth
> Relying on single annotator labels is a big threat to construct validity. You rightfully point out that defining "design work" is inherently ambiguous and context-dependent. Because of that ambiguity, ground truth really needs to be established by multiple annotators or consensus.

**Status:** Not started. No inter-rater reliability data (e.g. Cohen's kappa) exists yet anywhere in the repo. Requires new human labeling work: recruiting a second annotator and a scope decision (how many issues to double-label, sampling strategy, agreement metric).

### R2.6 — Generalizability looks weak (LOPO variance)
> The LOPO variance is really concerning. Seeing F1 scores swing from 0.83 on one project all the way down to 0.48 on another shows major inconsistency. You note that project-specific language variation is the primary barrier to generalization. If the tool breaks down on new projects, its operational viability is questionable.

**Status:** Not started — needs a discussion/limitations write-up grounded in the per-project LOPO breakdown already computed (`gcp_results/lopo_*_results.json`), ideally tied to the qualitative error analysis in [R1.3](#r13--no-qualitative-classification-error-analysis) and the attribution deep-dive in [R2.4](#r24--downstream-applications-need-fleshing-out) (i.e., is the variance actually explained by project-specific vocabulary?).

---

## Cross-reference: code comments → reviewer points

Two files already contain shorthand references to these points from before this document existed:
- `significance_tests.py` — "R1#4 ... R1#1/R2#3" → **R1.4**, **R1.1**, **R2.3** above.
- `compute_overhead_table.py` — "reviewer request for a computational-cost comparison" → **R1.2** above.

New code comments referencing reviewer feedback should cite point IDs from this document (e.g. `# Addresses R2.2`) rather than free text, so they stay traceable.
