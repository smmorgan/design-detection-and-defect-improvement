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

**Status:** Addressed. `compute_overhead_table.py` reports LLM cost ($/1k tickets, s/ticket), transformer Stage-1 pretraining wall-clock, decoder+LoRA fine-tuning wall-clock, and traditional-ML (SVM/GB) CPU timing, plus BERT-base's reproducible OOM failure. Output: `gcp_results/computational_overhead.json`. Extended 2026-09-23: the batch_size=32 configuration that OOM'd on the local 11.69 GiB GPU for all three encoders (bert/roberta/distilbert) was reproduced successfully on a rented GCP `g2-standard-4` instance (1x NVIDIA L4, 24 GiB) at a combined cost of ~$6, but yields no measurable accuracy improvement over the free batch_size=16 local configuration (F1 differences ≤0.0011, an order of magnitude below seed-to-seed variance). New Table~\ref{tab:batch32cost} in `paper/manuscript.tex` §5.3; also fixed a pre-existing inconsistency where the Stage 1 prose cited the batch_size=32 GCP bert F1/AUC (0.9056/0.9646) as if it were a batch_size=16 result in Table~\ref{tab:stage1} — corrected to the actual batch_size=16 bert number (0.9045/0.9634).

### R1.3 — No qualitative classification error analysis
> The study also does not provide a qualitative classification error analysis.

**Status:** Done and in the manuscript (2026-09-23). Manually reviewed every FP/FN for the two difficulty-extreme projects (CONFSERVER, SERVER — 100% coverage) plus DM (all FNs, a 15-of-29 sample of FPs) from the Base+Meta config, categorized into a 7-bucket codebook. Headline: 52.3% of sampled errors (34/65) are label-quality issues (`LABEL_AMBIGUOUS`/`LABEL_ERROR`), not model failures — converging with the independent R2.5 inter-rater finding. Each remaining failure mode has a distinct, citable signature: `DESIGN_VOCAB_IMPL` is FP-only (implementation work with design-sounding vocabulary), `IMPLICIT_DESIGN` is FN-only (design work without design vocabulary), `SPARSE_TEXT` skews FN, `PROJECT_JARGON` skews FP and only appears in CONFSERVER/DM. Also surfaced an actionable data bug: 3 DM tickets are boilerplate "Deploy for X" tickets mislabeled design, violating `LABELLING_METHODOLOGY.md`'s own hard-exclusion rule. Written into `paper/manuscript.tex` §5.2 "Cross-Project Generalization Challenges" (combined with R2.6, as recommended). Full source analysis in `ERROR_ANALYSIS_FINDINGS.md`.

### R1.4 — Lack of statistical significance testing
> Another issue is the lack of statistical tests. It is unclear whether there are statistically significant differences between models (among BERT variants as well as between BERTs and classical MLs).

**Status:** Addressed and in the manuscript. `significance_tests.py` runs paired Wilcoxon signed-rank tests + Cohen's d + Holm-Bonferroni correction across all 10 LOPO folds, both within BERT variants and against SVM/Gradient Boosting/LLM baselines. Written into §4.7 "Statistical Significance Testing" (`tab:significance`). As of 2026-09-23, the underlying RoBERTa baseline used for this test was corrected from the winner's-curse-selected `Base+Meta+SqrtW` single run to the seed-averaged `Base+Meta` config (see the seed-variance note below) — `gcp_results/significance_tests_model_family_seedavg.json` was regenerated against this corrected baseline, and now all 11 comparisons (not just 5) are significant after Holm correction.

### R1.5 — "Design Mining" naming sets the wrong expectation
> "Design Mining" - I expected it to extract architectural designs, design decisions, etc. However, the focus is on classifying texts.

**Status:** Done and in the manuscript. §3.1 "Overview" now states explicitly: "design mining ... refers specifically to the binary text classification of individual JIRA tickets as design-related or not ... it does not extract design artifacts, decisions, or architectural models."

---

## Reviewer #2

**Summary:** This paper investigates whether transformer-based models (like BERT, RoBERTa, and DistilBERT) can beat out traditional machine learning techniques for "design mining". Specifically, the authors fine-tune these models on a Stack Overflow dataset and then use transfer learning to automatically classify design-related discussions within open-source JIRA issues from the TAWOS dataset.

> Applying transformers to scale design mining across operational software engineering artifacts is a really relevant direction, and demonstrating a 10-15% performance improvement over traditional baselines (like SVM and Gradient Boosting) is definitely a promising result. It is an interesting problem space that tackles a real need in understanding architectural effort in software projects.

However, there are several major issues that need to be addressed before it is ready for publication:

### R2.1 — Missing a workflow diagram
> The overall study approach lacks a clear overview. You outline the sequential methodology in the text—fine-tuning on Stack Overflow, applying transfer learning to TAWOS, and running cross-project validation—but burying this in text makes it hard to follow. Adding a visual flowchart of the pipeline will massively improve the clarity of the paper.

**Status:** Done and in the manuscript. A TikZ pipeline figure (`fig:pipeline`) is now in §3.1 "Overview," showing SO pretraining → TAWOS transfer → LOPO cross-validation → traditional-ML/modern-encoder/LLM comparison.

### R2.2 — Cherry-picking heuristic is too vague
> You mention using this to fix the self-affirmation bias and address the class imbalance, where only 28% of the manual data was design-related. You briefly describe filtering out bugs and using threshold based on architectural phrases to grab likely design tickets. However, you end up generating over 9,000 labels this way and admit the heuristic introduces noise and potentially systematic biases. Why exactly is this needed, how are the phrases collected and weighted, and how much is this noise driving the final results? More details are absolutely needed here.

**Status:** Done and in the manuscript. §3.4 "Labeling the TAWOS Dataset" now spells out the exact hard-exclusion rules, the weighted keyword lexicon (37 strong / 31 medium / 10 weak phrases with point values and worked examples), the selection threshold, and a manual audit of 100 heuristic labels (~95% agreement). The augmentation-ablation table (`tab:lopo`) is called out explicitly as the noise-quantification the reviewer asked for. Label counts in `tab:labels` and the prose were also corrected to match the underlying CSVs (496 manual design / 1,290 manual non-design; 4,989 cherry-picked design / 4,995 cherry-picked non-design — the manuscript previously reported mismatched totals here).

### R2.3 — Where are the LLM baselines?
> Given the current landscape, how do modern LLMs handle this design mining task? You justify using BERT because it can be fine-tuned with small datasets. However, completely ignoring zero-shot or few-shot generative LLMs is a massive omission. Why not just directly use LLMs as a baseline?

**Status:** Addressed — same work as [R1.1](#r11--model-lineup-is-limited-to-traditional-bert-no-llm-comparison). Zero-shot and few-shot baselines added for Claude, GPT-4o, and a local open model (Llama3.1:8b) across all 10 LOPO projects, with significance tests against the best fine-tuned model (RoBERTa). Notable finding: RoBERTa is *not* statistically significantly better than Claude few-shot (p=0.11) despite a positive mean F1 gap — worth reporting directly rather than glossing over.

### R2.4 — Downstream applications need fleshing out
> What is the actual downstream application of this design mining? You briefly mention using it to screen JIRA repositories or as a tool to measure if defect rates change after a redesign, but you relegate the actual analysis to future work. This section needs much more discussion. Please provide a qualitative review of the classification results and connect it with related work in downstream applications. The attribution analysis highlights that the model over-indexes on project-specific terms, like "docker", "helm", or "plugin". This deserves a deeper dive.

**Status:** Done and in the manuscript. §4.6 "Attribution Analysis" now includes the per-project vocabulary-breadth deep dive (narrow but strong design-class signals vs. broad, high-frequency implementation terms like "docker"/"plugin"), and explicit deployment guidance (re-run attribution on your own project before reuse). The "qualitative review of the classification results" half of this request is satisfied by R1.3's error analysis, now in §5.2.

### R2.5 — Single-annotator ground truth
> Relying on single annotator labels is a big threat to construct validity. You rightfully point out that defining "design work" is inherently ambiguous and context-dependent. Because of that ambiguity, ground truth really needs to be established by multiple annotators or consensus.

**Status:** Done and in the manuscript (2026-09-23). A second annotator independently labeled the same 250-ticket stratified sample. Cohen's $\kappa$ = 0.436 (moderate), raw agreement 72.8%, driven by a systematic and statistically significant (McNemar $p<10^{-7}$) prevalence gap — the second annotator labeled design 46% of the time vs. 27.6% for the primary annotator, concentrated almost entirely in "Improvement"/"Suggestion" issue types (100% design rate from the second annotator on both), likely because no shared written definition of "design" was given. Written into §5.4 "Threats to Validity" (Construct validity paragraph) as a genuine limitation, cross-referenced with R1.3's independent finding that 52% of the model's own errors are also label-quality issues. Full source analysis in `INTER_RATER_FINDINGS.md`; its optional adjudication pass on the 68 disagreements (§7) was not performed.

### R2.6 — Generalizability looks weak (LOPO variance)
> The LOPO variance is really concerning. Seeing F1 scores swing from 0.83 on one project all the way down to 0.48 on another shows major inconsistency. You note that project-specific language variation is the primary barrier to generalization. If the tool breaks down on new projects, its operational viability is questionable.

**Status:** Done and in the manuscript (2026-09-23), combined with R1.3 in §5.2 "Cross-Project Generalization Challenges." Reports that project difficulty is consistent across all 13 evaluated model families (Kendall's $W$=0.70 on F1, $p<10^{-13}$), correlates with out-of-vocabulary rate ($\rho=-0.77$) and empty-description rate ($\rho=-0.64$) across projects, and that the per-project error analysis gives a concrete mechanism per project (SERVER=sparse descriptions, DM=vocabulary shift, CONFSERVER=residual ambiguity) rather than a bare F1-range assertion.

---

## Cross-reference: code comments → reviewer points

Two files already contain shorthand references to these points from before this document existed:
- `significance_tests.py` — "R1#4 ... R1#1/R2#3" → **R1.4**, **R1.1**, **R2.3** above.
- `compute_overhead_table.py` — "reviewer request for a computational-cost comparison" → **R1.2** above.

New code comments referencing reviewer feedback should cite point IDs from this document (e.g. `# Addresses R2.2`) rather than free text, so they stay traceable.
