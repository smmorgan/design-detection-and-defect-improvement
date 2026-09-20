# Resubmission Guide — Where to Focus First

Ordered guidance for updating `paper/manuscript.tex` in response to `REVIEWER_FEEDBACK.md`. This is guidance only — it describes *where* and *what*, not the actual LaTeX prose. Point IDs (`R1.n`/`R2.n`) match `REVIEWER_FEEDBACK.md`.

## 0. Fix first: Table 1 has three unsupported rows (blocking)

`\label{tab:stage1}` (manuscript.tex:189–212, "Stage 1 Results: Training on Stack Overflow Data") reports these rows:

```
roberta     32   2e-5   3   0.907   0.966   <- bolded as best AUC in the table
bert        32   2e-5   3   0.906   0.965
distilbert  32   2e-5   3   0.906   0.964
```

I could not find a completed run backing any of these three in `gcp_results/` or `.archive/`. Specifically:
- All three `bert-base-uncased` batch_size=32 attempts found (`bert_baseline_0308_1555`, `_1558`, `_2027`, and `bert_conservative_large_batch_0308_2027`) end in `torch.OutOfMemoryError: CUDA out of memory` on the 11.69 GiB GPU — none completed.
- No batch_size=32 run of any kind exists for `roberta-base` or `distilbert-base-uncased` in the results directory.
- Every successful Stage-1 run found (7 total, across roberta/distilbert/bert) is at batch_size=16.
- `TRAINING_REPORT.md` (an internal note predating the manuscript) states directly: *"Batch size 16 outperforms batch size 32 by 0.03–0.04 F1 points across all models."* This is the opposite of what the bolded bs=32 row in Table 1 claims.
- The prose right above the table (manuscript.tex:187) already says the *best F1* came from `roberta-base` at bs=16/lr=1e-5 — consistent with the real data — but the table then separately bolds a bs=32 row as best-AUC, which is the discrepancy.

**Before touching anything else in this section**, decide one of:
1. **Re-run bs=32** (with gradient checkpointing / a smaller GPU-friendly variant) to see if those three rows can become real. Slowest option.
2. **Drop the three bs=32 rows** and republish Table 1 with only the 7 verified bs=16 rows (I have the exact F1/AUC for all 7 from this session if you want them — roberta bs16/lr1e-5 is F1=0.9066/AUC=0.9650, matching the prose's claimed best). Fastest, and self-consistent with `TRAINING_REPORT.md`.
3. **Search elsewhere** (another machine, an untracked results folder, a checkpoint bucket) for the missing artifacts, in case they exist outside this repo and just weren't committed.

Whichever you pick changes what Table 1 and its prose (lines 185–187) should say, so do this before drafting any new subsection that sits near it (R1.1/R2.3 additions below land right after this table).

---

## 1. Ready to write now (data already exists, just needs prose + a table)

### R1.4 — Statistical significance testing
- **Where:** New subsection after 4.6 "Attribution Analysis" (manuscript.tex:326–332), e.g. "4.7 Statistical Significance," or fold into 5.1 "Transformer Advantage for Design Mining" (line 336) where F1 gaps vs. SVM/GB are currently asserted without a test.
- **What:** A table of paired Wilcoxon signed-rank p-values + Cohen's d for RoBERTa (best) vs. each of: ModernBERT, Qwen+LoRA (cold-start), Qwen+LoRA (Stage-1-pretrained), SVM, Gradient Boosting, and the 6 LLM configs. Explicitly call out the one non-obvious result: RoBERTa is *not* significantly better than Claude few-shot (p=0.11) despite a positive mean F1 gap.
- **Data source:** `gcp_results/significance_tests_model_family.json`.
- **Caveat to state in text:** n=10 folds is small; note whether you're applying a Bonferroni correction across the ~8 comparisons or reporting uncorrected p-values with that limitation flagged.

### R1.2 — Computational overhead
- **Where:** New subsection, either in Results (e.g. "4.7 Computational Overhead," alongside or after the significance subsection) or in 5.3 "Practical Implications" (line 348), since overhead is directly relevant to the "screening tool" deployment claim already made there.
- **What:** LLM cost ($/1k tickets, s/ticket), transformer Stage-1 pretraining wall-clock, decoder+LoRA fine-tuning wall-clock, traditional-ML CPU timing, and BERT-base's reproducible OOM failure at bs=32 (which is now also the resolution to item 0 above — worth cross-referencing).
- **Data source:** `gcp_results/computational_overhead.json`.
- **Open sub-item:** Qwen's Stage-1 pretraining wall-clock isn't in `computational_overhead.json` yet — `compute_overhead_table.py`'s `transformer_pretraining_overhead()` only covers roberta-base/distilbert-base/modernbert-base. The raw timing is in the repo-root `training.log` (Qwen Stage-1 run finished 2026-09-04); this needs to be extracted and added to the script before this subsection can cite a Qwen pretraining time. Low effort, do this before drafting the paragraph.

### R1.1 / R2.3 — Model lineup + LLM baselines
- **Where:** This is the biggest structural addition. Two natural landing spots:
  - Extend 4.4 "Comparison with SVM and Gradient Boosting" (line 269, `tab:lopo_trad`) into a broader "Comparison with Traditional, Modern-Encoder, and LLM Baselines" — add ModernBERT and Qwen+LoRA (both variants) as new rows in a LOPO comparison table alongside SVM/GB, and a second table (or an extra block in the same one) for the 6 LLM zero-/few-shot configs.
  - Update the Stage 1 discussion (5.1, line 340) which currently only compares roberta/bert/distilbert F1 (0.9045–0.9066) — ModernBERT's Stage-1 result (F1=0.9104/AUC=0.9657) should be folded into that sentence since it's now the best Stage-1 F1, not just a LOPO-stage entrant.
- **What:** ModernBERT LOPO (F1=0.6688/AUC=0.8891), Qwen+LoRA cold-start LOPO (F1=0.5263/AUC=0.7291), Qwen+LoRA Stage-1-pretrained LOPO (F1=0.5782/AUC=0.8331), and the 6 LLM baseline configs (zero-/few-shot × Claude/GPT-4o/local Llama3.1:8b). Worth a sentence noting Qwen+LoRA trails both encoders and most prompting baselines despite pretraining helping (+0.052 F1, +0.104 AUC from pretraining alone).
- **Data source:** `gcp_results/lopo_modernbert_results.json`, `gcp_results/lopo_qwen25_1.5b_lora_results.json`, `gcp_results/lopo_qwen25_1.5b_lora_pretrained_results.json`, `gcp_results/llm_*_results.json` (6 files).

---

## 2. Needs expansion of existing content (partial credit already in the manuscript)

### R2.2 — Cherry-picking heuristic detail
- **Where:** 3.4 "Labeling the TAWOS Dataset" (line 119) is where the heuristic is currently described; 5.4 "Threats to Validity" (line 356) already acknowledges the heuristic "introduced noise and potentially systematic biases."
- **What's already there:** Table `tab:lopo` (line 246) already functions as a partial ablation — Baseline (0 augmented) vs. Aug-250/500/750/1000/all — showing performance degrades past 250 samples. That data point (noise increases with augmentation volume) directly answers part of the reviewer's "how much is this noise driving results" question and should be called out explicitly as such in 3.4, not left implicit.
- **What's still missing:** How the architectural phrases themselves were collected and weighted (the reviewer's specific ask). This lives in `LABELLING_METHODOLOGY.md` — check whether that level of detail (phrase list source, weighting scheme) is already written there and can be summarized into 3.4, or whether it needs to be written fresh.

### R2.4 — Downstream applications / attribution deep-dive
- **Where:** 4.6 "Attribution Analysis" (line 326) and 5.3 "Practical Implications" (line 348) already discuss the docker/helm/plugin/connector over-indexing finding and gesture at downstream use (defect-rate screening).
- **What's still missing:** The reviewer wants this "fleshed out," not just mentioned. Check whether `gcp_results/attribution_analysis.pdf` has findings beyond what's already summarized in prose (e.g., per-project attribution differences, not just aggregate word lists) that could extend 4.6. Also consider whether 5.3's downstream-application paragraph (line 354, currently one sentence ending in "left to future work") should absorb more of R1.3's qualitative analysis once that's done, since the two overlap.

---

## 3. Fast, low-effort fix — do whenever convenient

### R1.5 — "Design Mining" naming
- **Where:** Title, Abstract, and the opening of 2.2 "Design Mining" (line 79) and 3.1 "Overview" (line 105).
- **What:** A framing decision, not new analysis — either (a) add one or two sentences early (abstract + intro) explicitly scoping "design mining" as ticket-level text classification rather than design-artifact extraction, or (b) consider a title/terminology adjustment if you want to preempt the expectation mismatch entirely. This is a wording decision only you can make; the guide's job stops at flagging where it needs to land.

---

## 4. Larger write-ups needing new analysis (start after items 1–2 land)

### R2.6 — LOPO variance / generalizability discussion
- **Where:** 5.2 "Cross-Project Generalization Challenges" (line 342) already states the F1 range (0.48–0.83) and attributes it to project-specific language — reviewer wants this argument substantiated, not just asserted.
- **What's needed:** Per-project breakdown already exists in `gcp_results/lopo_*_results.json` — pull the full per-project F1 table (not just the two examples cited in text, SERVER/DM/CONFSERVER/DNN) and connect it explicitly to the attribution findings in 4.6/R2.4 (i.e., show that the low-F1 projects are the ones where attribution reveals heavy reliance on project-specific terms). This ties R2.6, R2.4, and R1.3 together — doing them as one combined analysis pass is more efficient than three separate ones.

### R1.3 — Qualitative classification error analysis
- **Where:** New subsection in Results (after 4.6) or folded into 5.2, depending on how deep it goes.
- **What's needed:** Actual false-positive/false-negative example inspection — pull misclassified tickets from the LOPO runs for the worst-performing projects (SERVER, DM) and best (CONFSERVER), and characterize what's driving the errors (vocabulary, ticket-type confusion, ticket length, etc.). This requires re-inspecting raw predictions, not just aggregate metrics — no shortcut from existing artifacts.

### R2.1 — Workflow diagram
- **Where:** Likely early in 3.1 "Overview" (line 105), before the prose walks through Stage 1 → Stage 2 → LOPO.
- **What's needed:** Purely visual — a flowchart of SO pretraining → TAWOS transfer → LOPO cross-validation. No data dependency; can be done independently and in parallel with everything else. Low risk, whenever you have bandwidth for a diagramming pass.

---

## 5. Start now, finishes last: second-annotator study

### R2.5 — Single-annotator ground truth / inter-rater reliability
- **Where:** 3.4 "Labeling the TAWOS Dataset" (line 119) for methodology, 5.4 "Threats to Validity" (line 356, which already flags this exact concern in one sentence — "manually labeled data would be performed by committee") for the discussion.
- **What's needed:** This is the one item that isn't a writing or code task — it requires recruiting a second human annotator, deciding a sampling strategy (how many of the 1,786 manually labeled tickets get double-labeled), and computing an agreement statistic (Cohen's kappa is standard). Because this has real-world lead time (finding a rater, scheduling their time), it's worth kicking off the logistics now even though the write-up will be the last thing drafted — everything else on this list can be done with code/writing alone.

---

## Suggested order of attack

1. **Resolve Table 1** (§0) — blocking, and touches the same section as item 3 below.
2. **Kick off the R2.5 annotator logistics** (§5) in parallel — longest lead time, no code dependency on anything else.
3. **Write up R1.4, R1.2** (§1) — data is ready, these are the fastest wins.
4. **Write up R1.1/R2.3** (§1) — biggest addition, but no missing data, just needs the Qwen-overhead gap (§1, R1.2 sub-item) filled first if you want the overhead table complete before citing it.
5. **Expand R2.2, R2.4** (§2) — check what's already in `LABELLING_METHODOLOGY.md` / `attribution_analysis.pdf` before drafting new text, since some of this may already exist and just need surfacing into the manuscript.
6. **R1.5 naming fix** (§3) — quick, do whenever.
7. **R2.6 + R1.3 combined pass** (§4) — do together, they share the same per-project data.
8. **R2.1 diagram** (§4) — independent, whenever there's bandwidth.
9. **R2.5 write-up** (§5) — once the second-annotator data (started in step 2) comes back.
