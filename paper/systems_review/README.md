# Chapter 2 → standalone review article

Draft of dissertation Chapter 2 ("Software System Design") reworked as a
standalone academic article.

- **Target venue:** *Systems* (MDPI) — Review article
- **Framing:** critical review + research agenda
- **Files:** `chapter2_article.tex`, `chapter2_refs.bib`
- **Length:** ~10,400 words of body text (MDPI *Systems* reviews typically run
  8,000–12,000; no hard limit, but trimmable — see "Where to cut" below)

## What changed from the chapter

The chapter was a survey organized by tradition, written to establish
background for the dissertation's research questions. A journal review needs a
contribution of its own, so the draft adds an argument the chapter only gestured
at and restructures the material to support it.

**The thesis:** all eight traditions share three failure modes, and those are
three views of one absence — there is no observational instrument for design
activity.

1. *Boundary blind spot* — every tradition locates difficulty at socio-technical
   boundaries and then delegates that region elsewhere.
2. *Measurement deficit* — every instrument for design value requires expert
   elicitation, so it does not scale, does not replicate, and is not independent
   of the designers being evaluated.
3. *Evidentiary asymmetry* — a literature of successes with no comparable record
   of failures, so effect sizes are unknown.

**Structural changes:**

- Added §2 (scope and method), §11 (synthesis + comparative table), §12 (the
  measurement gap), §13 (five-part research agenda), §14 (threats to validity).
- Each tradition section now ends with an explicit **Assessment** paragraph
  tying it to the three failure modes. These are new; the chapter's critiques
  were scattered mid-paragraph.
- Added Table 1: the eight traditions against a common comparative frame
  (unit of analysis, human agency, emergence, evidence base, falsifiability).
  This is new analysis, not in the chapter.
- Added Figure 1 (original TikZ): the measurement gap and how RA1–RA5 stage
  across it.
- The dissertation's Chapter 3 material (Aleti et al.'s optimization review,
  Zadeh on soft computing, design mining, rationale extraction) was pulled
  forward into §12 because the argument needs it — the article has to show the
  instrument is plausible, not just that it is missing.
- Project management (§10) gained an argument the chapter did not make: the
  waterfall/agile artifact asymmetry is a *measurement* confound, not just a
  methodological difference. This becomes RA3's main threat.

**Author position is disclosed** in §14. The agenda advocates the direction the
authors are already pursuing, and RA1 (construct validity) is stated at full
strength because it is where that program is most vulnerable.

## Before submitting — open items

- [ ] **Two references need verification.** `papageorgiou2008` and `kunc2016`
      are cited in the dissertation text but were missing from its reference
      list. The `.bib` entries are placeholders marked `% VERIFY` with
      reconstructed titles. Confirm or replace — do not submit as-is.
- [ ] **Figures 1–4 of the chapter were dropped deliberately.** The Zachman
      Framework, TOGAF ADM, the SEBoK value creation process, and Hieronymi's
      role-of-systems-science diagram are third-party copyrighted figures.
      MDPI is open access (CC BY), which makes reuse permission harder to
      obtain than for a subscription venue. Either secure permission or keep
      them out — the current draft describes them in prose instead.
- [ ] **Compile check.** No LaTeX toolchain was available in the drafting
      environment, so this has **not been compiled**. Structure was verified by
      script: 58 bib entries / 58 distinct cite keys with no mismatch either
      direction, no undefined `\ref`, balanced braces and environments. Run
      `pdflatex` + `bibtex` before trusting the layout.
- [ ] **Convert to the MDPI template.** Draft uses standard `article` class for
      portability. Download `mdpi.cls` from https://www.mdpi.com/authors/latex
      and move the frontmatter into `\Title` / `\Author` / `\abstract` /
      `\keyword`. Section order already matches MDPI's expected layout,
      including back matter (Abbreviations, Author Contributions, Funding, Data
      Availability, Conflicts of Interest).
- [ ] **Confirm co-authorship** with Jeff Daniels, and confirm the Author
      Contributions statement reflects the actual split.
- [ ] **Check dissertation embargo / prior publication rules.** MDPI generally
      permits material from a thesis, but the divergence from Chapter 2 should
      be documented in the cover letter.
- [ ] **Consider adding** the technical debt literature to §14's coverage gaps —
      it attacks an adjacent measurement problem from inside software
      engineering and a reviewer may well ask why it is absent.

## Where to cut, if length becomes an issue

In rough order of expendability:

1. §10 (project management) — longest section relative to its load-bearing
   role; the waterfall/agile artifact asymmetry is the only part RA3 needs.
2. §7 (design thinking) — weakest link to the measurement thesis.
3. §9 (systems science) — could fold into §11's synthesis, keeping Klir's
   "agreed object of study" criterion, which is the part that does work.

Do not cut §11–§13; they are the contribution.

## Relationship to the other manuscript

`paper/manuscript.tex` is the RQ1 empirical paper (design mining with
transformers on TAWOS, targeted at *Journal of Systems and Software*). This
article is its conceptual counterpart: it argues *why* the measurement problem
matters and what would have to be true for artifact-based measurement to count,
without depending on the empirical results. They can be submitted independently.
If both land, each should cite the other.
