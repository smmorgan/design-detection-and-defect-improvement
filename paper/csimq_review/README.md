# Chapter 2 → standalone review article (CSIMQ)

Dissertation Chapter 2 ("Software System Design") reworked as a standalone
critical review + research agenda.

- **Target venue:** *Complex Systems Informatics and Modeling Quarterly*
  (CSIMQ), RTU Press. Diamond open access (no APC), CC BY 4.0.
  Author guidelines: https://csimq-journals.rtu.lv
- **Supersedes:** `../systems_review/` (MDPI *Systems*), which is frozen. The
  argument, table, and figure are carried over unchanged. The prose differs
  from the MDPI version in title-cased headings, IEEE-style `\cite`, wider
  Table 1 columns, sole authorship, and the removal of first-person phrasing
  and em dashes. See that
  directory's README for what changed relative to the dissertation chapter and
  where to cut if length becomes an issue.
- **Author:** sole-authored (Steven M. Morgan). The text avoids first-person
  references and em dashes; keep it that way in revisions.
- **Files:** `chapter2_csimq.tex`, `chapter2_refs.bib`, plus the CSIMQ template
  files `csimq.cls`, `csimq-header.pdf`, `IEEEtran.bst`.

## Building

XeLaTeX is required (the class loads `fontspec` + Times New Roman):
`xelatex` → `bibtex` → `xelatex` ×2. Compiles on Overleaf with the compiler set
to XeLaTeX (Menu → Settings → Compiler) and the main document set to
`chapter2_csimq.tex`. If Times New Roman is unavailable, `TeX Gyre Termes` is a
layout-equivalent fallback, but restore Times New Roman before submitting.

## Before submitting: open items

- [ ] **Layout check.** `csimq.cls` disables hyphenation (`hyphenat[none]`);
      check Table 1 (`tab:landscape`) for text overflowing its narrow columns,
      plus Figure 1 (TikZ) width and long URLs in the reference list.
- [x] **Front matter.** Sole-authored; ORCID iD filled in.
      Issue/pages/DOI/PII/dates stay as `xx` (the editors fill them in).
- [ ] **Back matter.** The MDPI back matter (Abbreviations, Author
      Contributions, Funding, Data Availability, Conflicts of Interest) was
      dropped. Check the CSIMQ guidelines for which statements, if any, are
      expected; at minimum consider an Acknowledgements section.
- [ ] **CSIMQ requirements.** Confirm against the author guidelines: length
      limit (body is ~10,200 words), abstract length (235 words), keyword count
      (6), cover letter.
- [ ] **Two references need verification.** `papageorgiou2008` and `kunc2016`
      are placeholders marked `% VERIFY` with reconstructed titles. Confirm or
      replace; do not submit as-is.
- [ ] **Re-pitch for CSIMQ's audience.** §1 and §1.1 were written for
      *Systems*; sharpen the enterprise-modeling/information-systems angle and
      consider citing relevant CSIMQ papers.
- [ ] **Dissertation prior-publication rules.** Document the divergence from
      Chapter 2 in the cover letter.
- [ ] **Consider** expanding the technical debt coverage gap in §14 beyond a
      one-line mention.
