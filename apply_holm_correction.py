#!/usr/bin/env python3
"""
Holm-Bonferroni correction across the full pre-registered spec-grid family
(RQ2_PREREGISTRATION.md Sec 5, corrected 2026-09-27): 6 spec cells (1 project
x quarter cell x 2 corrections, plus 2 missing-component handling rules x 2
corrections at the project x component x quarter unit) x 2 hypotheses (H1,
H2) = 12 tests. H3 is excluded from this family -- see the Sec 5 correction
note for why.

For H1, uses the permutation-null p-value where available (the pre-
registration's own stated preference over the chi-square LR p-value, given
strong within-series autocorrelation -- RQ2_INITIAL_RESULTS.md Sec 3). For H2,
uses the chi-square LR p-value (no permutation null was pre-registered for H2).

This is a plain data-entry script, not a pipeline -- the p-values below are
transcribed from each spec's fit_rq2_models.py / fit_rq2_component_models.py
run and must be updated by hand if any spec is re-run.
"""

import numpy as np
import pandas as pd


def holm_correction(df, p_col='p_value'):
    df = df.sort_values(p_col).reset_index(drop=True)
    n = len(df)
    raw_adj = [(n - i) * p for i, p in enumerate(df[p_col])]
    # Enforce monotonicity (each adjusted p >= the previous one) and cap at 1,
    # per the standard Holm step-down procedure: a running MAXIMUM from the
    # smallest p upward. (A reverse running minimum -- used here before
    # 2026-09-29 -- is Hochberg's step-up form, which is anti-conservative
    # relative to Holm and was not what the pre-registration specifies.)
    adj = np.maximum.accumulate(raw_adj)
    df['holm_adjusted_p'] = np.minimum(adj, 1.0)
    df['significant_at_0.05'] = df['holm_adjusted_p'] < 0.05
    return df


TESTS = [
    # unit, correction, missing_handling, hypothesis, p_value, note
    # All permutation p-values below use (exceed + 1) / (n_ok + 1) and the
    # pre-registered scheme='shuffle' null (within-series random permutation).
    # Rerun 2026-10-05 after fixing permutation_null's row-loss bug (see
    # RQ2_INITIAL_RESULTS.md: the original version lost each series' first 4
    # rows in every permuted refit) -- every H1 p-value below is new; H2
    # (chi-square, not permutation-based) is unaffected by that bug and
    # unchanged from the prior transcription. Primary unit:
    # logs/primary_{raw,corrected}.log. Component unit:
    # logs/component_{drop,bucket}_{raw,corrected}.log (2000 reps). A
    # circular-shift null (preserves each series' own autocorrelation,
    # which shuffle destroys) was also run for the 3 raw-ratio cells as a
    # non-pre-registered robustness check -- see RQ2_PREREGISTRATION.md Sec 11
    # and RQ2_INITIAL_RESULTS.md; it is reported alongside but does not
    # replace the shuffle p-value in this Holm family. Superseded logs are in
    # logs/superseded_2026-10-04_permfix/.
    ('project x quarter', 'raw', None, 'H1', 0.0979, 'permutation null (shuffle), 1000 reps'),
    ('project x quarter', 'raw', None, 'H2', 0.3126, 'chi-square LR, df=2'),
    ('project x quarter', 'corrected', None, 'H1', 0.2348, 'permutation null (shuffle), 1000 reps'),
    ('project x quarter', 'corrected', None, 'H2', 0.1734, 'chi-square LR, df=2'),
    ('project x component x quarter', 'raw', 'drop', 'H1', 0.0345, 'permutation null (shuffle), 2000 reps'),
    ('project x component x quarter', 'raw', 'drop', 'H2', 0.3768, 'chi-square LR, df=2'),
    ('project x component x quarter', 'corrected', 'drop', 'H1', 0.0435, 'permutation null (shuffle), 2000 reps'),
    ('project x component x quarter', 'corrected', 'drop', 'H2', 0.3948, 'chi-square LR, df=2'),
    ('project x component x quarter', 'raw', 'bucket', 'H1', 0.0070, 'permutation null (shuffle), 2000 reps'),
    ('project x component x quarter', 'raw', 'bucket', 'H2', 0.2074, 'chi-square LR, df=2'),
    ('project x component x quarter', 'corrected', 'bucket', 'H1', 0.0060, 'permutation null (shuffle), 2000 reps'),
    ('project x component x quarter', 'corrected', 'bucket', 'H2', 0.2288, 'chi-square LR, df=2'),
]

# Circular-shift null, raw-ratio cells only (power_equivalence_rq2.py hardcodes
# design_ratio_raw; not run for the corrected ratio -- the uncorrected ratio is
# already the primary reading per Sec 8/8b, so this gap doesn't affect the
# headline result). Reported for comparison, NOT part of the Holm family above.
CIRCULAR_NULL_RAW = [
    ('project x quarter', 'raw', None, 'H1', 0.2038, 'permutation null (circular), 1000 reps'),
    ('project x component x quarter', 'raw', 'drop', 'H1', 0.1948, 'permutation null (circular), 1000 reps'),
    ('project x component x quarter', 'raw', 'bucket', 'H1', 0.0679, 'permutation null (circular), 1000 reps'),
]


def main():
    df = pd.DataFrame(TESTS, columns=['unit', 'correction', 'missing_handling', 'hypothesis', 'p_value', 'note'])
    if len(df) < 12:
        print(f"WARNING: only {len(df)}/12 pre-registered tests populated -- "
              f"this is a partial family, not the final Holm correction. "
              f"Fill in the project x component x quarter cells before treating this as final.")
    result = holm_correction(df)
    pd.set_option('display.width', 140)
    print(result.to_string(index=False))

    circ = pd.DataFrame(CIRCULAR_NULL_RAW, columns=['unit', 'correction', 'missing_handling', 'hypothesis', 'p_value', 'note'])
    print("\nCircular-shift null (raw ratio only, robustness check -- not Holm-corrected, not pre-registered):")
    print(circ.to_string(index=False))


if __name__ == '__main__':
    main()
