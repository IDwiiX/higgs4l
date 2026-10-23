# Reference values for validation

Authoritative test targets. Closed-form numbers are recomputed by
`references/verify_reference_values.py`. Numbers from the official ATLAS notebook are its
stored outputs (HZZAnalysis.ipynb, release 2025e-13tev-beta, 36.6 fb^-1, skim exactly4lep).
Never loosen a tolerance on your own.

## Phase 1: reproduce the official notebook, then correct it

| ID | Quantity | Target | Tolerance |
| --- | --- | --- | --- |
| P1-a | Notebook selection, weighted signal + background in 122.5-130 GeV (bins 17-19 of 2.5 GeV from 80 GeV) | 28.729 | 0.1% |
| P1-b | Same window, weighted background only | 10.301 | 0.1% |
| P1-c | Notebook metric (S+B)/sqrt(B + 0.3 B^2) | 4.426 | 0.1% |
| P1-d | Signal alone in the window, (P1-a) - (P1-b) | 18.428 | 0.1% |
| P1-e | Invariant mass of hand-built four-vectors (unit test) | exact | 1e-9 relative |
| P1-f | Z1/Z2 pairing on hand-built events, incl. e+ e+ mu- mu- rejected (unit test) | exact | exact |
| P1-g | Event weight formula on one hand-filled record (unit test) | exact | 1e-12 relative |

If P1-a or P1-b fail with the notebook's own code path, check the release, the file list and
the atlasopenmagic version before touching the selection: the beta files may have changed.
Record what you find.

Counting-approximation significances for the window (for the comparison table, not a fit):

| ID | Method | Value |
| --- | --- | --- |
| P1-h | Naive S/sqrt(B) | 5.742 |
| P1-i | Asimov median significance, no background uncertainty | 4.699 |
| P1-j | Asimov, 10% background uncertainty | 4.377 |
| P1-k | Asimov, 30% background uncertainty | 3.075 |

## Phase 2: mass fit

| ID | Check | Target | Tolerance |
| --- | --- | --- | --- |
| P2-a | Pull (m_hat - m_true) / sigma_hat over 2000 toys at expected yields | mean 0, width 1 | mean within +/- 0.05, width within +/- 0.05 |
| P2-b | Fit to simulation-only pseudo-data after calibration | 125.0 GeV | within the fitted uncertainty |
| P2-c | Fit converges (valid minimum, accurate covariance) on data and on every toy used | all | no failures, or failures counted and reported |
| P2-d | Comparison, not a test: ATLAS 36.1 fb^-1 four-lepton result | 124.79 +/- 0.37 GeV | report only |
| P2-e | Comparison, not a test: PDG average (2026 edition, pdglive S126M) | 125.13 +/- 0.11 GeV | report only |

Sources: ATLAS, Phys. Lett. B 784 (2018) 345, arXiv:1806.00242; PDG Live, Higgs mass.

## Phase 3: profile-likelihood significance

Single-bin pyhf models with signal s and background b must reproduce the closed-form Asimov
median significance (Cowan, Cranmer, Gross, Vitells, EPJC 71 (2011) 1554). With background
uncertainty, use a Poisson-constrained background (pyhf `shapesys`), which is the on/off
problem the formula describes; the script checks that equivalence numerically.

| ID | s | b | Background uncertainty | Z_A target | Tolerance |
| --- | --- | --- | --- | --- | --- |
| P3-a | 10 | 5 | none | 3.5998 | 1e-3 relative |
| P3-b | 10 | 5 | 10% | 3.4691 | 1e-3 relative |
| P3-c | 10 | 5 | 30% | 2.7826 | 1e-3 relative |
| P3-d | 5 | 20 | none | 1.0757 | 1e-3 relative |
| P3-e | 5 | 20 | 30% | 0.6273 | 1e-3 relative |
| P3-f | 18.428 | 10.301 | none | 4.6986 | 1e-3 relative |
| P3-g | 18.428 | 10.301 | 30% | 3.0754 | 1e-3 relative |
| P3-h | Toy-based vs asymptotic q0 on P3-a | | | agree | within the toys' statistical error |

## Phase 4: classifier

| ID | Check | Target |
| --- | --- | --- |
| P4-a | Train and test score distributions, signal and background separately | KS p-value reported; no target |
| P4-b | Background m4l shape in each category vs inclusive | chi2 p-value reported with the shape figure; no target |
| P4-c | Expected significance with categories vs without, on the test half | reported with both numbers; no target |
| P4-d | m4l is not among the features | exact |

P4-a and P4-b were pass/fail checks (p > 0.05) until the first classifier, gradient-boosted
trees, failed P4-a on simulation. They are now reported with their figures, as decided by the
project owner; the reasons are in docs/learning/phase4.md, section 9.
