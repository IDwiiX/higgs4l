# Phase 2: measuring the Higgs mass with an unbinned likelihood fit

## 1. Extended unbinned likelihood

We observe n events in 110 < m4l < 140 GeV with masses m_1 ... m_n. The model is a signal
density f_s(m | m_H) and a background density f_b(m), both normalized to 1 on the fit range,
with expected yields N_s and N_b. Two things are random: how many events there are,
n ~ Poisson(N_s + N_b), and where each one falls. The extended likelihood includes both:

    L(m_H, N_s, N_b) = e^{-(N_s + N_b)} / n! · Π_i [ N_s f_s(m_i | m_H) + N_b f_b(m_i) ]

Fits minimize the negative log-likelihood, dropping the constant ln n!:

    NLL = (N_s + N_b) − Σ_i ln[ N_s f_s(m_i | m_H) + N_b f_b(m_i) ]

Why "unbinned": each event enters with its exact mass. With only about 70 expected events in
the range, binning would throw away information and make the result depend on the bin edges.

Why "extended": the fitted yields carry the Poisson fluctuation of the total count, so N̂_s
comes with a correct uncertainty. The fitted m_H is the same as in a non-extended fit with a
free signal fraction.

## 2. The double-sided Crystal Ball

The reconstructed four-lepton mass of a Higgs boson is not Gaussian. Final-state radiation and
electron bremsstrahlung lose energy and build a low-mass tail. Rare mismeasurements build a
smaller high-mass tail. The double-sided Crystal Ball (DSCB) has a Gaussian core with a
power-law tail on each side. With t = (m − μ)/σ:

    f(t) ∝ exp(−t²/2)                          for −α_L ≤ t ≤ α_R
         ∝ A_L · (B_L − t)^(−n_L)              for t < −α_L
         ∝ A_R · (B_R + t)^(−n_R)              for t >  α_R

    A = (n/α)^n · exp(−α²/2),   B = n/α − α     (with α, n of the corresponding side)

A and B make the density and its first derivative continuous where the tails join the core.
The density is normalized numerically on 110-140 GeV.

**Which parameters are fixed.** The shape parameters σ, α_L, n_L, α_R and n_R are fitted once
to the signal simulation, then frozen. There are only about 40 expected signal events (results/mass_model.json), which
cannot constrain tails. The background shape (an exponential or a low-order polynomial,
chosen on simulation) is fixed the same way. Only m_H, N_s and N_b float in the fit to data.

Fits to weighted simulation maximize Σ_i w_i ln f(m_i). This gives consistent shape estimates
even with negative weights. Their naive uncertainties are not meaningful, but frozen shapes
do not need them.

## 3. Calibrating the peak position

The simulation is generated at m_H = 125.0 GeV, but the fitted DSCB mean μ_MC differs from
125.0. The tails are asymmetric, so the core does not sit at the true mass, and lepton energy
scales are not perfect. We measure the offset on simulation:

    δ = μ_MC − 125.0 GeV

In the fit to data the peak is placed at μ = m_H + δ, so the free parameter m_H is directly
the Higgs mass. This assumes δ is the same for masses near 125 GeV. The release has a single
mass point, so that is a stated limitation. The statistical uncertainty of δ, from the size of
the simulation sample, is reported. It is much smaller than the data uncertainty.

## 4. Profile-likelihood uncertainty

To quote an uncertainty on m_H alone, the other parameters (N_s, N_b) are profiled. At each
fixed m_H they take the values that maximize L. The profile likelihood ratio is

    λ(m_H) = L(m_H, N̂̂_s, N̂̂_b) / L(m̂_H, N̂_s, N̂_b)

By Wilks' theorem, −2 ln λ(m_H) behaves asymptotically like (m_H − m̂_H)²/σ². The 68%
interval is where −2 ln λ ≤ 1, that is where the NLL rises by 0.5 above its minimum. This is
what MINOS computes. The interval can be asymmetric, which matters with about 40 signal
events. HESSE instead takes the curvature at the minimum and is always symmetric. The
likelihood scan plots −2 ln λ against m_H, so the reader sees the shape of the interval.

## 5. Toys and pulls

To check that the fit is unbiased and that its uncertainties have the right coverage:

1. Generate a pseudo-experiment (toy) from the frozen model at m_true = 125 GeV. Draw
   n_s ~ Poisson(N_s,exp) and n_b ~ Poisson(N_b,exp), then sample the masses from f_s and
   f_b.
2. Fit it exactly as data would be fitted.
3. Compute the pull g = (m̂_H − m_true)/σ̂, where σ̂ is the MINOS error on the side towards
   the truth: the upper error if m̂_H < m_true, the lower error otherwise.

If the estimator is unbiased and the errors are right, g follows a standard normal
distribution. A mean away from 0 means a bias. A width above 1 means the errors are too small
(undercoverage); below 1, too large. With 2000 toys the statistical precision is
1/√2000 ≈ 0.022 on the mean and 1/√4000 ≈ 0.016 on the width, so the ±0.05 tolerance
(REFERENCE_VALUES P2-a) allows only small real deviations.

Every toy fit is checked for a valid minimum and an accurate covariance matrix. Failures are
counted and reported, never silently dropped (P2-c). With so few signal events, some toys
fluctuate to N̂_s ≈ 0. Then m_H is barely constrained, and the pull distribution can grow
non-Gaussian tails. That is a real property of a low-statistics measurement, not a bug to
hide.

## 6. Cross-checks

- **Simulation pseudo-data (P2-b).** Fit simulated events that were not used for the shapes
  or the calibration, and check that the calibrated fit returns 125.0 GeV within its
  uncertainty. Using the same events for both would make the test circular.
- **Gaussian signal.** Repeating the fit with a Gaussian signal shape shows how much the
  result depends on the tail model.

## 7. Choices fixed for this analysis (before any fit to data)

- **Selection and range.** Z-paired selection, 110 < m4l < 140 GeV. Free parameters: m_H, N_s
  and N_b. All shape parameters are frozen in `results/mass_model.json`.
- **Signal shape and calibration.** The DSCB is fitted on the H → ZZ* → 4ℓ samples (ggH 345060
  and VBF 346228), using only even eventNumber. This simulation has 163k events with small,
  nearly equal weights. The small VH and ttH samples are left out of the shape: one W⁺H event
  alone carries 2.3% of the window signal, and their inclusive Higgs decays do not all peak
  at 125 GeV. They still count in the expected N_s, and the toys generate all of N_s with the
  DSCB peak. That approximation concerns the 5% of N_s from these small samples.
- **Background shape.** Fitted on the ZZ* simulation (all 164 events in the range, 20.1 of the 27.6
  expected background events; `expected_background_by_sample` in results/mass_model.json). The candidates are an exponential (one parameter) and a second-order
  Bernstein polynomial (two). The exponential is kept unless the polynomial improves the
  weighted −2 ln L by more than 4. Refitting with all backgrounds, including one Z → μμ + jets
  event with weight 5.8, is a cross-check of the m_H shift.
- **Pull.** The pull is (m̂_H − 125)/σ, where σ is the MINOS error on the side towards the
  truth. Every toy is counted. Failed fits are reported, not dropped.
- **Check on simulation (P2-b).** The odd-eventNumber simulation, never used for the shape or
  the calibration, is fitted as weighted pseudo-data with weights ×2. The fit must return
  125.0 GeV within the expected data uncertainty, the median toy MINOS error. Its own
  MC-statistical error is reported too.
- **Weighted fits.** A weighted likelihood Σ w ln f gives consistent central values, but its
  Hesse errors refer to the sum of weights, not to the simulation's statistical power. Before
  any weighted fit whose error is quoted (δ, and the P2-b fit), we rescale the weights so that
  they sum to the effective number of events, n_eff = (Σw)² / Σw². The Hesse error is then
  the MC-statistical error. This is exact for equal weights and a good approximation for the
  nearly equal signal weights. For the mixed signal + background pseudo-data of P2-b it is
  only approximate, and it is quoted as such.

## 8. What the validation found

- **Toys (P2-a, P2-c).** 2000 toys give a pull mean of −0.010 ± 0.023 and a width of
  1.013 ± 0.016, and every fit converged (results/mass_toys.json). The expected uncertainty
  is 0.50 GeV.
- **Background content.** The expected background of 27.6 events includes 6.6 from three
  Z → μμ + jets events (sample 700323), one of them with weight 5.8. So a quarter of the
  expected background, and therefore part of the 0.50 GeV, rests on a single simulated event.
  The toys at simulated yields also ignore that data exceed simulation by a factor of 1.26 in
  110-140 GeV (results/phase1.json). The real uncertainty on data will be somewhat larger.
- **Held-out simulation (P2-b).** All three 700323 events have even eventNumber. The
  odd-half pseudo-data therefore contain 40.6 signal and only 21.9 background events instead
  of 39.1 and 27.6 (results/mass_simulation.json). The fit returns 125.15 GeV, within the
  0.50 GeV expected uncertainty.
- **Calibration sign.** Fitting the odd half of the H → ZZ* → 4ℓ simulation alone returns
  124.989 ± 0.007 GeV. A wrong sign of δ would give about 124.3 GeV.
- **Model dependence.** A Gaussian signal shape moves m̂ by +0.23 GeV, and the all-background
  shape by +0.09 GeV. Both are well below the statistical uncertainty.
- **MC-statistical error of P2-b.** The ≈ 0.26 GeV from the n_eff rescaling is approximate,
  because the pseudo-data mix very different weights.

## 9. Result on data (unblind step)

The frozen model was fitted once to the 84 data events in 110-140 GeV
(`scripts/mass_data.py` → results/mass_fit.json): m_H = 125.06 +0.93 −0.92 GeV.

- **Larger error than expected.** The fit finds N̂_s = 24.9 ± 7.9 and N̂_b = 59.1 ± 9.8,
  against 39.1 and 27.6 in simulation (−1.8 and +3.2 standard deviations). Counting in
  10 GeV ranges shows where this comes from. Data and simulated S+B agree in 110-120 GeV
  (16 vs 16.7) and in 120-130 GeV (42 vs 42.6), but 130-140 GeV has 26 events against 7.4.
  With the background shape frozen, the fit can only follow that sideband by raising N_b over
  the whole range, which takes events out of the peak. Fewer signal events over more
  background widen the likelihood.
- **How unusual the error is.** Of the 2000 P2-a toys at simulated yields, 0.9% have a MINOS
  error of 0.93 GeV or more. Toys at the fitted yields are an a-posteriori diagnostic: their
  median error is 0.78 GeV and 30% reach 0.93 GeV. So the error is ordinary given the
  yields, and the yields are what is unusual. Two of these 500 toys have a failed MINOS
  (N̂_s = 6.3 and 3.1, one interval reaching the 140 GeV edge). They are counted as failures
  and kept in the error fraction, and left out of the shape-shift statistics.
- **Goodness of fit.** The binned Poisson deviance in 12 bins of 2.5 GeV is 5.9 for 9 degrees
  of freedom (p = 0.75, approximate with these small counts). The frozen shapes describe the
  data; it is the normalization that departs from simulation.
- **Shape cross-checks.** A Gaussian signal shifts m̂_H by +0.91 GeV. In toys at the fitted
  yields the same shift has a mean of +0.43 GeV and a spread of 0.67 GeV: with a peak of 25
  events the two shapes weigh individual events differently enough to disagree by that much.
  The all-background shape shifts m̂_H by +0.41 GeV.
- **Comparison.** ATLAS (36.1 fb⁻¹, 4ℓ): 124.79 ± 0.37 GeV; PDG: 125.13 ± 0.11 GeV. The
  differences are +0.27 and −0.07 GeV. The ATLAS result comes from the same collisions, so the
  agreement is expected rather than an independent confirmation, and the two errors are not
  combined in quadrature.
