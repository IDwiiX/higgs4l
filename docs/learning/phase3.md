# Phase 3: significance with a profile likelihood

## 1. Binned Poisson likelihood with nuisance parameters

The m4l spectrum in 110-160 GeV is split into bins i = 1…N. Simulation gives the expected
signal s_i and background b_i at 36.6 fb⁻¹. The expected count in bin i is

    ν_i(μ, θ) = μ · s_i(θ) + b_i(θ)

μ is the signal strength: μ = 0 means no Higgs, μ = 1 means the Standard Model rate. The
nuisance parameters θ are the quantities we do not know exactly but do not want to measure:
the luminosity, the background level and the finite size of the simulation. Each one either
floats freely or has an auxiliary measurement, a constraint term, that pins it down:

    L(μ, θ) = Π_i Pois(n_i | ν_i(μ, θ)) · Π_j c_j(a_j | θ_j)

The n_i are the observed counts and the a_j the auxiliary data. A Gaussian c_j means "θ_j
was measured as a_j with error σ_j". A Poisson c_j means "θ_j was measured by counting
events".

The modifiers in our model:
- **μ**, normfactor on the signal, the parameter of interest.
- **Luminosity**, one factor on the simulated signal, Gaussian constraint with σ = 0.87%
  (section 6). It is not put on the background. There, it would be degenerate with the free
  background normalization (section 7). With μ free it hardly matters for the significance. Under μ = 0 there is no
  signal to scale. It does enter the uncertainty on μ̂.
- **Background normalization**, one factor on the summed background (see section 7 for the
  choice).
- **MC statistics**, one factor γ_i per bin multiplying the total expected yield, Gaussian
  with relative width √(Σw²)/Σw of the simulated events in that bin (Barlow-Beeston
  "lite": one parameter per bin instead of one per sample per bin).

## 2. The profile likelihood ratio and q0

To test a value μ while letting the nuisance parameters adjust, the likelihood is profiled:

    λ(μ) = L(μ, θ̂̂(μ)) / L(μ̂, θ̂)

θ̂̂(μ) maximizes L at fixed μ, and (μ̂, θ̂) is the global maximum. λ lies between 0 and 1, and
values near 1 mean μ fits the data about as well as the best value.

For discovery the null hypothesis is μ = 0 and the statistic is

    q0 = −2 ln λ(0)   if μ̂ ≥ 0,   q0 = 0   if μ̂ < 0.

The one-sided form encodes that a deficit of events is not evidence for a signal. Large q0
means the background-only hypothesis fits badly. The p-value is p0 = P(q0 ≥ q0,obs | μ = 0),
and the significance is Z = Φ⁻¹(1 − p0).

## 3. Asymptotic distribution

By Wilks' theorem, and its boundary version from Chernoff because μ̂ is cut at 0, q0 under
μ = 0 follows asymptotically a half-and-half mixture of a spike at 0 and a χ² with one degree of
freedom. Then √q0 is the positive half of a standard normal, and

    p0 = 1 − Φ(√q0),   Z = √q0.

"Asymptotically" means large expected counts. With b = 5 the Poisson counts are coarse and
the formula is only approximate (section 8).

## 4. The Asimov dataset

The expected significance is the median Z we would get if the signal is really there
(μ = 1). Running toys to find it is slow. Cowan, Cranmer, Gross and Vitells (EPJC 71 (2011) 1554)
showed that one artificial dataset gives it. The Asimov dataset sets every count, including
the auxiliary data, equal to its expectation under μ = 1 with nominal nuisance parameters:
n_i = s_i + b_i, a_j = nominal. On that dataset the fit returns μ̂ = 1 and θ̂ = nominal
exactly, and

    Z_A = √q0,A

is the median significance in the asymptotic limit. Counts need not be integers, because the
Poisson term is written with Γ(n + 1).

**Single bin, known background.** L(μ) = Pois(n | μs + b). With n = s + b, μ̂ = 1 and the
fit at μ = 0 has nothing to adjust:

    q0,A = 2 [ (s + b) ln((s + b)/b) − s ]    ⇒   Z_A = √(2[(s + b) ln(1 + s/b) − s]).

For s ≪ b, a Taylor expansion of the logarithm gives back s/√b. For s = 10, b = 5:
Z_A = 3.5998 (P3-a), against s/√b = 4.47. The naive formula overstates the significance
whenever s is not small compared with b.

## 5. Background uncertainty: the on/off problem and why shapesys matches

Cowan et al. model an uncertain background as an auxiliary counting measurement, the "off"
region. It has m events with expectation τ·b, and τ is chosen so that the off region alone
measures b with error σ_b: the estimate b̂ = m/τ has variance b/τ = σ_b², so τ = b/σ_b².

    L(μ, b) = Pois(n | μs + b) · Pois(m | τ b)

Asimov: n = s + b, m = τb. The global fit gives μ̂ = 1 and b̂ = b. At μ = 0 the background
is fitted to both regions: setting ∂ln L/∂b = 0 with μ = 0 gives

    b̂̂ = (n + m)/(1 + τ).

The linear terms of the two Poisson logs cancel, because n + m − (1 + τ)b̂̂ = 0. That leaves

    q0,A = 2 [ n ln(n/b̂̂) + m ln(m/(τ b̂̂)) ].

Substituting n = s + b, m = b²/σ_b², τ = b/σ_b² and b̂̂ = (b² + (s + b)σ_b²)/(b + σ_b²):

    n ln(n/b̂̂)     = (s + b) ln[ (s + b)(b + σ_b²) / (b² + (s + b)σ_b²) ]
    m ln(m/(τb̂̂)) = −(b²/σ_b²) ln[ 1 + σ_b² s / (b(b + σ_b²)) ]

which is the closed formula in `stats.z_asimov_unc` and in REFERENCE_VALUES P3-b, c, e, g.
As σ_b → 0 it returns to the known-background formula.

**pyhf `shapesys` is this likelihood.** For a background b with uncertainty σ_b, shapesys
multiplies b by γ and adds the constraint Pois(a | γ·τ_s) with τ_s = (b/σ_b)² and
auxiliary data a = τ_s. Write b' = γb, the true background. Then the constraint is
Pois(a | τ b') with τ = τ_s/b = b/σ_b², and the Asimov auxiliary count a = τ_s = τ b = m.
These are the same likelihood and the same Asimov data, only parametrized by γ instead of b,
so they give the same q0. The pyhf test therefore checks the code, not an approximation.

Other pyhf modifiers would not match. `normsys` and `staterror` use Gaussian constraints
(normsys with an exponential interpolation). They agree with the formula only when σ_b ≪ b,
and differ visibly at 30%.

## 6. Luminosity uncertainty

ATLAS calibrated the Run 2 luminosity with van der Meer scans
(Eur. Phys. J. C 83 (2023) 982, arXiv:2212.09379). The paper's Table 8 gives 3.24 fb⁻¹
(1.13%) for 2015 and 33.40 fb⁻¹ (0.89%) for 2016. The text below the table gives the absolute
uncertainties as 0.0367 and 0.296 fb⁻¹, with a correlation of 0.579 between the two years.
Adding the years with the correlation:

    σ² = 0.0367² + 0.296² + 2 · 0.579 · 0.0367 · 0.296   ⇒   σ = 0.32 fb⁻¹ on 36.64 fb⁻¹ = 0.87%.

The sum, 36.64 fb⁻¹, is the 36.6 fb⁻¹ of the release. Most Run 2 papers used an earlier,
preliminary calibration. The same paper states that the final calibration raised the 2015-16
luminosity by 1.2% and roughly halved the uncertainty. For example, arXiv:1902.01636 quotes
36.1 fb⁻¹ with 2.1%.

## 7. Choices fixed for this analysis (before any fit to data)

- **Selection.** Z-paired selection, sorted pT, signed weights, all 8 signal samples (as in
  phase 2).
- **Range and binning.** 110-160 GeV in 20 bins of 2.5 GeV: the SPEC default range on the
  phase 1 histogram grid, with bins close to the 2.1 GeV mass resolution. The 130-140 GeV data
  excess seen in phase 2 played no part in this choice. 5 GeV bins are a reported
  cross-check only.
- **Background normalization.** A free factor on the summed background, fixed by the
  sidebands of the range. The simulation is about 40% below data at 180-250 GeV, where ZZ*
  dominates (phase 1). A tight constraint would therefore assume something our own control
  region contradicts. The price is a lower expected significance, and the table also shows
  the fixed-background number.
- **Modifiers.** μ (bounds 0-10), luminosity 0.87% on the signal, the free background
  factor, and staterror per bin. The fixed-background comparison model has no background
  normalization uncertainty at all.
- **MC statistics are a real limitation.** The background simulation in 110-160 GeV is
  sparse. The relative MC error of the background in each 2.5 GeV bin is 29-127%. Combined
  with the signal, the width of the staterror factor is 6-115% (`mc_stat_rel` in
  results/significance_expected.json). One Z → μμ + jets event (sample 700323, weight 5.8)
  sits in the 117.5-120 GeV bin, next to the peak, and carries about 11% of the background in
  the whole range. 53 of the 567 background events have negative weights. Together they
  sum to −10.7 events, and they partly cancel the others in some bins. The
  Gaussian staterror constraint is a poor approximation at these widths, so the background
  shape is only loosely constrained bin by bin. This choice was fixed before unblinding and
  is kept. A Poisson (shapesys) treatment of MC statistics would only be a cross-check.
- **Toys.** 1000 s+b and 1000 b-only toys for the full model, seed 20251023. For P3-h,
  10⁷ single-bin toys with the same seed.

## 8. Where asymptotics are only approximate

For P3-a (s = 10, b = 5), q0 depends only on the integer count n. Under μ = 1 the median count is
15, so the median q0 is exactly q0,A: the Asimov statement holds. Under μ = 0, however, the
true tail probability of n ≥ 15 is 2.26e-4 (Z = 3.51), while the asymptotic formula gives
1.59e-4 (Z = 3.60). Toys show this directly. The asymptotic formula overstates the
significance slightly for a background of 5 events. For our full model, the background
in the peak bins is also only a few events, so the toy cross-check is reported next to the
asymptotic number.

For the full model the toys (results/significance_expected.json) show the same direction.
The median √q0 of 1000 s+b toys is 6.50 ± 0.04, against Z_A = 6.64, which is 3.4 standard errors
low. The Asimov number overstates the expected significance by about 2%, probably because
of the small counts per bin and the Gaussian MC-statistics constraints. Among 1000 b-only
toys, the fractions with q0 ≥ 1, 4 and 9 are 0.139 ± 0.011, 0.030 ± 0.005 and
0.002 ± 0.001, against 0.159, 0.023 and 0.0013 asymptotically. Toys of this size test the
null distribution only up to Z ≈ 3, so every Z above 3 below is asymptotic.

## 9. Expected results (simulation and Asimov data)

From `scripts/significance.py`, results/significance_expected.json, fixed before the data fit:

- **Shape fit, background normalization free:** Z_A = 6.64, with μ̂ = 1 +0.21 −0.19 on
  Asimov data.
- **Background fixed to simulation:** 8.74. The free normalization therefore costs 2.1 units
  of expected Z. This is the price of not trusting a simulation that is 40% low at
  180-250 GeV.
- **5 GeV bins (cross-check):** 6.37.
- **Counting in the 122.5-130 GeV window, corrected selection:** Asimov 7.82 with a known
  background, 5.38 with a 30% background uncertainty.

## 10. Result on data (unblind step)

The frozen model was fitted once to the 124 data events in 110-160 GeV
(`scripts/significance_data.py` → results/significance.json):

- **Observed significance: Z = 3.96** (asymptotic, p0 = 3.8e-5). The expected value is 6.64.
- **Signal strength: μ̂ = 0.72 +0.22 −0.20.** It is 1.3 standard deviations below the
  Standard Model value of 1. This matches phase 2, where the fit found N_s = 24.9 against
  39.1 expected.
- **The background is much larger than simulated.** μ̂_b = 2.09. The staterror factors pull
  some bins down, so the fitted background is 96.6 events against 53.7 simulated (×1.80), and
  the fitted signal is 27.4 events. This is the same effect phase 2 found in 110-140 GeV.
- **Why the observed Z is below the expected Z.** An a-posteriori diagnostic helps here: the
  Asimov Z at μ = 1, with every nuisance parameter at its fitted value, is 5.18. The larger
  background alone takes the expectation from 6.64 to 5.18. The low μ̂ takes it the rest
  of the way to 3.96.
- **Background fixed to simulation:** Z = 8.63 and μ̂ = 1.04. This is not a valid result.
  The data reject that background normalization. The fit absorbs the excess through the
  MC-statistics factors and the signal. It is shown only because the comparison table needs
  the row.
- **5 GeV bins:** Z = 4.18 and μ̂ = 0.77, consistent with the nominal binning.
