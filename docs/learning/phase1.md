# Phase 1: event weights, the notebook selection, Z pairing, and the notebook's N_sig

## 1. Event weights

A simulated sample stands for a process with cross-section σ. To compare it with data
recorded with integrated luminosity L, each simulated event gets a weight such that the sum of
weights of selected events is the expected number of selected events in data:

    w_i = L · 1000 · σ · ε_filter · k · (mcWeight_i · SF_i) / Σ_all mcWeight

- **L** is in fb⁻¹ and **σ** in pb. The factor 1000 converts fb⁻¹ to pb⁻¹.
- **ε_filter** is the generator filter efficiency. **k** is a higher-order correction to σ.
- **mcWeight** is the generator weight of the event.
- **SF_i** = SF_PILEUP · SF_ELE · SF_MUON · SF_LepTRIGGER. These scale factors correct the
  simulation's pile-up profile and its lepton identification and trigger efficiencies to those
  measured in data.
- **Σ_all mcWeight** (`sum_of_weights`) is summed over all generated events, before any skim
  or selection. Dividing by it makes the selection efficiency, including the 4-lepton skim,
  appear automatically.

**Negative weights.** NLO generators (Sherpa, MadGraph5_aMC@NLO) produce events with
negative mcWeight. In the ZZ sample 700600, 8.7% of events are negative. Only the signed sum
is an unbiased estimate of the expected yield and shape. The notebook takes `abs()` of every
factor. Because |a·b·c| = |a|·|b|·|c| and the normalization L·1000/Σw is positive, the
notebook weight is exactly |w_i|. Using |w_i| inflates the yield of samples with negative
weights and distorts their shapes. We keep it only to reproduce the notebook. Everywhere else
we use signed weights. The one exception is classifier training (Phase 4), where the loss
needs non-negative weights.

**MC statistical uncertainty.** The uncertainty on a bin content Σw is √(Σw²).

## 2. The notebook selection

The `exactly4lep` skim keeps events with exactly four reconstructed leptons. The notebook
applies these cuts, in order:

1. **Trigger.** A single-electron or single-muon trigger fired (`trigE or trigM`). Without a
   trigger the event would not have been recorded in data, so simulation must require it too.
2. **Trigger match.** At least one lepton is matched to the trigger object. This ensures the
   trigger fired on a selected lepton, so trigger scale factors apply.
3. **pT thresholds** of 20, 15 and 10 GeV, intended for the three leading leptons. These sit
   above the trigger and identification thresholds and reduce fake leptons.
4. **Identification and isolation** for all four leptons. Electrons: loose ID and loose
   isolation. Muons: medium ID and loose isolation. This suppresses non-prompt leptons from
   heavy-flavour decays and jets faking leptons (the Z+jets and tt̄ backgrounds).
5. **Flavour content.** The sum of `lep_type` (11 = e, 13 = μ) is 44 (4e), 48 (2e2μ) or
   52 (4μ). This rejects 3e1μ and 1e3μ.
6. **Charge.** The sum of charges is zero.

The four-lepton mass is m4l = √((ΣE)² − |Σp|²) of the four-vector sum.

**Quirk 1: "leading" is not leading.** The notebook applies the pT cuts to `lep_pt[:, 0]`,
`[:, 1]` and `[:, 2]`, assuming the leptons are pT-ordered. In this release they are stored
electrons first, then muons, and are not sorted by pT within a flavour. Only about a quarter of
events are pT-ordered. The cut actually applied is "the leptons in stored positions 0, 1, 2
have pT above 20, 15, 10 GeV".

This is strictly tighter than the intended cut. If positions 0, 1, 2 pass, then the largest pT
is above 20, the second largest is at least min(pT₀, pT₁) > 15, and the third largest is at
least min(pT₀, pT₁, pT₂) > 10. So every event passing the index cut passes the sorted cut. On
the ZZ simulation, after the other cuts, the index cut keeps 6365 events where the sorted cut
keeps 7870. We keep the index cut only to reproduce the notebook. The corrected selection sorts
by pT.

**Quirk 2: charge and flavour sums are not pairs.** A charge sum of zero and a type sum of 48
also accept e⁺e⁺μ⁻μ⁻. That event cannot be two Z bosons, since a Z decays to a same-flavour,
opposite-charge (SFOS) pair. Section 3 fixes this.

**Quirk 3 (cosmetic).** The notebook computes a weight for data too, because the check
`'data' not in s` is always true when the key is `'Data'`. It also prints unweighted MC counts
per batch, because every URL contains "opendata". Neither affects the plot or the numbers.

## 3. Z1/Z2 pairing

There are three ways to split four leptons into two pairs: (12)(34), (13)(24) and (14)(23). A
split is valid if both pairs are SFOS.

- **4e and 4μ** (two positive, two negative): two of the three splits are valid.
- **2e2μ** with e⁺e⁻μ⁺μ⁻: exactly one split is valid.
- **e⁺e⁺μ⁻μ⁻**: no split is valid, so the event is rejected.

Among the pairs in valid splits, **Z1** is the one with mass closest to m_Z = 91.1876 GeV, and
**Z2** is its partner. For a Higgs boson at 125 GeV, one Z is nearly on shell and the other is
off shell (m34 between about 12 and 50 GeV). Choosing the pair closest to m_Z picks the
on-shell one.

The cuts are then:
- **50 < m12 < 106 GeV.** Z1 is compatible with a real Z.
- **m34 > 12 GeV.** This removes low-mass Drell-Yan pairs, quarkonia, and leptons from photon
  conversions or heavy-flavour decays.

The full ATLAS analysis also uses an m4l-dependent m34 threshold, a ΔR separation between
leptons, and a J/ψ veto on alternative pairings. They are not used here; they are listed as
limitations.

## 4. Why the notebook's N_sig includes the background

The notebook draws the signal on top of the background with
`signal_heights = ax.hist(signal_x, ..., bottom=mc_x_tot)`. matplotlib's `hist` returns the
histogram counts **without** the `bottom` offset; checked with matplotlib 3.11.2, where counts
[0, 5] with bottom [10, 10] return [0, 5]. The notebook then adds the background back:
`signal_tot = signal_heights[0] + mc_x_tot`. So

    N_sig = Σ_{bins 17-19} signal_tot = S + B = 28.729,  N_bg = B = 10.301,  S = 18.428

in the window 122.5-130 GeV (bins 17 to 19 of 2.5 GeV from 80 GeV). The notebook's text says
"115 to 130 GeV", but the code sums 122.5-130 GeV.

The printed significance is (S + B)/√(B + 0.3 B²) = 4.426. Two things are wrong with it. The
numerator includes the background. And the 0.3 B² term is unexplained. If it was meant to be
a background uncertainty σ_B, then σ_B² = 0.3 B² means σ_B ≈ 0.55 B, not 30%. A 30%
uncertainty would be (0.3 B)² = 0.09 B².

The stored outputs were also produced before signal sample 346310 was added to the notebook
(commit eae7616). We reproduce them with the 7 samples that were used, and report the
8-sample numbers too.

## 5. Counting significances done properly

For a counting experiment with expected signal s and background b, the median discovery
significance under the signal hypothesis (Cowan, Cranmer, Gross, Vitells, EPJC 71 (2011) 1554)
is

    Z_A = √(2[(s + b) ln(1 + s/b) − s])

This reduces to s/√b only when s ≪ b. Here s/b ≈ 1.8, so s/√b = 5.742 overstates the
sensitivity (Z_A = 4.699).

If the background is known only to ±σ_b, from an auxiliary Poisson measurement (the "on/off"
problem), then

    Z_A = √(2[(s+b) ln((s+b)(b+σ_b²)/(b² + (s+b)σ_b²)) − (b²/σ_b²) ln(1 + σ_b² s/(b(b+σ_b²)))])

This gives 4.377 for σ_b = 10% and 3.075 for σ_b = 30%. These are counting approximations
for the comparison table. The full shape fit comes in Phase 3.

## 6. What the corrected selection changes, one step at a time

`results/phase1.json` (`selection_comparison`) changes one thing at a time. It gives S, B and
Z_A in the 122.5-130 GeV window:

| Stage | S | B | Z_A |
| --- | ---: | ---: | ---: |
| 1. Notebook, abs weights, 7 signal samples (the reproduction) | 18.43 | 10.30 | 4.70 |
| 2. Same with all 8 signal samples | 19.01 | 10.30 | 4.83 |
| 3. Signed weights | 18.94 | 9.68 | 4.91 |
| 4. pT cut on pT-sorted leptons | 31.53 | 15.54 | 6.42 |
| 5. Z pairing, 50 < m12 < 106, m34 > 12 GeV | 28.84 | 6.64 | 7.82 |

Most of the gain comes from fixing the pT ordering, not from the pairing. The pairing itself
removes no data event: after the lepton cuts, no data event is e⁺e⁺μ⁻μ⁻ (see
`results/cutflow.md`). The m12 and m34 cuts then cut the background in the window by more than half.
Most of what they remove is ZZ* (14.3 to 6.6 events); the other backgrounds fall from 1.2 to
0.1 events.

## 7. Data and simulation do not agree in normalization

With signed weights the simulation is well below data where ZZ* dominates. In 180-250 GeV,
data/MC is 1.42 for the notebook selection and 1.40 for the corrected one. With the
notebook's abs weights it is 1.10.

The weights themselves are correct. `scripts/check_normalization.py` sums mcWeight over the
unskimmed ZZ* file (248,881 of the 250,000 generated events). That sum is 0.996 of
`sum_of_weights`, so the denominator is the signed sum over the same sample.
`results/provenance.json` shows the release metadata agree with the file branches for this
sample.

The likely reason is what the background simulation contains. The notebook's ZZ* sample
(700600, Sherpa 2.2.12 llll) is quark-initiated ZZ* only, with k-factor 1. The gluon-initiated
gg → ZZ* process, which ATLAS adds as a separate sample, is not in the notebook's list, and
neither are further higher-order corrections. We have not quantified these here, so this is
an explanation, not a measured result. The better agreement with abs weights is a
coincidence: abs inflates the Sherpa yield by Σ|w|/Σw.

**Consequence for the next phases.** The ZZ* normalization from simulation cannot be trusted
at the 30-40% level. This was seen in Phase 1, before any mass fit or significance on data.
- The mass fit (Phase 2) is unaffected: N_b floats freely there.
- For Phase 3, the background normalization treatment must be decided in its planning step.
  It must be justified by this known mismodelling, not by the observed significance.

## 8. Cross-section metadata

`results/provenance.json` compares the file's xsec × filteff × kfac with the release metadata.
Background samples agree to rounding. Five signal samples do not:
- 346310: −13.9%. The branch value is the quark-initiated ZH cross-section.
- ttH samples 346340, 346341, 346342: +3.6%, +2.8% and −19.7%.
- 346228: +0.1%.

Like the notebook, we use the file branches. In the window 122.5-130 GeV of the corrected
selection, 346310 is 2% of the signal (from 2 simulated events) and the ttH samples together
are below 0.5%. Over the whole mass range 346310 is 15% of the "signal", because it includes
every Higgs decay: most of its selected events lie far from 125 GeV, as expected for Higgs
decays other than H → ZZ* that still give four leptons with the Z. Later phases must keep this in mind when they call all Higgs samples
signal. Small samples also carry large weights: 346311 (W⁺H) adds 2.3% of
the window signal with a single simulated event (`paired_signal_share_in_window` in
`results/phase1.json`).
