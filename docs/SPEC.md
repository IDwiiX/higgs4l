# higgs4l specification

H -> ZZ* -> 4l with the ATLAS 2025 Open Data for education (release `2025e-13tev-beta`,
13 TeV, 2015-2016, 36.6 fb^-1, skim `exactly4lep`). Four analysis phases and a write-up.
Reference numbers are in `docs/REFERENCE_VALUES.md`.

## 1. Starting point and what this adds

The official ATLAS notebook `HZZAnalysis.ipynb` (atlas-outreach-data-tools/
notebooks-collection-opendata, `13-TeV-examples/uproot_python/`) selects four-lepton events,
plots m4l with stacked simulation and quotes a significance of 4.43 from
`N/sqrt(B + 0.3 B^2)` in a 122.5-130 GeV window, where `N` is signal plus background and the
0.3 term is unexplained in the notebook itself.

This project:
1. reproduces that notebook exactly, then fixes and documents its shortcuts;
2. measures the Higgs mass with a likelihood fit and a calibrated uncertainty;
3. computes the expected and observed significance with a profile likelihood (pyhf);
4. trains a classifier and measures how much it improves the expected significance.

Out of scope: systematic uncertainties beyond a few stated, simple ones; production-mode
or coupling measurements; anything requiring the 2024 research release.

## 2. Data and samples

Access through `atlasopenmagic` (`set_release('2025e-13tev-beta')`,
`build_dataset(defs, skim='exactly4lep', protocol='https', cache=True)`), as in the notebook.
Sample IDs (from the official notebook), kept in `config/analysis.yaml`:

- Data: `data`
- Signal, m_H = 125 GeV: 345060, 346228, 346310, 346311, 346312, 346340, 346341, 346342
- ZZ* background: 700600
- Other backgrounds (Z+jets, ttbar, ttV, VVV): 410470, 410155, 410218, 410219, 412043,
  364243, 364242, 364246, 364248, 700320, 700321, 700322, 700323, 700324, 700325

Event weight for simulation: `lumi * 1000 * xsec * filteff * kfac * mcWeight *
ScaleFactor_PILEUP * ScaleFactor_ELE * ScaleFactor_MUON * ScaleFactor_LepTRIGGER /
sum_of_weights`, lumi in fb^-1, xsec in pb.

Cache files locally under `data/cache/` (gitignored). After the selection, write the selected
events (four-vectors, weights, sample ID, eventNumber) to compact files under `data/selected/`.
Check the Open Data licence on the record page (opendata.cern.ch/record/93910); if it allows
redistribution, commit `data/selected/` so every later phase and CI run without the network.

## 3. Phases

### Phase 1: selection and baseline

Build:
- Reproduce the notebook selection exactly: trigger (trigE or trigM), at least one
  trigger-matched lepton, pT > 20, 15, 10 GeV for the three leading leptons, ID and
  isolation (electrons loose ID + loose iso, muons medium ID + loose iso), lepton types
  summing to 44, 48 or 52, charges summing to zero, m4l from the four-vector sum.
- A cutflow table (data counts and weighted MC per sample group after each cut).
- The m4l plot, 80-250 GeV in 2.5 GeV bins, data vs stacked simulation with MC stat band.
- Then a second, documented selection that pairs leptons into Z1 (same flavour, opposite
  charge, closest to the Z mass) and Z2, and adds 50 < m12 < 106 GeV and m34 > 12 GeV.
  The notebook's charge and type sums let through events such as e+ e+ mu- mu-, which cannot
  form two Z candidates; the pairing removes them. Compare its yields to the notebook selection.

Validation (Phase 1 in REFERENCE_VALUES.md): reproduce the notebook's window numbers
exactly, unit tests on invariant mass, pairing and weights, and the corrected significance
numbers in the counting approximation.

Done when: phase1 tests pass; cutflow table and m4l figure exist; the notebook numbers are
reproduced to the stated tolerance.

### Phase 2: Higgs mass measurement

Build:
- Unbinned extended maximum-likelihood fit (iminuit) of m4l in 110-140 GeV:
  signal = double-sided Crystal Ball (mean m_H + shift, width and tails fixed from signal
  simulation), background = low-order polynomial or exponential with shape fixed from
  background simulation; free parameters m_H, N_sig, N_bkg.
- Calibrate the shift between the fitted peak and the generated 125.0 GeV on signal simulation.
- Uncertainty from the profile likelihood (MINOS), with a likelihood scan plot.
- Blinded first: develop and validate on simulation and toys; fit data only in the unblind step.

Validation: toys generated from the fitted model at the expected yields give a pull mean of
0 and width of 1 within the stated tolerances; a fit to simulation-only pseudo-data returns
125.0 GeV within its uncertainty after calibration; a stated cross-check with a Gaussian
signal shape.

Done when: phase2 tests pass; data result reported with its uncertainty next to the ATLAS
36.1 fb^-1 four-lepton result and the PDG average, with a short honest comparison.

### Phase 3: significance with a profile likelihood

Build:
- pyhf model of the m4l histogram (110-160 GeV or as decided in planning), signal strength
  mu as the parameter of interest, background normalization uncertainty (stated assumption),
  luminosity uncertainty (look up and cite the 2015-2016 value), MC statistical uncertainty.
- Expected significance (Asimov) and observed significance (q0, asymptotic), best-fit mu
  with uncertainty. A toy-based cross-check of the asymptotic result.
- A table comparing: notebook metric, naive S/sqrt(B), counting Asimov, full shape fit.

Validation: single-bin pyhf models reproduce the closed-form Asimov significance with and
without background uncertainty (Phase 3 rows); toys agree with asymptotics within their
statistical error.

Done when: phase3 tests pass; the comparison table and a likelihood scan in mu exist.

### Phase 4: machine-learning classifier

Build:
- Gradient-boosted decision trees (scikit-learn `HistGradientBoostingClassifier`), signal =
  all Higgs samples, background = all others, on simulation only.
- Features: the four lepton pT, m12, m34, pT and rapidity of the 4l system, cos(theta*) of Z1
  in the 4l rest frame, smallest delta-R between leptons. m4l is NOT a feature.
- Train on even eventNumber, test on odd; abs weights for training only (say so), signed
  weights for evaluation; scale test yields by 2.
- Choose one score threshold on the training half to define two categories; rerun the phase 3
  model with both categories on the test half; report the expected-significance gain.
- Unblind step: observed significance with categories on data.

Validation: ROC AUC on the test half reported; train vs test score distributions agree
(KS test p > 0.05); the background m4l shape after the category split is unchanged within
MC statistics (no sculpting); the gain is measured on events the classifier never saw.

Done when: phase4 tests pass; ROC, score and sculpting figures exist; gain reported honestly
even if small.

### Phase 5: write-up and polish

- README: what the project does in one paragraph, a results table (quantity, value, method,
  script), the three key figures, how to run, limitations, credits (ATLAS Open Data, official
  notebook).
- Short report in `report/` (LaTeX, 6 to 8 pages): data, selection, mass fit, significance,
  classifier, limitations, references. Every citation verified against its source.
- `scripts/make_all.py` regenerates every number and figure. CI runs the tests that do not
  need the network on every push.

## 4. Definition of done

A stranger can clone the repo, create the environment, run `scripts/make_all.py`, and get
every number in the README; every claim is backed by a test or a stored result.
