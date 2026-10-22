# higgs4l

A small, reproducible analysis of the Higgs boson in the H → ZZ* → 4ℓ channel with the
ATLAS 2025 Open Data for education (13 TeV, 2015-2016, 36.6 fb⁻¹). It starts from the official
ATLAS notebook, reproduces its numbers, and then goes further in three ways: a likelihood
measurement of the Higgs mass with a calibrated uncertainty, a profile-likelihood significance,
and a gradient-boosted classifier whose gain in expected significance is measured on events it
never saw.

Status: phases 1 (selection, notebook reproduction) and 2 (mass fit, validated on simulation;
data not yet fitted) are done; phase 3 (profile-likelihood significance) is next.

## Results

All numbers are in the 122.5-130 GeV window used by the notebook, unless stated otherwise.

| Quantity | Value | Method | Script → result |
| --- | --- | --- | --- |
| Notebook N_sig as printed (= S + B) | 28.729 | notebook selection, abs weights, 7 signal samples | `phase1_numbers.py` → `results/phase1.json` |
| Notebook N_bg (B) | 10.301 | same | same |
| Notebook metric (S+B)/√(B+0.3B²) | 4.426 | same | same |
| True signal S | 18.428 | same | same |
| S/√B | 5.742 | counting, same S and B | same |
| Asimov Z, no background uncertainty | 4.699 | Cowan et al. 2011, same S and B | same |
| Asimov Z, 10% / 30% background uncertainty | 4.377 / 3.075 | same | same |
| Asimov Z, corrected selection | 7.82 (S = 28.84, B = 6.64) | sorted pT + Z pairing, signed weights, 8 signal samples | same |
| Data/MC in 180-250 GeV, corrected selection | 1.40 | signed weights | same |
| Peak calibration δ = fitted DSCB mean − 125 GeV | −0.350 ± 0.010 GeV | DSCB fit, H→ZZ*→4ℓ simulation (even events) | `mass_model.py` → `results/mass_model.json` |
| Expected uncertainty on m_H | 0.50 GeV | median MINOS error, 2000 toys at simulated yields | `mass_toys.py` → `results/mass_toys.json` |
| Toy pulls (mean, width) | −0.010 ± 0.023, 1.013 ± 0.016; 0 failed fits | 2000 toys | same |
| m_H fitted on held-out simulation | 125.15 GeV (generated 125.0; MC stat. ≈ 0.26, expected data error 0.50) | odd-event simulation as weighted pseudo-data | `mass_simulation.py` → `results/mass_simulation.json` |
| Shift with a Gaussian signal shape / all-background shape | +0.23 / +0.09 GeV | same pseudo-data | same |

The notebook's numbers are reproduced to machine precision. Its stored outputs come from 7 of
the 8 signal samples now in the notebook (346310 was added later without rerunning). With all
8, S = 19.010.

Per-cut yields for both selections are in `results/cutflow.md`. The derivations and every
finding are in `docs/learning/phase1.md`.

![m4l, corrected selection](figures/m4l_paired.png)

## What phase 1 found

- **The notebook's "leading lepton" pT cut is applied to stored positions 0, 1, 2.** Leptons
  are stored electrons first, not by pT. Applying the cut to the pT-sorted leptons is the
  largest single improvement, taking the Asimov significance from 4.91 to 6.42.
- **The notebook's printed N_sig is signal plus background**, and its 0.3 B² term corresponds
  to a 55% background uncertainty, not 30%.
- **Z1/Z2 pairing removes no data event.** No selected data event is e⁺e⁺μ⁻μ⁻. The m12 and
  m34 cuts do the work.

## Limitations

- **With signed weights the simulation is about 40% below data** where ZZ* dominates
  (180-250 GeV). The weight normalization is verified (`scripts/check_normalization.py` →
  `results/normalization_check.json`). The likely cause is the background sample list, which
  has no gg → ZZ* sample; this is not quantified here.
- **Signal normalizations differ from the release metadata.** For 346310 and the ttH samples
  the file's cross-section differs from the release metadata by 3-20%. These samples are under
  3% of the signal in the window (`results/provenance.json`).
- **The expected mass uncertainty is optimistic.** The 0.50 GeV comes from toys at
  simulated yields. Data exceed simulation by a factor of 1.26 in 110-140 GeV
  (`results/phase1.json`). A quarter of the simulated background there (6.6 of 27.6 events)
  comes from three Z+jets events, one with weight 5.8 (`results/mass_model.json`).
- **The calibration comes from one simulated mass point.** δ is measured at 125.0 GeV and
  assumed to be the same nearby.
- **Simplified selection.** It does not use the m4l-dependent m34 threshold, the ΔR cut or
  the J/ψ veto of the full ATLAS analysis.
- **Partial blinding.** Data were plotted in phase 1. Blinding here means every later choice
  is fixed on simulation before any fit to data.

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
.venv/bin/pytest -q -m "not network"          # uses the committed data/selected
.venv/bin/python scripts/select_events.py      # downloads about 1 GB, rebuilds data/selected
.venv/bin/python scripts/phase1_numbers.py
.venv/bin/python scripts/plot_m4l.py
.venv/bin/python scripts/mass_model.py         # simulation only; freezes the fit model
.venv/bin/python scripts/mass_toys.py          # 2000 toys, about 2 minutes
.venv/bin/python scripts/mass_simulation.py    # needs the two outputs above
```

## Credits

- ATLAS Open Data for education and outreach, release `2025e-13tev-beta`, released under CC0:
  ATLAS Collaboration, doi:10.7483/OPENDATA.ATLAS.B5M9.44TN
  ([record 93910](https://opendata.cern.ch/record/93910)).
- The official notebook `HZZAnalysis.ipynb` from
  [atlas-outreach-data-tools/notebooks-collection-opendata](https://github.com/atlas-outreach-data-tools/notebooks-collection-opendata/tree/master/13-TeV-examples/uproot_python).
- Asimov significance: G. Cowan, K. Cranmer, E. Gross, O. Vitells, Eur. Phys. J. C 71 (2011) 1554.
