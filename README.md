# higgs4l

A small, reproducible analysis of the Higgs boson in the H → ZZ* → 4ℓ channel with the
ATLAS 2025 Open Data for education (13 TeV, 2015-2016, 36.6 fb⁻¹). It starts from the official
ATLAS notebook, reproduces its numbers, and then goes further in three ways: a likelihood
measurement of the Higgs mass with a calibrated uncertainty, a profile-likelihood significance,
and a gradient-boosted classifier whose gain in expected significance is measured on events it
never saw.

Status: in progress, phase 1 (selection and notebook reproduction) is next.

## Results

| Quantity | Value | Method | Script |
| --- | --- | --- | --- |

## How to run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
.venv/bin/pytest -q -m "not network"
```

## Credits

- ATLAS Open Data for education and outreach, release `2025e-13tev-beta`, released under CC0:
  ATLAS Collaboration, doi:10.7483/OPENDATA.ATLAS.B5M9.44TN
  ([record 93910](https://opendata.cern.ch/record/93910)).
- The official notebook `HZZAnalysis.ipynb` from
  [atlas-outreach-data-tools/notebooks-collection-opendata](https://github.com/atlas-outreach-data-tools/notebooks-collection-opendata/tree/master/13-TeV-examples/uproot_python).
