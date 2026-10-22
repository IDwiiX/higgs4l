import numpy as np
import pytest

from higgs4l.weights import event_weight

pytestmark = pytest.mark.phase1

RECORD = {
    "xsec": np.array([0.5]),
    "filteff": np.array([0.8]),
    "kfac": np.array([1.25]),
    "mcWeight": np.array([-2.0]),
    "ScaleFactor_PILEUP": np.array([0.9]),
    "ScaleFactor_ELE": np.array([1.1]),
    "ScaleFactor_MUON": np.array([1.05]),
    "ScaleFactor_LepTRIGGER": np.array([0.95]),
    "sum_of_weights": np.array([400.0]),
}
NOTEBOOK_FACTORS = ["filteff", "kfac", "xsec", "mcWeight"] + [
    "ScaleFactor_PILEUP",
    "ScaleFactor_ELE",
    "ScaleFactor_MUON",
    "ScaleFactor_LepTRIGGER",
]


def test_p1g_event_weight():
    """P1-g: event weight on one hand-filled record, 1e-12 relative."""
    # 36.6 * 1000 * 0.5 * 0.8 * 1.25 * (-2) / 400 = -91.5; scale factors give 0.987525.
    w = event_weight(RECORD, 36.6)
    assert w[0] == pytest.approx(-90.3585375, rel=1e-12)

    # The notebook's calc_weight: abs() of every factor.
    notebook = 36.6 * 1000 / RECORD["sum_of_weights"]
    for factor in NOTEBOOK_FACTORS:
        notebook = notebook * abs(RECORD[factor])
    assert np.abs(w)[0] == pytest.approx(notebook[0], rel=1e-12)
