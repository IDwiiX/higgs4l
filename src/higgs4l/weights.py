"""Simulation event weights normalized to the data luminosity."""

import numpy as np

SCALE_FACTORS = [
    "ScaleFactor_PILEUP",
    "ScaleFactor_ELE",
    "ScaleFactor_MUON",
    "ScaleFactor_LepTRIGGER",
]


def event_weight(events: dict, lumi_fb: float) -> np.ndarray:
    """Signed weight: expected events in data with lumi_fb (fb^-1), xsec in pb.

    The notebook takes abs() of every factor; since the normalization is positive,
    that product equals abs() of this weight.
    """
    w = lumi_fb * 1000 * np.asarray(events["xsec"], dtype=np.float64)
    w = w * events["filteff"] * events["kfac"] * events["mcWeight"]
    for sf in SCALE_FACTORS:
        w = w * events[sf]
    return w / events["sum_of_weights"]
