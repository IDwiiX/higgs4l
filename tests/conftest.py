import numpy as np
import pytest

from higgs4l.data import load_config, load_selected
from higgs4l.selection import select
from higgs4l.stats import bin_edges, window_yield


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def notebook_window(cfg):
    """Signal and background in the notebook window, with its selection and abs weights."""
    hist = cfg["histogram"]
    edges = bin_edges(hist["m4l_min"], hist["m4l_max"], hist["bin_width"])
    lo, hi = hist["notebook_window"]

    def window(dids):
        events = select(load_selected(dids), cfg, "notebook")
        return window_yield(events["m4l"], np.abs(events["weight"]), edges, lo, hi)

    background = cfg["samples"]["zz"] + cfg["samples"]["other_bkg"]
    return {
        "signal": window(cfg["notebook_outputs_signal"]),
        "background": window(background),
        "total": window(cfg["notebook_outputs_signal"] + background),
    }
