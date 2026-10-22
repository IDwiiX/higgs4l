import json

import numpy as np
import pytest

from higgs4l.data import (
    ROOT,
    STORED_DTYPES,
    fetch,
    file_urls,
    load_selected,
    read_events,
)
from higgs4l.selection import preselection
from higgs4l.stats import notebook_metric
from higgs4l.weights import event_weight

pytestmark = pytest.mark.phase1

# P1 tolerance: 0.1% relative.
TOL = 1e-3


def test_p1a_window_total(notebook_window):
    """P1-a: notebook selection, signal + background in 122.5-130 GeV."""
    assert notebook_window["total"] == pytest.approx(28.729, rel=TOL)


def test_p1b_window_background(notebook_window):
    """P1-b: notebook selection, background in 122.5-130 GeV."""
    assert notebook_window["background"] == pytest.approx(10.301, rel=TOL)


def test_p1c_notebook_metric(notebook_window):
    """P1-c: the notebook's (S+B)/sqrt(B + 0.3 B^2)."""
    metric = notebook_metric(notebook_window["total"], notebook_window["background"])
    assert metric == pytest.approx(4.426, rel=TOL)


def test_p1d_window_signal(notebook_window):
    """P1-d: signal alone in the window, summed directly, equals (P1-a) - (P1-b)."""
    signal = notebook_window["signal"]
    assert signal == pytest.approx(18.428, rel=TOL)
    total, background = notebook_window["total"], notebook_window["background"]
    assert signal == pytest.approx(total - background, rel=1e-9)


def test_provenance_lists_all_files(cfg):
    provenance = json.loads((ROOT / "results" / "provenance.json").read_text())
    assert provenance["release"] == cfg["release"]
    assert provenance["versions"]["atlasopenmagic"]
    dids = {r["did"] for r in provenance["files"]}
    assert dids == {d for ids in cfg["samples"].values() for d in ids}
    assert len(provenance["files"]) == 40


@pytest.mark.network
def test_selected_matches_source(cfg):
    """Rerunning the preselection on the source files reproduces data/selected exactly."""
    for group, dids in cfg["samples"].items():
        for did in dids:
            parts = []
            for url in file_urls(did, cfg["release"], cfg["skim"]):
                events = read_events(fetch(url))
                keep = preselection(events, cfg)
                part = {k: events[k][keep].astype(t) for k, t in STORED_DTYPES.items()}
                if group != "data":
                    part["weight"] = event_weight(events, cfg["lumi_fb"])[keep]
                parts.append(part)
            stored = load_selected([did])
            for key in parts[0]:
                fresh = np.concatenate([p[key] for p in parts])
                np.testing.assert_array_equal(
                    fresh, stored[key], err_msg=f"{did} {key}"
                )
