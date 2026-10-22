"""Notebook window numbers, counting significances and selection yields.

Reads data/selected and writes results/phase1.json.
"""

import json

import numpy as np

from higgs4l.data import ROOT, load_config, load_selected
from higgs4l.selection import select
from higgs4l.stats import (
    bin_edges,
    naive_z,
    notebook_metric,
    window_yield,
    z_asimov,
    z_asimov_unc,
)

RANGES = {
    "80-250": (80.0, 250.0),
    "110-140": (110.0, 140.0),
    "122.5-130": (122.5, 130.0),
    "180-250": (180.0, 250.0),
}
GROUPS = ("signal", "zz", "other_bkg")


def counting(s, b):
    return {
        "S": s,
        "B": b,
        "naive_S_over_sqrtB": naive_z(s, b),
        "asimov": z_asimov(s, b),
        "asimov_bkg_unc_10pct": z_asimov_unc(s, b, 0.1 * b),
        "asimov_bkg_unc_30pct": z_asimov_unc(s, b, 0.3 * b),
    }


def main():
    cfg = load_config()
    hist = cfg["histogram"]
    edges = bin_edges(hist["m4l_min"], hist["m4l_max"], hist["bin_width"])
    lo, hi = hist["notebook_window"]
    samples = cfg["samples"]
    events = {
        (selection, group): select(load_selected(dids), cfg, selection)
        for selection in ("notebook", "sorted", "paired")
        for group, dids in samples.items()
    }

    def window(selection, group, absolute, a, b, dids=None):
        ev = events[selection, group]
        keep = np.isin(ev["did"], dids) if dids else np.ones(len(ev["m4l"]), bool)
        w = None if group == "data" else ev["weight"][keep]
        w = np.abs(w) if absolute and w is not None else w
        return window_yield(ev["m4l"][keep], w, edges, a, b)

    def notebook_numbers(signal_dids):
        s = window("notebook", "signal", True, lo, hi, signal_dids)
        b = sum(window("notebook", g, True, lo, hi) for g in ("zz", "other_bkg"))
        return {
            "signal_dids": signal_dids,
            "printed_N_sig_S_plus_B": s + b,
            "printed_N_bg_B": b,
            "signal_S": s,
            "notebook_metric": notebook_metric(s + b, b),
        }

    # One change at a time, from the notebook reproduction to the corrected selection.
    stages = {
        "1 notebook, abs weights, 7 signal samples": (
            "notebook",
            True,
            cfg["notebook_outputs_signal"],
        ),
        "2 notebook, abs weights": ("notebook", True, samples["signal"]),
        "3 notebook, signed weights": ("notebook", False, samples["signal"]),
        "4 sorted pT, signed weights": ("sorted", False, samples["signal"]),
        "5 sorted pT + Z pairing and mass cuts, signed weights": (
            "paired",
            False,
            samples["signal"],
        ),
    }
    comparison = {}
    for stage, (selection, absolute, signal) in stages.items():
        rows = {}
        for label, (a, b) in RANGES.items():
            row = {"data": int(window(selection, "data", False, a, b))}
            row["signal"] = window(selection, "signal", absolute, a, b, signal)
            row["zz"] = window(selection, "zz", absolute, a, b)
            row["other_bkg"] = window(selection, "other_bkg", absolute, a, b)
            row["data_over_mc"] = row["data"] / sum(row[g] for g in GROUPS)
            rows[label] = row
        w = rows["122.5-130"]
        comparison[stage] = {
            "signal_dids": signal,
            "yields": rows,
            "counting_window": counting(w["signal"], w["zz"] + w["other_bkg"]),
        }

    signal = events["paired", "signal"]
    in_window = (signal["m4l"] >= lo) & (signal["m4l"] < hi)
    w, did = signal["weight"][in_window], signal["did"][in_window]
    signal_share = {
        str(d): float(w[did == d].sum() / w.sum()) for d in samples["signal"]
    }

    reproduction = notebook_numbers(cfg["notebook_outputs_signal"])
    results = {
        "notebook_window_GeV": [lo, hi],
        "notebook_reproduction": reproduction,
        "notebook_with_346310": notebook_numbers(samples["signal"]),
        "counting_notebook_selection": counting(
            reproduction["signal_S"], reproduction["printed_N_bg_B"]
        ),
        "selection_comparison": comparison,
        "paired_signal_share_in_window": signal_share,
    }
    out = ROOT / "results" / "phase1.json"
    out.write_text(json.dumps(results, indent=1) + "\n")
    for stage, c in comparison.items():
        r, z = c["yields"], c["counting_window"]
        print(
            f"{stage:55s} S={z['S']:6.2f} B={z['B']:6.2f} Z_A={z['asimov']:5.2f}"
            f"  data/MC 180-250={r['180-250']['data_over_mc']:.3f}"
        )


if __name__ == "__main__":
    main()
