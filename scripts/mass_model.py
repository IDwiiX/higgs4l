"""Freeze the mass-fit model on simulation: signal shape, calibration, background shape.

Writes results/mass_model.json and the shape figures. Uses simulation only.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from higgs4l.data import ROOT, load_config
from higgs4l.massfit import bernstein2, dscb, exponential, freeze_model, simulation
from higgs4l.plotting import shape_fit


def main():
    cfg = load_config()
    fit_cfg = cfg["mass_fit"]
    lo, hi = fit_cfg["range"]
    model = freeze_model(cfg)
    (ROOT / "results" / "mass_model.json").write_text(
        json.dumps(model, indent=1) + "\n"
    )

    signal = simulation(cfg, fit_cfg["signal_shape_dids"])
    even = signal["eventNumber"] % 2 == 0
    m_sig, w_sig = signal["m4l"][even], signal["weight"][even]
    edges = np.linspace(lo, hi, 61)
    x = np.linspace(lo, hi, 600)
    fig, ax = plt.subplots(figsize=(8, 6))
    p = model["signal"]["dscb"]["params"]
    curve = dscb(x, **p) / np.trapezoid(dscb(x, **p), x)
    shape_fit(ax, m_sig, w_sig, edges, x, curve, "H→ZZ*→4ℓ simulation, even events")
    fig.savefig(ROOT / "figures" / "mass_signal_shape.png", dpi=150)
    plt.close(fig)

    bkg = simulation(cfg, fit_cfg["background_shape_dids"])
    p = model["background"]["nominal"]["params"]
    if model["background"]["function"] == "exponential":
        y = exponential(x, lo=lo, **p)
    else:
        y = bernstein2(x, lo=lo, hi=hi, **p)
    curve = y / np.trapezoid(y, x)
    fig, ax = plt.subplots(figsize=(8, 6))
    shape_fit(
        ax,
        bkg["m4l"],
        bkg["weight"],
        np.linspace(lo, hi, 13),
        x,
        curve,
        "ZZ* simulation",
    )
    fig.savefig(ROOT / "figures" / "mass_background_shape.png", dpi=150)
    plt.close(fig)
    print(json.dumps(model["expected_yields"]))


if __name__ == "__main__":
    main()
