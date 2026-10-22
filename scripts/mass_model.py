"""Freeze the mass-fit model on simulation: signal shape, calibration, background shape.

Writes results/mass_model.json and the shape figures. Uses simulation only.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from higgs4l.data import ROOT, load_config, load_selected
from higgs4l.massfit import bernstein2, dscb, exponential, gauss, weighted_shape_fit
from higgs4l.plotting import shape_fit
from higgs4l.selection import in_range, select

DSCB_START = {
    "mu": 124.6,
    "sigma": 1.7,
    "alpha_l": 1.2,
    "n_l": 3.0,
    "alpha_r": 1.5,
    "n_r": 5.0,
}
DSCB_LIMITS = {
    "sigma": (0.5, 5),
    "alpha_l": (0.3, 5),
    "n_l": (1.01, 100),
    "alpha_r": (0.3, 5),
    "n_r": (1.01, 100),
}


def simulation(cfg, dids):
    """Paired-selection simulation in the fit range."""
    return in_range(
        select(load_selected(dids), cfg, "paired"), *cfg["mass_fit"]["range"]
    )


def background_breakdown(cfg, dids):
    """Simulated background per sample in the fit range: events, yield, largest weight."""
    bkg = simulation(cfg, dids)
    out = {}
    for did in dids:
        w = bkg["weight"][bkg["did"] == did]
        if len(w):
            out[str(did)] = {
                "events": len(w),
                "yield": float(w.sum()),
                "max_weight": float(w.max()),
            }
    return out


def summary(minuit, m_true=None):
    out = {
        "params": dict(zip(minuit.parameters, minuit.values)),
        "errors": dict(zip(minuit.parameters, minuit.errors)),
        "valid": minuit.fmin.is_valid,
        "m2lnl": 2 * minuit.fval,
    }
    if m_true is not None:
        out["delta"] = minuit.values["mu"] - m_true
        out["delta_mc_error"] = minuit.errors["mu"]
    return out


def main():
    cfg = load_config()
    fit_cfg = cfg["mass_fit"]
    lo, hi = fit_cfg["range"]
    m_true = fit_cfg["m_true"]

    signal = simulation(cfg, fit_cfg["signal_shape_dids"])
    even = signal["eventNumber"] % 2 == 0
    m_sig, w_sig = signal["m4l"][even], signal["weight"][even]
    dscb_fit = weighted_shape_fit(dscb, m_sig, w_sig, lo, hi, DSCB_START, DSCB_LIMITS)
    gauss_fit = weighted_shape_fit(
        gauss, m_sig, w_sig, lo, hi, {"mu": 124.6, "sigma": 2.0}, {"sigma": (0.5, 5)}
    )

    def exp_pdf(m, slope):
        return exponential(m, slope, lo)

    def bern_pdf(m, c1, c2):
        return bernstein2(m, c1, c2, lo, hi)

    def fit_background(dids, function):
        bkg = simulation(cfg, dids)
        if function == "exponential":
            return weighted_shape_fit(
                exp_pdf, bkg["m4l"], bkg["weight"], lo, hi, {"slope": 0.0}, {}
            )
        return weighted_shape_fit(
            bern_pdf,
            bkg["m4l"],
            bkg["weight"],
            lo,
            hi,
            {"c1": 1.0, "c2": 1.0},
            {"c1": (0, None), "c2": (0, None)},
        )

    shape_dids = fit_cfg["background_shape_dids"]
    candidates = {
        f: fit_background(shape_dids, f) for f in ("exponential", "bernstein2")
    }
    improvement = 2 * (candidates["exponential"].fval - candidates["bernstein2"].fval)
    # The extra Bernstein parameter must improve -2 ln L by more than 4 to be kept.
    function = "bernstein2" if improvement > 4 else "exponential"
    all_bkg = cfg["samples"]["zz"] + cfg["samples"]["other_bkg"]

    def total(dids):
        return float(simulation(cfg, dids)["weight"].sum())

    model = {
        "range": [lo, hi],
        "m_true": m_true,
        "signal": {
            "source_dids": fit_cfg["signal_shape_dids"],
            "events_even": int(even.sum()),
            "n_eff": float(w_sig.sum() ** 2 / (w_sig**2).sum()),
            "dscb": summary(dscb_fit, m_true),
            "gauss": summary(gauss_fit, m_true),
        },
        "background": {
            "source_dids": shape_dids,
            "candidates": {f: summary(c) for f, c in candidates.items()},
            "bernstein2_improvement_m2lnl": improvement,
            "function": function,
            "nominal": summary(candidates[function]),
            "all_backgrounds": summary(fit_background(all_bkg, function)),
        },
        "expected_yields": {
            "signal": total(cfg["samples"]["signal"]),
            "background": total(all_bkg),
        },
        "expected_background_by_sample": background_breakdown(cfg, all_bkg),
    }
    (ROOT / "results" / "mass_model.json").write_text(
        json.dumps(model, indent=1) + "\n"
    )

    edges = np.linspace(lo, hi, 61)
    x = np.linspace(lo, hi, 600)
    fig, ax = plt.subplots(figsize=(8, 6))
    p = model["signal"]["dscb"]["params"]
    curve = dscb(x, **p) / np.trapezoid(dscb(x, **p), x)
    shape_fit(ax, m_sig, w_sig, edges, x, curve, "H→ZZ*→4ℓ simulation, even events")
    fig.savefig(ROOT / "figures" / "mass_signal_shape.png", dpi=150)
    plt.close(fig)

    bkg = simulation(cfg, shape_dids)
    fit = candidates[function]
    pdf = exp_pdf if function == "exponential" else bern_pdf
    curve = pdf(x, *fit.values) / np.trapezoid(pdf(x, *fit.values), x)
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
