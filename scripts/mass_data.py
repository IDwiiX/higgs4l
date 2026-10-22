"""Unblind phase 2: fit the data with the frozen mass model.

Reads results/mass_model.json and results/mass_toys.json; writes results/mass_fit.json,
figures/mass_fit_data.png and figures/mass_scan_data.png.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2

from higgs4l.data import ROOT, load_config, load_selected
from higgs4l.massfit import (
    fit_mass,
    generate_toy,
    integral,
    load_model,
    run_toys,
    scan_mass,
)
from higgs4l.plotting import likelihood_scan, mass_fit_overlay
from higgs4l.selection import in_range, select

# REFERENCE_VALUES.md P2-d (arXiv:1806.00242) and P2-e (PDG Live, S126M).
REFERENCES = {
    "atlas_4l_36fb": {"m_h": 124.79, "error": 0.37},
    "pdg_2026": {"m_h": 125.13, "error": 0.11},
}
N_TOYS_FITTED_YIELDS = 500


def converged(fit):
    return fit["valid"] and fit["accurate"] and fit["minos_valid"]


def minos_error(fit):
    """Symmetrized MINOS error."""
    return (fit["minos_upper"] - fit["minos_lower"]) / 2


def goodness_of_fit(model, fit, m, edges):
    """Binned Poisson deviance of the fitted model, with 3 fitted parameters.

    The counts per bin are small, so the chi2 p-value is approximate.
    """
    n, _ = np.histogram(m, edges)

    def density(x):
        return fit["n_s"] * model.signal_pdf(x, fit["m_h"]) + fit[
            "n_b"
        ] * model.background_pdf(x)

    mu = np.array([integral(density, a, b) for a, b in zip(edges[:-1], edges[1:])])
    deviance = 2 * np.sum(mu - n + xlogy(n, n / mu))
    ndof = len(n) - 3
    return {"deviance": deviance, "ndof": ndof, "p_value": chi2.sf(deviance, ndof)}


def data_vs_simulation(cfg, m, edges):
    """Data counts and simulated signal and background yields in coarse mass ranges."""
    samples = cfg["samples"]
    mc = select(
        load_selected(samples["signal"] + samples["zz"] + samples["other_bkg"]),
        cfg,
        "paired",
    )
    is_signal = np.isin(mc["did"], samples["signal"])

    def simulated(mask):
        return np.histogram(mc["m4l"][mask], edges, weights=mc["weight"][mask])[0]

    return {
        "edges": edges,
        "data": np.histogram(m, edges)[0].tolist(),
        "signal": simulated(is_signal).tolist(),
        "background": simulated(~is_signal).tolist(),
    }


def toys_at_fitted_yields(nominal, gaussian, fit, observed_error, n_toys, seed):
    """A-posteriori diagnostic: toys generated at the yields and mass fitted in data.

    Judges the observed error and the Gaussian-shape shift against what these yields
    give. A toy whose nominal fit fails is still counted in the error fraction (a failed
    MINOS comes with a small N_s and a wide interval); the shape shift uses only toys
    where both fits converged.
    """
    rng = np.random.default_rng(seed)
    errors, shifts, failed_n_s, failed_gaussian = [], [], [], 0
    for _ in range(n_toys):
        m = generate_toy(rng, nominal, fit["n_s"], fit["n_b"], fit["m_h"])
        a, b = fit_mass(nominal, m), fit_mass(gaussian, m)
        errors.append(minos_error(a))
        if not converged(a):
            failed_n_s.append(a["n_s"])
        elif not converged(b):
            failed_gaussian += 1
        else:
            shifts.append(b["m_h"] - a["m_h"])
    errors = np.array(errors)
    return {
        "n_toys": n_toys,
        "n_failed": len(failed_n_s),
        "failed_n_s": failed_n_s,
        "n_failed_gaussian_only": failed_gaussian,
        "median_error": np.median(errors),
        "fraction_error_at_least_observed": np.mean(errors >= observed_error),
        "gaussian_shift_mean": np.mean(shifts),
        "gaussian_shift_std": np.std(shifts, ddof=1),
    }


def main():
    cfg = load_config()
    fit_cfg = cfg["mass_fit"]
    lo, hi = fit_cfg["range"]
    frozen = json.loads((ROOT / "results" / "mass_model.json").read_text())
    toys = json.loads((ROOT / "results" / "mass_toys.json").read_text())
    m = in_range(select(load_selected(["data"]), cfg, "paired"), lo, hi)["m4l"]

    nominal, gaussian = load_model(frozen), load_model(frozen, signal="gauss")
    fits = {
        "nominal": fit_mass(nominal, m),
        "gaussian_signal": fit_mass(gaussian, m),
        "all_background_shape": fit_mass(
            load_model(frozen, background="all_backgrounds"), m
        ),
    }
    best = fits["nominal"]
    error = minos_error(best)
    expected = frozen["expected_yields"]
    # The P2-a toys (same seed), to place the observed error among the expected ones.
    p2a_errors = np.array(
        [
            minos_error(t)
            for t in run_toys(
                nominal,
                expected["signal"],
                expected["background"],
                fit_cfg["m_true"],
                fit_cfg["n_toys"],
                fit_cfg["seed"],
            )
        ]
    )
    result = {
        "events": len(m),
        "fits": fits,
        "m_h": best["m_h"],
        "error_lower": best["minos_lower"],
        "error_upper": best["minos_upper"],
        "expected_error": toys["median_minos_error"],
        "fraction_expected_toys_error_at_least_observed": np.mean(p2a_errors >= error),
        "expected_yields": expected,
        "yield_pulls": {
            "n_s": (best["n_s"] - expected["signal"]) / best["n_s_error"],
            "n_b": (best["n_b"] - expected["background"]) / best["n_b_error"],
        },
        "data_vs_simulation": data_vs_simulation(cfg, m, [lo, 120, 130, hi]),
        "goodness_of_fit": goodness_of_fit(nominal, best, m, np.linspace(lo, hi, 13)),
        "toys_at_fitted_yields": toys_at_fitted_yields(
            nominal, gaussian, best, error, N_TOYS_FITTED_YIELDS, fit_cfg["seed"]
        ),
        "shift_gaussian_signal": fits["gaussian_signal"]["m_h"] - best["m_h"],
        "shift_all_background_shape": fits["all_background_shape"]["m_h"] - best["m_h"],
        # Differences only: the ATLAS result uses the same collisions, so its error and
        # ours cannot be added in quadrature.
        "references": REFERENCES,
        "difference_to_reference": {
            name: best["m_h"] - ref["m_h"] for name, ref in REFERENCES.items()
        },
    }
    (ROOT / "results" / "mass_fit.json").write_text(
        json.dumps(result, indent=1, default=float) + "\n"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    mass_fit_overlay(ax, m, nominal, best, np.linspace(lo, hi, 13), cfg["lumi_fb"])
    fig.savefig(ROOT / "figures" / "mass_fit_data.png", dpi=150)
    plt.close(fig)

    grid = np.linspace(122.0, 128.0, 121)
    fig, ax = plt.subplots(figsize=(8, 6))
    likelihood_scan(ax, grid, scan_mass(nominal, m, grid), "Data")
    fig.savefig(ROOT / "figures" / "mass_scan_data.png", dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
