"""Fit the odd-eventNumber simulation as pseudo-data (P2-b), with shape cross-checks.

The odd half was not used for the signal shape or the calibration. Writes
results/mass_simulation.json and figures/mass_scan_simulation.png. No data are used.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from higgs4l.data import ROOT, load_config, load_selected
from higgs4l.massfit import fit_mass, load_model, scan_mass
from higgs4l.plotting import likelihood_scan
from higgs4l.selection import in_range, select


def odd_half_pseudodata(cfg):
    """All simulation in the fit range with odd eventNumber, weights doubled.

    Also returns the signal and background content, which differ from the expected
    yields because a few background events carry large weights.
    """
    samples = cfg["samples"]
    dids = samples["signal"] + samples["zz"] + samples["other_bkg"]
    events = select(load_selected(dids), cfg, "paired")
    events = in_range(events, *cfg["mass_fit"]["range"])
    odd = events["eventNumber"] % 2 == 1
    m, w = events["m4l"][odd], 2 * events["weight"][odd]
    is_signal = np.isin(events["did"][odd], samples["signal"])
    content = {
        "signal": float(w[is_signal].sum()),
        "background": float(w[~is_signal].sum()),
    }
    return m, w, content


def main():
    cfg = load_config()
    m_true = cfg["mass_fit"]["m_true"]
    frozen = json.loads((ROOT / "results" / "mass_model.json").read_text())
    toys = json.loads((ROOT / "results" / "mass_toys.json").read_text())
    m, w, content = odd_half_pseudodata(cfg)

    nominal = load_model(frozen)
    fits = {
        "nominal": fit_mass(nominal, m, w),
        "gaussian_signal": fit_mass(load_model(frozen, signal="gauss"), m, w),
        "all_background_shape": fit_mass(
            load_model(frozen, background="all_backgrounds"), m, w
        ),
    }
    # Signal-only closure on the odd half of the shape samples: sensitive to the sign of
    # delta (a flipped sign would move m_hat by 2 |delta| = 0.7 GeV).
    signal = in_range(
        select(load_selected(cfg["mass_fit"]["signal_shape_dids"]), cfg, "paired"),
        *cfg["mass_fit"]["range"],
    )
    odd = signal["eventNumber"] % 2 == 1
    w_sig = signal["weight"][odd]
    closure = fit_mass(
        nominal, signal["m4l"][odd], w_sig * w_sig.sum() / (w_sig**2).sum()
    )

    # Weights rescaled to the effective number of events give the MC-statistical error.
    n_eff_scale = w.sum() / (w**2).sum()
    mc_stat = fit_mass(nominal, m, w * n_eff_scale)["hesse"]
    expected_error = toys["median_minos_error"]
    result = {
        "pseudodata": {"events": len(m), "sum_weights": float(w.sum())} | content,
        "fits": fits,
        "signal_closure": {"m_h": closure["m_h"], "hesse": closure["hesse"]},
        "m_hat_minus_true": fits["nominal"]["m_h"] - m_true,
        "expected_data_error": expected_error,
        "mc_stat_error_approx": mc_stat,
        "shift_gaussian_signal": fits["gaussian_signal"]["m_h"]
        - fits["nominal"]["m_h"],
        "shift_all_background_shape": fits["all_background_shape"]["m_h"]
        - fits["nominal"]["m_h"],
    }
    (ROOT / "results" / "mass_simulation.json").write_text(
        json.dumps(result, indent=1, default=float) + "\n"
    )

    grid = np.linspace(123.0, 127.0, 81)
    fig, ax = plt.subplots(figsize=(8, 6))
    likelihood_scan(
        ax, grid, scan_mass(nominal, m, grid, w), "Odd-half simulation as pseudo-data"
    )
    fig.savefig(ROOT / "figures" / "mass_scan_simulation.png", dpi=150)
    plt.close(fig)
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
