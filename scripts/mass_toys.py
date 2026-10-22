"""Toy experiments from the frozen mass model: pulls, coverage and fit convergence.

Writes results/mass_toys.json and figures/mass_pulls.png. No data are used.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from higgs4l.data import ROOT, load_config
from higgs4l.massfit import load_model, pulls, run_toys
from higgs4l.plotting import pull_histogram


def toy_summary(results, m_true):
    """Pull mean and width with their statistical errors, and the convergence count."""
    ok = np.array([r["valid"] and r["accurate"] and r["minos_valid"] for r in results])
    pull = pulls([r for r, good in zip(results, ok) if good], m_true)
    n = len(pull)
    width = pull.std(ddof=1)
    minos = np.array([(r["minos_upper"] - r["minos_lower"]) / 2 for r in results])
    return {
        "n_toys": len(results),
        "n_failed": int((~ok).sum()),
        "pull_mean": float(pull.mean()),
        "pull_mean_error": float(width / np.sqrt(n)),
        "pull_width": float(width),
        "pull_width_error": float(width / np.sqrt(2 * (n - 1))),
        "median_minos_error": float(np.median(minos[ok])),
        "median_m_hat": float(
            np.median([r["m_h"] for r, good in zip(results, ok) if good])
        ),
    }, pull


def main():
    cfg = load_config()
    fit_cfg = cfg["mass_fit"]
    frozen = json.loads((ROOT / "results" / "mass_model.json").read_text())
    yields = frozen["expected_yields"]
    results = run_toys(
        load_model(frozen),
        yields["signal"],
        yields["background"],
        fit_cfg["m_true"],
        fit_cfg["n_toys"],
        fit_cfg["seed"],
    )
    summary, pull = toy_summary(results, fit_cfg["m_true"])
    summary |= {"expected_yields": yields, "seed": fit_cfg["seed"]}
    (ROOT / "results" / "mass_toys.json").write_text(
        json.dumps(summary, indent=1) + "\n"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    pull_histogram(ax, pull, summary["pull_mean"], summary["pull_width"])
    fig.savefig(ROOT / "figures" / "mass_pulls.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
