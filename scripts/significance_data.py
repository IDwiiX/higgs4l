"""Unblind phase 3: observed significance and signal strength with the frozen model.

Reads results/phase1.json and results/significance_expected.json; writes
results/significance.json, results/significance_table.md and
figures/significance_scan_data.png.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from higgs4l.data import ROOT, load_config, load_selected
from higgs4l.plotting import likelihood_scan
from higgs4l.selection import select
from higgs4l.stats import bin_edges, discovery_q0, mu_profile

from significance import build_model, simulation

MU_GRID = np.linspace(0, 3, 121)


def observed(model, counts):
    """Observed q0, Z, p0, and the profile fit of mu."""
    data = list(counts) + model.config.auxdata
    q0 = discovery_q0(model, data)
    fit, scan, pars = mu_profile(model, data, MU_GRID)
    out = {"q0": q0, "z": float(np.sqrt(q0)), "p0": float(norm.sf(np.sqrt(q0)))} | fit
    by_sample = model.main_model.expected_data(pars, return_by_sample=True)
    out |= {
        f"fitted_{name}": float(np.sum(rates))
        for name, rates in zip(model.config.samples, by_sample)
    }
    if "mu_b" in model.config.par_order:
        # A-posteriori diagnostic: the Asimov Z at mu = 1 with every nuisance parameter at
        # its fitted value, to separate a small signal from a large background.
        post = np.array(pars)
        post[model.config.poi_index] = 1.0
        z_post = np.sqrt(discovery_q0(model, model.expected_data(post)))
        out |= {
            "mu_b_hat": float(pars[model.config.par_slice("mu_b")][0]),
            "z_asimov_at_fitted_background": float(z_post),
        }
    return out, scan


CAPTION = """Discovery significance Z. The notebook column is the notebook selection (7 signal
samples, abs weights, pT cuts on stored lepton order); the corrected column is the paired
selection (8 signal samples, signed weights, sorted pT, Z pairing). "Window" is 122.5-130 GeV.
Shape-fit Z values are asymptotic (Z = sqrt(q0)); toys test the q0 distribution only up to
Z = 3. The fixed-background observed Z is not a valid result: the free fit to the same data
finds a background {mu_b:.2f} times the simulation.
"""


def table(phase1, expected, obs):
    nb = phase1["counting_notebook_selection"]
    paired = expected["window_counting_paired"]
    toys = expected["toys_free_background"]
    rows = [
        (
            "Notebook metric (S+B)/√(B+0.3B²), window",
            f"{phase1['notebook_reproduction']['notebook_metric']:.2f}",
            f"{paired['notebook_metric']:.2f}",
            "-",
        ),
        (
            "S/√B, window",
            f"{nb['naive_S_over_sqrtB']:.2f}",
            f"{paired['naive_S_over_sqrtB']:.2f}",
            "-",
        ),
        (
            "Counting Asimov, known B, window",
            f"{nb['asimov']:.2f}",
            f"{paired['asimov']:.2f}",
            "-",
        ),
        (
            "Counting Asimov, 30% B uncertainty, window",
            f"{nb['asimov_bkg_unc_30pct']:.2f}",
            f"{paired['asimov_bkg_unc_30pct']:.2f}",
            "-",
        ),
        (
            "Shape fit 110-160 GeV, B fixed to simulation",
            "-",
            f"{expected['fixed_background']['z_asimov']:.2f}",
            f"({obs['fixed_background']['z']:.2f}, not valid)",
        ),
        (
            "**Shape fit 110-160 GeV, B normalization free**",
            "-",
            f"{expected['free_background']['z_asimov']:.2f}",
            f"**{obs['free_background']['z']:.2f}**",
        ),
        (
            "Same, median of 1000 s+b toys",
            "-",
            f"{toys['median_z_sb_toys']:.2f} ± {toys['median_z_sb_toys_error']:.2f}",
            "-",
        ),
        (
            "Shape fit, 5 GeV bins, B free (cross-check)",
            "-",
            f"{expected['free_background_5GeV']['z_asimov']:.2f}",
            f"{obs['free_background_5GeV']['z']:.2f}",
        ),
    ]
    lines = [
        CAPTION.format(mu_b=obs["free_background"]["mu_b_hat"]),
        "| Method | Notebook selection | Corrected, expected | Corrected, observed |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(lines) + "\n"


def main():
    cfg = load_config()
    sig = cfg["significance"]
    expected = json.loads((ROOT / "results" / "significance_expected.json").read_text())
    phase1 = json.loads((ROOT / "results" / "phase1.json").read_text())
    mc = simulation(cfg)
    data = select(load_selected(["data"]), cfg, "paired")

    results = {"config": sig}
    scans = {}
    for label, width, free in (
        ("free_background", sig["bin_width"], True),
        ("fixed_background", sig["bin_width"], False),
        ("free_background_5GeV", sig["cross_check_bin_width"], True),
    ):
        model, _ = build_model(cfg, mc, width, free)
        edges = bin_edges(*sig["range"], width)
        counts = np.histogram(data["m4l"], edges)[0]
        results[label], scans[label] = observed(model, counts)
        results[label]["z_expected"] = expected[label]["z_asimov"]
        results[label]["data_counts"] = counts.tolist()
        results[label]["simulated_s_plus_b"] = (
            expected[label]["signal_yield"] + expected[label]["background_yield"]
        )

    (ROOT / "results" / "significance.json").write_text(
        json.dumps(results, indent=1) + "\n"
    )
    (ROOT / "results" / "significance_table.md").write_text(
        table(phase1, expected, results)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    likelihood_scan(ax, MU_GRID, scans["free_background"], "Data, B normalization free")
    ax.set_xlabel(r"Signal strength $\mu$")
    fig.savefig(ROOT / "figures" / "significance_scan_data.png", dpi=150)
    plt.close(fig)
    print(json.dumps(results, indent=1))
    print((ROOT / "results" / "significance_table.md").read_text())


if __name__ == "__main__":
    main()
