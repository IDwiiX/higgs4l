"""Phase 3, blinded: expected significance of the m4l template fit, on simulation only.

Writes results/significance_expected.json and figures/significance_scan_asimov.png.
No data are used.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from higgs4l.data import ROOT, load_config, load_selected
from higgs4l.plotting import likelihood_scan
from higgs4l.selection import select
from higgs4l.stats import (
    asimov_data,
    bin_edges,
    counting_toys,
    discovery_q0,
    mu_profile,
    naive_z,
    notebook_metric,
    shape_inputs,
    shape_model,
    window_yield,
    z_asimov,
    z_asimov_unc,
)

MU_GRID = np.linspace(0, 2.5, 101)


def simulation(cfg):
    samples = cfg["samples"]
    return select(
        load_selected(samples["signal"] + samples["zz"] + samples["other_bkg"]),
        cfg,
        "paired",
    )


def build_model(cfg, mc, width, free_background):
    sig = cfg["significance"]
    edges = bin_edges(*sig["range"], width)
    inputs = shape_inputs(mc, cfg["samples"]["signal"], edges)
    return shape_model(*inputs, sig["lumi_rel_unc"], free_background), inputs


def model_toys(model, n_toys, seed):
    """q0 on toys generated at mu = 1 and at mu = 0, other parameters nominal.

    The auxiliary data are sampled too, so each toy is a full repetition of the experiment.
    """
    rng = np.random.default_rng(seed)
    init = np.array(model.config.suggested_init())
    out = {}
    for name, mu in (("sb", 1.0), ("b", 0.0)):
        pars = init.copy()
        pars[model.config.poi_index] = mu
        pdf = model.make_pdf(pars)
        q0 = []
        for _ in range(n_toys):
            # pyhf's numpy backend samples through scipy, which takes the global seed.
            np.random.seed(rng.integers(2**32))
            q0.append(discovery_q0(model, pdf.sample()))
        out[name] = np.array(q0)
    return out


def toy_summary(toys, z_a):
    n = len(toys["b"])
    z_sb = np.sqrt(toys["sb"])
    tails = {}
    for z in (1, 2, 3):
        p = float(np.mean(toys["b"] >= z * z))
        tails[f"q0>={z * z}"] = {
            "toys": p,
            "toys_error": float(np.sqrt(max(p, 1 / n) * (1 - p) / n)),
            "asymptotic": float(norm.sf(z)),
        }
    median = float(np.median(z_sb))
    # Standard error of a median, sqrt(pi/2) sigma / sqrt(n) for a near-normal spread.
    error = float(np.sqrt(np.pi / 2) * z_sb.std(ddof=1) / np.sqrt(n))
    return {
        "n_toys": n,
        "median_z_sb_toys": median,
        "median_z_sb_toys_error": error,
        "z_asimov": z_a,
        "median_minus_z_asimov_in_errors": (median - z_a) / error,
        "b_only_tails": tails,
        "note": "b-only toys test the q0 distribution only up to q0 = 9 (Z = 3); "
        "the s+b toy median tests the Asimov approximation of the expected Z",
    }


def counting_rows(s, b):
    return {
        "S": s,
        "B": b,
        "notebook_metric": notebook_metric(s + b, b),
        "naive_S_over_sqrtB": naive_z(s, b),
        "asimov": z_asimov(s, b),
        "asimov_bkg_unc_30pct": z_asimov_unc(s, b, 0.3 * b),
    }


def main():
    cfg = load_config()
    sig = cfg["significance"]
    mc = simulation(cfg)

    results = {"config": sig, "selection": "paired, sorted pT, signed weights"}
    models = {}
    for label, width, free in (
        ("free_background", sig["bin_width"], True),
        ("fixed_background", sig["bin_width"], False),
        ("free_background_5GeV", sig["cross_check_bin_width"], True),
    ):
        model, inputs = build_model(cfg, mc, width, free)
        models[label] = model
        results[label] = {
            "z_asimov": float(np.sqrt(discovery_q0(model, asimov_data(model)))),
            "signal_yield": float(inputs[0].sum()),
            "background_yield": float(inputs[2].sum()),
            # Relative MC statistical width of the staterror factor in each bin.
            "mc_stat_rel": (
                np.hypot(inputs[1], inputs[3]) / (inputs[0] + inputs[2])
            ).tolist(),
        }

    model = models["free_background"]
    fit, scan, _ = mu_profile(model, asimov_data(model), MU_GRID)
    results["free_background"] |= fit
    results["toys_free_background"] = toy_summary(
        model_toys(model, sig["n_toys"], sig["seed"]),
        results["free_background"]["z_asimov"],
    )

    p3h = counting_toys(10, 5, sig["p3h_toys"], sig["seed"])
    p3h.pop("q0_pyhf")
    results["p3h_counting_toys"] = p3h

    hist = cfg["histogram"]
    edges = bin_edges(hist["m4l_min"], hist["m4l_max"], hist["bin_width"])
    is_signal = np.isin(mc["did"], cfg["samples"]["signal"])
    s, b = (
        window_yield(
            mc["m4l"][mask], mc["weight"][mask], edges, *hist["notebook_window"]
        )
        for mask in (is_signal, ~is_signal)
    )
    results["window_counting_paired"] = counting_rows(s, b)

    (ROOT / "results" / "significance_expected.json").write_text(
        json.dumps(results, indent=1) + "\n"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    likelihood_scan(ax, MU_GRID, scan, "Asimov, simulation")
    ax.set_xlabel(r"Signal strength $\mu$")
    fig.savefig(ROOT / "figures" / "significance_scan_asimov.png", dpi=150)
    plt.close(fig)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
