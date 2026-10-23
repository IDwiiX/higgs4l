"""Counting significances and profile-likelihood models."""

from math import log, sqrt

import numpy as np
import pyhf
from scipy.stats import norm


def bin_edges(lo: float, hi: float, width: float) -> np.ndarray:
    """Histogram edges from lo to hi inclusive, built as in the notebook."""
    return np.arange(lo, hi + width, width)


def window_yield(m4l, weights, edges, lo: float, hi: float) -> float:
    """Weighted histogram content summed over the whole bins between edges lo and hi."""
    counts, _ = np.histogram(m4l, bins=edges, weights=weights)
    i, j = np.searchsorted(edges, [lo, hi])
    return float(counts[i:j].sum())


def naive_z(s: float, b: float) -> float:
    """S / sqrt(B)."""
    return s / sqrt(b)


def notebook_metric(n: float, b: float) -> float:
    """The notebook's N / sqrt(B + 0.3 B^2), with N = S + B as it prints it."""
    return n / sqrt(b + 0.3 * b**2)


def z_asimov(s: float, b: float) -> float:
    """Median discovery significance, known background (Cowan et al., EPJC 71 (2011) 1554)."""
    return sqrt(2 * ((s + b) * log(1 + s / b) - s))


def z_asimov_unc(s: float, b: float, sigma_b: float) -> float:
    """Same, with background uncertainty sigma_b from an auxiliary Poisson measurement."""
    v = sigma_b**2
    t1 = (s + b) * log((s + b) * (b + v) / (b * b + (s + b) * v))
    t2 = (b * b / v) * log(1 + v * s / (b * (b + v)))
    return sqrt(2 * (t1 - t2))


MU = {"name": "mu", "type": "normfactor", "data": None}


def counting_model(s: float, b: float, rel_unc: float | None = None) -> pyhf.Model:
    """One bin, mu * s + b; with rel_unc the background is Poisson-constrained (shapesys).

    shapesys is the on/off likelihood behind z_asimov_unc (docs/learning/phase3.md, section 5).
    """
    background = {"name": "background", "data": [b], "modifiers": []}
    if rel_unc is not None:
        background["modifiers"].append(
            {"name": "b_unc", "type": "shapesys", "data": [rel_unc * b]}
        )
    signal = {"name": "signal", "data": [s], "modifiers": [MU]}
    channel = {"name": "count", "samples": [signal, background]}
    return pyhf.Model({"channels": [channel]}, poi_name="mu")


def shape_inputs(
    events: dict, signal_dids, edges: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Signal and background histograms with their MC statistical errors sqrt(sum w^2)."""
    is_signal = np.isin(events["did"], signal_dids)

    def hist(mask):
        m, w = events["m4l"][mask], events["weight"][mask]
        return (
            np.histogram(m, edges, weights=w)[0],
            np.sqrt(np.histogram(m, edges, weights=w**2)[0]),
        )

    return *hist(is_signal), *hist(~is_signal)


def shape_model(
    signal,
    signal_err,
    background,
    background_err,
    lumi_rel_unc: float,
    free_background: bool,
) -> pyhf.Model:
    """m4l template model: mu on the signal, luminosity on the signal, staterror per bin.

    With free_background the summed background has a free normalization mu_b. The
    luminosity factor is on the signal only: on the background it would be degenerate with
    mu_b, and the fixed-background model is a comparison with no background normalization
    uncertainty at all.
    """
    inputs = (signal, signal_err, background, background_err)
    mu_b = "mu_b" if free_background else None
    return _template_model([_channel("m4l", inputs, mu_b)], lumi_rel_unc)


def category_model(categories, lumi_rel_unc: float) -> pyhf.Model:
    """One m4l channel per category, sharing mu and the luminosity factor.

    categories is a list of shape_inputs tuples. Each category has its own free background
    normalization and its own staterror factors (docs/learning/phase4.md, section 7).
    """
    channels = [
        _channel(f"category_{i}", inputs, f"mu_b_{i}")
        for i, inputs in enumerate(categories)
    ]
    return _template_model(channels, lumi_rel_unc)


def _channel(name: str, inputs, mu_b: str | None) -> dict:
    signal, signal_err, background, background_err = inputs
    stat = {"name": f"mc_stat_{name}", "type": "staterror"}
    signal_mods = [
        MU,
        {"name": "lumi", "type": "lumi", "data": None},
        stat | {"data": list(signal_err)},
    ]
    background_mods = [stat | {"data": list(background_err)}]
    if mu_b is not None:
        background_mods.append({"name": mu_b, "type": "normfactor", "data": None})
    samples = [
        {"name": "signal", "data": list(signal), "modifiers": signal_mods},
        {"name": "background", "data": list(background), "modifiers": background_mods},
    ]
    return {"name": name, "samples": samples}


def _template_model(channels: list, lumi_rel_unc: float) -> pyhf.Model:
    lumi = {
        "name": "lumi",
        "auxdata": [1.0],
        "sigmas": [lumi_rel_unc],
        "inits": [1.0],
        "bounds": [[1 - 10 * lumi_rel_unc, 1 + 10 * lumi_rel_unc]],
    }
    spec = {"channels": channels, "parameters": [lumi]}
    return pyhf.Model(spec, poi_name="mu")


def asimov_data(model: pyhf.Model) -> np.ndarray:
    """Expected main and auxiliary data at mu = 1 with every nuisance parameter nominal."""
    return model.expected_data(model.config.suggested_init())


def discovery_q0(model: pyhf.Model, data) -> float:
    """One-sided discovery test statistic q0 (Cowan et al. 2011)."""
    config = model.config
    return float(
        pyhf.infer.test_statistics.q0(
            0.0,
            data,
            model,
            config.suggested_init(),
            config.suggested_bounds(),
            config.suggested_fixed(),
        )
    )


def mu_profile(
    model: pyhf.Model, data, grid: np.ndarray
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Best-fit mu, its -2 Delta ln L = 1 interval from a profile scan, the scan, all parameters."""
    pars, best = pyhf.infer.mle.fit(data, model, return_fitted_val=True)
    mu_hat = float(pars[model.config.poi_index])
    scan = np.array(
        [
            float(
                pyhf.infer.mle.fixed_poi_fit(mu, data, model, return_fitted_val=True)[1]
            )
            for mu in grid
        ]
    ) - float(best)
    left, right = grid < mu_hat, grid > mu_hat
    lo = np.interp(1.0, scan[left][::-1], grid[left][::-1])
    hi = np.interp(1.0, scan[right], grid[right])
    summary = {
        "mu_hat": mu_hat,
        "mu_error_down": float(mu_hat - lo),
        "mu_error_up": float(hi - mu_hat),
    }
    return summary, scan, pars


def counting_toys(s: float, b: float, n_toys: int, seed: int, n_max: int = 60) -> dict:
    """Toys for a known-background counting model.

    q0 depends only on the integer count, so pyhf computes it once per count 0..n_max and
    each toy looks it up. P(n > 60) is below 1e-17 for the means used here.
    """
    model = counting_model(s, b)
    q0_table = np.array([discovery_q0(model, [n]) for n in range(n_max + 1)])
    rng = np.random.default_rng(seed)
    q0_sb = q0_table[rng.poisson(s + b, n_toys)]
    q0_b = q0_table[rng.poisson(b, n_toys)]
    q0_asimov = discovery_q0(model, asimov_data(model))
    p0_toys = float(np.mean(q0_b >= q0_asimov - 1e-9))
    return {
        "n_toys": n_toys,
        "seed": seed,
        "q0_pyhf": q0_table,
        "median_z_sb": float(np.median(np.sqrt(q0_sb))),
        "z_asimov": float(np.sqrt(q0_asimov)),
        "p0_toys": p0_toys,
        "p0_asymptotic": float(norm.sf(np.sqrt(q0_asimov))),
        "z_toys": float(norm.isf(p0_toys)),
    }
