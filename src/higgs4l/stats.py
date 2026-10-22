"""Counting significances and profile-likelihood models."""

from math import log, sqrt

import numpy as np


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
