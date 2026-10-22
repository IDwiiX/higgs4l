"""Unbinned extended likelihood fit of the four-lepton mass."""

from dataclasses import dataclass

import numpy as np
from iminuit import Minuit
from scipy.integrate import simpson

GRID_POINTS = 2001
SAMPLING_POINTS = 20001


def dscb(m, mu, sigma, alpha_l, n_l, alpha_r, n_r):
    """Double-sided Crystal Ball, unnormalized: Gaussian core, power-law tails."""
    t = (np.asarray(m, dtype=np.float64) - mu) / sigma
    out = np.exp(-0.5 * t**2)
    left, right = t < -alpha_l, t > alpha_r
    # A and B make the function and its derivative continuous at the joins.
    a_l = (n_l / alpha_l) ** n_l * np.exp(-0.5 * alpha_l**2)
    a_r = (n_r / alpha_r) ** n_r * np.exp(-0.5 * alpha_r**2)
    out[left] = a_l * (n_l / alpha_l - alpha_l - t[left]) ** -n_l
    out[right] = a_r * (n_r / alpha_r - alpha_r + t[right]) ** -n_r
    return out


def gauss(m, mu, sigma):
    """Gaussian, unnormalized."""
    return np.exp(-0.5 * ((np.asarray(m, dtype=np.float64) - mu) / sigma) ** 2)


def exponential(m, slope, lo):
    """exp(slope (m - lo)), unnormalized."""
    return np.exp(slope * (np.asarray(m, dtype=np.float64) - lo))


def bernstein2(m, c1, c2, lo, hi):
    """Second-order Bernstein polynomial with c0 = 1, non-negative for c1, c2 >= 0."""
    x = (np.asarray(m, dtype=np.float64) - lo) / (hi - lo)
    return (1 - x) ** 2 + 2 * c1 * x * (1 - x) + c2 * x**2


def integral(f, lo, hi):
    """Integral of a vectorized function on [lo, hi] (Simpson on a fixed grid)."""
    grid = np.linspace(lo, hi, GRID_POINTS)
    return simpson(f(grid), x=grid)


def weighted_shape_fit(pdf, m, w, lo, hi, start: dict, limits: dict) -> Minuit:
    """Fit the shape parameters of pdf(m, **params) to weighted events by -sum w ln f.

    Weights are rescaled to sum to n_eff = (sum w)^2 / sum w^2, so that the Hesse errors
    approximate the statistical errors of the weighted sample.
    """
    w = w * w.sum() / (w**2).sum()

    def nll(*p):
        params = dict(zip(start, p))
        norm = integral(lambda x: pdf(x, **params), lo, hi)
        return -np.sum(w * np.log(pdf(m, **params) / norm))

    minuit = Minuit(nll, *start.values(), name=list(start))
    minuit.errordef = Minuit.LIKELIHOOD
    for name, limit in limits.items():
        minuit.limits[name] = limit
    minuit.migrad()
    minuit.hesse()
    return minuit


@dataclass
class MassModel:
    """Frozen signal and background shapes; only m_H, N_s and N_b are free."""

    lo: float
    hi: float
    signal: str  # "dscb" or "gauss"
    signal_shape: dict  # shape parameters except the mean
    delta: float  # fitted peak position minus the generated 125.0 GeV
    background: str  # "exponential" or "bernstein2"
    background_shape: dict

    def signal_pdf(self, m, m_h):
        f = dscb if self.signal == "dscb" else gauss
        g = lambda x: f(x, m_h + self.delta, **self.signal_shape)
        return g(m) / integral(g, self.lo, self.hi)

    def __post_init__(self):
        self._background_norm = integral(self._background, self.lo, self.hi)

    def _background(self, m):
        if self.background == "exponential":
            return exponential(m, lo=self.lo, **self.background_shape)
        return bernstein2(m, lo=self.lo, hi=self.hi, **self.background_shape)

    def background_pdf(self, m):
        return self._background(m) / self._background_norm

    def nll(self, m, w, m_h, n_s, n_b):
        """Extended NLL; with weights w it is the weighted version (sum w replaces n)."""
        density = n_s * self.signal_pdf(m, m_h) + n_b * self.background_pdf(m)
        return n_s + n_b - np.sum(w * np.log(density))


def _minuit(model, m, w):
    n = w.sum()
    minuit = Minuit(
        lambda m_h, n_s, n_b: model.nll(m, w, m_h, n_s, n_b),
        m_h=125.0,
        n_s=0.5 * n,
        n_b=0.5 * n,
    )
    minuit.errordef = Minuit.LIKELIHOOD
    minuit.limits["m_h"] = (model.lo, model.hi)
    minuit.limits["n_s"] = (0, None)
    minuit.limits["n_b"] = (0, None)
    return minuit


def fit_mass(model: MassModel, m, w=None) -> dict:
    """Fit m_H, N_s, N_b; MINOS on m_H. Returns values, errors and validity flags."""
    w = np.ones(len(m)) if w is None else w
    minuit = _minuit(model, m, w)
    minuit.migrad()
    minuit.hesse()
    result = {
        "m_h": minuit.values["m_h"],
        "n_s": minuit.values["n_s"],
        "n_b": minuit.values["n_b"],
        "hesse": minuit.errors["m_h"],
        "valid": minuit.fmin.is_valid,
        "accurate": minuit.fmin.has_accurate_covar,
        "minos_lower": np.nan,
        "minos_upper": np.nan,
        "minos_valid": False,
        "nll": minuit.fval,
    }
    if minuit.fmin.is_valid:
        minuit.minos("m_h")
        error = minuit.merrors["m_h"]
        result |= {
            "minos_lower": error.lower,
            "minos_upper": error.upper,
            "minos_valid": error.is_valid,
        }
    return result


def scan_mass(model: MassModel, m, grid, w=None) -> np.ndarray:
    """Profile -2 Delta ln L in m_H on a grid (N_s and N_b refitted at each point)."""
    w = np.ones(len(m)) if w is None else w
    minuit = _minuit(model, m, w)
    minuit.migrad()
    best = minuit.fval
    minuit.fixed["m_h"] = True
    values = []
    for m_h in grid:
        minuit.values["m_h"] = m_h
        minuit.migrad()
        values.append(2 * (minuit.fval - best))
    return np.array(values)


def generate_toy(rng, model: MassModel, n_s: float, n_b: float, m_h: float):
    """One pseudo-experiment: Poisson counts, masses by inverse CDF of each density."""
    grid = np.linspace(model.lo, model.hi, SAMPLING_POINTS)
    masses = []
    for density, mean in (
        (model.signal_pdf(grid, m_h), n_s),
        (model.background_pdf(grid), n_b),
    ):
        cdf = np.concatenate([[0], np.cumsum((density[1:] + density[:-1]) / 2)])
        masses.append(np.interp(rng.random(rng.poisson(mean)), cdf / cdf[-1], grid))
    return np.concatenate(masses)


def run_toys(model: MassModel, n_s, n_b, m_true, n_toys, seed) -> list[dict]:
    """Generate and fit toys at the true mass; one result dict per toy."""
    rng = np.random.default_rng(seed)
    return [
        fit_mass(model, generate_toy(rng, model, n_s, n_b, m_true))
        for _ in range(n_toys)
    ]


def pulls(results: list[dict], m_true: float) -> np.ndarray:
    """(m_hat - m_true) / MINOS error on the side towards the truth."""
    m_hat = np.array([r["m_h"] for r in results])
    lower = np.array([-r["minos_lower"] for r in results])
    upper = np.array([r["minos_upper"] for r in results])
    sigma = np.where(m_hat < m_true, upper, lower)
    return (m_hat - m_true) / sigma


def load_model(frozen: dict, signal="dscb", background="nominal") -> MassModel:
    """MassModel from results/mass_model.json; alternatives are cross-checks."""
    sig = frozen["signal"][signal]
    bkg = frozen["background"][background]
    return MassModel(
        lo=frozen["range"][0],
        hi=frozen["range"][1],
        signal=signal,
        signal_shape={k: v for k, v in sig["params"].items() if k != "mu"},
        delta=sig["delta"],
        background=frozen["background"]["function"],
        background_shape=bkg["params"],
    )
