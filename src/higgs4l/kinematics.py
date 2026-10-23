"""Lepton four-vectors and derived event kinematics."""

import numpy as np


def lepton_p4(pt, eta, phi, e) -> np.ndarray:
    """Four-vectors (E, px, py, pz) in GeV from collider coordinates, as float64."""
    pt, eta, phi, e = (np.asarray(x, dtype=np.float64) for x in (pt, eta, phi, e))
    return np.stack([e, pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], axis=-1)


def event_p4(events: dict) -> np.ndarray:
    """Lepton four-vectors of every event, shape (n, 4 leptons, 4)."""
    return lepton_p4(
        events["lep_pt"], events["lep_eta"], events["lep_phi"], events["lep_e"]
    )


def invariant_mass(p4: np.ndarray) -> np.ndarray:
    """Invariant mass of four-vectors (..., 4)."""
    m2 = p4[..., 0] ** 2 - p4[..., 1] ** 2 - p4[..., 2] ** 2 - p4[..., 3] ** 2
    # Rounding can push massless single leptons slightly below zero.
    return np.sqrt(np.maximum(m2, 0.0))


def four_lepton_mass(events: dict) -> np.ndarray:
    """m4l of every event."""
    return invariant_mass(event_p4(events).sum(axis=1))


def transverse_momentum(p4: np.ndarray) -> np.ndarray:
    """pT of four-vectors (..., 4)."""
    return np.hypot(p4[..., 1], p4[..., 2])


def rapidity(p4: np.ndarray) -> np.ndarray:
    """Rapidity y = 1/2 ln((E + pz) / (E - pz))."""
    return 0.5 * np.log((p4[..., 0] + p4[..., 3]) / (p4[..., 0] - p4[..., 3]))


def cos_theta_star(z1: np.ndarray, system: np.ndarray) -> np.ndarray:
    """Cosine of the angle between Z1 and the beam (z) axis in the 4l rest frame.

    Z1 is taken to the rest frame by the pure boost -beta of the 4l system.
    """
    beta = system[..., 1:] / system[..., :1]
    b2 = (beta**2).sum(axis=-1)
    gamma = 1 / np.sqrt(1 - b2)
    bp = (beta * z1[..., 1:]).sum(axis=-1)
    scale = (gamma - 1) * bp / b2 - gamma * z1[..., 0]
    p = z1[..., 1:] + scale[..., None] * beta
    return p[..., 2] / np.linalg.norm(p, axis=-1)


def min_delta_r(eta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Smallest delta-R between any two of the four leptons, shape (n,)."""
    i, j = np.triu_indices(4, k=1)
    deta = eta[:, i] - eta[:, j]
    dphi = (phi[:, i] - phi[:, j] + np.pi) % (2 * np.pi) - np.pi
    return np.hypot(deta, dphi).min(axis=1)
