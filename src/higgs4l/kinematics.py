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
