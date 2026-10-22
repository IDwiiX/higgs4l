import numpy as np
import pytest

from higgs4l.kinematics import invariant_mass, lepton_p4

pytestmark = pytest.mark.phase1


def boost(p4, beta, axis):
    """Boost four-vectors (E, px, py, pz) along a coordinate axis (1, 2 or 3)."""
    gamma = 1 / np.sqrt(1 - beta**2)
    out = p4.copy()
    out[:, 0] = gamma * (p4[:, 0] + beta * p4[:, axis])
    out[:, axis] = gamma * (p4[:, axis] + beta * p4[:, 0])
    return out


def to_collider(p4):
    """(E, px, py, pz) to (pt, eta, phi, E)."""
    pt = np.hypot(p4[:, 1], p4[:, 2])
    return pt, np.arcsinh(p4[:, 3] / pt), np.arctan2(p4[:, 2], p4[:, 1]), p4[:, 0]


def test_p1e_invariant_mass():
    """P1-e: invariant mass of hand-built four-vectors, 1e-9 relative."""
    mz = 91.1876
    pair = lepton_p4([mz / 2, mz / 2], [0.0, 0.0], [0.0, np.pi], [mz / 2, mz / 2])
    assert invariant_mass(pair.sum(axis=0)) == pytest.approx(mz, rel=1e-9)

    # Four massless leptons at rest-frame energies 40, 40, 22.5, 22.5 GeV (M = 125 GeV),
    # back to back in pairs along two directions, then boosted along z and x.
    u = np.array([0.36, 0.48, 0.8])
    v = np.array([0.6, -0.8, 0.0])
    rest = np.array(
        [[40, *(40 * u)], [40, *(-40 * u)], [22.5, *(22.5 * v)], [22.5, *(-22.5 * v)]]
    )
    lab = boost(boost(rest, 0.6, 3), 0.3, 1)
    p4 = lepton_p4(*to_collider(lab))
    assert invariant_mass(p4.sum(axis=0)) == pytest.approx(125.0, rel=1e-9)
