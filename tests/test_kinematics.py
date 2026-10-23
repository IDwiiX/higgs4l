import numpy as np
import pytest

from higgs4l.kinematics import (
    cos_theta_star,
    invariant_mass,
    lepton_p4,
    min_delta_r,
    rapidity,
)


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


@pytest.mark.phase1
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


@pytest.mark.phase4
def test_rapidity():
    assert rapidity(np.array([10.0, 3.0, 1.0, 6.0])) == pytest.approx(np.log(2), rel=1e-12)


@pytest.mark.phase4
@pytest.mark.parametrize("cos", [1.0, 0.6, 0.0, -0.6])
@pytest.mark.parametrize("axis", [1, 3])
def test_cos_theta_star(cos, axis):
    """Z1 at a known angle to z in the 4l rest frame; a boost along z or x keeps it.

    Combined boosts along different axes are not tested: the pure boost back to the rest
    frame then differs by a Wigner rotation, which changes the angle to the beam axis.
    """
    sin = np.sqrt(1 - cos**2)
    z1 = np.array([[50.0, 30 * sin, 0.0, 30 * cos]])
    z2 = np.array([[40.0, -30 * sin, 0.0, -30 * cos]])
    lab1, lab2 = boost(z1, 0.6, axis), boost(z2, 0.6, axis)
    assert cos_theta_star(lab1, lab1 + lab2)[0] == pytest.approx(cos, abs=1e-12)


@pytest.mark.phase4
def test_min_delta_r_wraps_phi():
    eta = np.array([[0.0, 0.05, 2.0, -2.0]])
    phi = np.array([[3.1, -3.1, 1.0, -1.0]])
    expected = np.hypot(0.05, 2 * np.pi - 6.2)
    assert min_delta_r(eta, phi)[0] == pytest.approx(expected, rel=1e-12)
