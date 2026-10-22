import numpy as np
import pytest

from higgs4l.selection import pt_index_cut, pt_sorted_cut, z_cuts, z_pairs

pytestmark = pytest.mark.phase1

MZ = 91.1876

# Massless leptons (E, px, py, pz). Pair masses follow from m^2 = 2 E1 E2 (1 - cos):
# m(0,2) = MZ (back to back), m(0,1) = 49.0, m(0,3) = 40.0, m(1,2) = 55.8,
# m(1,3) = 34.6, m(2,3) = 45.6 GeV.
E2 = MZ**2 / 160
LEPTONS = np.array(
    [[40, 40, 0, 0], [30, 0, 30, 0], [E2, -E2, 0, 0], [20, 0, 0, 20]], dtype=float
)
# Two separated pairs for 2e2mu: ee at MZ along x, mumu at 30 GeV along y, then shuffled.
EE = [[MZ / 2, MZ / 2, 0, 0], [MZ / 2, -MZ / 2, 0, 0]]
MUMU = [[15, 0, 15, 0], [15, 0, -15, 0]]
SHUFFLED = np.array([MUMU[0], EE[0], MUMU[1], EE[1]], dtype=float)

CASES = {
    # 4mu, charges ++--: valid splits (0,2)(1,3) and (0,3)(1,2); (0,2) is at MZ.
    "4mu": (LEPTONS, [13] * 4, [1, 1, -1, -1], (0, 2), (1, 3)),
    # 4e, charges +-+-: (0,2) is same-sign, so the best valid pair is (1,2) at 55.8 GeV.
    "4e": (LEPTONS, [11] * 4, [1, -1, 1, -1], (1, 2), (0, 3)),
    "2e2mu shuffled": (SHUFFLED, [13, 11, 13, 11], [1, -1, -1, 1], (1, 3), (0, 2)),
}


@pytest.mark.parametrize("name", CASES)
def test_p1f_pairing(name):
    """P1-f: Z1/Z2 pairing on hand-built events, exact indices."""
    p4, lep_type, charge, expect_z1, expect_z2 = CASES[name]
    z1, z2, ok = z_pairs(np.array([lep_type]), np.array([charge]), p4[None], MZ)
    assert ok[0]
    assert tuple(z1[0]) == expect_z1
    assert tuple(z2[0]) == expect_z2


def test_p1f_same_sign_pairs_rejected():
    """P1-f: e+ e+ mu- mu- passes the notebook's sums but has no Z pairing."""
    lep_type = np.array([[11, 11, 13, 13]])
    charge = np.array([[1, 1, -1, -1]])
    assert lep_type.sum() == 48 and charge.sum() == 0
    _, _, ok = z_pairs(lep_type, charge, LEPTONS[None], MZ)
    assert not ok[0]


def test_index_pt_cut_is_subset_of_sorted():
    pt = np.array([[12.0, 25.0, 18.0, 30.0], [25.0, 16.0, 11.0, 5.0]])
    assert pt_index_cut(pt, [20, 15, 10]).tolist() == [False, True]
    assert pt_sorted_cut(pt, [20, 15, 10]).tolist() == [True, True]


def as_event(p4, lep_type, charge):
    """One stored-format event from massless four-vectors (E, px, py, pz)."""
    pt = np.hypot(p4[:, 1], p4[:, 2])
    return {
        "lep_pt": pt[None],
        "lep_eta": np.arcsinh(p4[:, 3] / pt)[None],
        "lep_phi": np.arctan2(p4[:, 2], p4[:, 1])[None],
        "lep_e": p4[:, 0][None],
        "lep_type": np.array([lep_type]),
        "lep_charge": np.array([charge]),
    }


@pytest.mark.parametrize(
    "ee_mass, mumu_mass, passes",
    [
        (MZ, 30.0, {"m12": True, "m34": True}),
        (MZ, 10.0, {"m12": True, "m34": False}),
        (45.0, 30.0, {"m12": False, "m34": True}),
    ],
)
def test_z_mass_cuts(cfg, ee_mass, mumu_mass, passes):
    ee = [[ee_mass / 2, ee_mass / 2, 0, 0], [ee_mass / 2, -ee_mass / 2, 0, 0]]
    mumu = [[mumu_mass / 2, 0, mumu_mass / 2, 0], [mumu_mass / 2, 0, -mumu_mass / 2, 0]]
    p4 = np.array([ee[0], ee[1], mumu[0], mumu[1]], dtype=float)
    # With m(ee) = 45 GeV the mumu pair at 30 GeV is further from MZ, so ee stays Z1.
    cuts = z_cuts(
        as_event(p4, [11, 11, 13, 13], [1, -1, 1, -1]), cfg["paired_selection"]
    )
    assert cuts["Z pairing"][0]
    assert {k: bool(cuts[k][0]) for k in ("m12", "m34")} == passes
