"""Notebook selection, Z1/Z2 pairing and cutflow."""

import numpy as np

from higgs4l.kinematics import event_p4, four_lepton_mass, invariant_mass

ELECTRON, MUON = 11, 13
# The three ways to split four leptons into two pairs.
SPLITS = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]


def pt_index_cut(pt: np.ndarray, pt_min) -> np.ndarray:
    """The notebook's pT cut on stored positions 0, 1, 2 (leptons are not pT-ordered)."""
    return (pt[:, 0] > pt_min[0]) & (pt[:, 1] > pt_min[1]) & (pt[:, 2] > pt_min[2])


def pt_sorted_cut(pt: np.ndarray, pt_min) -> np.ndarray:
    """pT thresholds on the leading, subleading and third lepton."""
    leading = -np.sort(-pt, axis=1)
    return pt_index_cut(leading, pt_min)


def lepton_cuts(events: dict, cfg: dict) -> dict[str, np.ndarray]:
    """Notebook cuts other than pT, each on its own (not cumulative)."""
    t = events["lep_type"]
    ele, mu = cfg["electron"], cfg["muon"]
    good = ((t == ELECTRON) & events[ele["id"]] & events[ele["iso"]]) | (
        (t == MUON) & events[mu["id"]] & events[mu["iso"]]
    )
    return {
        "trigger": events["trigE"] | events["trigM"],
        "trigger match": events["lep_isTrigMatched"].any(axis=1),
        "ID and isolation": good.all(axis=1),
        "flavour sum": np.isin(t.sum(axis=1), cfg["lep_type_sums"]),
        "charge sum": events["lep_charge"].sum(axis=1) == 0,
    }


def z_pairs(lep_type, lep_charge, p4, mz: float):
    """Lepton indices of Z1 and Z2, shape (n, 2), and whether a pairing exists.

    A split is valid when both pairs are same flavour and opposite charge; Z1 is the
    pair of a valid split with mass closest to mz, Z2 the other pair of that split.
    """
    n = len(lep_type)
    best = np.full(n, np.inf)
    z1 = np.zeros((n, 2), dtype=int)
    z2 = np.zeros((n, 2), dtype=int)

    def sfos(i, j):
        return (lep_type[:, i] == lep_type[:, j]) & (
            lep_charge[:, i] + lep_charge[:, j] == 0
        )

    for a, b in SPLITS:
        valid = sfos(*a) & sfos(*b)
        for first, second in ((a, b), (b, a)):
            m = invariant_mass(p4[:, first[0]] + p4[:, first[1]])
            distance = np.where(valid, np.abs(m - mz), np.inf)
            closer = distance < best
            best[closer] = distance[closer]
            z1[closer] = first
            z2[closer] = second
    return z1, z2, np.isfinite(best)


def z_masses(events: dict, mz: float):
    """m12, m34 and whether a valid pairing exists (masses are meaningless otherwise)."""
    p4 = event_p4(events)
    z1, z2, ok = z_pairs(events["lep_type"], events["lep_charge"], p4, mz)
    rows = np.arange(len(ok))[:, None]
    m12 = invariant_mass(p4[rows, z1].sum(axis=1))
    m34 = invariant_mass(p4[rows, z2].sum(axis=1))
    return m12, m34, ok


def z_cuts(events: dict, cfg: dict) -> dict[str, np.ndarray]:
    """Pairing and Z mass cuts of the corrected selection, each on its own."""
    m12, m34, ok = z_masses(events, cfg["mz"])
    lo, hi = cfg["m12"]
    return {
        "Z pairing": ok,
        "m12": ok & (m12 > lo) & (m12 < hi),
        "m34": ok & (m34 > cfg["m34_min"]),
    }


def notebook_cuts(events: dict, cfg: dict) -> dict[str, np.ndarray]:
    """The notebook selection in its own order, each cut on its own."""
    cuts = lepton_cuts(events, cfg["notebook_selection"])
    pt = pt_index_cut(events["lep_pt"], cfg["notebook_selection"]["lep_pt_min"])
    return _insert_after(cuts, "trigger match", "lepton pT", pt)


def paired_cuts(events: dict, cfg: dict) -> dict[str, np.ndarray]:
    """The corrected selection: sorted pT cut, then Z pairing and Z mass cuts."""
    cuts = lepton_cuts(events, cfg["notebook_selection"])
    pt = pt_sorted_cut(events["lep_pt"], cfg["notebook_selection"]["lep_pt_min"])
    cuts = _insert_after(cuts, "trigger match", "lepton pT (sorted)", pt)
    return cuts | z_cuts(events, cfg["paired_selection"])


def preselection(events: dict, cfg: dict) -> np.ndarray:
    """Events kept in data/selected: all lepton cuts with the sorted pT cut.

    The index pT cut implies the sorted one, so this is a superset of both selections.
    """
    cuts = lepton_cuts(events, cfg["notebook_selection"])
    pt = pt_sorted_cut(events["lep_pt"], cfg["notebook_selection"]["lep_pt_min"])
    return np.logical_and.reduce(list(cuts.values())) & pt


def select(events: dict, cfg: dict, name: str) -> dict[str, np.ndarray]:
    """Stored events passing a selection, with m4l, m12 and m34 added.

    'notebook': the notebook's cuts; 'sorted': the same with the sorted pT cut (all stored
    events); 'paired': 'sorted' plus Z pairing and the Z mass cuts.
    """
    m12, m34, _ = z_masses(events, cfg["paired_selection"]["mz"])
    if name == "notebook":
        mask = pt_index_cut(events["lep_pt"], cfg["notebook_selection"]["lep_pt_min"])
    elif name == "sorted":
        mask = np.ones(len(m12), dtype=bool)
    else:
        mask = np.logical_and.reduce(
            list(z_cuts(events, cfg["paired_selection"]).values())
        )
    out = {k: v[mask] for k, v in events.items()}
    out["m4l"] = four_lepton_mass(out)
    out["m12"], out["m34"] = m12[mask], m34[mask]
    return out


def cumulative(cuts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Running AND of cuts, in order."""
    out, mask = {}, None
    for name, cut in cuts.items():
        mask = cut if mask is None else mask & cut
        out[name] = mask
    return out


def _insert_after(cuts: dict, after: str, name: str, cut) -> dict:
    out = {}
    for k, v in cuts.items():
        out[k] = v
        if k == after:
            out[name] = cut
    return out


def in_range(events: dict, lo: float, hi: float) -> dict[str, np.ndarray]:
    """Selected events with lo <= m4l < hi."""
    keep = (events["m4l"] >= lo) & (events["m4l"] < hi)
    return {k: v[keep] for k, v in events.items()}
