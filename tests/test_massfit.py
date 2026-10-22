import json

import numpy as np
import pytest
from scipy.integrate import simpson

from higgs4l.data import ROOT, load_selected
from higgs4l.massfit import fit_mass, generate_toy, load_model, pulls, run_toys
from higgs4l.selection import in_range, select

pytestmark = pytest.mark.phase2


@pytest.fixture(scope="module")
def frozen():
    return json.loads((ROOT / "results" / "mass_model.json").read_text())


@pytest.fixture(scope="module")
def recorded_toys():
    return json.loads((ROOT / "results" / "mass_toys.json").read_text())


@pytest.fixture(scope="module")
def toys(cfg, frozen):
    """The 2000 toys of P2-a, regenerated from the frozen model and the config seed."""
    fit_cfg = cfg["mass_fit"]
    yields = frozen["expected_yields"]
    return run_toys(
        load_model(frozen),
        yields["signal"],
        yields["background"],
        fit_cfg["m_true"],
        fit_cfg["n_toys"],
        fit_cfg["seed"],
    )


def converged(results):
    return np.array(
        [r["valid"] and r["accurate"] and r["minos_valid"] for r in results]
    )


def test_p2a_pulls(cfg, toys):
    """P2-a: pulls over 2000 toys at expected yields have mean 0 and width 1 (+-0.05)."""
    pull = pulls(
        [t for t, ok in zip(toys, converged(toys)) if ok], cfg["mass_fit"]["m_true"]
    )
    print(f"P2-a pull mean {pull.mean():.4f}, width {pull.std(ddof=1):.4f}")
    assert abs(pull.mean()) < 0.05
    assert abs(pull.std(ddof=1) - 1) < 0.05


def test_p2b_simulation_pseudodata(cfg, frozen, recorded_toys):
    """P2-b: the calibrated fit to odd-half simulation returns 125.0 GeV within the
    expected data uncertainty (median toy MINOS error)."""
    samples = cfg["samples"]
    events = select(
        load_selected(samples["signal"] + samples["zz"] + samples["other_bkg"]),
        cfg,
        "paired",
    )
    events = in_range(events, *cfg["mass_fit"]["range"])
    odd = events["eventNumber"] % 2 == 1
    result = fit_mass(load_model(frozen), events["m4l"][odd], 2 * events["weight"][odd])
    assert result["valid"] and result["accurate"] and result["minos_valid"]
    assert (
        abs(result["m_h"] - cfg["mass_fit"]["m_true"])
        < recorded_toys["median_minos_error"]
    )


def test_p2c_convergence(toys, recorded_toys):
    """P2-c: every toy converges (valid minimum, accurate covariance, valid MINOS), and
    the count matches results/mass_toys.json. Any failure fails the test, so that it
    is looked at and reported rather than absorbed."""
    n_failed = int((~converged(toys)).sum())
    print(f"P2-c failed toy fits: {n_failed} of {len(toys)}")
    assert n_failed == recorded_toys["n_failed"]
    assert n_failed == 0


def test_toys_match_recorded(cfg, toys, recorded_toys):
    pull = pulls(
        [t for t, ok in zip(toys, converged(toys)) if ok], cfg["mass_fit"]["m_true"]
    )
    assert pull.mean() == pytest.approx(recorded_toys["pull_mean"], abs=1e-12)


def test_normalization(frozen):
    model = load_model(frozen)
    grid = np.linspace(model.lo, model.hi, 200001)
    assert simpson(model.signal_pdf(grid, 125.0), x=grid) == pytest.approx(1, abs=1e-6)
    assert simpson(model.background_pdf(grid), x=grid) == pytest.approx(1, abs=1e-6)


def test_fit_recovers_known_toy(frozen):
    """A large toy generated at m_H = 124 GeV is fitted back at 124 GeV (generation and
    fit share the model, so this checks the fit machinery, not the calibration)."""
    model = load_model(frozen)
    m = generate_toy(np.random.default_rng(1), model, 50000, 50000, 124.0)
    result = fit_mass(model, m)
    assert result["valid"]
    assert abs(result["m_h"] - 124.0) < 3 * result["hesse"]


def test_signal_closure(cfg, frozen):
    """Odd-half H->ZZ*->4l simulation, never used for the shape or delta, is fitted at
    125.0 GeV within 3 MC-statistical sigma. This is the check on the sign of delta."""
    signal = in_range(
        select(load_selected(cfg["mass_fit"]["signal_shape_dids"]), cfg, "paired"),
        *cfg["mass_fit"]["range"],
    )
    odd = signal["eventNumber"] % 2 == 1
    w = signal["weight"][odd]
    result = fit_mass(
        load_model(frozen), signal["m4l"][odd], w * w.sum() / (w**2).sum()
    )
    assert result["valid"]
    assert abs(result["m_h"] - cfg["mass_fit"]["m_true"]) < 3 * result["hesse"]


def test_p2c_data_fit(cfg, frozen):
    """P2-c on data: the nominal and cross-check fits converge, and the quoted values
    match results/mass_fit.json."""
    recorded = json.loads((ROOT / "results" / "mass_fit.json").read_text())
    events = select(load_selected(["data"]), cfg, "paired")
    m = in_range(events, *cfg["mass_fit"]["range"])["m4l"]
    for signal, background in [
        ("dscb", "nominal"),
        ("gauss", "nominal"),
        ("dscb", "all_backgrounds"),
    ]:
        result = fit_mass(load_model(frozen, signal, background), m)
        assert result["valid"] and result["accurate"] and result["minos_valid"]
    result = fit_mass(load_model(frozen), m)
    for key in ("m_h", "minos_lower", "minos_upper", "n_s", "n_b"):
        assert result[key] == pytest.approx(recorded["fits"]["nominal"][key], abs=1e-6)
