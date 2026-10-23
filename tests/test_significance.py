import numpy as np
import pyhf
import pytest
from scipy import stats

from higgs4l.data import load_selected
from higgs4l.selection import select
from higgs4l.stats import (
    asimov_data,
    bin_edges,
    counting_model,
    counting_toys,
    discovery_q0,
    shape_inputs,
    shape_model,
    z_asimov,
    z_asimov_unc,
)

pytestmark = pytest.mark.phase3

# REFERENCE_VALUES.md, Phase 3: s, b, relative background uncertainty, Z_A target.
SINGLE_BIN = {
    "P3-a": (10, 5, None, 3.5998),
    "P3-b": (10, 5, 0.1, 3.4691),
    "P3-c": (10, 5, 0.3, 2.7826),
    "P3-d": (5, 20, None, 1.0757),
    "P3-e": (5, 20, 0.3, 0.6273),
    "P3-f": (18.428, 10.301, None, 4.6986),
    "P3-g": (18.428, 10.301, 0.3, 3.0754),
}


@pytest.mark.parametrize("ref", SINGLE_BIN)
def test_single_bin_asimov(ref):
    """P3-a to P3-g: pyhf q0 on Asimov data reproduces the closed-form Z_A."""
    s, b, rel_unc, target = SINGLE_BIN[ref]
    model = counting_model(s, b, rel_unc)
    z = np.sqrt(discovery_q0(model, asimov_data(model)))
    closed = z_asimov(s, b) if rel_unc is None else z_asimov_unc(s, b, rel_unc * b)
    print(f"{ref}: pyhf {z:.5f}, closed form {closed:.5f}, target {target}")
    assert z == pytest.approx(target, rel=1e-3)
    assert z == pytest.approx(closed, rel=1e-3)


def test_p3h_toys_vs_asymptotic(cfg):
    """P3-h: toys vs asymptotics for the P3-a model (s = 10, b = 5).

    Split into what toys can test exactly (docs/learning/phase3.md, section 8): pyhf q0
    equals the closed form for every count; the median significance of s+b toys equals Z_A;
    the b-only toy p-value equals the exact Poisson tail. The gap between that tail and the
    asymptotic p-value is Poisson discreteness at b = 5 and is reported, not tested.
    """
    s, b = 10, 5
    toys = counting_toys(
        s, b, cfg["significance"]["p3h_toys"], cfg["significance"]["seed"]
    )
    n = np.arange(len(toys["q0_pyhf"]))
    closed = np.where(n > b, 2 * (n * np.log(np.maximum(n, 1) / b) - (n - b)), 0.0)
    np.testing.assert_allclose(toys["q0_pyhf"], closed, atol=1e-6)

    # The count is an integer and 15 is the median of Poisson(15), so the toy median of
    # sqrt(q0) is exactly the Asimov value: its statistical error is zero.
    assert toys["median_z_sb"] == pytest.approx(z_asimov(s, b), rel=1e-6)

    exact = stats.poisson.sf(s + b - 1, b)
    error = np.sqrt(exact * (1 - exact) / toys["n_toys"])
    print(
        f"P3-h: p0 toys {toys['p0_toys']:.3e} +- {error:.1e}, exact {exact:.3e}, "
        f"asymptotic {toys['p0_asymptotic']:.3e}"
    )
    assert abs(toys["p0_toys"] - exact) < 3 * error


def test_shape_model_asimov(cfg):
    """The full model returns the simulated yields at mu = 1 and fits mu = 1 on Asimov data."""
    sig = cfg["significance"]
    samples = cfg["samples"]
    mc = select(
        load_selected(samples["signal"] + samples["zz"] + samples["other_bkg"]),
        cfg,
        "paired",
    )
    edges = bin_edges(*sig["range"], sig["bin_width"])
    inputs = shape_inputs(mc, samples["signal"], edges)
    model = shape_model(*inputs, sig["lumi_rel_unc"], free_background=True)
    data = asimov_data(model)
    expected = model.expected_data(model.config.suggested_init(), include_auxdata=False)
    np.testing.assert_allclose(expected, inputs[0] + inputs[2], rtol=1e-12)
    pars = pyhf.infer.mle.fit(data, model)
    assert pars[model.config.poi_index] == pytest.approx(1.0, abs=1e-4)
