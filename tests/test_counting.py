import pytest

from higgs4l.stats import naive_z, z_asimov, z_asimov_unc

pytestmark = pytest.mark.phase1

# The notebook's stored window numbers (REFERENCE_VALUES.md, P1-a and P1-b).
REF_TOTAL, REF_B = 28.729140024845144, 10.300928352079978
REF_S = REF_TOTAL - REF_B

# Closed forms reproduce the table to 1e-3; with our own S and B, to 0.1% (P1-a tolerance).
METHODS = {
    "P1-h": (naive_z, 5.742),
    "P1-i": (z_asimov, 4.699),
    "P1-j": (lambda s, b: z_asimov_unc(s, b, 0.1 * b), 4.377),
    "P1-k": (lambda s, b: z_asimov_unc(s, b, 0.3 * b), 3.075),
}


def check(method, notebook_window):
    z, target = METHODS[method]
    assert z(REF_S, REF_B) == pytest.approx(target, rel=1e-3)
    ours = z(notebook_window["signal"], notebook_window["background"])
    assert ours == pytest.approx(target, rel=1e-3)


def test_p1h_naive(notebook_window):
    """P1-h: naive S/sqrt(B) in the notebook window."""
    check("P1-h", notebook_window)


def test_p1i_asimov(notebook_window):
    """P1-i: Asimov significance, no background uncertainty."""
    check("P1-i", notebook_window)


def test_p1j_asimov_10pct(notebook_window):
    """P1-j: Asimov significance, 10% background uncertainty."""
    check("P1-j", notebook_window)


def test_p1k_asimov_30pct(notebook_window):
    """P1-k: Asimov significance, 30% background uncertainty."""
    check("P1-k", notebook_window)
