import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from higgs4l.ml import (
    FEATURES,
    asimov_z,
    category_inputs,
    chi2_weighted,
    choose_threshold,
    classifier_sample,
    fit_classifier,
    train_test_ks,
    weighted_ks,
    weighted_roc,
)
from higgs4l.stats import bin_edges, category_model, shape_inputs, shape_model

pytestmark = pytest.mark.phase4

SPEC_FEATURES = [
    "lep_pt_1",
    "lep_pt_2",
    "lep_pt_3",
    "lep_pt_4",
    "m12",
    "m34",
    "pt_4l",
    "y_4l",
    "cos_theta_star",
    "min_delta_r",
]


@pytest.fixture(scope="module")
def sample(cfg):
    return classifier_sample(cfg)


@pytest.fixture(scope="module")
def classifier(cfg, sample):
    """Trained on even eventNumber with the config seed, as in scripts/classifier.py."""
    clf, kind, tried = fit_classifier(sample, cfg["classifier"])
    return clf, kind, clf.predict_proba(sample["X"])[:, 1], tried


def test_chi2_weighted_exact():
    """Identical shapes give 0; a 3-bin unit-weight example matches the formula by hand."""
    h = np.array([10.0, 20.0, 30.0])
    assert chi2_weighted(h, h, 2 * h, 4 * h)[0] == pytest.approx(0.0, abs=1e-12)
    other = np.array([5.0, 5.0, 10.0])
    expected = 10000 / 22000 + 10000 / 26000
    assert chi2_weighted(h, h, other, other)[0] == pytest.approx(expected, rel=1e-12)


def test_weighted_ks_unit_weights():
    """With unit weights the statistic is the ordinary two-sample KS statistic."""
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=300), rng.normal(0.2, size=200)
    d, _ = weighted_ks(a, np.ones(300), b, np.ones(200))
    assert d == pytest.approx(stats.ks_2samp(a, b).statistic, rel=1e-12)


def test_weighted_roc_matches_sklearn_for_positive_weights():
    rng = np.random.default_rng(2)
    y = rng.random(500) < 0.4
    score, w = rng.random(500) + 0.3 * y, rng.random(500)
    _, _, auc = weighted_roc(y, score, w)
    assert auc == pytest.approx(roc_auc_score(y, score, sample_weight=w), rel=1e-12)


def test_category_background_is_inclusive_shape(cfg, sample):
    """The category backgrounds add up to the inclusive template, bin by bin."""
    edges = bin_edges(*cfg["significance"]["range"], cfg["significance"]["bin_width"])
    signal_dids = cfg["samples"]["signal"]
    high = sample["m12"] > 80
    cats = category_inputs(sample, high, signal_dids, edges, 1.0)
    _, _, b, _ = shape_inputs(sample, signal_dids, edges)
    np.testing.assert_allclose(cats[0][2] + cats[1][2], b, rtol=1e-12)
    np.testing.assert_allclose(cats[0][2] / cats[0][2].sum(), b / b.sum(), rtol=1e-12)


def test_train_test_scores_reported(cfg, sample, classifier):
    """P4-a (reported, no target): the boosted trees fail the train vs test KS rule, so
    the analysis uses logistic regression (docs/learning/phase4.md, section 9)."""
    _, kind, score, tried = classifier
    p_min = cfg["classifier"]["ks_p_min"]
    assert min(r["ks_p"] for r in tried["gradient_boosting"].values()) <= p_min
    assert kind == "logistic"
    ks = train_test_ks(sample, score)
    for name in ("signal", "background"):
        print(f"P4-a {kind} {name}: KS p {ks[name]['ks_p']:.3g}")


def test_background_shape_reported(cfg, sample, classifier):
    """P4-b (reported, no target): background m4l, high vs low category, test half,
    with the threshold chosen on the training half as in scripts/classifier.py."""
    _, _, score, _ = classifier
    c = cfg["classifier"]
    sig = cfg["significance"]
    even = sample["even"]
    threshold, _ = choose_threshold(
        {k: v[even] for k, v in sample.items()},
        score[even],
        c["signal_efficiencies"],
        cfg["samples"]["signal"],
        bin_edges(*sig["range"], sig["bin_width"]),
        sig["lumi_rel_unc"],
    )
    edges = bin_edges(*c["range"], c["sculpting_bin_width"])
    bkg = ~sample["is_signal"] & ~even
    m, w = sample["m4l"], sample["weight"]
    high = score > threshold

    def hist(mask):
        return (
            np.histogram(m[mask], edges, weights=w[mask])[0],
            np.histogram(m[mask], edges, weights=w[mask] ** 2)[0],
        )

    x2, p = chi2_weighted(*hist(bkg & high), *hist(bkg & ~high))
    print(
        f"P4-b: threshold {threshold:.3f}, chi2 {x2:.2f} / {len(edges) - 2}, p {p:.3f}"
    )
    assert np.isfinite(x2)


def test_uninformative_split_has_no_gain(cfg, sample):
    """P4-c: a split by a fixed fraction reproduces the inclusive Z_A exactly."""
    sig = cfg["significance"]
    edges = bin_edges(*sig["range"], sig["bin_width"])
    s, s_err, b, b_err = shape_inputs(sample, cfg["samples"]["signal"], edges)
    f = 0.3
    split = [(g * s, np.sqrt(g) * s_err, g * b, np.sqrt(g) * b_err) for g in (f, 1 - f)]
    inclusive = asimov_z(shape_model(s, s_err, b, b_err, sig["lumi_rel_unc"], True))
    categorized = asimov_z(category_model(split, sig["lumi_rel_unc"]))
    print(f"P4-c: inclusive {inclusive:.6f}, proportional split {categorized:.6f}")
    assert categorized == pytest.approx(inclusive, rel=1e-6)


def test_m4l_not_a_feature(sample, classifier):
    """P4-d: the features are the spec list, and none of them is m4l."""
    clf = classifier[0]
    assert FEATURES == SPEC_FEATURES
    assert "m4l" not in FEATURES
    assert sample["X"].shape[1] == 10
    for column in sample["X"].T:
        assert not np.allclose(column, sample["m4l"])
    assert clf.n_features_in_ == 10
