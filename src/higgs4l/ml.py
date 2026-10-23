"""Gradient-boosted classifier and its validation."""

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from higgs4l.data import load_selected
from higgs4l.kinematics import (
    cos_theta_star,
    event_p4,
    min_delta_r,
    rapidity,
    transverse_momentum,
)
from higgs4l.selection import in_range, select, z_pairs
from higgs4l.stats import asimov_data, category_model, discovery_q0, shape_inputs

FEATURES = [
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


def features(events: dict, mz: float) -> np.ndarray:
    """Classifier inputs of paired events, shape (n, 10), columns as in FEATURES."""
    p4 = event_p4(events)
    z1, _, _ = z_pairs(events["lep_type"], events["lep_charge"], p4, mz)
    z1_p4 = p4[np.arange(len(p4))[:, None], z1].sum(axis=1)
    system = p4.sum(axis=1)
    pt_sorted = -np.sort(-events["lep_pt"].astype(np.float64), axis=1)
    return np.column_stack(
        [
            pt_sorted,
            events["m12"],
            events["m34"],
            transverse_momentum(system),
            rapidity(system),
            cos_theta_star(z1_p4, system),
            min_delta_r(
                events["lep_eta"].astype(np.float64),
                events["lep_phi"].astype(np.float64),
            ),
        ]
    )


def classifier_sample(cfg: dict) -> dict[str, np.ndarray]:
    """Paired simulation in the classifier range with features, class and half."""
    samples = cfg["samples"]
    mc = select(
        load_selected(samples["signal"] + samples["zz"] + samples["other_bkg"]),
        cfg,
        "paired",
    )
    mc = in_range(mc, *cfg["classifier"]["range"])
    mc["X"] = features(mc, cfg["paired_selection"]["mz"])
    mc["is_signal"] = np.isin(mc["did"], samples["signal"])
    mc["even"] = mc["eventNumber"] % 2 == 0
    return mc


def train(X, is_signal, weight, kind: str, params: dict, seed: int):
    """Fit the classifier with |w|, each class rescaled to half the number of events.

    Negative weights make the cross-entropy unbounded below (docs/learning/phase4.md,
    section 2). The rescaling balances the classes and keeps leaf hessian sums of order one.
    """
    w = np.abs(weight)
    for cls in (is_signal, ~is_signal):
        w[cls] *= 0.5 * len(w) / w[cls].sum()
    if kind == "gradient_boosting":
        clf = HistGradientBoostingClassifier(**params, random_state=seed)
        return clf.fit(X, is_signal, sample_weight=w)
    clf = make_pipeline(StandardScaler(), LogisticRegression(**params))
    return clf.fit(X, is_signal, logisticregression__sample_weight=w)


def weighted_roc(is_signal, score, weight) -> tuple[np.ndarray, np.ndarray, float]:
    """Background and signal efficiency above each score, and the AUC, signed weights.

    sklearn rejects the non-monotonic curve that negative weights produce; the trapezoid
    area is still the weighted probability that signal scores above background.
    """
    order = np.argsort(-score, kind="stable")
    w, sig = weight[order], is_signal[order]
    tpr = np.concatenate([[0], np.cumsum(np.where(sig, w, 0)) / w[sig].sum()])
    fpr = np.concatenate([[0], np.cumsum(np.where(sig, 0, w)) / w[~sig].sum()])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def train_test_ks(mc: dict, score) -> dict:
    """Train vs test scores per class: unweighted KS p (P4-a) and |w|-weighted KS p."""
    out = {}
    for name, cls in (("signal", mc["is_signal"]), ("background", ~mc["is_signal"])):
        tr, te = cls & mc["even"], cls & ~mc["even"]
        w = np.abs(mc["weight"])
        ks = stats.ks_2samp(score[tr], score[te])
        d_w, p_w = weighted_ks(score[tr], w[tr], score[te], w[te])
        out[name] = {
            "ks_statistic": float(ks.statistic),
            "ks_p": float(ks.pvalue),
            "weighted_ks_statistic": d_w,
            "weighted_ks_p_approx": p_w,
        }
    return out


def fit_classifier(mc: dict, cfg: dict):
    """Gradient boosting on the even half; logistic regression if its scores overtrain.

    Returns the model used, its kind, and the train/test KS results of every model tried.
    """
    even = mc["even"]
    tried = {}
    for kind in ("gradient_boosting", "logistic"):
        clf = train(
            mc["X"][even],
            mc["is_signal"][even],
            mc["weight"][even],
            kind,
            cfg[kind],
            cfg["seed"],
        )
        tried[kind] = train_test_ks(mc, clf.predict_proba(mc["X"])[:, 1])
        if min(r["ks_p"] for r in tried[kind].values()) > cfg["ks_p_min"]:
            break
    return clf, kind, tried


def weighted_ks(a, wa, b, wb) -> tuple[float, float]:
    """Two-sample KS statistic with weighted CDFs; approximate p-value from n_eff."""
    x = np.concatenate([a, b])
    order = np.argsort(x, kind="stable")
    step_a = np.concatenate([wa / wa.sum(), np.zeros(len(b))])[order]
    step_b = np.concatenate([np.zeros(len(a)), wb / wb.sum()])[order]
    # Compare the CDFs only after the last of a run of tied values.
    last = np.append(np.diff(x[order]) > 0, True)
    d = float(np.abs(np.cumsum(step_a) - np.cumsum(step_b))[last].max())
    n_a, n_b = (w.sum() ** 2 / (w**2).sum() for w in (wa, wb))
    n = n_a * n_b / (n_a + n_b)
    return d, float(stats.kstwobign.sf(d * np.sqrt(n)))


def chi2_weighted(w1, s1, w2, s2) -> tuple[float, float]:
    """Shape compatibility of two weighted histograms (Gagunashvili, arXiv:0905.4221).

    w are the bin sums of weights, s the bin sums of squared weights; ndf = bins - 1.
    """
    t1, t2 = w1.sum(), w2.sum()
    x2 = float(np.sum((t1 * w2 - t2 * w1) ** 2 / (t1**2 * s2 + t2**2 * s1)))
    return x2, float(stats.chi2.sf(x2, len(w1) - 1))


def category_inputs(events: dict, high, signal_dids, edges, scale: float) -> list:
    """Templates of the high- and low-score categories, weights times scale.

    Signal comes from each category's own events. The background is the inclusive shape
    times the category's background fraction f, with errors times sqrt(f): the simulated
    background is too sparse for per-category shapes (docs/learning/phase4.md, section 7).
    """
    weighted = {"m4l": events["m4l"], "did": events["did"]}
    weighted["weight"] = scale * events["weight"]
    _, _, background, background_err = shape_inputs(weighted, signal_dids, edges)
    is_bkg = ~np.isin(events["did"], signal_dids)
    out = []
    for mask in (high, ~high):
        part = {k: v[mask] for k, v in weighted.items()}
        signal, signal_err, _, _ = shape_inputs(part, signal_dids, edges)
        f = events["weight"][mask & is_bkg].sum() / events["weight"][is_bkg].sum()
        out.append((signal, signal_err, f * background, np.sqrt(f) * background_err))
    return out


def asimov_z(model) -> float:
    return float(np.sqrt(discovery_q0(model, asimov_data(model))))


def choose_threshold(
    events: dict, score, efficiencies, signal_dids, edges, lumi_rel_unc: float
) -> tuple[float, list]:
    """Score threshold maximizing the two-category Asimov Z on these events, weights x2.

    Candidates keep the given fractions of signal events above the threshold.
    """
    signal_scores = score[events["is_signal"]]
    scan = []
    for eff in efficiencies:
        t = float(np.quantile(signal_scores, 1 - eff))
        inputs = category_inputs(events, score > t, signal_dids, edges, 2.0)
        z = asimov_z(category_model(inputs, lumi_rel_unc))
        scan.append({"signal_efficiency": eff, "threshold": t, "z_asimov": z})
    best = max(scan, key=lambda row: row["z_asimov"])
    return best["threshold"], scan
