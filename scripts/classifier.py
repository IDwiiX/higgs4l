"""Phase 4, blinded: classifier, category threshold and expected gain, on simulation only.

Trains on even eventNumber, evaluates on odd. Writes results/ml.json and the ROC, score and
sculpting figures. No data are used.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from higgs4l.data import ROOT, load_config
from higgs4l.ml import (
    FEATURES,
    asimov_z,
    category_inputs,
    chi2_weighted,
    choose_threshold,
    classifier_sample,
    fit_classifier,
    weighted_roc,
)
from higgs4l.stats import bin_edges, category_model, shape_inputs, shape_model


def half(events, mask):
    return {k: v[mask] for k, v in events.items()}


def sculpting(test, high, edges):
    """Background m4l shape, high vs low category, on the test half (P4-b, reported)."""
    bkg = ~test["is_signal"]
    m, w = test["m4l"], test["weight"]

    def hist(mask):
        return (
            np.histogram(m[mask], edges, weights=w[mask])[0],
            np.histogram(m[mask], edges, weights=w[mask] ** 2)[0],
        )

    (w1, s1), (w2, s2) = hist(bkg & high), hist(bkg & ~high)
    x2, p = chi2_weighted(w1, s1, w2, s2)
    cov = np.cov(test["score"][bkg], m[bkg], aweights=np.abs(w[bkg]))
    return {
        "edges": edges.tolist(),
        "events_high": int((bkg & high).sum()),
        "events_low": int((bkg & ~high).sum()),
        "chi2": x2,
        "ndf": len(edges) - 2,
        "p": p,
        "mean_m4l_high": float(np.average(m[bkg & high], weights=w[bkg & high])),
        "mean_m4l_low": float(np.average(m[bkg & ~high], weights=w[bkg & ~high])),
        "score_m4l_correlation_abs_weights": float(
            cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ),
        "high": [w1.tolist(), s1.tolist()],
        "low": [w2.tolist(), s2.tolist()],
    }


def plot_roc(curves, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, (fpr, tpr, auc), style in curves:
        ax.plot(fpr, tpr, style, label=f"{label}, AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", linewidth=1)
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.text(0.35, 0.1, "signed weights, 110-160 GeV", transform=ax.transAxes)
    ax.legend(loc="lower right", fontsize="small")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scores(mc, ks, threshold, path):
    edges = np.linspace(0, 1, 26)
    centres = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, cls, colour in (
        ("signal", mc["is_signal"], "r"),
        ("background", ~mc["is_signal"], "b"),
    ):
        train = np.histogram(mc["score"][cls & mc["even"]], edges, density=True)[0]
        ax.stairs(
            train, edges, color=colour, fill=True, alpha=0.3, label=f"{name}, train"
        )
        s = mc["score"][cls & ~mc["even"]]
        h, _ = np.histogram(s, edges, density=True)
        n, _ = np.histogram(s, edges)
        ax.errorbar(
            centres,
            h,
            yerr=h / np.sqrt(np.maximum(n, 1)),
            fmt="o",
            color=colour,
            markersize=3,
            label=f"{name}, test (KS p = {ks[name]['ks_p']:.2f})",
        )
    ax.axvline(threshold, color="k", linestyle="--", linewidth=1, label="Threshold")
    ax.set_xlabel("Classifier score")
    ax.set_ylabel("Normalized events (unweighted)")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize="small")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sculpting(result, path):
    edges = np.array(result["edges"])
    centres = (edges[:-1] + edges[1:]) / 2
    high, low = np.array(result["high"]), np.array(result["low"])
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, (w, s), colour, shift in (
        ("high score", high, "r", -1),
        ("low score", low, "b", 1),
        ("inclusive", high + low, "k", 0),
    ):
        ax.errorbar(
            centres + shift,
            w / w.sum(),
            yerr=np.sqrt(s) / w.sum(),
            fmt="o",
            color=colour,
            markersize=4,
            label=label,
        )
    ax.set_xlabel(r"$m_{4\ell}$ [GeV]")
    ax.set_ylabel("Fraction of background (test half)")
    ax.set_ylim(bottom=0)
    ax.text(
        0.05,
        0.92,
        rf"high vs low: $\chi^2$/ndf = {result['chi2']:.1f}/{result['ndf']}, "
        f"p = {result['p']:.2f}",
        transform=ax.transAxes,
    )
    ax.legend(loc="upper right", fontsize="small")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    c = cfg["classifier"]
    sig = cfg["significance"]
    signal_dids = cfg["samples"]["signal"]
    edges = bin_edges(*sig["range"], sig["bin_width"])

    mc = classifier_sample(cfg)
    clf, kind, tried = fit_classifier(mc, c)
    mc["score"] = clf.predict_proba(mc["X"])[:, 1]
    train_half, test = half(mc, mc["even"]), half(mc, ~mc["even"])

    threshold, scan = choose_threshold(
        train_half,
        train_half["score"],
        c["signal_efficiencies"],
        signal_dids,
        edges,
        sig["lumi_rel_unc"],
    )
    high = test["score"] > threshold

    scaled = {"m4l": test["m4l"], "did": test["did"], "weight": 2 * test["weight"]}
    inclusive_inputs = shape_inputs(scaled, signal_dids, edges)
    inclusive = asimov_z(
        shape_model(*inclusive_inputs, sig["lumi_rel_unc"], free_background=True)
    )
    categories = category_inputs(test, high, signal_dids, edges, 2.0)
    categorized = asimov_z(category_model(categories, sig["lumi_rel_unc"]))

    roc = {
        name: weighted_roc(h["is_signal"], h["score"], h["weight"])
        for name, h in (("test", test), ("train", train_half))
    }
    bkg = ~mc["is_signal"]
    heaviest = np.argmax(np.where(bkg, np.abs(mc["weight"]), 0))
    results = {
        "config": c,
        "features": FEATURES,
        "classifier": kind,
        "train_test_ks_by_model": tried,
        "train_test_ks": tried[kind],
        "auc_test_signed": roc["test"][2],
        "auc_train_signed": roc["train"][2],
        "threshold": threshold,
        "threshold_scan_train": scan,
        "z_asimov_test_inclusive": inclusive,
        "z_asimov_test_categories": categorized,
        "z_gain": categorized - inclusive,
        "test_yields_x2": {
            name: {"signal": float(inp[0].sum()), "background": float(inp[2].sum())}
            for name, inp in (
                ("high", categories[0]),
                ("low", categories[1]),
                ("inclusive", inclusive_inputs),
            )
        },
        "sculpting": sculpting(
            test, high, bin_edges(*c["range"], c["sculpting_bin_width"])
        ),
        "largest_background_weight": {
            "did": int(mc["did"][heaviest]),
            "weight": float(mc["weight"][heaviest]),
            "half": "train" if mc["even"][heaviest] else "test",
            "score": float(mc["score"][heaviest]),
        },
        "note": "simulation only; test-half yields and errors scaled by 2; inclusive and "
        "category Z both from the test half; category background = inclusive shape times "
        "the category's background fraction",
    }
    (ROOT / "results" / "ml.json").write_text(json.dumps(results, indent=1) + "\n")

    figures = ROOT / "figures"
    plot_roc(
        [("test (odd)", roc["test"], "k-"), ("train (even)", roc["train"], "r--")],
        figures / "classifier_roc.png",
    )
    plot_scores(mc, tried[kind], threshold, figures / "classifier_scores.png")
    plot_sculpting(results["sculpting"], figures / "classifier_sculpting.png")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
