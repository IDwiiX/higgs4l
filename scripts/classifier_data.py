"""Unblind phase 4: observed significance with the two classifier categories.

Uses the frozen classifier, threshold and test-half templates of results/ml.json and writes
results/ml_data.json.
"""

import json

import numpy as np

from higgs4l.data import ROOT, load_config, load_selected
from higgs4l.ml import category_inputs, classifier_sample, features, fit_classifier
from higgs4l.selection import in_range, select
from higgs4l.stats import bin_edges, category_model, shape_inputs, shape_model

from significance_data import observed


def main():
    cfg = load_config()
    c = cfg["classifier"]
    sig = cfg["significance"]
    signal_dids = cfg["samples"]["signal"]
    edges = bin_edges(*sig["range"], sig["bin_width"])
    expected = json.loads((ROOT / "results" / "ml.json").read_text())

    mc = classifier_sample(cfg)
    clf, kind, _ = fit_classifier(mc, c)
    assert kind == expected["classifier"]
    mc["score"] = clf.predict_proba(mc["X"])[:, 1]
    test = {k: v[~mc["even"]] for k, v in mc.items()}
    threshold = expected["threshold"]

    data = in_range(select(load_selected(["data"]), cfg, "paired"), *c["range"])
    data_high = clf.predict_proba(features(data, cfg["paired_selection"]["mz"]))[:, 1]
    data_high = data_high > threshold

    scaled = {"m4l": test["m4l"], "did": test["did"], "weight": 2 * test["weight"]}
    inclusive = shape_model(
        *shape_inputs(scaled, signal_dids, edges), sig["lumi_rel_unc"], True
    )
    categories = category_model(
        category_inputs(test, test["score"] > threshold, signal_dids, edges, 2.0),
        sig["lumi_rel_unc"],
    )

    counts = {
        name: np.histogram(data["m4l"][mask], edges)[0]
        for name, mask in (("high", data_high), ("low", ~data_high))
    }
    results = {"threshold": threshold, "classifier": kind}
    results["inclusive"], _ = observed(inclusive, counts["high"] + counts["low"])
    results["categories"], _ = observed(
        categories, np.concatenate([counts["high"], counts["low"]])
    )
    results["inclusive"]["z_expected"] = expected["z_asimov_test_inclusive"]
    results["categories"]["z_expected"] = expected["z_asimov_test_categories"]
    results["data_events"] = {k: int(v.sum()) for k, v in counts.items()}
    results["data_counts"] = {k: v.tolist() for k, v in counts.items()}
    results["note"] = (
        "templates from the odd simulation half x2, as in results/ml.json; the phase 3 "
        "observed Z with full-sample templates is in results/significance.json"
    )
    (ROOT / "results" / "ml_data.json").write_text(json.dumps(results, indent=1) + "\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
