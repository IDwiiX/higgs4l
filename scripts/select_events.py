"""Download the samples, apply both selections, and store the selected events.

Writes data/selected/*.npz, results/cutflow.json, results/cutflow.md and
results/provenance.json. Needs the network on the first run (about 1 GB).
"""

import json
import platform
from importlib.metadata import version

import atlasopenmagic as atom
import numpy as np

from higgs4l.data import (
    ROOT,
    STORED_DTYPES,
    fetch,
    file_record,
    file_urls,
    load_config,
    read_events,
    save_selected,
)
from higgs4l.selection import cumulative, notebook_cuts, paired_cuts, preselection
from higgs4l.weights import event_weight

NOTEBOOK = {
    "url": "https://github.com/atlas-outreach-data-tools/notebooks-collection-opendata/"
    "blob/56f3ea775c2925199be8ef7e4ff7dff4662b2170/13-TeV-examples/uproot_python/"
    "HZZAnalysis.ipynb",
    "commit": "56f3ea775c2925199be8ef7e4ff7dff4662b2170",
}


def add_cutflow(table, cuts, weight):
    """Accumulate count, signed and abs weighted sums after each cumulative cut."""
    stages = {"all events": np.ones(len(weight), dtype=bool)} | cumulative(cuts)
    for name, mask in stages.items():
        row = table.setdefault(name, [0, 0.0, 0.0])
        row[0] += int(mask.sum())
        row[1] += float(weight[mask].sum())
        row[2] += float(np.abs(weight[mask]).sum())


def metadata_check(did, record):
    """Compare the file's normalization with the release metadata.

    The two split sigma * filter efficiency * k-factor differently between the three
    factors, so only the product is comparable. The analysis uses the file branches.
    """
    meta = atom.get_metadata(str(did))
    branch = record["xsec"] * record["filteff"] * record["kfac"]
    metadata = meta["cross_section_pb"] * meta["genFiltEff"] * meta["kFactor"]
    return {
        "effective_xsec_pb": {"branch": branch, "metadata": metadata},
        "effective_xsec_rel_diff": branch / metadata - 1,
        "sum_of_weights": {
            "branch": record["sum_of_weights"],
            "metadata": meta["sumOfWeights"],
        },
    }


def cutflow_markdown(cutflow):
    lines = []
    for selection, groups in cutflow.items():
        lines += [
            f"## {selection} selection",
            "",
            "Data are event counts; MC columns are weighted yields with signed weights,"
            " and with the notebook's abs weights in brackets.",
            "",
            "| Cut | Data | Signal | ZZ* | Other bkg |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for stage in groups["data"]:
            cells = [str(groups["data"][stage][0])]
            for group in ("signal", "zz", "other_bkg"):
                _, signed, absolute = groups[group][stage]
                cells.append(f"{signed:.3f} ({absolute:.3f})")
            lines.append(f"| {stage} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    cfg = load_config()
    lumi = cfg["lumi_fb"]
    cutflow = {"notebook": {}, "paired": {}}
    files = []
    for group, dids in cfg["samples"].items():
        for name in cutflow:
            cutflow[name][group] = {}
        for did in dids:
            parts = []
            for url in file_urls(did, cfg["release"], cfg["skim"]):
                path = fetch(url)
                events = read_events(path)
                record = {"did": did} | file_record(path, url, events)
                if group != "data":
                    record["metadata"] = metadata_check(did, record)
                files.append(record)
                if group == "data":
                    weight = np.ones(len(events["eventNumber"]))
                else:
                    weight = event_weight(events, lumi)
                add_cutflow(
                    cutflow["notebook"][group], notebook_cuts(events, cfg), weight
                )
                add_cutflow(cutflow["paired"][group], paired_cuts(events, cfg), weight)
                keep = preselection(events, cfg)
                part = {k: events[k][keep] for k in STORED_DTYPES}
                if group != "data":
                    part["weight"] = weight[keep]
                parts.append(part)
            save_selected(
                did, {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            )
            print(
                f"{group:10s} {did}: {sum(len(p['eventNumber']) for p in parts)} stored"
            )

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "cutflow.json").write_text(json.dumps(cutflow, indent=1) + "\n")
    (results / "cutflow.md").write_text(cutflow_markdown(cutflow))
    provenance = {
        "release": cfg["release"],
        "skim": cfg["skim"],
        "lumi_fb": lumi,
        "versions": {
            p: version(p) for p in ("atlasopenmagic", "uproot", "awkward", "numpy")
        }
        | {"python": platform.python_version()},
        "notebook": NOTEBOOK,
        "files": files,
    }
    (results / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")


if __name__ == "__main__":
    main()
