"""m4l of data and simulation for the notebook selection and the paired selection."""

import matplotlib.pyplot as plt
import numpy as np

from higgs4l.data import ROOT, load_config, load_selected
from higgs4l.plotting import m4l_stack
from higgs4l.selection import select
from higgs4l.stats import bin_edges


def main():
    cfg = load_config()
    hist = cfg["histogram"]
    edges = bin_edges(hist["m4l_min"], hist["m4l_max"], hist["bin_width"])
    samples = cfg["samples"]
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    # The notebook figure reproduces the notebook (abs weights, the signal samples behind
    # its stored outputs); the paired figure is the corrected analysis (signed weights).
    figures = {
        "notebook": (cfg["notebook_outputs_signal"], np.abs, "Notebook selection"),
        "paired": (samples["signal"], lambda w: w, "Z-paired selection"),
    }
    for name, (signal, to_weight, title) in figures.items():
        groups = [
            (
                r"Other bkg ($Z$+jets, $t\bar{t}$, $t\bar{t}V$, $VVV$)",
                samples["other_bkg"],
                "#6b59d3",
            ),
            (r"$ZZ^*$", samples["zz"], "#ff0000"),
            (r"Signal ($m_H$ = 125 GeV)", signal, "#00cdff"),
        ]
        mc = []
        for label, dids, colour in groups:
            events = select(load_selected(dids), cfg, name)
            mc.append((label, events["m4l"], to_weight(events["weight"]), colour))
        data = select(load_selected(["data"]), cfg, name)
        fig, ax = plt.subplots(figsize=(9, 6))
        m4l_stack(ax, edges, data["m4l"], mc, cfg["lumi_fb"], title)
        fig.savefig(out / f"m4l_{name}.png", dpi=150)
        fig.savefig(out / f"m4l_{name}.pdf", metadata={"CreationDate": None})
        plt.close(fig)


if __name__ == "__main__":
    main()
