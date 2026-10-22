"""Figures in the ATLAS Open Data style."""

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

plt.style.use(hep.style.ATLAS)


def m4l_stack(ax, edges, data_m4l, mc, lumi_fb: float, title: str):
    """Data over stacked simulation with the MC statistical band.

    mc is a list of (label, m4l, weights, colour), stacked bottom to top.
    """
    centres = (edges[:-1] + edges[1:]) / 2
    width = np.diff(edges)
    total = np.zeros(len(centres))
    sumw2 = np.zeros(len(centres))
    for label, m4l, weights, colour in mc:
        h, _ = np.histogram(m4l, edges, weights=weights)
        ax.bar(centres, h, width, bottom=total, color=colour, label=label)
        total += h
        sumw2 += np.histogram(m4l, edges, weights=weights**2)[0]
    error = np.sqrt(sumw2)
    ax.bar(
        centres,
        2 * error,
        width,
        bottom=total - error,
        fill=False,
        hatch="////",
        linewidth=0,
        label="MC stat. unc.",
    )
    n, _ = np.histogram(data_m4l, edges)
    ax.errorbar(centres, n, yerr=np.sqrt(n), fmt="ko", markersize=4, label="Data")

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, 2.0 * max(n.max(), total.max()))
    ax.set_xlabel(r"$m_{4\ell}$ [GeV]")
    ax.set_ylabel(f"Events / {width[0]:g} GeV")
    hep.atlas.label(ax=ax, text="Open Data", data=True, lumi=lumi_fb, com=13)
    ax.text(0.05, 0.80, "for education", transform=ax.transAxes, style="italic")
    ax.text(0.05, 0.73, title, transform=ax.transAxes)
    ax.legend(loc="upper right", fontsize="small")
