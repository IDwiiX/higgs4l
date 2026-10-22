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


def shape_fit(ax, m, w, edges, x, density, title):
    """Weighted simulation histogram with a fitted density scaled to it."""
    centres = (edges[:-1] + edges[1:]) / 2
    h, _ = np.histogram(m, edges, weights=w)
    err = np.sqrt(np.histogram(m, edges, weights=w**2)[0])
    width = edges[1] - edges[0]
    ax.errorbar(centres, h, yerr=err, fmt="ko", markersize=3, label="Simulation")
    ax.plot(x, density * w.sum() * width, "r-", label="Fit")
    ax.set_xlabel(r"$m_{4\ell}$ [GeV]")
    ax.set_ylabel(f"Events / {width:g} GeV")
    ax.set_ylim(bottom=0)
    ax.text(0.05, 0.92, title, transform=ax.transAxes)
    ax.legend(loc="upper right", fontsize="small")


def pull_histogram(ax, pull, mean, width):
    """Toy pulls against the standard normal."""
    edges = np.linspace(-4, 4, 41)
    ax.hist(pull, edges, histtype="step", color="k", label="Toys")
    x = np.linspace(-4, 4, 400)
    scale = len(pull) * (edges[1] - edges[0]) / np.sqrt(2 * np.pi)
    ax.plot(x, scale * np.exp(-0.5 * x**2), "r-", label="N(0, 1)")
    ax.set_xlabel(r"Pull $(\hat{m}_H - m_H^{true}) / \sigma$")
    ax.set_ylabel("Toys")
    ax.text(0.05, 0.92, f"mean {mean:.3f}, width {width:.3f}", transform=ax.transAxes)
    ax.legend(loc="upper right", fontsize="small")


def likelihood_scan(ax, grid, m2dnll, label):
    """Profile -2 Delta ln L against m_H, with the 1 sigma line."""
    ax.plot(grid, m2dnll, "k-", label=label)
    ax.axhline(1.0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$m_H$ [GeV]")
    ax.set_ylabel(r"$-2\,\Delta\ln L$")
    ax.set_ylim(0, 6)
    ax.legend(loc="upper right", fontsize="small")


def mass_fit_overlay(ax, m, model, fit: dict, edges, lumi_fb: float):
    """Data histogram with the fitted signal + background density and the background."""
    centres = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    n, _ = np.histogram(m, edges)
    x = np.linspace(edges[0], edges[-1], 600)
    background = fit["n_b"] * model.background_pdf(x) * width
    signal = fit["n_s"] * model.signal_pdf(x, fit["m_h"]) * width
    ax.errorbar(centres, n, yerr=np.sqrt(n), fmt="ko", markersize=4, label="Data")
    ax.plot(x, signal + background, "r-", label="Fit, signal + background")
    ax.plot(x, background, "b--", label="Fit, background")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, 1.8 * n.max())
    ax.set_xlabel(r"$m_{4\ell}$ [GeV]")
    ax.set_ylabel(f"Events / {width:g} GeV")
    hep.atlas.label(ax=ax, text="Open Data", data=True, lumi=lumi_fb, com=13)
    ax.text(0.05, 0.80, rf"$m_H$ = {fit['m_h']:.2f} GeV", transform=ax.transAxes)
    ax.legend(loc="upper right", fontsize="small")
