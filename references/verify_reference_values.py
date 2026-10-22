#!/usr/bin/env python3
"""Recompute the reference numbers in docs/REFERENCE_VALUES.md.

Needs only Python 3 and numpy/scipy. Run: python3 references/verify_reference_values.py
"""
from math import log, sqrt

from scipy import special


def z_asimov(s, b):
    """Median discovery significance, no background uncertainty (Cowan, Cranmer, Gross, Vitells, EPJC 71 (2011) 1554)."""
    return sqrt(2 * ((s + b) * log(1 + s / b) - s))


def z_asimov_unc(s, b, sb):
    """Same, with background uncertainty sb from an auxiliary Poisson measurement."""
    sb2 = sb * sb
    t1 = (s + b) * log((s + b) * (b + sb2) / (b * b + (s + b) * sb2))
    t2 = (b * b / sb2) * log(1 + sb2 * s / (b * (b + sb2)))
    return sqrt(2 * (t1 - t2))


def pois_ll(n, mu):
    # continuous in n so Asimov (non-integer) counts work
    return n * log(mu) - mu - special.gammaln(n + 1)


def z_onoff_profile(n_on, n_off, tau):
    """Observed significance for the on/off problem by direct profile likelihood (numerical check)."""
    def nll(s, b):
        return -(pois_ll(n_on, s + b) + pois_ll(n_off, tau * b))

    b0 = (n_on + n_off) / (1 + tau)
    nll0 = nll(0.0, b0)
    s_hat, b_hat = n_on - n_off / tau, n_off / tau
    nll1 = nll(s_hat, b_hat)
    return sqrt(max(0.0, 2 * (nll0 - nll1)))


def main():
    print("== Official ATLAS notebook (HZZAnalysis.ipynb, 2025e-13tev-beta, 36.6 fb^-1) ==")
    n_tot, n_bkg = 28.729140024845144, 10.300928352079978
    s = n_tot - n_bkg
    print(f"window 122.5-130 GeV (bins 17:20 of 2.5 GeV from 80 GeV)")
    print(f"printed N_sig = {n_tot:.3f}  (this is signal + background)")
    print(f"printed N_bg  = {n_bkg:.3f}")
    print(f"true signal S = {s:.3f}")
    print(f"notebook metric N_sig/sqrt(B + 0.3 B^2) = {n_tot / sqrt(n_bkg + 0.3 * n_bkg**2):.3f} (notebook prints 4.426)")
    print(f"naive S/sqrt(B) = {s / sqrt(n_bkg):.3f}")
    print(f"Asimov Z, no bkg uncertainty = {z_asimov(s, n_bkg):.3f}")
    for frac in (0.1, 0.3):
        print(f"Asimov Z, bkg uncertainty {int(frac*100)}% = {z_asimov_unc(s, n_bkg, frac * n_bkg):.3f}")

    print("\n== Counting-experiment test cases for the likelihood code ==")
    for s_, b_ in [(10, 5), (5, 20), (s, n_bkg)]:
        print(f"s={s_:.3f} b={b_:.3f}: Z_A = {z_asimov(s_, b_):.4f}, "
              f"Z_A(10%) = {z_asimov_unc(s_, b_, 0.1 * b_):.4f}, Z_A(30%) = {z_asimov_unc(s_, b_, 0.3 * b_):.4f}")

    print("\n== Cross-check: Asimov formula vs direct on/off profile likelihood ==")
    s_, b_, frac = 18.428, 10.301, 0.3
    tau = b_ / (frac * b_) ** 2
    n_on, n_off = s_ + b_, tau * b_
    print(f"on/off with Asimov counts: Z = {z_onoff_profile(n_on, n_off, tau):.4f} "
          f"vs formula {z_asimov_unc(s_, b_, frac * b_):.4f}")


if __name__ == "__main__":
    main()
