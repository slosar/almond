"""Pure-NumPy reference for spin-2 HEALPix synthesis (ducc0's native spin path).

Port of ducc0's sxdata/Ylmgen(s=2) machinery: two scaled Wigner-d chains
(lambda^+ = d^l_{m,-s}-like, lambda^- = d^l_{m,s}-like) advanced with a
Delta-l=1 three-term recursion in cos(theta), eight real accumulator pairs,
and the parity-interleaved E/B accumulation.  Spin alm input is (a_E, a_B),
output maps are (Q, U), healpy conventions (validated against ducc0).
"""

from __future__ import annotations

import numpy as np

from .geometry import ring_geometry, pair_geometry
from .reference import FBIG, FSMALL, FBIGHALF, FTOL, _mypow_scaled, phase_to_map

SPIN = 2


def spin_norm_l(lmax: int) -> np.ndarray:
    """get_norm(lmax, 2): -0.5 sqrt((2l+1)/4pi), zero below l=2."""
    l = np.arange(lmax + 1, dtype=np.float64)
    n = -0.5 * np.sqrt((2 * l + 1) / (4 * np.pi))
    n[:SPIN] = 0.0
    return n


def spin_prefac(mmax: int):
    """prefac[m] (with 2^800 scale): sqrt((2 mhi)! / ((mhi+mlo)! (mhi-mlo)!))."""
    n = 2 * (max(mmax, SPIN)) + 1
    fac = np.empty(n)
    fsc = np.zeros(n, dtype=np.int64)
    fac[0] = 1.0
    for i in range(1, n):
        v, s = fac[i - 1] * np.sqrt(i), fsc[i - 1]
        while abs(v) > FBIGHALF:
            v *= FSMALL
            s += 1
        fac[i], fsc[i] = v, s
    def norm2(v, s):
        # two-sided (ducc's normalize): ratios can under- OR overflow the band
        while abs(v) > FBIGHALF:
            v *= FSMALL
            s += 1
        while 0.0 < abs(v) < FBIGHALF * FSMALL:
            v *= FBIG
            s -= 1
        return v, s

    pre = np.empty(mmax + 1)
    psc = np.empty(mmax + 1, dtype=np.int64)
    for m in range(mmax + 1):
        mlo, mhi = min(m, SPIN), max(m, SPIN)
        v = fac[2 * mhi] / fac[mhi + mlo]
        s = fsc[2 * mhi] - fsc[mhi + mlo]
        v, s = norm2(v, s)
        v /= fac[mhi - mlo]
        s -= fsc[mhi - mlo]
        v, s = norm2(v, s)
        pre[m], psc[m] = v, s
    return pre, psc


def spin_coeffs(m: int, lmax: int):
    """fx (a,b) for l = mhi..lmax+1 and alpha[l]; ducc Ylmgen::prepare, s!=0."""
    s = SPIN
    mhi = max(m, s)
    fx_a = np.zeros(lmax + 3)
    fx_b = np.zeros(lmax + 3)
    alpha = np.zeros(lmax + 3)
    flm1 = lambda i: np.sqrt(1.0 / (i + 1.0))
    flm2 = lambda i: np.sqrt(i / (i + 1.0))
    alpha[mhi] = 1.0
    for l in range(mhi, lmax + 1):
        t = flm1(l + m) * flm1(l - m) * flm1(l + s) * flm1(l - s)
        lt = 2.0 * l + 1.0
        l1 = l + 1.0
        flp10 = l1 * lt * t
        flp11 = (m * s / (l * l1)) if l > 0 else 0.0
        t2 = flm2(l + m) * flm2(l - m) * flm2(l + s) * flm2(l - s)
        flp12 = t2 * l1 / l if l > 0 else 0.0
        alpha[l + 1] = alpha[l - 1] * flp12 if l > mhi else 1.0
        fx_a[l + 1] = flp10 * alpha[l] / alpha[l + 1]
        fx_b[l + 1] = flp11 * fx_a[l + 1]
    return fx_a, fx_b, alpha


def synthesis_spin2(alm_E: np.ndarray, alm_B: np.ndarray, nside: int,
                    lmax: int) -> np.ndarray:
    """Spin-2 synthesis: healpy (a_E, a_B) -> (2, npix) maps (Q, U)."""
    geom = ring_geometry(nside)
    pairs = pair_geometry(geom, lmax, spin=SPIN)
    mmax = lmax
    norm = spin_norm_l(lmax)
    pre, psc = spin_prefac(mmax)
    mstart = (np.arange(mmax + 1) * (2 * lmax + 1 - np.arange(mmax + 1)) // 2)
    phase = np.zeros((2, geom.nring, mmax + 1), dtype=np.complex128)

    for m in range(mmax + 1):
        act = pairs.mlim >= m
        if not act.any():
            continue
        cth = pairs.cth[act]
        sth = pairs.sth[act]
        mhi = max(m, SPIN)
        mlo = min(m, SPIN)
        fx_a, fx_b, alpha = spin_coeffs(m, lmax)

        # prepped alm: aG/aC(l) = alm * norm_l * alpha[l], zero-padded
        ls = np.arange(m, lmax + 1)
        aG = np.zeros(lmax + 2, dtype=np.complex128)
        aC = np.zeros(lmax + 2, dtype=np.complex128)
        aG[ls] = alm_E[mstart[m] + ls] * norm[ls]
        aC[ls] = alm_B[mstart[m] + ls] * norm[ls]
        aG[mhi: lmax + 1] *= alpha[mhi: lmax + 1]
        aC[mhi: lmax + 1] *= alpha[mhi: lmax + 1]

        # prefactor powers of the half-angle sines/cosines
        if mhi == m:
            cosPow, sinPow = mhi + SPIN, mhi - SPIN
            pm_p = pm_m = bool((mhi - SPIN) & 1)
        else:
            cosPow, sinPow = mhi + m, mhi - m
            pm_p, pm_m = False, bool((mhi + m) & 1)

        cth2 = np.maximum(np.sqrt((1.0 + cth) * 0.5), 1e-15)
        sth2 = np.maximum(np.sqrt((1.0 - cth) * 0.5), 1e-15)
        ccp, ccps = _mypow_scaled(cth2, cosPow)
        ssp, ssps = _mypow_scaled(sth2, sinPow)
        csp, csps = _mypow_scaled(cth2, sinPow)
        scp, scps = _mypow_scaled(sth2, cosPow)

        def norm_half(v, sc):
            big = np.abs(v) > FBIGHALF
            while big.any():
                v[big] *= FSMALL
                sc[big] += 1
                big = np.abs(v) > FBIGHALF
            sm = (np.abs(v) < FBIGHALF * FSMALL) & (v != 0)
            while sm.any():
                v[sm] *= FBIG
                sc[sm] -= 1
                sm = (np.abs(v) < FBIGHALF * FSMALL) & (v != 0)
            return v, sc

        l2p = pre[m] * ccp
        scp_ = psc[m] + ccps
        l2m = pre[m] * csp
        scm_ = psc[m] + csps
        l2p, scp_ = norm_half(l2p, scp_)
        l2m, scm_ = norm_half(l2m, scm_)
        l2p *= ssp
        scp_ = scp_ + ssps
        l2m *= scp
        scm_ = scm_ + scps
        if pm_p:
            l2p = -l2p
        if pm_m:
            l2m = -l2m

        def norm_ftol(v, sc):
            big = np.abs(v) > FTOL
            while big.any():
                v[big] *= FSMALL
                sc[big] += 1
                big = np.abs(v) > FTOL
            sm = (np.abs(v) < FTOL * FSMALL) & (v != 0)
            while sm.any():
                v[sm] *= FBIG
                sc[sm] -= 1
                sm = (np.abs(v) < FTOL * FSMALL) & (v != 0)
            return v, sc

        l2p, scp_ = norm_ftol(l2p, scp_)
        l2m, scm_ = norm_ftol(l2m, scm_)
        l1p = np.zeros_like(l2p)
        l1m = np.zeros_like(l2m)

        p1p = np.zeros(cth.shape, dtype=np.complex128)
        p1m = np.zeros_like(p1p)
        p2p = np.zeros_like(p1p)
        p2m = np.zeros_like(p1p)

        for l in range(mhi, lmax + 1):
            cfp = np.where(scp_ < 0, 0.0, np.where(scp_ > 0, FBIG, 1.0))
            cfm = np.where(scm_ < 0, 0.0, np.where(scm_ > 0, FBIG, 1.0))
            vp = l2p * cfp
            vm = l2m * cfm
            if ((l - mhi) & 1) == 0:
                p1p += aG[l] * vp
                p1m += aC[l] * vp
                p2p += 1j * aC[l] * vm
                p2m += -1j * aG[l] * vm
            else:
                p1p += -1j * aC[l] * vp
                p1m += 1j * aG[l] * vp
                p2p += aG[l] * vm
                p2m += aC[l] * vm
            newp = (cth * fx_a[l + 1] - fx_b[l + 1]) * l2p - l1p
            newm = (cth * fx_a[l + 1] + fx_b[l + 1]) * l2m - l1m
            l1p, l2p = l2p, newp
            l1m, l2m = l2m, newm
            need = (np.abs(l2p) > FTOL) & (scp_ < 0)
            l1p[need] *= FSMALL
            l2p[need] *= FSMALL
            scp_[need] += 1
            need = (np.abs(l2m) > FTOL) & (scm_ < 0)
            l1m[need] *= FSMALL
            l2m[need] *= FSMALL
            scm_[need] += 1

        fct = -1.0 if ((mhi - m + SPIN) & 1) else 1.0
        q1p = p1p + 1j * p2m
        q2p = p2p - 1j * p1m
        q1m = p1m - 1j * p2p
        q2m = p2m + 1j * p1p
        ph0N = q1p + q2p
        ph1N = q1m + q2m
        ph0S = fct * (q1p - q2p)
        ph1S = fct * (q1m - q2m)

        inorth = pairs.inorth[act]
        isouth = pairs.isouth[act]
        phase[0, inorth, m] = ph0N
        phase[1, inorth, m] = ph1N
        south = isouth != inorth
        phase[0, isouth[south], m] = ph0S[south]
        phase[1, isouth[south], m] = ph1S[south]

    Q = phase_to_map(phase[0], geom, mmax)
    U = phase_to_map(phase[1], geom, mmax)
    return np.stack([Q, U])
