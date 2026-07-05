"""Pure-NumPy reference implementation of spin-0 HEALPix synthesis.

This is an exact port of ducc0's spin-0 ``alm2map`` algorithm (the libsharp2
lineage) and serves two purposes:

1. executable documentation for the CUDA kernels in :mod:`almond.kernels` --
   every array and every step here has a one-to-one counterpart on the GPU;
2. the convention bridge in the test suite: it is validated against ducc0 to
   ~1e-14, and the GPU path is validated against *it* (and against ducc0).

Algorithm (see report/ for the derivation):

Stage 1 (Legendre).  For each m, the sums over l are evaluated with a
recursion in ``x = cos^2(theta)`` that only visits every *other* l.  The
odd-offset terms are folded into the even-offset ones beforehand using the
three-term recurrence of the normalized associated Legendre functions
``lambda_lm``:

    cth * lam_l = eps_{l+1} lam_{l+1} + eps_l lam_{l-1},
    eps_l = sqrt((l^2 - m^2) / (4 l^2 - 1))

With ``A_il = alpha_il (eps_{l+1} a_lm + eps_{l+2} a_{l+2,m})``,
``B_il = alpha_il a_{l+1,m}`` (l = m + 2 il), and the scaled functions
``mu_il = lam_{m+2il} / alpha_il`` satisfying

    mu_{il+1} = (a_il x + b_il) mu_il + mu_{il-1}

one gets the symmetric/antisymmetric parts

    p1 = sum_il mu_il A_il,   p2 = sum_il mu_il B_il,
    F_m(north) = p1 + cth p2,   F_m(south) = p1 - cth p2.

``lam_mm ~ sin^m(theta)`` underflows near the poles, so the recursion carries
a per-ring integer ``scale``: the true function is ``lam * 2^(800*scale)``.
While ``scale < 0`` contributions are exactly zero at double precision and are
skipped; rings with ``m > mlim(theta)`` are skipped entirely.

Stage 2 (FFT).  ``f_ring(phi_j) = sum_{m=-mmax}^{mmax} F_m e^{i m phi_j}``
with ``F_{-m} = conj(F_m)`` and ``phi_j = phi0 + 2 pi j / nphi``.  Because
``nphi < 2 mmax + 1`` on every HEALPix ring at ``lmax = 3 nside - 1``, each
``F_m`` is folded (aliased) onto FFT bin ``m mod nphi`` with twiddle
``e^{i m phi0}``, the conjugate onto bin ``-m mod nphi``; a length-``nphi``
inverse DFT then yields the ring.  ``Im F_0`` is discarded (ducc convention).
"""

from __future__ import annotations

import numpy as np

from .geometry import RingGeometry, PairGeometry, pair_geometry, ring_geometry

FBIG = 2.0 ** 800
FSMALL = 2.0 ** -800
FBIGHALF = 2.0 ** 400
FTOL = 2.0 ** -60


# ----------------------------------------------------------------------------
# Coefficient tables (host side; the GPU plan builds the same tables)
# ----------------------------------------------------------------------------

def mfac_table(mmax: int) -> np.ndarray:
    """lam_mm(theta) = mfac[m] * sin^m(theta);  mfac[m] = sqrt((2m+1)!!/(4 pi (2m)!!))."""
    m = np.arange(1, mmax + 1)
    return np.concatenate([[1.0], np.cumprod(np.sqrt((2 * m + 1.0) / (2 * m)))]) / np.sqrt(4 * np.pi)


def eps_table(m: int, lmax: int) -> np.ndarray:
    """eps_l for l = 0..lmax+3 (entries below l=m+1 unused; eps_m set to 0)."""
    l = np.arange(lmax + 4, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        e = np.sqrt(np.maximum(l * l - m * m, 0.0) / (4.0 * l * l - 1.0))
    e[: m + 1] = 0.0
    e[m + 1:] = np.sqrt(
        ((l[m + 1:] + m) * (l[m + 1:] - m)) / ((2 * l[m + 1:] + 1) * (2 * l[m + 1:] - 1))
    )
    return e


def recursion_coeffs(m: int, lmax: int):
    """alpha_il and (a_il, b_il) for l = m, m+2, ... (ducc0 Ylmgen::prepare, s=0).

    Returns (alpha, coef_a, coef_b) with ``nil = number of il steps``
    (enough to cover l = m .. lmax+1).
    """
    eps = eps_table(m, lmax)
    nil = (lmax + 1 - m) // 2 + 2  # matches ducc's lmax/2+2 sizing, per m
    alpha = np.zeros(nil)
    coef_a = np.zeros(nil)
    coef_b = np.zeros(nil)

    alpha[0] = 1.0 / eps[m + 1]
    if nil > 1:
        alpha[1] = eps[m + 1] / (eps[m + 2] * eps[m + 3])
    il = 1
    for l in range(m + 2, lmax + 1, 2):
        if il + 1 < nil:
            sgn = -1.0 if (il & 1) else 1.0
            alpha[il + 1] = sgn / (eps[l + 2] * eps[l + 3] * alpha[il])
        il += 1
    il = 0
    for l in range(m, lmax + 2, 2):
        if il < nil:
            sgn = -1.0 if (il & 1) else 1.0
            coef_a[il] = sgn * alpha[il] * alpha[il]
            coef_b[il] = -coef_a[il] * (eps[l + 2] ** 2 + eps[l + 1] ** 2)
        il += 1
    return alpha, coef_a, coef_b


def prefold_alm_m(alm_m: np.ndarray, m: int, lmax: int, alpha: np.ndarray):
    """Fold odd-l coefficients into the even-l stream (ducc's inner_loop_a2m).

    ``alm_m``: complex (lmax+1-m,) values a_lm for l = m..lmax.
    Returns (A, B): complex arrays over il (l = m + 2 il).
    """
    eps = eps_table(m, lmax)
    a = np.zeros(lmax + 3 - m, dtype=np.complex128)
    a[: lmax + 1 - m] = alm_m
    nil = (lmax - m) // 2 + 1
    A = np.empty(nil, dtype=np.complex128)
    B = np.empty(nil, dtype=np.complex128)
    for il in range(nil):
        l = m + 2 * il
        A[il] = alpha[il] * (eps[l + 1] * a[l - m] + eps[l + 2] * a[l + 2 - m])
        B[il] = alpha[il] * a[l + 1 - m]
    return A, B


# ----------------------------------------------------------------------------
# Stage 1: Legendre recursion -> phase(ring, m) = F_m(theta_ring)
# ----------------------------------------------------------------------------

def _mypow_scaled(x: np.ndarray, n: int):
    """x^n by repeated squaring with 2^(800*scale) scale tracking (vectorized)."""
    val = x.copy()
    vscale = np.zeros(x.shape, dtype=np.int64)
    res = np.ones_like(x)
    rscale = np.zeros(x.shape, dtype=np.int64)

    def normalize(v, s):
        mask = np.abs(v) > FBIGHALF
        while mask.any():
            v[mask] *= FSMALL
            s[mask] += 1
            mask = np.abs(v) > FBIGHALF
        mask = (np.abs(v) < FBIGHALF * FSMALL) & (v != 0)
        while mask.any():
            v[mask] *= FBIG
            s[mask] -= 1
            mask = (np.abs(v) < FBIGHALF * FSMALL) & (v != 0)

    normalize(val, vscale)
    while n:
        if n & 1:
            res *= val
            rscale += vscale
            normalize(res, rscale)
        val *= val
        vscale += vscale
        normalize(val, vscale)
        n >>= 1
    return res, rscale


def legendre_phase(alm: np.ndarray, geom: RingGeometry, pairs: PairGeometry,
                   lmax: int, mmax: int | None = None) -> np.ndarray:
    """Compute phase[ring, m] = F_m(theta_ring) for all rings and m."""
    if mmax is None:
        mmax = lmax
    mfac = mfac_table(mmax)
    nring = geom.nring
    phase = np.zeros((nring, mmax + 1), dtype=np.complex128)

    mstart = (np.arange(mmax + 1) * (2 * lmax + 1 - np.arange(mmax + 1)) // 2)

    for m in range(mmax + 1):
        act = pairs.mlim >= m
        if not act.any():
            continue
        csq = pairs.csq[act]
        sth = pairs.sth[act]
        cth = pairs.cth[act]

        alpha, ca, cb = recursion_coeffs(m, lmax)
        alm_m = alm[mstart[m] + np.arange(m, lmax + 1)]
        A, B = prefold_alm_m(alm_m, m, lmax, alpha)
        nil = A.size

        # init: lam2 = (-1)^m mfac[m] sin^m theta (scaled), lam1 = 0
        lam2, scale = _mypow_scaled(sth, m)
        lam2 *= mfac[m] if (m % 2 == 0) else -mfac[m]
        # normalize into (FTOL*FSMALL, FTOL]
        big = np.abs(lam2) > FTOL
        while big.any():
            lam2[big] *= FSMALL
            scale[big] += 1
            big = np.abs(lam2) > FTOL
        small = (np.abs(lam2) < FTOL * FSMALL) & (lam2 != 0)
        while small.any():
            lam2[small] *= FBIG
            scale[small] -= 1
            small = (np.abs(lam2) < FTOL * FSMALL) & (lam2 != 0)
        lam1 = np.zeros_like(lam2)

        p1 = np.zeros(lam2.shape, dtype=np.complex128)
        p2 = np.zeros(lam2.shape, dtype=np.complex128)

        for il in range(nil):
            # corfac: 0 below IEEE range, 1 at scale 0, 2^800 at scale 1
            corfac = np.where(scale < 0, 0.0, np.where(scale > 0, FBIG, 1.0))
            lam = lam2 * corfac
            p1 += lam * A[il]
            p2 += lam * B[il]
            new_lam1 = (ca[il] * csq + cb[il]) * lam2 + lam1
            lam1 = lam2
            lam2 = new_lam1
            # rescale while below IEEE comfort zone
            need = (np.abs(lam2) > FTOL) & (scale < 0)
            lam1[need] *= FSMALL
            lam2[need] *= FSMALL
            scale[need] += 1
            # once scale reaches 0, unscale and run plain (mimic by corfac=1)
            emerged = scale == 0
            # nothing to do: corfac handles it uniformly (slower than ducc,
            # but bit-identical in exact arithmetic and this is a reference)
            del emerged, need

        t2 = cth * p2
        n_idx = geom.nring - 1 - pairs.isouth[act]  # == inorth
        phase[pairs.inorth[act], m] = p1 + t2
        south = pairs.isouth[act] != pairs.inorth[act]
        phase[pairs.isouth[act][south], m] = (p1 - t2)[south]
    return phase


# ----------------------------------------------------------------------------
# Stage 2: fold + per-ring inverse DFT -> map
# ----------------------------------------------------------------------------

def phase_to_map(phase: np.ndarray, geom: RingGeometry, mmax: int) -> np.ndarray:
    """f_j = sum_m F_m e^{i m (phi0 + 2 pi j / nphi)}, folded per ring."""
    out = np.empty(geom.npix, dtype=np.float64)
    for r in range(geom.nring):
        n = int(geom.nphi[r])
        phi0 = geom.phi0[r]
        G = np.zeros(n, dtype=np.complex128)
        F = phase[r]
        G[0] = F[0].real  # ducc discards Im F_0
        m = np.arange(1, mmax + 1)
        t = F[1:] * np.exp(1j * m * phi0)
        np.add.at(G, m % n, t)
        np.add.at(G, (-m) % n, np.conj(t))
        ring = np.fft.ifft(G) * n
        out[geom.ringstart[r]: geom.ringstart[r] + n] = ring.real
    return out


# ----------------------------------------------------------------------------
# Adjoint synthesis (map -> alm), ducc0 conventions
# ----------------------------------------------------------------------------
#
# ducc's adjoint is the transpose of synthesis under the alm inner product
# that counts m>0 modes twice (the reality-extended sum over +-m):
#     <map, synthesis(a)>_pix = a'_l0 . a_l0 + 2 Re sum_{m>0} a'_lm a*_lm
# which works out to the simple prescription (no factors of 2 anywhere):
#     Ghat_k(ring)  = sum_j f_j e^{-2 pi i j k / n}          (forward DFT)
#     F'_m(ring)    = e^{-i m phi0} Ghat_{m mod n}           (unfold; Im F'_0 -> 0)
#     a'_lm         = sum_rings lambda_lm(theta) F'_m(theta) (Legendre adjoint)
# The Legendre adjoint reuses the mu recursion: with p1' = F'_N + F'_S and
# p2' = cth (F'_N - F'_S) per ring pair,
#     A'_i = sum_pairs mu_i p1',   B'_i = sum_pairs mu_i p2',
#     a'_{l}   += alpha_i eps_{l+1} A'_i  (+ alpha_{i-1} eps_l A'_{i-1}),  l-m even
#     a'_{l}    = alpha_i B'_i,                                            l-m odd.


def map_to_phase(maps: np.ndarray, geom: RingGeometry, mmax: int) -> np.ndarray:
    """Adjoint of :func:`phase_to_map` in the ducc convention."""
    phase = np.zeros((geom.nring, mmax + 1), dtype=np.complex128)
    for r in range(geom.nring):
        n = int(geom.nphi[r])
        f = maps[geom.ringstart[r]: geom.ringstart[r] + n]
        Ghat = np.fft.fft(f)                     # sum_j f_j e^{-2 pi i jk/n}
        m = np.arange(mmax + 1)
        vals = Ghat[m % n] * np.exp(-1j * m * geom.phi0[r])
        vals[0] = vals[0].real
        phase[r] = vals
    return phase


def adjoint_legendre(phase: np.ndarray, geom: RingGeometry, pairs: PairGeometry,
                     lmax: int, mmax: int | None = None) -> np.ndarray:
    """a'_lm = sum_rings lambda_lm F'_m, via the scaled mu recursion."""
    if mmax is None:
        mmax = lmax
    mfac = mfac_table(mmax)
    nalm = (lmax + 1) * (lmax + 2) // 2
    alm = np.zeros(nalm, dtype=np.complex128)
    mstart = (np.arange(mmax + 1) * (2 * lmax + 1 - np.arange(mmax + 1)) // 2)

    for m in range(mmax + 1):
        act = pairs.mlim >= m
        if not act.any():
            continue
        csq = pairs.csq[act]
        sth = pairs.sth[act]
        cth = pairs.cth[act]
        Fn = phase[pairs.inorth[act], m]
        Fs = phase[pairs.isouth[act], m]
        # the self-paired equator ring must not be double counted
        selfp = pairs.isouth[act] == pairs.inorth[act]
        Fs = np.where(selfp, 0.0, Fs)
        p1 = Fn + Fs
        p2 = cth * (Fn - Fs)

        alpha, ca, cb = recursion_coeffs(m, lmax)
        eps = eps_table(m, lmax)
        nil = (lmax - m) // 2 + 1

        lam2, scale = _mypow_scaled(sth, m)
        lam2 *= mfac[m] if (m % 2 == 0) else -mfac[m]
        big = np.abs(lam2) > FTOL
        while big.any():
            lam2[big] *= FSMALL
            scale[big] += 1
            big = np.abs(lam2) > FTOL
        small = (np.abs(lam2) < FTOL * FSMALL) & (lam2 != 0)
        while small.any():
            lam2[small] *= FBIG
            scale[small] -= 1
            small = (np.abs(lam2) < FTOL * FSMALL) & (lam2 != 0)
        lam1 = np.zeros_like(lam2)

        Ap = np.zeros(nil, dtype=np.complex128)
        Bp = np.zeros(nil, dtype=np.complex128)
        for il in range(nil):
            corfac = np.where(scale < 0, 0.0, np.where(scale > 0, FBIG, 1.0))
            lam = lam2 * corfac
            Ap[il] = np.sum(lam * p1)
            Bp[il] = np.sum(lam * p2)
            new = (ca[il] * csq + cb[il]) * lam2 + lam1
            lam1 = lam2
            lam2 = new
            need = (np.abs(lam2) > FTOL) & (scale < 0)
            lam1[need] *= FSMALL
            lam2[need] *= FSMALL
            scale[need] += 1

        # postfold gather
        for l in range(m, lmax + 1):
            if (l - m) % 2 == 0:
                il = (l - m) // 2
                v = alpha[il] * eps[l + 1] * Ap[il]
                if il >= 1:
                    v += alpha[il - 1] * eps[l] * Ap[il - 1]
            else:
                il = (l - m - 1) // 2
                v = alpha[il] * Bp[il]
            alm[mstart[m] + l] = v
    return alm


def adjoint_synthesis(maps: np.ndarray, nside: int, lmax: int) -> np.ndarray:
    """Spin-0 adjoint synthesis: RING map -> healpy triangular alm."""
    maps = np.asarray(maps, dtype=np.float64).ravel()
    geom = ring_geometry(nside)
    pairs = pair_geometry(geom, lmax, spin=0)
    phase = map_to_phase(maps, geom, lmax)
    return adjoint_legendre(phase, geom, pairs, lmax)


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

def synthesis(alm: np.ndarray, nside: int, lmax: int) -> np.ndarray:
    """Spin-0 synthesis: healpy triangular alm -> HEALPix RING map (npix,)."""
    alm = np.asarray(alm, dtype=np.complex128).ravel()
    geom = ring_geometry(nside)
    pairs = pair_geometry(geom, lmax, spin=0)
    phase = legendre_phase(alm, geom, pairs, lmax)
    return phase_to_map(phase, geom, lmax)
