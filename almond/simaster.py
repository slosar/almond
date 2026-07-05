"""Drop-in GPU replacement for SiMaster's ``RealSHT`` (see simaster/sht.py).

``AlmondRealSHT`` exposes the same interface — ``synth(a)`` / ``adjoint(m)`` on
real-basis coefficient vectors ``(ncol, B)`` and observed-pixel maps
``(nrow, B)`` — but runs on the GPU through :class:`almond.SynthesisPlan`.
Accepts and returns numpy arrays (drop-in for the ``pure_callback`` path);
pass cupy arrays to stay on the device.

Columns are processed through the plan's grid-batched pipelines
(``synthesis_device_batch`` / ``adjoint_device_batch``), chunked by the
device-memory budget, and the real-basis <-> healpy conversion is a pure
(deterministic) gather: each healpy (l, m>=0) entry receives exactly one
+m and at most one -m real mode, so the inverse map is precomputed once.
"""

from __future__ import annotations

import numpy as np

from .plan import SynthesisPlan


class AlmondRealSHT:
    """Real-basis synthesis Y and its exact transpose, GPU-accelerated.

    Parameters mirror ``simaster.sht.RealSHT``: ``index`` is a
    ``simaster.utils.RealAlmIndex`` (provides l, m arrays and lmax), ``spin``
    is 0 or 2, ``obs_pix`` the observed-pixel subset of the RING map.
    """

    def __init__(self, nside: int, index, spin: int, obs_pix):
        import cupy as cp

        self.cp = cp
        self.nside = int(nside)
        self.index = index
        self.spin = int(spin)
        self.lmax = int(index.lmax)
        self.plan = SynthesisPlan(nside, self.lmax, spin=self.spin)
        self.npix = self.plan.npix
        self.nalm = self.plan.nalm
        self.obs_pix = np.asarray(obs_pix)
        self.nobs = self.obs_pix.size
        self.ncomp = 1 if self.spin == 0 else 2
        self.K = index.nmodes
        self.ncol = self.ncomp * self.K
        self.nrow = self.ncomp * self.nobs
        self.d_obs = cp.asarray(self.obs_pix.astype(np.int64))

        # real mode k -> healpy index and complex weight (simaster convention)
        l = np.asarray(index.l)
        m = np.asarray(index.m)
        mu = np.abs(m)
        sgn = (-1.0) ** mu
        idx_h = (mu * (2 * self.lmax + 1 - mu) // 2 + l).astype(np.int64)
        val = np.where(m == 0, 1.0 + 0j,
                       np.where(m > 0, sgn / np.sqrt(2.0) + 0j,
                                1j * sgn / np.sqrt(2.0)))
        self.d_idx = cp.asarray(idx_h)
        self.d_val = cp.asarray(val)
        self.d_cfac = cp.asarray(np.where(m == 0, 1.0, 2.0))

        # inverse map: healpy triangle index -> contributing real modes.
        # m >= 0 modes and m < 0 modes each hit a given healpy index at most
        # once, so real->healpy is a pure two-term gather (no scatter-add).
        inv_pos = np.zeros(self.nalm, dtype=np.int64)
        w_pos = np.zeros(self.nalm, dtype=np.complex128)
        inv_neg = np.zeros(self.nalm, dtype=np.int64)
        w_neg = np.zeros(self.nalm, dtype=np.complex128)
        pos = m >= 0
        inv_pos[idx_h[pos]] = np.nonzero(pos)[0]
        w_pos[idx_h[pos]] = val[pos]
        neg = ~pos
        inv_neg[idx_h[neg]] = np.nonzero(neg)[0]
        w_neg[idx_h[neg]] = val[neg]
        self.d_inv_pos = cp.asarray(inv_pos)
        self.d_w_pos = cp.asarray(w_pos)
        self.d_inv_neg = cp.asarray(inv_neg)
        self.d_w_neg = cp.asarray(w_neg)

    # ---- real basis <-> healpy triangle, batched (device) ------------------

    def _real_to_healpy_b(self, a):
        """a (nb, ncomp, K) real -> healpy alm (nb, ncomp, nalm) complex."""
        return (self.d_w_pos * a[:, :, self.d_inv_pos]
                + self.d_w_neg * a[:, :, self.d_inv_neg])

    def _healpy_to_real_b(self, alm):
        """healpy alm (nb, ncomp, nalm) -> real coefficients (nb, ncomp, K)."""
        cp = self.cp
        g = alm[:, :, self.d_idx]
        return self.d_cfac * (cp.conj(self.d_val) * g).real

    # ---- public API --------------------------------------------------------

    def synth(self, a):
        """a: (ncol, B) -> observed-pixel maps (nrow, B)."""
        cp = self.cp
        a_d = cp.asarray(a, dtype=cp.float64)
        B = a_d.shape[-1]
        aT = cp.ascontiguousarray(a_d.T).reshape(B, self.ncomp, self.K)
        out = cp.empty((self.nrow, B), dtype=cp.float64)
        C = max(1, self.plan._zbatch_cols(B, adjoint=False))
        for c0 in range(0, B, C):
            nb = min(C, B - c0)
            alm = self._real_to_healpy_b(aT[c0: c0 + nb])
            if self.spin == 0:
                mp = self.plan.synthesis_device_batch(alm[:, 0, :])
                out[:, c0: c0 + nb] = mp[:, self.d_obs].T
            else:
                mp = self.plan.synthesis_device_batch(alm)
                out[:, c0: c0 + nb] = \
                    mp[:, :, self.d_obs].reshape(nb, self.nrow).T
            del alm, mp
        return out.get() if isinstance(a, np.ndarray) else out

    def adjoint(self, maps):
        """maps: (nrow, B) -> coefficients (ncol, B); exact transpose."""
        cp = self.cp
        m_d = cp.asarray(maps, dtype=cp.float64)
        B = m_d.shape[-1]
        mT = cp.ascontiguousarray(m_d.T).reshape(B, self.ncomp, self.nobs)
        out = cp.empty((self.ncol, B), dtype=cp.float64)
        C = max(1, self.plan._zbatch_cols(B, adjoint=True))
        full = cp.zeros((C, self.ncomp, self.npix), dtype=cp.float64)
        for c0 in range(0, B, C):
            nb = min(C, B - c0)
            if c0:
                full[:nb] = 0.0
            full[:nb, :, self.d_obs] = mT[c0: c0 + nb]
            if self.spin == 0:
                alm = self.plan.adjoint_device_batch(full[:nb, 0, :])
                alm = alm[:, None, :]
            else:
                alm = self.plan.adjoint_device_batch(full[:nb])
            out[:, c0: c0 + nb] = \
                self._healpy_to_real_b(alm).reshape(nb, self.ncol).T
            del alm
        return out.get() if isinstance(maps, np.ndarray) else out
