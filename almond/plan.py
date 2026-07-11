"""GPU synthesis plan: precomputed tables + persistent buffers on the device.

Usage::

    import almond
    plan = almond.SynthesisPlan(nside=256, lmax=767)
    m = plan.synthesis(alm_coeffs)            # numpy in -> numpy out
    m_dev = plan.synthesis_device(alm_dev)    # cupy in -> cupy out (no copies)

The plan owns everything that depends only on (nside, lmax): ring/pair
geometry, the recursion coefficient table, and the work buffers, so repeated
transforms allocate nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import get_mlim, pair_geometry, ring_geometry

_KERNEL_SOURCE = (Path(__file__).with_name("kernels.cu").read_text()
                  + "\n" + Path(__file__).with_name("kernels_spin.cu").read_text())

_module_cache = {}


def _get_module(chunk: int = 1):
    import cupy as cp

    key = (cp.cuda.Device().id, int(chunk))
    if key not in _module_cache:
        _module_cache[key] = cp.RawModule(
            code=_KERNEL_SOURCE,
            options=("--std=c++17", f"-DCHUNK={int(chunk)}"),
            name_expressions=None,
        )
    return _module_cache[key]


class SynthesisPlan:
    """Spin-0 HEALPix synthesis (alm -> RING map) on the GPU, float64."""

    def __init__(self, nside: int, lmax: int, chunk: int | None = None,
                 spin: int = 0):
        import cupy as cp

        self.cp = cp
        self.nside = int(nside)
        self.lmax = int(lmax)
        self.mmax = self.lmax
        lmax, mmax = self.lmax, self.mmax
        # batch chunk: columns per legendre_batch launch in
        # synthesis_device_batch. Measured on A100: chunk=1 (loop the
        # single-transform pipeline, whose legendre2 kernel already has the
        # ILP of two recursion chains) beats the chunked kernel, which pays
        # 4*chunk accumulator registers per thread (150 regs at chunk=8) and
        # double the relative L1 load pressure. Chunked mode is kept for
        # experimentation (chunk>1).
        self.chunk = int(chunk) if chunk else 1

        if spin not in (0, 2):
            raise ValueError("spin must be 0 or 2")
        self.spin = int(spin)
        self.ncomp = 1 if spin == 0 else 2
        geom = ring_geometry(nside)
        pairs = pair_geometry(geom, lmax, spin=self.spin)
        self.geom, self.pairs = geom, pairs
        self.npix = geom.npix
        self.nring = geom.nring
        self.npair = pairs.npair
        self.nalm = (lmax + 1) * (lmax + 2) // 2

        mod = _get_module(self.chunk)
        self._k_build = mod.get_function("build_coef")
        self._k_prefold = mod.get_function("prefold")
        self._k_legendre = mod.get_function("legendre")
        self._k_legendre2 = mod.get_function("legendre2")
        self._k_legendre_b = mod.get_function("legendre_batch")
        self._k_fold = mod.get_function("fold")
        self._k_capdft = mod.get_function("cap_dft")
        self._k_beltfin = mod.get_function("belt_finish")
        self._k_beltfin_b = mod.get_function("belt_finish_b")
        self._k_blu_b = mod.get_function("bluestein_b")
        self._k_blu_pre = mod.get_function("bluestein_pre")
        self._k_blu_post = mod.get_function("bluestein_post")
        self._k_capdft_adj = mod.get_function("cap_dft_adj")
        self._k_blu_pre_adj = mod.get_function("bluestein_pre_adj")
        self._k_blu_post_adj = mod.get_function("bluestein_post_adj")
        self._k_unfold = mod.get_function("unfold")
        self._k_legendre_adj = mod.get_function("legendre_adj")
        self._k_legendre_adj2 = mod.get_function("legendre_adj2")
        self._k_postfold = mod.get_function("postfold")
        # adjoint Legendre implementation: 'v2' (staged bulk reduction,
        # default) or 'v1' (per-il warp shuffles); env override for A/B runs
        import os as _os
        self.adj_impl = _os.environ.get("ALMOND_ADJ_IMPL", "v2")

        # ---- static tables -------------------------------------------------
        m = np.arange(mmax + 1)
        nil_tab = (lmax - m) // 2 + 2
        moff = np.concatenate([[0], np.cumsum(nil_tab)]).astype(np.int64)
        self.ncoef = int(moff[-1])
        self.d_moff = cp.asarray(moff)
        self.d_mstart = cp.asarray((m * (2 * lmax + 1 - m) // 2).astype(np.int64))

        mm = np.arange(1, mmax + 1)
        mfac = np.concatenate([[1.0], np.cumprod(np.sqrt((2 * mm + 1.0) / (2 * mm)))])
        mfac /= np.sqrt(4.0 * np.pi)
        self.d_mfac = cp.asarray(mfac)
        with np.errstate(divide="ignore"):
            powlimit = np.exp(-400.0 * np.log(2.0) / np.maximum(m, 1))
        powlimit[0] = 0.0
        self.d_powlimit = cp.asarray(powlimit)

        # pair geometry
        self.d_csq = cp.asarray(pairs.csq)
        self.d_cth = cp.asarray(pairs.cth)
        self.d_sth = cp.asarray(pairs.sth)
        self.d_mlim = cp.asarray(pairs.mlim.astype(np.int32))
        self.d_inorth = cp.asarray(pairs.inorth.astype(np.int32))
        self.d_isouth = cp.asarray(pairs.isouth.astype(np.int32))

        # ring geometry (for the FFT stage)
        self.d_ringstart = cp.asarray(geom.ringstart)
        self.d_nphi = cp.asarray(geom.nphi.astype(np.int32))
        ring_mlim = np.empty(geom.nring, dtype=np.int32)
        ring_mlim[pairs.inorth] = pairs.mlim
        ring_mlim[pairs.isouth] = pairs.mlim
        self.d_ring_mlim = cp.asarray(ring_mlim)

        # phi0 = pi * num / den (exact integer reduction for sincospi)
        i = np.arange(1, geom.nring + 1)
        northcap = i < nside
        southcap = i > 3 * nside
        ir = np.where(southcap, 4 * nside - i, i)
        num = np.where(northcap | southcap, 1, ((i - nside + 1) % 2)).astype(np.int32)
        den = np.where(northcap | southcap, 4 * ir, 4 * nside).astype(np.int32)
        self.d_phi0num = cp.asarray(num)
        self.d_phi0den = cp.asarray(den)

        # cap rings (both hemispheres), and the belt slice.
        # Small cap rings (i < 64) go through the direct Hermitian DFT
        # kernel; larger ones through Bluestein classes (one batched cuFFT
        # per power-of-two size class; see kernels.cu).
        cap = np.where(northcap | southcap)[0].astype(np.int32)
        cap_ir = ir[cap]  # mirrored ring number 1..nside-1
        small = cap[cap_ir < min(64, nside)]
        self.ncap = small.size
        self.d_capring = cp.asarray(small)
        self.max_cap_nphi = int(geom.nphi[small].max()) if small.size else 0

        self.blu_classes = []
        j = 6
        while 2**j < nside:
            lo, hi = 2**j, min(2 ** (j + 1), nside)
            mem = cap[(cap_ir >= lo) & (cap_ir < hi)].astype(np.int32)
            if mem.size:
                M = 2 ** (j + 4)   # >= 2*(4*(hi-1)) > 2n-1 for all members
                self.blu_classes.append({
                    "M": M, "nmem": int(mem.size),
                    "ring": cp.asarray(mem),
                    "max_n": int(geom.nphi[mem].max()),
                })
            j += 1
        self.belt_rows = 2 * nside + 1
        self.belt_nphi = 4 * nside
        self.belt_start = int(geom.ringstart[nside - 1])  # first belt pixel

        # ---- work buffers --------------------------------------------------
        self.d_coef = cp.empty(self.ncoef, dtype=cp.complex128)  # (a,b) pairs
        self.d_AB = cp.empty(2 * self.ncoef, dtype=cp.complex128)
        self.d_phase = None  # fused fold/unfold: only the chunked batch path
        #                      materialises a phase array (see _batch_buffers)
        self.d_G = cp.empty(self.npix, dtype=cp.complex128)
        self.d_map = cp.empty(self.npix, dtype=cp.float64)

        # build the coefficient table once
        bs = 128
        self._k_build(((mmax + 1 + bs - 1) // bs,), (bs,),
                      (np.int32(lmax), np.int32(mmax), self.d_moff,
                       self.d_coef))

        # build the Bluestein chirp-kernel FFTs (persistent, per class)
        for c in self.blu_classes:
            B = cp.empty((c["nmem"], c["M"]), dtype=cp.complex128)
            self._k_blu_b(((c["M"] + bs - 1) // bs, c["nmem"]), (bs,),
                          (np.int32(c["nmem"]), np.int32(c["M"]),
                           np.float64(-1.0), c["ring"], self.d_nphi, B))
            c["Bhat"] = cp.fft.fft(B, axis=1)
            del B
        cp.cuda.runtime.deviceSynchronize()

        # ---- adjoint-specific geometry -------------------------------------
        # adjoint Legendre blockDim: cover all pairs in one base iteration
        # (blockDim*PPT >= npair) with the smallest block >= 64 threads --
        # oversized blocks burn FMA slots on inactive zero-chains at small
        # nside; blocks > 256 lose to barrier costs (measured).
        def _adj_bd(ppt):
            return int(min(256, max(64, 32 * ((self.npair + 32 * ppt - 1)
                                              // (32 * ppt)))))
        self._bd_adj0 = _adj_bd(4)   # ADJ2_PPT
        self._bd_adj2 = _adj_bd(2)   # SADJ2_PPT
        # half-spectrum Ghat offsets and the first active pair per m
        hlen = (geom.nphi // 2 + 1).astype(np.int64)
        hstart = np.concatenate([[0], np.cumsum(hlen)[:-1]]).astype(np.int64)
        self.nghalf = int(hlen.sum())
        self.d_hstart = cp.asarray(hstart)
        self.belt_hstart = int(hstart[nside - 1])
        self.belt_hlen = 4 * nside // 2 + 1
        self.d_pstart = cp.asarray(
            np.searchsorted(pairs.mlim, np.arange(mmax + 1)).astype(np.int32))

        # ---- spin-2 tables and buffers -------------------------------------
        if self.spin == 2:
            self._k_build_s = mod.get_function("build_coef_spin")
            self._k_prefold_s = mod.get_function("prefold_spin")
            self._k_leg_s = mod.get_function("legendre_spin")
            self._k_leg_s_2p = mod.get_function("legendre_spin_2p")
            self._k_leg_s_adj = mod.get_function("legendre_spin_adj")
            self._k_leg_s_adj2 = mod.get_function("legendre_spin_adj2")
            # spin-2 forward implementation: '2p' (two ring pairs per
            # thread, default) or 'v1'; env override for A/B runs
            import os as _os2
            self.spin_fwd_impl = _os2.environ.get("ALMOND_SPINFWD", "2p")
            self._k_postfold_s = mod.get_function("postfold_spin")
            nls = (lmax + 2 - m).astype(np.int64)          # l = m..lmax+1
            soff = np.concatenate([[0], np.cumsum(nls)]).astype(np.int64)
            self.nscoef = int(soff[-1])
            self.d_soff = cp.asarray(soff)
            # prefac with 2^800 scale (host, exact cumulative products)
            from .reference_spin import spin_prefac
            pre, psc = spin_prefac(mmax)
            self.d_sprefac = cp.asarray(pre)
            self.d_sprescale = cp.asarray(psc.astype(np.int32))
            self.d_fx = cp.empty(self.nscoef, dtype=cp.complex128)
            self.d_walpha = cp.empty(self.nscoef, dtype=cp.float64)
            self._k_build_s(((mmax + 1 + bs - 1) // bs,), (bs,),
                            (np.int32(lmax), np.int32(mmax), self.d_soff,
                             self.d_fx, self.d_walpha))
            self.d_GC = cp.empty(2 * self.nscoef, dtype=cp.complex128)
            # component-doubled G/map buffers (phase is fused away)
            self.d_G = cp.empty((2, self.npix), dtype=cp.complex128)
            self.d_map = cp.empty((2, self.npix), dtype=cp.float64)
            cp.cuda.runtime.deviceSynchronize()

    # ------------------------------------------------------------------ API

    def synthesis_device(self, alm, out=None, timings: dict | None = None):
        """alm: cupy complex128 (nalm,) healpy layout -> cupy float64 (npix,).

        If ``timings`` is a dict, per-stage wall times (s, device-synced) are
        stored into it (slow; for profiling only).
        """
        cp = self.cp
        if self.spin == 2:
            if alm.shape != (2, self.nalm):
                raise ValueError(f"spin-2 alm must be (2, {self.nalm})")
            return self._synthesis_spin2(cp.ascontiguousarray(alm), out)
        lmax, mmax = self.lmax, self.mmax
        nring, npair = self.nring, self.npair
        if alm.dtype != cp.complex128 or alm.shape != (self.nalm,):
            raise ValueError(f"alm must be complex128 ({self.nalm},)")
        if out is None:
            if getattr(self, "d_map", None) is None:
                self.d_map = cp.empty(self.npix, dtype=cp.float64)
            out = self.d_map

        if timings is not None:
            import time as _time

            def _mark(label, _last=[None]):
                cp.cuda.runtime.deviceSynchronize()
                now = _time.perf_counter()
                if _last[0] is not None:
                    timings[label] = now - _last[0][1]
                _last[0] = (label, now)
                return _mark
            _mark("start")
        else:
            def _mark(label):
                return None

        # stage 1a: prefold alm -> (A,B)
        bs = 128
        max_nil = (lmax - 0) // 2 + 2
        self._k_prefold(((max_nil + bs - 1) // bs, mmax + 1), (bs,),
                        (np.int32(lmax), np.int32(mmax), self.d_moff,
                         self.d_coef, self.d_mstart, np.int64(self.nalm),
                         np.int32(1), np.int64(1), np.int64(0),
                         alm, self.d_AB))
        _mark("prefold")

        # stage 1b: Legendre recursion with the fold fused into the extract:
        # F_m is scattered straight onto the ring FFT bins (no phase array)
        self.d_G.fill(0)
        nthread = (npair + 1) // 2
        self._k_legendre2(((nthread + bs - 1) // bs, mmax + 1), (bs,),
                          (np.int32(lmax), np.int32(mmax), np.int32(npair),
                           np.int32(nring), self.d_moff, self.d_coef, self.d_AB,
                           self.d_mfac, self.d_powlimit, self.d_csq, self.d_cth,
                           self.d_sth, self.d_mlim, self.d_inorth,
                           self.d_isouth, self.d_ringstart, self.d_nphi,
                           self.d_phi0num, self.d_phi0den,
                           np.int64(0), np.int64(0), self.d_G))
        _mark("legendre_fold")

        # stage 2b: belt = batched inverse FFT (cuFFT), caps = direct DFT.
        # In-place: the belt region of G is not read again (cap stages touch
        # disjoint rings), so the transform may overwrite it.
        import cupyx.scipy.fft as _cufft
        belt = self.d_G[self.belt_start:
                        self.belt_start + self.belt_rows * self.belt_nphi]
        belt = belt.reshape(self.belt_rows, self.belt_nphi)
        z = _cufft.ifft(belt, axis=1, overwrite_x=True)
        ntot = z.size
        self._k_beltfin(((ntot + 255) // 256,), (256,),
                        (np.int64(ntot), np.float64(self.belt_nphi), z,
                         out[self.belt_start:]))
        _mark("belt_fft")

        if self.ncap:
            gx = (self.max_cap_nphi + bs - 1) // bs
            self._k_capdft((gx, self.ncap, 1), (bs,),
                           (np.int32(self.ncap), np.int64(self.npix),
                            self.d_capring, self.d_ringstart, self.d_nphi,
                            self.d_G, out))
        _mark("cap_small_dft")

        import cupyx.scipy.fft as cufft
        for c in self.blu_classes:
            nmem, M = c["nmem"], c["M"]
            A = cp.empty((nmem, M), dtype=cp.complex128)
            self._k_blu_pre(((M + bs - 1) // bs, nmem), (bs,),
                            (np.int32(nmem), np.int32(M), np.int64(self.npix),
                             c["ring"], self.d_ringstart, self.d_nphi,
                             self.d_G, A))
            Ahat = cufft.fft(A, axis=1, overwrite_x=True)
            Ahat *= c["Bhat"]
            conv = cufft.ifft(Ahat, axis=1, overwrite_x=True)
            self._k_blu_post(((c["max_n"] + bs - 1) // bs, nmem), (bs,),
                             (np.int32(nmem), np.int32(M), np.int64(self.npix),
                              c["ring"], self.d_ringstart, self.d_nphi,
                              conv, out))
        _mark("cap_bluestein")
        return out

    def _batch_buffers(self):
        """Lazily allocate the chunked-batch work buffers.

        The single-transform phase/G/map buffers are re-pointed to the first
        chunk slice of the batch buffers (identical layouts), so batch mode
        does not double the footprint.
        """
        cp = self.cp
        if not hasattr(self, "d_AB_b"):
            C = self.chunk
            self.d_AB_b = cp.empty(2 * self.ncoef * C, dtype=cp.complex128)
            self.d_phase_b = cp.empty(C * (self.mmax + 1) * self.nring,
                                      dtype=cp.complex128)
            self.d_G_b = cp.empty(C * self.npix, dtype=cp.complex128)
            self.d_phase = self.d_phase_b[: (self.mmax + 1) * self.nring]
            self.d_G = self.d_G_b[: self.npix]
            self.d_map = None  # realloc lazily if the single path is used
            cp.get_default_memory_pool().free_all_blocks()

    def _remainder_buffers(self):
        cp = self.cp
        if not hasattr(self, "d_alm_b"):
            self.d_alm_b = cp.empty((self.chunk, self.nalm),
                                    dtype=cp.complex128)
            self.d_map_b = cp.empty((self.chunk, self.npix), dtype=cp.float64)

    # -------------------------------------------------- grid-z batched paths

    def _zbatch_cols(self, B, adjoint):
        """Columns per grid-z chunk, from the device-memory budget
        (env ALMOND_BATCH_MEM, default 4 GiB)."""
        import os
        budget = int(os.environ.get("ALMOND_BATCH_MEM", 4 << 30))
        beltpix = self.belt_rows * self.belt_nphi
        if self.spin == 0:
            spec = self.nghalf if adjoint else self.npix
            coefpl = 2 * self.ncoef
        else:
            spec = 2 * (self.nghalf if adjoint else self.npix)
            coefpl = 2 * self.nscoef
            beltpix *= 2
        # plane + coef plane + belt transient + Bluestein scratch (crude)
        percol = 16 * (spec + coefpl + beltpix + 4 * self.nside * self.nside)
        return max(1, min(B, budget // percol))

    def _zbuf(self, C, adjoint):
        cp = self.cp
        ncomp = self.ncomp
        if adjoint:
            if getattr(self, "d_Gh_z", None) is None or \
                    self.d_Gh_z.shape[0] < C * ncomp:
                self.d_Gh_z = cp.empty((C * ncomp, self.nghalf),
                                       dtype=cp.complex128)
        else:
            if getattr(self, "d_G_z", None) is None or \
                    self.d_G_z.shape[0] < C * ncomp:
                self.d_G_z = cp.empty((C * ncomp, self.npix),
                                      dtype=cp.complex128)
        ncoefpl = 2 * (self.ncoef if self.spin == 0 else self.nscoef)
        if getattr(self, "d_coef_z", None) is None or \
                self.d_coef_z.size < C * ncoefpl:
            self.d_coef_z = cp.empty(C * ncoefpl, dtype=cp.complex128)

    def _synth_zchunk(self, alm_c, out_c):
        """Grid-z batched synthesis of one chunk: alm_c (nb[,2],nalm) ->
        out_c (nb[,2],npix), all device-contiguous."""
        cp = self.cp
        import cupyx.scipy.fft as cufft

        lmax, mmax = self.lmax, self.mmax
        nring, npair, npix = self.nring, self.npair, self.npix
        nb = alm_c.shape[0]
        ncomp = self.ncomp
        bs = 128
        G = self.d_G_z[: nb * ncomp]
        G.fill(0)
        if self.spin == 0:
            max_nil = lmax // 2 + 2
            self._k_prefold(((max_nil + bs - 1) // bs, mmax + 1), (bs,),
                            (np.int32(lmax), np.int32(mmax), self.d_moff,
                             self.d_coef, self.d_mstart, np.int64(self.nalm),
                             np.int32(nb), np.int64(1), np.int64(self.ncoef),
                             alm_c, self.d_coef_z))
            nthread = (npair + 1) // 2
            self._k_legendre2(((nthread + bs - 1) // bs, mmax + 1, nb), (bs,),
                              (np.int32(lmax), np.int32(mmax), np.int32(npair),
                               np.int32(nring), self.d_moff, self.d_coef,
                               self.d_coef_z, self.d_mfac, self.d_powlimit,
                               self.d_csq, self.d_cth, self.d_sth, self.d_mlim,
                               self.d_inorth, self.d_isouth, self.d_ringstart,
                               self.d_nphi, self.d_phi0num, self.d_phi0den,
                               np.int64(2 * self.ncoef), np.int64(npix), G))
        else:
            self._k_prefold_s(((lmax + 2 + bs - 1) // bs, mmax + 1, nb), (bs,),
                              (np.int32(lmax), np.int32(mmax), self.d_soff,
                               self.d_walpha, self.d_mstart,
                               np.int64(self.nalm), np.int64(2 * self.nalm),
                               np.int64(2 * self.nscoef), alm_c,
                               self.d_coef_z))
            args_s = (np.int32(lmax), np.int32(mmax), np.int32(npair),
                      np.int32(nring), self.d_soff, self.d_fx,
                      self.d_coef_z, self.d_sprefac, self.d_sprescale,
                      self.d_cth, self.d_sth, self.d_mlim, self.d_inorth,
                      self.d_isouth, self.d_ringstart, self.d_nphi,
                      self.d_phi0num, self.d_phi0den, np.int64(npix),
                      np.int64(2 * self.nscoef), np.int64(2 * npix), G)
            if self.spin_fwd_impl == "2p":
                nthread = (npair + 1) // 2
                self._k_leg_s_2p(((nthread + bs - 1) // bs, mmax + 1, nb),
                                 (bs,), args_s)
            else:
                self._k_leg_s(((npair + bs - 1) // bs, mmax + 1, nb), (bs,),
                              args_s)

        nch = nb * ncomp
        beltpix = self.belt_rows * self.belt_nphi
        belt = cp.ascontiguousarray(
            G[:, self.belt_start: self.belt_start + beltpix])
        z = cufft.ifft(belt.reshape(nch * self.belt_rows, self.belt_nphi),
                       axis=1, overwrite_x=True)
        out_flat = out_c.reshape(nch, npix)
        self._k_beltfin_b((((nch * beltpix) + 255) // 256,), (256,),
                          (np.int64(beltpix), np.int64(npix),
                           np.int64(self.belt_start), np.int32(nch),
                           np.float64(self.belt_nphi), z, out_flat))
        del belt, z
        if self.ncap:
            gx = (self.max_cap_nphi + bs - 1) // bs
            self._k_capdft((gx, self.ncap, nch), (bs,),
                           (np.int32(self.ncap), np.int64(npix),
                            self.d_capring, self.d_ringstart, self.d_nphi,
                            G, out_flat))
        for cl in self.blu_classes:
            nmem, M = cl["nmem"], cl["M"]
            A = cp.empty((nch * nmem, M), dtype=cp.complex128)
            self._k_blu_pre(((M + bs - 1) // bs, nch * nmem), (bs,),
                            (np.int32(nmem), np.int32(M), np.int64(npix),
                             cl["ring"], self.d_ringstart, self.d_nphi, G, A))
            Ahat = cufft.fft(A, axis=1, overwrite_x=True)
            Ahat = Ahat.reshape(nch, nmem, M)
            Ahat *= cl["Bhat"][None, :, :]
            conv = cufft.ifft(Ahat.reshape(nch * nmem, M), axis=1,
                              overwrite_x=True)
            self._k_blu_post(((cl["max_n"] + bs - 1) // bs, nch * nmem), (bs,),
                             (np.int32(nmem), np.int32(M), np.int64(npix),
                              cl["ring"], self.d_ringstart, self.d_nphi,
                              conv, out_flat))
            del A, Ahat, conv

    def _adj_zchunk(self, maps_c, out_c):
        """Grid-z batched adjoint of one chunk: maps_c (nb[,2],npix) ->
        out_c (nb[,2],nalm), all device-contiguous."""
        cp = self.cp
        import cupyx.scipy.fft as cufft

        lmax, mmax = self.lmax, self.mmax
        nring, npair, npix = self.nring, self.npair, self.npix
        nb = maps_c.shape[0]
        ncomp = self.ncomp
        nch = nb * ncomp
        bs = 128
        Gh = self.d_Gh_z[:nch]
        maps_flat = maps_c.reshape(nch, npix)

        beltpix = self.belt_rows * self.belt_nphi
        belt = cp.ascontiguousarray(
            maps_flat[:, self.belt_start: self.belt_start + beltpix])
        gh = cufft.rfft(belt.reshape(nch * self.belt_rows, self.belt_nphi),
                        axis=1)
        Gh[:, self.belt_hstart:
           self.belt_hstart + self.belt_rows * self.belt_hlen] = \
            gh.reshape(nch, self.belt_rows * self.belt_hlen)
        del belt, gh
        if self.ncap:
            gx = (self.max_cap_nphi // 2 + 1 + bs - 1) // bs
            self._k_capdft_adj((gx, self.ncap, nch), (bs,),
                               (np.int32(self.ncap), np.int64(self.nghalf),
                                self.d_capring, self.d_ringstart,
                                self.d_hstart, self.d_nphi, maps_flat,
                                np.int64(npix), Gh))
        for c in self.blu_classes:
            nmem, M = c["nmem"], c["M"]
            A = cp.empty((nch * nmem, M), dtype=cp.complex128)
            self._k_blu_pre_adj(((M + bs - 1) // bs, nch * nmem), (bs,),
                                (np.int32(nmem), np.int32(M), np.int64(npix),
                                 c["ring"], self.d_ringstart, self.d_nphi,
                                 maps_flat, A))
            Ahat = cufft.fft(A, axis=1, overwrite_x=True)
            Ahat = Ahat.reshape(nch, nmem, M)
            Ahat *= c["Bhat_adj"][None, :, :]
            conv = cufft.ifft(Ahat.reshape(nch * nmem, M), axis=1,
                              overwrite_x=True)
            self._k_blu_post_adj(((c["max_n"] // 2 + 1 + bs - 1) // bs,
                                  nch * nmem), (bs,),
                                 (np.int32(nmem), np.int32(M),
                                  np.int64(self.nghalf), c["ring"],
                                  self.d_hstart, self.d_nphi, conv, Gh))
            del A, Ahat, conv

        TR = 8
        if self.spin == 0:
            bd = self._bd_adj0
            nw = bd // 32
            shmem = (nw * TR * 4 * 33 + 2 * nw * (TR * 4 + 1)) * 8
            AB = self.d_coef_z[: nb * 2 * self.ncoef]
            AB.fill(0)
            if shmem > getattr(self, "_adj2_shmem_set", 0):
                self._k_legendre_adj2.max_dynamic_shared_size_bytes = shmem
                self._adj2_shmem_set = shmem
            self._k_legendre_adj2(
                (mmax + 1, nb), (bd,),
                (np.int32(lmax), np.int32(mmax), np.int32(npair),
                 np.int32(nring), self.d_moff, self.d_coef, self.d_mfac,
                 self.d_powlimit, self.d_csq, self.d_cth, self.d_sth,
                 self.d_mlim, self.d_inorth, self.d_isouth, self.d_pstart,
                 Gh, self.d_hstart, self.d_nphi, self.d_phi0num,
                 self.d_phi0den, np.int64(self.nghalf),
                 np.int64(2 * self.ncoef), AB), shared_mem=shmem)
            max_dl = lmax + 1
            self._k_postfold(((max_dl + bs - 1) // bs, mmax + 1, nb), (bs,),
                             (np.int32(lmax), np.int32(mmax), self.d_moff,
                              self.d_coef, self.d_mstart,
                              np.int64(2 * self.ncoef), np.int64(self.nalm),
                              AB, out_c))
        else:
            bd = self._bd_adj2
            nw = bd // 32
            shmem = (nw * TR * 4 * 33 + 2 * nw * (TR * 4 + 1)) * 8
            GC = self.d_coef_z[: nb * 2 * self.nscoef]
            GC.fill(0)
            if shmem > getattr(self, "_sadj2_shmem_set", 0):
                self._k_leg_s_adj2.max_dynamic_shared_size_bytes = shmem
                self._sadj2_shmem_set = shmem
            self._k_leg_s_adj2(
                (mmax + 1, nb), (bd,),
                (np.int32(lmax), np.int32(mmax), np.int32(npair),
                 np.int32(nring), self.d_soff, self.d_fx, self.d_sprefac,
                 self.d_sprescale, self.d_cth, self.d_sth, self.d_mlim,
                 self.d_inorth, self.d_isouth, self.d_pstart, Gh,
                 self.d_hstart, self.d_nphi, self.d_phi0num, self.d_phi0den,
                 np.int64(self.nghalf), np.int64(2 * self.nghalf),
                 np.int64(2 * self.nscoef), GC), shared_mem=shmem)
            self._k_postfold_s(((lmax + 1 + bs - 1) // bs, mmax + 1, nb),
                               (bs,),
                               (np.int32(lmax), np.int32(mmax), self.d_soff,
                                self.d_walpha, self.d_mstart,
                                np.int64(self.nalm),
                                np.int64(2 * self.nscoef),
                                np.int64(2 * self.nalm), GC, out_c))

    def synthesis_device_batch(self, alm, out=None):
        """Batched synthesis: alm (B, nalm) -> maps (B, npix) for spin 0,
        alm (B, 2, nalm) -> maps (B, 2, npix) for spin 2.

        Columns are batched on the kernel grid (blockIdx.z), chunked by the
        device-memory budget; falls back to looping the single-transform
        pipeline when only one column fits (large nside).  ``chunk>1`` keeps
        the experimental per-thread chunked path (spin 0 only).
        """
        cp = self.cp
        import cupyx.scipy.fft as cufft

        lmax, mmax = self.lmax, self.mmax
        nring, npair, npix = self.nring, self.npair, self.npix
        C = self.chunk
        alm = cp.ascontiguousarray(alm)
        want = (self.nalm,) if self.spin == 0 else (2, self.nalm)
        if alm.shape[1:] != want:
            raise ValueError(f"alm must be (B, {want})")
        B = alm.shape[0]
        if out is None:
            out = cp.empty((B,) + ((npix,) if self.spin == 0
                                   else (2, npix)), dtype=cp.float64)
        if C == 1:
            Cz = self._zbatch_cols(B, adjoint=False)
            if Cz <= 1:
                for b in range(B):
                    self.synthesis_device(alm[b], out=out[b])
                return out
            self._zbuf(Cz, adjoint=False)
            for c0 in range(0, B, Cz):
                nb = min(Cz, B - c0)
                self._synth_zchunk(alm[c0: c0 + nb], out[c0: c0 + nb])
            return out
        self._batch_buffers()
        bs = 128
        max_nil = lmax // 2 + 2

        for c0 in range(0, B, C):
            nb = min(C, B - c0)
            if nb == C:
                alm_c = alm[c0: c0 + C]
                out_c = out[c0: c0 + C]
            else:                       # remainder: pad with zeros
                self._remainder_buffers()
                self.d_alm_b[:nb] = alm[c0: c0 + nb]
                self.d_alm_b[nb:] = 0
                alm_c = self.d_alm_b
                out_c = self.d_map_b

            self._k_prefold(((max_nil + bs - 1) // bs, mmax + 1), (bs,),
                            (np.int32(lmax), np.int32(mmax), self.d_moff,
                             self.d_coef, self.d_mstart, np.int64(self.nalm),
                             np.int32(C), np.int64(C), np.int64(1),
                             alm_c, self.d_AB_b))

            self.d_phase_b.fill(0)
            self._k_legendre_b(((npair + bs - 1) // bs, mmax + 1), (bs,),
                               (np.int32(lmax), np.int32(mmax),
                                np.int32(npair), np.int32(nring),
                                self.d_moff, self.d_coef, self.d_AB_b,
                                self.d_mfac, self.d_powlimit, self.d_csq,
                                self.d_cth, self.d_sth, self.d_mlim,
                                self.d_inorth, self.d_isouth, self.d_phase_b))

            self.d_G_b.fill(0)
            self._k_fold(((nring + bs - 1) // bs, mmax + 1), (bs,),
                         (np.int32(nring), np.int32(mmax), np.int32(C),
                          np.int64(npix), self.d_phase_b,
                          self.d_ringstart, self.d_nphi, self.d_phi0num,
                          self.d_phi0den, self.d_ring_mlim, self.d_G_b))

            # belt: one batched cuFFT over (C * belt_rows, belt_nphi)
            beltpix = self.belt_rows * self.belt_nphi
            Gb = self.d_G_b.reshape(C, npix)
            belt = Gb[:, self.belt_start: self.belt_start + beltpix]
            belt = cp.ascontiguousarray(belt).reshape(C * self.belt_rows,
                                                      self.belt_nphi)
            z = cufft.ifft(belt, axis=1, overwrite_x=True)
            tot = C * beltpix
            self._k_beltfin_b(((tot + 255) // 256,), (256,),
                              (np.int64(beltpix), np.int64(npix),
                               np.int64(self.belt_start), np.int32(C),
                               np.float64(self.belt_nphi), z, out_c))
            del belt, z   # free ~4 GB of transients before the Bluestein stage

            if self.ncap:
                gx = (self.max_cap_nphi + bs - 1) // bs
                self._k_capdft((gx, self.ncap, C), (bs,),
                               (np.int32(self.ncap), np.int64(npix),
                                self.d_capring, self.d_ringstart,
                                self.d_nphi, self.d_G_b, out_c))

            for cl in self.blu_classes:
                nmem, M = cl["nmem"], cl["M"]
                A = cp.empty((C * nmem, M), dtype=cp.complex128)
                self._k_blu_pre(((M + bs - 1) // bs, C * nmem), (bs,),
                                (np.int32(nmem), np.int32(M), np.int64(npix),
                                 cl["ring"], self.d_ringstart, self.d_nphi,
                                 self.d_G_b, A))
                Ahat = cufft.fft(A, axis=1, overwrite_x=True)
                Ahat = Ahat.reshape(C, nmem, M)
                Ahat *= cl["Bhat"][None, :, :]
                conv = cufft.ifft(Ahat.reshape(C * nmem, M), axis=1,
                                  overwrite_x=True)
                self._k_blu_post(((cl["max_n"] + bs - 1) // bs, C * nmem),
                                 (bs,),
                                 (np.int32(nmem), np.int32(M), np.int64(npix),
                                  cl["ring"], self.d_ringstart, self.d_nphi,
                                  conv, out_c))
                del A, Ahat, conv

            if nb < C:
                out[c0: c0 + nb] = self.d_map_b[:nb]
        return out

    # ------------------------------------------------------------- spin 2

    def _synthesis_spin2(self, alm, out=None):
        """alm (2, nalm) [E, B] -> maps (2, npix) [Q, U], device."""
        cp = self.cp
        import cupyx.scipy.fft as cufft

        lmax, mmax = self.lmax, self.mmax
        nring, npair, npix = self.nring, self.npair, self.npix
        out = self.d_map if out is None else out
        bs = 128

        self._k_prefold_s(((lmax + 2 + bs - 1) // bs, mmax + 1), (bs,),
                          (np.int32(lmax), np.int32(mmax), self.d_soff,
                           self.d_walpha, self.d_mstart, np.int64(self.nalm),
                           np.int64(0), np.int64(0), alm, self.d_GC))
        self.d_G.fill(0)
        args_s = (np.int32(lmax), np.int32(mmax), np.int32(npair),
                  np.int32(nring), self.d_soff, self.d_fx, self.d_GC,
                  self.d_sprefac, self.d_sprescale, self.d_cth,
                  self.d_sth, self.d_mlim, self.d_inorth, self.d_isouth,
                  self.d_ringstart, self.d_nphi, self.d_phi0num,
                  self.d_phi0den, np.int64(npix),
                  np.int64(0), np.int64(0), self.d_G)
        if self.spin_fwd_impl == "2p":
            nthread = (npair + 1) // 2
            self._k_leg_s_2p(((nthread + bs - 1) // bs, mmax + 1), (bs,),
                             args_s)
        else:
            self._k_leg_s(((npair + bs - 1) // bs, mmax + 1), (bs,), args_s)

        beltpix = self.belt_rows * self.belt_nphi
        belt = self.d_G.reshape(2, npix)[:, self.belt_start:
                                         self.belt_start + beltpix]
        belt = cp.ascontiguousarray(belt).reshape(2 * self.belt_rows,
                                                  self.belt_nphi)
        z = cufft.ifft(belt, axis=1, overwrite_x=True)
        self._k_beltfin_b(((2 * beltpix + 255) // 256,), (256,),
                          (np.int64(beltpix), np.int64(npix),
                           np.int64(self.belt_start), np.int32(2),
                           np.float64(self.belt_nphi), z, out))
        del belt, z
        if self.ncap:
            gx = (self.max_cap_nphi + bs - 1) // bs
            self._k_capdft((gx, self.ncap, 2), (bs,),
                           (np.int32(self.ncap), np.int64(npix),
                            self.d_capring, self.d_ringstart, self.d_nphi,
                            self.d_G, out))
        for cl in self.blu_classes:
            nmem, M = cl["nmem"], cl["M"]
            A = cp.empty((2 * nmem, M), dtype=cp.complex128)
            self._k_blu_pre(((M + bs - 1) // bs, 2 * nmem), (bs,),
                            (np.int32(nmem), np.int32(M), np.int64(npix),
                             cl["ring"], self.d_ringstart, self.d_nphi,
                             self.d_G, A))
            Ahat = cufft.fft(A, axis=1, overwrite_x=True)
            Ahat = Ahat.reshape(2, nmem, M)
            Ahat *= cl["Bhat"][None, :, :]
            conv = cufft.ifft(Ahat.reshape(2 * nmem, M), axis=1,
                              overwrite_x=True)
            self._k_blu_post(((cl["max_n"] + bs - 1) // bs, 2 * nmem), (bs,),
                             (np.int32(nmem), np.int32(M), np.int64(npix),
                              cl["ring"], self.d_ringstart, self.d_nphi,
                              conv, out))
            del A, Ahat, conv
        return out

    def _adjoint_spin2(self, maps, out=None):
        """maps (2, npix) [Q, U] -> alm (2, nalm) [E, B], device."""
        cp = self.cp
        import cupyx.scipy.fft as cufft

        lmax, mmax = self.lmax, self.mmax
        nring, npair, npix = self.nring, self.npair, self.npix
        bs = 128
        if not hasattr(self, "d_Gh2"):
            self.d_Gh2 = cp.empty((2, self.nghalf), dtype=cp.complex128)
            self.d_alm2_out = cp.empty((2, self.nalm), dtype=cp.complex128)
            for c in self.blu_classes:
                if "Bhat_adj" not in c:
                    B = cp.empty((c["nmem"], c["M"]), dtype=cp.complex128)
                    self._k_blu_b(((c["M"] + bs - 1) // bs, c["nmem"]), (bs,),
                                  (np.int32(c["nmem"]), np.int32(c["M"]),
                                   np.float64(1.0), c["ring"], self.d_nphi, B))
                    c["Bhat_adj"] = cp.fft.fft(B, axis=1)
                    del B
        out = self.d_alm2_out if out is None else out

        beltpix = self.belt_rows * self.belt_nphi
        belt = cp.ascontiguousarray(
            maps[:, self.belt_start: self.belt_start + beltpix])
        gh = cufft.rfft(belt.reshape(2 * self.belt_rows, self.belt_nphi),
                        axis=1)
        gh = gh.reshape(2, self.belt_rows * self.belt_hlen)
        self.d_Gh2[:, self.belt_hstart:
                   self.belt_hstart + self.belt_rows * self.belt_hlen] = gh
        del belt, gh
        if self.ncap:
            gx = (self.max_cap_nphi // 2 + 1 + bs - 1) // bs
            self._k_capdft_adj((gx, self.ncap, 2), (bs,),
                               (np.int32(self.ncap), np.int64(self.nghalf),
                                self.d_capring, self.d_ringstart,
                                self.d_hstart, self.d_nphi, maps,
                                np.int64(npix), self.d_Gh2))
        for c in self.blu_classes:
            nmem, M = c["nmem"], c["M"]
            A = cp.empty((2 * nmem, M), dtype=cp.complex128)
            self._k_blu_pre_adj(((M + bs - 1) // bs, 2 * nmem), (bs,),
                                (np.int32(nmem), np.int32(M), np.int64(npix),
                                 c["ring"], self.d_ringstart, self.d_nphi,
                                 maps, A))
            Ahat = cufft.fft(A, axis=1, overwrite_x=True)
            Ahat = Ahat.reshape(2, nmem, M)
            Ahat *= c["Bhat_adj"][None, :, :]
            conv = cufft.ifft(Ahat.reshape(2 * nmem, M), axis=1,
                              overwrite_x=True)
            self._k_blu_post_adj(((c["max_n"] // 2 + 1 + bs - 1) // bs,
                                  2 * nmem), (bs,),
                                 (np.int32(nmem), np.int32(M),
                                  np.int64(self.nghalf), c["ring"],
                                  self.d_hstart, self.d_nphi, conv,
                                  self.d_Gh2))
            del A, Ahat, conv

        self.d_GC.fill(0)
        args = (np.int32(lmax), np.int32(mmax), np.int32(npair),
                np.int32(nring), self.d_soff, self.d_fx,
                self.d_sprefac, self.d_sprescale, self.d_cth,
                self.d_sth, self.d_mlim, self.d_inorth,
                self.d_isouth, self.d_pstart, self.d_Gh2,
                self.d_hstart, self.d_nphi, self.d_phi0num,
                self.d_phi0den, np.int64(self.nghalf))
        if self.adj_impl == "v2":
            bd = self._bd_adj2
            nw, TR = bd // 32, 8
            shmem = (nw * TR * 4 * 33 + 2 * nw * (TR * 4 + 1)) * 8
            if shmem > getattr(self, "_sadj2_shmem_set", 0):
                self._k_leg_s_adj2.max_dynamic_shared_size_bytes = shmem
                self._sadj2_shmem_set = shmem
            self._k_leg_s_adj2((mmax + 1,), (bd,),
                               args + (np.int64(0), np.int64(0), self.d_GC),
                               shared_mem=shmem)
        else:
            shmem = 8 * 64 * 4 * 8   # nwarp * SADJ_TILE * 4 doubles
            self._k_leg_s_adj((mmax + 1,), (256,), args + (self.d_GC,),
                              shared_mem=shmem)
        self._k_postfold_s(((lmax + 1 + bs - 1) // bs, mmax + 1), (bs,),
                           (np.int32(lmax), np.int32(mmax), self.d_soff,
                            self.d_walpha, self.d_mstart, np.int64(self.nalm),
                            np.int64(0), np.int64(0), self.d_GC, out))
        return out

    # ------------------------------------------------------------- adjoint

    def _adjoint_buffers(self):
        cp = self.cp
        if not hasattr(self, "d_Gh"):
            self.d_Gh = cp.empty(self.nghalf, dtype=cp.complex128)
            self.d_alm_out = cp.empty(self.nalm, dtype=cp.complex128)
            bs = 128
            for c in self.blu_classes:
                B = cp.empty((c["nmem"], c["M"]), dtype=cp.complex128)
                self._k_blu_b(((c["M"] + bs - 1) // bs, c["nmem"]), (bs,),
                              (np.int32(c["nmem"]), np.int32(c["M"]),
                               np.float64(1.0), c["ring"], self.d_nphi, B))
                c["Bhat_adj"] = cp.fft.fft(B, axis=1)
                del B

    def adjoint_device(self, maps, out=None):
        """Adjoint synthesis: cupy float64 map (npix,) -> alm (nalm,) complex.

        Exact transpose of :meth:`synthesis_device` under ducc0's convention
        (matches ``ducc0.sht.experimental.adjoint_synthesis``).
        """
        cp = self.cp
        import cupyx.scipy.fft as cufft

        if self.spin == 2:
            if maps.shape != (2, self.npix):
                raise ValueError(f"spin-2 map must be (2, {self.npix})")
            return self._adjoint_spin2(cp.ascontiguousarray(maps), out)
        lmax, mmax = self.lmax, self.mmax
        nring, npair, npix = self.nring, self.npair, self.npix
        if maps.dtype != cp.float64 or maps.shape != (npix,):
            raise ValueError(f"map must be float64 ({npix},)")
        self._adjoint_buffers()
        out = self.d_alm_out if out is None else out
        bs = 128

        # stage 1: Ghat (half spectrum) per ring
        belt = maps[self.belt_start:
                    self.belt_start + self.belt_rows * self.belt_nphi]
        gh_belt = cufft.rfft(belt.reshape(self.belt_rows, self.belt_nphi),
                             axis=1)
        self.d_Gh[self.belt_hstart:
                  self.belt_hstart + self.belt_rows * self.belt_hlen] = \
            gh_belt.ravel()
        del gh_belt

        if self.ncap:
            gx = (self.max_cap_nphi // 2 + 1 + bs - 1) // bs
            self._k_capdft_adj((gx, self.ncap, 1), (bs,),
                               (np.int32(self.ncap), np.int64(self.nghalf),
                                self.d_capring, self.d_ringstart,
                                self.d_hstart, self.d_nphi, maps,
                                np.int64(npix), self.d_Gh))

        for c in self.blu_classes:
            nmem, M = c["nmem"], c["M"]
            A = cp.empty((nmem, M), dtype=cp.complex128)
            self._k_blu_pre_adj(((M + bs - 1) // bs, nmem), (bs,),
                                (np.int32(nmem), np.int32(M), np.int64(npix),
                                 c["ring"], self.d_ringstart, self.d_nphi,
                                 maps, A))
            Ahat = cufft.fft(A, axis=1, overwrite_x=True)
            Ahat *= c["Bhat_adj"]
            conv = cufft.ifft(Ahat, axis=1, overwrite_x=True)
            self._k_blu_post_adj(((c["max_n"] // 2 + 1 + bs - 1) // bs, nmem),
                                 (bs,),
                                 (np.int32(nmem), np.int32(M),
                                  np.int64(self.nghalf), c["ring"],
                                  self.d_hstart, self.d_nphi, conv, self.d_Gh))
            del A, Ahat, conv

        # stage 2+3: Legendre adjoint with the unfold fused into the init
        # (F'_m gathered from Ghat on the fly); accumulates into ABadj
        self.d_AB.fill(0)
        nwarp = 256 // 32
        args = (np.int32(lmax), np.int32(mmax), np.int32(npair),
                np.int32(nring), self.d_moff, self.d_coef,
                self.d_mfac, self.d_powlimit, self.d_csq,
                self.d_cth, self.d_sth, self.d_mlim,
                self.d_inorth, self.d_isouth, self.d_pstart,
                self.d_Gh, self.d_hstart, self.d_nphi,
                self.d_phi0num, self.d_phi0den)
        if self.adj_impl == "v2":
            bd = self._bd_adj0
            nw = bd // 32
            # stage: nw*ADJ2_TR*4*ADJ2_W doubles; xtile: 2*nw*(TR*4+1)
            TR = 8
            shmem = (nw * TR * 4 * 33 + 2 * nw * (TR * 4 + 1)) * 8
            if shmem > getattr(self, "_adj2_shmem_set", 0):
                self._k_legendre_adj2.max_dynamic_shared_size_bytes = shmem
                self._adj2_shmem_set = shmem
            self._k_legendre_adj2((mmax + 1,), (bd,),
                                  args + (np.int64(0), np.int64(0), self.d_AB),
                                  shared_mem=shmem)
        else:
            shmem = nwarp * 128 * 4 * 8   # nwarp * ADJ_TILE * 4 doubles
            self._k_legendre_adj((mmax + 1,), (256,), args + (self.d_AB,),
                                 shared_mem=shmem)

        # stage 4: gather to healpy alm
        max_dl = lmax + 1
        self._k_postfold(((max_dl + bs - 1) // bs, mmax + 1), (bs,),
                         (np.int32(lmax), np.int32(mmax), self.d_moff,
                          self.d_coef, self.d_mstart,
                          np.int64(0), np.int64(0), self.d_AB, out))
        return out

    def adjoint_device_batch(self, maps, out=None):
        """Batched adjoint: maps (B, npix) -> alm (B, nalm) for spin 0,
        maps (B, 2, npix) -> alm (B, 2, nalm) for spin 2.

        Columns are batched on the kernel grid (blockIdx.y of the Legendre
        adjoint), chunked by the device-memory budget; falls back to looping
        singles when only one column fits."""
        cp = self.cp
        maps = cp.ascontiguousarray(maps)
        want = (self.npix,) if self.spin == 0 else (2, self.npix)
        if maps.shape[1:] != want:
            raise ValueError(f"maps must be (B, {want})")
        B = maps.shape[0]
        if out is None:
            out = cp.empty((B,) + ((self.nalm,) if self.spin == 0
                                   else (2, self.nalm)), dtype=cp.complex128)
        Cz = self._zbatch_cols(B, adjoint=True)
        if Cz <= 1:
            for b in range(B):
                self.adjoint_device(maps[b], out=out[b])
            return out
        # ensure the adjoint Bluestein chirps exist
        if self.spin == 0:
            self._adjoint_buffers()
        elif self.blu_classes and "Bhat_adj" not in self.blu_classes[0]:
            self._adjoint_spin2(maps[0])   # lazy-builds Bhat_adj + buffers
        self._zbuf(Cz, adjoint=True)
        for c0 in range(0, B, Cz):
            nb = min(Cz, B - c0)
            self._adj_zchunk(maps[c0: c0 + nb], out[c0: c0 + nb])
        return out

    # -------------------------------------------------------------- inverse

    def _alm_inner_batch(self, a, b):
        """Real-field harmonic inner product, independently per batch row."""
        cp = self.cp
        # In healpy's packed representation m=0 occupies the first lmax+1
        # entries. The independent real and imaginary parts at m>0 each carry
        # the factor two appearing in the exact synthesis-adjoint identity.
        z = (cp.conj(a) * b).real
        z[..., self.lmax + 1:] *= 2.0
        return z.reshape(z.shape[0], -1).sum(axis=1)

    def _canonicalize_alm_batch(self, alm):
        """Project packed coefficients onto the real spin-s field domain."""
        alm[..., :self.lmax + 1] = alm[..., :self.lmax + 1].real
        if self.spin == 2:
            # l=0,1 do not exist for a spin-2 field. In packed healpy order
            # these are (l,m)=(0,0),(1,0),(1,1).
            alm[..., :2] = 0.0
            alm[..., self.lmax + 1] = 0.0
        return alm

    def inverse_device_batch(self, maps, *, epsilon=1e-10, maxiter=20,
                             x0=None, return_info=False):
        """Iterative inverse synthesis (least-squares analysis), batched.

        This operation is intentionally distinct from :meth:`adjoint_device`.
        It solves ``min_a ||synthesis(a) - maps||_2`` with batched CGLS using
        Almond's exact synthesis/adjoint pair. For a band-limited input map it
        recovers the generating coefficients to ``epsilon``; for a general
        map it returns the Euclidean least-squares pseudoinverse, matching the
        mathematical target of ``ducc0.sht.experimental.pseudo_analysis``.

        Parameters
        ----------
        maps : cupy-compatible array
            ``(B,npix)`` for spin 0 or ``(B,2,npix)`` for spin 2.
        epsilon : float
            Relative normal-equation residual tolerance.
        maxiter : int
            Maximum CGLS iterations.
        x0 : array, optional
            Initial packed healpy coefficients with the corresponding batch
            shape.
        return_info : bool
            Also return a dictionary containing iteration count and residuals.
        """
        cp = self.cp
        maps = cp.ascontiguousarray(maps, dtype=cp.float64)
        want = (self.npix,) if self.spin == 0 else (2, self.npix)
        if maps.ndim != len(want) + 1 or maps.shape[1:] != want:
            raise ValueError(f"maps must be (B, {want})")
        B = maps.shape[0]
        ashape = (B, self.nalm) if self.spin == 0 else (B, 2, self.nalm)
        if x0 is None:
            x = cp.zeros(ashape, dtype=cp.complex128)
            r = maps.copy()
        else:
            x = cp.ascontiguousarray(x0, dtype=cp.complex128).copy()
            if x.shape != ashape:
                raise ValueError(f"x0 must have shape {ashape}")
            self._canonicalize_alm_batch(x)
            r = maps - self.synthesis_device_batch(x)

        s = self._canonicalize_alm_batch(self.adjoint_device_batch(r))
        p = s.copy()
        gamma = self._alm_inner_batch(s, s)
        gamma0 = cp.maximum(gamma, cp.finfo(cp.float64).tiny)
        tiny = cp.finfo(cp.float64).tiny
        rel_normal = cp.sqrt(gamma / gamma0)
        nit = 0

        for nit in range(1, int(maxiter) + 1):
            q = self.synthesis_device_batch(p)
            delta = cp.sum(q.reshape(B, -1) ** 2, axis=1)
            alpha = cp.where(delta > tiny, gamma / delta, 0.0)
            expand_map = (B,) + (1,) * (maps.ndim - 1)
            expand_alm = (B,) + (1,) * (x.ndim - 1)
            x += alpha.reshape(expand_alm) * p
            r -= alpha.reshape(expand_map) * q
            s = self._canonicalize_alm_batch(self.adjoint_device_batch(r))
            gamma_new = self._alm_inner_batch(s, s)
            rel_normal = cp.sqrt(gamma_new / gamma0)
            if bool(cp.all(rel_normal <= float(epsilon))):
                gamma = gamma_new
                break
            beta = cp.where(gamma > tiny, gamma_new / gamma, 0.0)
            p = s + beta.reshape(expand_alm) * p
            gamma = gamma_new

        self._canonicalize_alm_batch(x)
        if not return_info:
            if not bool(cp.all(rel_normal <= float(epsilon))):
                worst = float(cp.max(rel_normal))
                raise RuntimeError(
                    f"inverse did not converge in {maxiter} iterations "
                    f"(relative normal residual {worst:.3e}); increase "
                    "maxiter, lower lmax, or pass return_info=True to inspect "
                    "and explicitly accept an approximate solution")
            return x
        bnorm = cp.sqrt(cp.sum(maps.reshape(B, -1) ** 2, axis=1))
        rnorm = cp.sqrt(cp.sum(r.reshape(B, -1) ** 2, axis=1))
        info = {
            "niter": nit,
            "relative_normal_residual": rel_normal,
            "relative_map_residual": rnorm / cp.maximum(bnorm, tiny),
            "converged": rel_normal <= float(epsilon),
        }
        return x, info

    def inverse_device(self, maps, *, epsilon=1e-10, maxiter=20, x0=None,
                       return_info=False):
        """Single-map device inverse; see :meth:`inverse_device_batch`."""
        cp = self.cp
        maps = cp.ascontiguousarray(maps, dtype=cp.float64)
        x0b = None if x0 is None else cp.asarray(x0)[None]
        result = self.inverse_device_batch(
            maps[None], epsilon=epsilon, maxiter=maxiter, x0=x0b,
            return_info=return_info)
        if not return_info:
            return result[0]
        x, info = result
        info = {k: (v[0] if hasattr(v, "shape") and v.shape else v)
                for k, v in info.items()}
        return x[0], info

    def inverse(self, maps: np.ndarray, *, epsilon=1e-10, maxiter=20,
                x0=None, return_info=False):
        """Host convenience wrapper for iterative inverse synthesis."""
        cp = self.cp
        dmap = cp.asarray(np.ascontiguousarray(maps, dtype=np.float64))
        dx0 = None if x0 is None else cp.asarray(
            np.ascontiguousarray(x0, dtype=np.complex128))
        result = self.inverse_device(dmap, epsilon=epsilon, maxiter=maxiter,
                                     x0=dx0, return_info=return_info)
        if not return_info:
            return cp.asnumpy(result)
        alm, info = result
        return cp.asnumpy(alm), {
            k: (cp.asnumpy(v) if isinstance(v, cp.ndarray) else v)
            for k, v in info.items()
        }

    def adjoint(self, maps: np.ndarray) -> np.ndarray:
        """Host convenience wrapper: numpy map in, numpy alm out."""
        cp = self.cp
        d = cp.asarray(np.ascontiguousarray(maps, dtype=np.float64))
        if self.spin == 0:
            d = d.ravel()
        return cp.asnumpy(self.adjoint_device(d))

    def synthesis(self, alm: np.ndarray) -> np.ndarray:
        """Host convenience wrapper: numpy alm in, numpy map out."""
        cp = self.cp
        d_alm = cp.asarray(np.ascontiguousarray(alm, dtype=np.complex128))
        if self.spin == 0:
            d_alm = d_alm.ravel()
        out = self.synthesis_device(d_alm)
        return cp.asnumpy(out)

    def memory_bytes(self) -> dict:
        """Device memory footprint of the plan's persistent buffers."""
        batched = hasattr(self, "d_phase_b")
        b = {
            "coef": self.d_coef.nbytes,
            "AB": self.d_AB.nbytes,
            "map": self.d_map.nbytes if getattr(self, "d_map", None) is not None else 0,
            "bluestein_Bhat": sum(c["Bhat"].nbytes for c in self.blu_classes),
        }
        if batched:  # single-path G is a view into the batch buffers
            b["AB_b"] = self.d_AB_b.nbytes
            b["phase_b"] = self.d_phase_b.nbytes
            b["G_b"] = self.d_G_b.nbytes
        else:
            b["G"] = self.d_G.nbytes
        b["total"] = sum(b.values())
        return b
