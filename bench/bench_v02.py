#!/usr/bin/env python3
"""v0.2 benchmark: adjoint + spin-2 (synth & adjoint) vs ducc0 at 64 threads."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ducc0


def median(x):
    return float(np.median(np.asarray(x)))


def timeit(fn, n):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--ducc-threads", type=int, default=64)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    nside = args.nside
    lmax = 3 * nside - 1
    nt = args.ducc_threads
    import cupy as cp
    from almond.plan import SynthesisPlan

    rng = np.random.default_rng(3)
    nalm = (lmax + 1) * (lmax + 2) // 2
    npix = 12 * nside * nside
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    m = np.arange(lmax + 1)
    mstart = (m * (2 * lmax + 1 - m) // 2).astype(np.uint64)

    result = {"nside": nside, "lmax": lmax, "ducc_threads": nt,
              "hostname": socket.gethostname(),
              "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}

    def sync():
        cp.cuda.runtime.deviceSynchronize()

    # ---- spin 0 ----
    p0 = SynthesisPlan(nside, lmax)
    a0 = rng.standard_normal(nalm) + 1j * rng.standard_normal(nalm)
    a0[: lmax + 1] = a0[: lmax + 1].real
    f0 = rng.standard_normal(npix)
    d_a0, d_f0 = cp.asarray(a0), cp.asarray(f0)

    got = cp.asnumpy(p0.adjoint_device(d_f0))
    ref = ducc0.sht.experimental.adjoint_synthesis(
        map=f0.reshape(1, -1), lmax=lmax, spin=0, mstart=mstart,
        nthreads=nt, **geom)[0]
    result["acc_adj0"] = float(np.abs(got - ref).max() / np.abs(ref).max())
    result["alm_synth0_s"] = timeit(
        lambda: (p0.synthesis_device(d_a0), sync()), args.repeats)
    result["alm_adj0_s"] = timeit(
        lambda: (p0.adjoint_device(d_f0), sync()), args.repeats)
    result["ducc_adj0_s"] = timeit(
        lambda: ducc0.sht.experimental.adjoint_synthesis(
            map=f0.reshape(1, -1), lmax=lmax, spin=0, mstart=mstart,
            nthreads=nt, **geom), max(3, args.repeats // 2))
    result["ducc_synth0_s"] = timeit(
        lambda: ducc0.sht.experimental.synthesis(
            alm=a0.reshape(1, -1), lmax=lmax, spin=0, mstart=mstart,
            nthreads=nt, **geom), max(3, args.repeats // 2))
    del p0, d_a0, d_f0
    cp.get_default_memory_pool().free_all_blocks()

    # ---- spin 2 ----
    p2 = SynthesisPlan(nside, lmax, spin=2)
    a2 = rng.standard_normal((2, nalm)) + 1j * rng.standard_normal((2, nalm))
    a2[:, : lmax + 1] = a2[:, : lmax + 1].real
    a2[:, :2] = 0
    a2[:, lmax + 1] = 0
    f2 = rng.standard_normal((2, npix))
    d_a2, d_f2 = cp.asarray(a2), cp.asarray(f2)

    got = cp.asnumpy(p2.synthesis_device(d_a2))
    ref = ducc0.sht.experimental.synthesis(
        alm=a2, lmax=lmax, spin=2, mstart=mstart, nthreads=nt, **geom)
    result["acc_synth2"] = float(np.abs(got - ref).max() / np.abs(ref).max())
    gota = cp.asnumpy(p2.adjoint_device(d_f2))
    refa = ducc0.sht.experimental.adjoint_synthesis(
        map=f2, lmax=lmax, spin=2, mstart=mstart, nthreads=nt, **geom)
    result["acc_adj2"] = float(np.abs(gota - refa).max() / np.abs(refa).max())

    result["alm_synth2_s"] = timeit(
        lambda: (p2.synthesis_device(d_a2), sync()), args.repeats)
    result["alm_adj2_s"] = timeit(
        lambda: (p2.adjoint_device(d_f2), sync()), args.repeats)
    result["ducc_synth2_s"] = timeit(
        lambda: ducc0.sht.experimental.synthesis(
            alm=a2, lmax=lmax, spin=2, mstart=mstart, nthreads=nt, **geom),
        max(3, args.repeats // 2))
    result["ducc_adj2_s"] = timeit(
        lambda: ducc0.sht.experimental.adjoint_synthesis(
            map=f2, lmax=lmax, spin=2, mstart=mstart, nthreads=nt, **geom),
        max(3, args.repeats // 2))

    for k in ["synth0", "adj0", "synth2", "adj2"]:
        result[f"speedup_{k}"] = result[f"ducc_{k}_s"] / result[f"alm_{k}_s"]

    out = args.output or Path(__file__).resolve().parents[1] / "results" / \
        f"v02_n{nside}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
