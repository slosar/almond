#!/usr/bin/env python3
"""Benchmark all Almond transform modes: {synthesis, adjoint} x {spin 0, spin 2}.

Per (nside, mode): accuracy gate vs ducc0, then device-resident warm-call
median timing.  Writes one JSON per run to results/.

Usage:
    python bench/bench_modes.py --nsides 512 1024 2048
    python bench/bench_modes.py --nsides 1024 --skip-ducc   # GPU-only (dev loop)
"""

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


def ducc_geom(nside, lmax):
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    m = np.arange(lmax + 1)
    geom["mstart"] = (m * (2 * lmax + 1 - m) // 2).astype(np.uint64)
    geom["lmax"] = lmax
    return geom


def median(x):
    return float(np.median(np.asarray(x)))


def time_device(fn, arg, repeats, cp):
    fn(arg)  # warmup
    cp.cuda.runtime.deviceSynchronize()
    times = []
    for _ in range(repeats):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        fn(arg)
        cp.cuda.runtime.deviceSynchronize()
        times.append(time.perf_counter() - t0)
    return median(times)


def bench_nside(nside, lmax, repeats, skip_ducc, ducc_threads):
    import cupy as cp
    from almond.plan import SynthesisPlan

    lmax = lmax if lmax is not None else 3 * nside - 1
    rng = np.random.default_rng(42)
    nalm = (lmax + 1) * (lmax + 2) // 2
    npix = 12 * nside * nside

    def rand_alm(ncomp):
        a = rng.standard_normal((ncomp, nalm)) + 1j * rng.standard_normal((ncomp, nalm))
        a[:, : lmax + 1] = a[:, : lmax + 1].real
        return a

    geom = ducc_geom(nside, lmax)
    nt = min(os.cpu_count() or 16, 16)
    res = {"nside": nside, "lmax": lmax}

    # ---------------- spin 0 ----------------
    alm0 = rand_alm(1)
    plan0 = SynthesisPlan(nside, lmax)
    d_alm0 = cp.asarray(alm0[0])
    m_ref = ducc0.sht.experimental.synthesis(
        alm=alm0, spin=0, nthreads=nt, **geom)[0]
    got = cp.asnumpy(plan0.synthesis_device(d_alm0))
    res["synth0_err"] = float(np.abs(got - m_ref).max() / np.abs(m_ref).max())
    res["synth0_s"] = time_device(plan0.synthesis_device, d_alm0, repeats, cp)

    d_map0 = cp.asarray(m_ref)
    a_ref = ducc0.sht.experimental.adjoint_synthesis(
        map=m_ref[None, :], spin=0, nthreads=nt, **geom)[0]
    got = cp.asnumpy(plan0.adjoint_device(d_map0))
    res["adj0_err"] = float(np.abs(got - a_ref).max() / np.abs(a_ref).max())
    res["adj0_s"] = time_device(plan0.adjoint_device, d_map0, repeats, cp)

    if not skip_ducc:
        res["ducc"] = {}
        for dt in ducc_threads:
            if dt > (os.cpu_count() or 1):
                continue
            for tag, fn in (
                ("synth0", lambda: ducc0.sht.experimental.synthesis(
                    alm=alm0, spin=0, nthreads=dt, **geom)),
                ("adj0", lambda: ducc0.sht.experimental.adjoint_synthesis(
                    map=m_ref[None, :], spin=0, nthreads=dt, **geom)),
            ):
                fn()
                times = []
                for _ in range(max(3, repeats // 3)):
                    t0 = time.perf_counter()
                    fn()
                    times.append(time.perf_counter() - t0)
                res["ducc"][f"{tag}_{dt}t"] = median(times)
    del plan0, d_alm0, d_map0
    cp.get_default_memory_pool().free_all_blocks()

    # ---------------- spin 2 ----------------
    alm2 = rand_alm(2)
    alm2[:, : 2 * (lmax + 1)] = 0  # l<2 zero for spin-2 (both m=0,1 blocks ok)
    for m in (0, 1):
        ms = m * (2 * lmax + 1 - m) // 2
        alm2[:, ms + m: ms + 2] = 0
    plan2 = SynthesisPlan(nside, lmax, spin=2)
    d_alm2 = cp.asarray(alm2)
    qu_ref = ducc0.sht.experimental.synthesis(
        alm=alm2, spin=2, nthreads=nt, **geom)
    got = cp.asnumpy(plan2.synthesis_device(d_alm2))
    res["synth2_err"] = float(np.abs(got - qu_ref).max() / np.abs(qu_ref).max())
    res["synth2_s"] = time_device(plan2.synthesis_device, d_alm2, repeats, cp)

    d_qu = cp.asarray(qu_ref)
    a2_ref = ducc0.sht.experimental.adjoint_synthesis(
        map=qu_ref, spin=2, nthreads=nt, **geom)
    got = cp.asnumpy(plan2.adjoint_device(d_qu))
    res["adj2_err"] = float(np.abs(got - a2_ref).max() / np.abs(a2_ref).max())
    res["adj2_s"] = time_device(plan2.adjoint_device, d_qu, repeats, cp)

    if not skip_ducc:
        for dt in ducc_threads:
            if dt > (os.cpu_count() or 1):
                continue
            for tag, fn in (
                ("synth2", lambda: ducc0.sht.experimental.synthesis(
                    alm=alm2, spin=2, nthreads=dt, **geom)),
                ("adj2", lambda: ducc0.sht.experimental.adjoint_synthesis(
                    map=qu_ref, spin=2, nthreads=dt, **geom)),
            ):
                fn()
                times = []
                for _ in range(max(3, repeats // 3)):
                    t0 = time.perf_counter()
                    fn()
                    times.append(time.perf_counter() - t0)
                res["ducc"][f"{tag}_{dt}t"] = median(times)
    del plan2, d_alm2, d_qu
    cp.get_default_memory_pool().free_all_blocks()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsides", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--lmax", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--ducc-threads", type=int, nargs="*", default=[64])
    ap.add_argument("--skip-ducc", action="store_true")
    ap.add_argument("--tag", type=str, default="modes")
    args = ap.parse_args()

    import cupy as cp
    result = {
        "hostname": socket.gethostname(),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cpu_count": os.cpu_count(),
        "runs": [],
    }
    for nside in args.nsides:
        r = bench_nside(nside, args.lmax, args.repeats,
                        args.skip_ducc, args.ducc_threads)
        result["runs"].append(r)
        print(f"nside {nside}: "
              + "  ".join(f"{k}={v * 1e3:.1f}ms" for k, v in r.items()
                          if k.endswith("_s"))
              + "  errs: "
              + " ".join(f"{r[k]:.1e}" for k in
                         ("synth0_err", "adj0_err", "synth2_err", "adj2_err")),
              flush=True)

    out_path = Path(__file__).resolve().parents[1] / "results" / \
        f"{args.tag}_{'_'.join(str(n) for n in args.nsides)}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
