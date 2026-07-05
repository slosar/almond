#!/usr/bin/env python3
"""Batched benchmark, all modes: {synthesis, adjoint} x {spin 0, spin 2}.

Almond's grid-batched pipelines vs ducc0's ntrans mode (its strongest
many-transform configuration).  Reports per-column device-resident times.

    python bench/bench_batched_modes.py --nside 128 --batch 64
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


def time_it(fn, repeats, sync=None):
    fn()
    if sync:
        sync()
    ts = []
    for _ in range(repeats):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--ducc-threads", type=int, default=None)
    ap.add_argument("--skip-ducc", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    nside, B = args.nside, args.batch
    lmax = 3 * nside - 1
    nt = args.ducc_threads or (64 if os.environ.get("SLURM_JOB_ID") else 16)

    import cupy as cp
    from almond.plan import SynthesisPlan

    rng = np.random.default_rng(7)
    nalm = (lmax + 1) * (lmax + 2) // 2
    geom = ducc_geom(nside, lmax)
    sync = cp.cuda.runtime.deviceSynchronize

    result = {
        "mode": "batched_modes", "nside": nside, "lmax": lmax, "batch": B,
        "hostname": socket.gethostname(),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "ducc_threads": nt, "repeats": args.repeats,
    }

    # ---------------- spin 0 ----------------
    alm0 = rng.standard_normal((B, nalm)) + 1j * rng.standard_normal((B, nalm))
    alm0[:, : lmax + 1] = alm0[:, : lmax + 1].real
    plan0 = SynthesisPlan(nside, lmax)
    d_alm0 = cp.asarray(alm0)
    d_maps0 = plan0.synthesis_device_batch(d_alm0)

    # accuracy gates
    ref = ducc0.sht.experimental.synthesis(
        alm=alm0[:, None, :], spin=0, nthreads=nt, **geom)[:, 0, :]
    got = cp.asnumpy(d_maps0)
    result["synth0_err"] = float(np.abs(got - ref).max() / np.abs(ref).max())
    aref = ducc0.sht.experimental.adjoint_synthesis(
        map=ref[:, None, :], spin=0, nthreads=nt, **geom)[:, 0, :]
    agot = cp.asnumpy(plan0.adjoint_device_batch(cp.asarray(ref)))
    result["adj0_err"] = float(np.abs(agot - aref).max() / np.abs(aref).max())

    result["synth0_percol_s"] = time_it(
        lambda: plan0.synthesis_device_batch(d_alm0, out=d_maps0),
        args.repeats, sync) / B
    d_in0 = cp.asarray(ref)
    result["adj0_percol_s"] = time_it(
        lambda: plan0.adjoint_device_batch(d_in0),
        args.repeats, sync) / B
    if not args.skip_ducc:
        result["ducc_synth0_percol_s"] = time_it(
            lambda: ducc0.sht.experimental.synthesis(
                alm=alm0[:, None, :], spin=0, nthreads=nt, **geom),
            max(3, args.repeats // 2)) / B
        result["ducc_adj0_percol_s"] = time_it(
            lambda: ducc0.sht.experimental.adjoint_synthesis(
                map=ref[:, None, :], spin=0, nthreads=nt, **geom),
            max(3, args.repeats // 2)) / B
    del plan0, d_alm0, d_maps0, d_in0
    cp.get_default_memory_pool().free_all_blocks()

    # ---------------- spin 2 ----------------
    alm2 = (rng.standard_normal((B, 2, nalm))
            + 1j * rng.standard_normal((B, 2, nalm)))
    alm2[:, :, : lmax + 1] = alm2[:, :, : lmax + 1].real
    for m in (0, 1):
        ms = m * (2 * lmax + 1 - m) // 2
        alm2[:, :, ms + m: ms + 2] = 0
    plan2 = SynthesisPlan(nside, lmax, spin=2)
    d_alm2 = cp.asarray(alm2)
    d_maps2 = plan2.synthesis_device_batch(d_alm2)

    ref2 = ducc0.sht.experimental.synthesis(
        alm=alm2, spin=2, nthreads=nt, **geom)
    got2 = cp.asnumpy(d_maps2)
    result["synth2_err"] = float(np.abs(got2 - ref2).max()
                                 / np.abs(ref2).max())
    aref2 = ducc0.sht.experimental.adjoint_synthesis(
        map=ref2, spin=2, nthreads=nt, **geom)
    agot2 = cp.asnumpy(plan2.adjoint_device_batch(cp.asarray(ref2)))
    result["adj2_err"] = float(np.abs(agot2 - aref2).max()
                               / np.abs(aref2).max())

    result["synth2_percol_s"] = time_it(
        lambda: plan2.synthesis_device_batch(d_alm2, out=d_maps2),
        args.repeats, sync) / B
    d_in2 = cp.asarray(ref2)
    result["adj2_percol_s"] = time_it(
        lambda: plan2.adjoint_device_batch(d_in2),
        args.repeats, sync) / B
    if not args.skip_ducc:
        result["ducc_synth2_percol_s"] = time_it(
            lambda: ducc0.sht.experimental.synthesis(
                alm=alm2, spin=2, nthreads=nt, **geom),
            max(3, args.repeats // 2)) / B
        result["ducc_adj2_percol_s"] = time_it(
            lambda: ducc0.sht.experimental.adjoint_synthesis(
                map=ref2, spin=2, nthreads=nt, **geom),
            max(3, args.repeats // 2)) / B

    out_path = args.output or Path(__file__).resolve().parents[1] / \
        "results" / f"batchmodes_n{nside}_B{B}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
