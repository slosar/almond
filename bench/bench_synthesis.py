#!/usr/bin/env python3
"""Benchmark Almond GPU spin-0 synthesis vs ducc0 CPU.

Per nside: validates vs ducc0 (max abs rel error), then times
  - alm GPU synthesis, device-resident input/output (the QML use case)
  - alm GPU synthesis including host->device->host copies
  - ducc0 synthesis at the requested thread counts

Writes one JSON per run to results/.
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


def ducc_make(nside, lmax, nthreads):
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    m = np.arange(lmax + 1)
    mstart = (m * (2 * lmax + 1 - m) // 2).astype(np.uint64)

    def run(alm2d):
        return ducc0.sht.experimental.synthesis(
            alm=alm2d, lmax=lmax, spin=0, mstart=mstart,
            nthreads=nthreads, **geom)
    return run


def median(x):
    return float(np.median(np.asarray(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, required=True)
    ap.add_argument("--lmax", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--ducc-threads", type=int, nargs="*", default=[1, 16, 32, 64, 128])
    ap.add_argument("--skip-ducc", action="store_true")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    nside = args.nside
    lmax = args.lmax if args.lmax is not None else 3 * nside - 1
    repeats = args.repeats

    import cupy as cp
    from almond.plan import SynthesisPlan

    rng = np.random.default_rng(42)
    nalm = (lmax + 1) * (lmax + 2) // 2
    alm = rng.standard_normal(nalm) + 1j * rng.standard_normal(nalm)
    alm[: lmax + 1] = alm[: lmax + 1].real
    alm2d = np.ascontiguousarray(alm.reshape(1, -1))

    result = {
        "nside": nside, "lmax": lmax, "npix": 12 * nside * nside,
        "nalm": nalm, "repeats": repeats,
        "hostname": socket.gethostname(),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "cpu_count": os.cpu_count(),
        "slurm_cpus": os.environ.get("SLURM_CPUS_ON_NODE"),
    }

    # ---------------- Almond GPU ----------------
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    t0 = time.perf_counter()
    plan = SynthesisPlan(nside, lmax)
    cp.cuda.runtime.deviceSynchronize()
    result["alm_plan_build_s"] = time.perf_counter() - t0
    result["alm_plan_buffers_bytes"] = plan.memory_bytes()

    d_alm = cp.asarray(alm)
    # warmup (also compiles kernels on first plan in the process)
    out = plan.synthesis_device(d_alm)
    cp.cuda.runtime.deviceSynchronize()
    result["alm_gpu_peak_pool_bytes"] = pool.total_bytes()

    # accuracy vs ducc (reference at 16 threads)
    ref = ducc_make(nside, lmax, min(os.cpu_count() or 16, 16))(alm2d)[0]
    got = cp.asnumpy(out)
    result["max_abs_rel_err_vs_ducc"] = float(
        np.abs(got - ref).max() / np.abs(ref).max())

    # device-resident timing
    times = []
    for _ in range(repeats):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        plan.synthesis_device(d_alm)
        cp.cuda.runtime.deviceSynchronize()
        times.append(time.perf_counter() - t0)
    result["alm_synth_device_s"] = median(times)
    result["alm_synth_device_all_s"] = times

    # including host<->device copies
    times = []
    for _ in range(repeats):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        plan.synthesis(alm)
        cp.cuda.runtime.deviceSynchronize()
        times.append(time.perf_counter() - t0)
    result["alm_synth_host_s"] = median(times)

    # ---------------- ducc CPU ----------------
    if not args.skip_ducc:
        result["ducc"] = {}
        for nt in args.ducc_threads:
            if nt > (os.cpu_count() or 1):
                continue
            run = ducc_make(nside, lmax, nt)
            run(alm2d)  # warmup
            times = []
            for _ in range(max(3, repeats // 2) if nside >= 1024 else repeats):
                t0 = time.perf_counter()
                run(alm2d)
                times.append(time.perf_counter() - t0)
            result["ducc"][str(nt)] = median(times)

    out_path = args.output or Path(__file__).resolve().parents[1] / "results" / \
        f"synth_n{nside}_l{lmax}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
