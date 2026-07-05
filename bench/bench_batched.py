#!/usr/bin/env python3
"""High-throughput batched comparison: Almond batched GPU vs ducc0 ntrans mode.

ducc0 is run in its strongest configuration for many transforms: a single
synthesis call with alm (B, 1, nalm), which threads across the batch with
near-linear scaling.  Almond runs its chunked batched pipeline.  Reports
per-column times, device-resident and including host copies.
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


def median(x):
    return float(np.median(np.asarray(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--ducc-threads", type=int, default=None,
                    help="default: physical cores (SLURM) or 16")
    ap.add_argument("--chunk", type=int, default=None)
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
    alm = rng.standard_normal((B, nalm)) + 1j * rng.standard_normal((B, nalm))
    alm[:, : lmax + 1] = alm[:, : lmax + 1].real

    result = {
        "mode": "batched", "nside": nside, "lmax": lmax, "batch": B,
        "hostname": socket.gethostname(),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "ducc_threads": nt, "repeats": args.repeats,
    }

    # ---------------- Almond batched ----------------
    plan = SynthesisPlan(nside, lmax, chunk=args.chunk)
    result["chunk"] = plan.chunk
    d_alm = cp.asarray(alm)
    out = plan.synthesis_device_batch(d_alm)   # warmup + compile
    cp.cuda.runtime.deviceSynchronize()
    pool = cp.get_default_memory_pool()
    result["alm_gpu_peak_pool_bytes"] = pool.total_bytes()

    times = []
    for _ in range(args.repeats):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        plan.synthesis_device_batch(d_alm, out=out)
        cp.cuda.runtime.deviceSynchronize()
        times.append(time.perf_counter() - t0)
    result["alm_batch_device_s"] = median(times)
    result["alm_batch_device_per_col_s"] = median(times) / B
    got = cp.asnumpy(out)

    # host-inclusive: reuse the resident buffers, time H2D + compute + D2H
    times = []
    for _ in range(max(2, args.repeats // 2)):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        d_alm.set(alm)
        plan.synthesis_device_batch(d_alm, out=out)
        _ = cp.asnumpy(out)
        cp.cuda.runtime.deviceSynchronize()
        times.append(time.perf_counter() - t0)
    result["alm_batch_host_s"] = median(times)
    result["alm_batch_host_per_col_s"] = median(times) / B

    # ---------------- ducc ntrans batch ----------------
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    m = np.arange(lmax + 1)
    mstart = (m * (2 * lmax + 1 - m) // 2).astype(np.uint64)
    alm3 = np.ascontiguousarray(alm.reshape(B, 1, nalm))

    def ducc_run():
        return ducc0.sht.experimental.synthesis(
            alm=alm3, lmax=lmax, spin=0, mstart=mstart, nthreads=nt, **geom)

    ref = ducc_run()  # warmup + reference
    err = np.abs(got - ref[:, 0, :]).max() / np.abs(ref).max()
    result["max_abs_rel_err_vs_ducc"] = float(err)

    if not args.skip_ducc:
        times = []
        for _ in range(args.repeats if nside < 1024 else max(3, args.repeats // 2)):
            t0 = time.perf_counter()
            ducc_run()
            times.append(time.perf_counter() - t0)
        result["ducc_batch_s"] = median(times)
        result["ducc_batch_per_col_s"] = median(times) / B
        result["speedup_device"] = result["ducc_batch_s"] / result["alm_batch_device_s"]
        result["speedup_host"] = result["ducc_batch_s"] / result["alm_batch_host_s"]

    out_path = args.output or Path(__file__).resolve().parents[1] / "results" / \
        f"batch_n{nside}_B{B}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
