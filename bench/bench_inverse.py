#!/usr/bin/env python3
"""Benchmark Almond CGLS inverse against ducc0 ``pseudo_analysis``."""

from __future__ import annotations

import argparse
import json
import os
import time

import cupy as cp
import ducc0
import numpy as np

from almond.plan import SynthesisPlan


def geom(nside, lmax):
    g = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    m = np.arange(lmax + 1)
    g["mstart"] = (m * (2 * lmax + 1 - m) // 2).astype(np.uint64)
    g["lmax"] = lmax
    return g


def median_time(fn, reps, sync=None):
    fn()
    if sync:
        sync()
    vals = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        vals.append(time.perf_counter() - t0)
    return float(np.median(vals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nside", type=int, default=128)
    p.add_argument("--lmax", type=int)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epsilon", type=float, default=1e-10)
    p.add_argument("--maxiter", type=int, default=30)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--ducc-threads", type=int, default=64)
    args = p.parse_args()
    lmax = args.lmax or 3 * args.nside - 1
    nalm = (lmax + 1) * (lmax + 2) // 2
    rng = np.random.default_rng(505)
    g = geom(args.nside, lmax)
    result = {"nside": args.nside, "lmax": lmax, "batch": args.batch,
              "epsilon": args.epsilon, "maxiter": args.maxiter,
              "ducc_threads": args.ducc_threads,
              "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}

    for spin in (0, 2):
        shape = ((args.batch, nalm) if spin == 0 else
                 (args.batch, 2, nalm))
        alm = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        alm[..., :lmax + 1] = alm[..., :lmax + 1].real
        if spin == 2:
            alm[..., :2] = 0.0
            alm[..., lmax + 1] = 0.0
        plan = SynthesisPlan(args.nside, lmax, spin=spin)
        maps = plan.synthesis_device_batch(cp.asarray(alm))
        maps_h = cp.asnumpy(maps)
        dcall = lambda: plan.inverse_device_batch(
            maps, epsilon=args.epsilon, maxiter=args.maxiter,
            return_info=True)
        dresult = dcall()
        atime = median_time(dcall, args.reps, cp.cuda.runtime.deviceSynchronize)
        dinv, dinfo = dresult

        ducc_maps = maps_h[:, None, :] if spin == 0 else maps_h
        ucall = lambda: ducc0.sht.experimental.pseudo_analysis(
            map=ducc_maps, spin=spin, nthreads=args.ducc_threads,
            maxiter=args.maxiter, epsilon=args.epsilon, **g)
        uresult = ucall()
        utime = median_time(ucall, args.reps)
        uinv = uresult[0][:, 0, :] if spin == 0 else uresult[0]
        dinv_h = cp.asnumpy(dinv)
        result[f"spin{spin}"] = {
            "almond_s": atime,
            "ducc_s": utime,
            "speedup": utime / atime,
            "almond_niter": int(dinfo["niter"]),
            "ducc_niter": int(np.max(uresult[2])),
            "almond_map_residual": float(cp.max(
                dinfo["relative_map_residual"])),
            "ducc_stop_reason": int(np.max(uresult[1])),
            "solution_relative_difference": float(
                np.linalg.norm(dinv_h - uinv) / np.linalg.norm(uinv)),
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
