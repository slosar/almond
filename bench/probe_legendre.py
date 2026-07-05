#!/usr/bin/env python3
"""Probe: which m range makes the legendre kernel slow at nside=1024,
and is the coefficient table sane (no Inf/NaN, magnitude stats)."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cupy as cp
from almond.plan import SynthesisPlan

nside = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
lmax = 3 * nside - 1
plan = SynthesisPlan(nside, lmax)
print(f"nside={nside} lmax={lmax} npair={plan.npair}", flush=True)

# --- coef table sanity ---
coef = plan.d_coef
a = coef.real
b = coef.imag
print("coef finite:", bool(cp.isfinite(a).all()), bool(cp.isfinite(b).all()), flush=True)
print("max |a|:", float(cp.abs(a).max()), " max |b|:", float(cp.abs(b).max()), flush=True)

# --- prefold with random alm ---
rng = np.random.default_rng(0)
alm = rng.standard_normal(plan.nalm) + 1j * rng.standard_normal(plan.nalm)
d_alm = cp.asarray(alm)
bs = 128
max_nil = lmax // 2 + 2
plan._k_prefold(((max_nil + bs - 1) // bs, plan.mmax + 1), (bs,),
                (np.int32(lmax), np.int32(plan.mmax), plan.d_moff,
                 plan.d_coef, plan.d_mstart, d_alm, plan.d_AB))
cp.cuda.runtime.deviceSynchronize()
AB = plan.d_AB
print("AB finite:", bool(cp.isfinite(AB.real).all()), "max|AB|:",
      float(cp.abs(AB).max()), flush=True)

# --- legendre with m restricted to [0, mtest] via truncated grid+mmax ---
plan.d_phase.fill(0)
cp.cuda.runtime.deviceSynchronize()
for mtest in [255, 511, 1023, 1535, 2047, 2559, 2815, 3071]:
    if mtest > plan.mmax:
        break
    t0 = time.perf_counter()
    plan._k_legendre(((plan.npair + bs - 1) // bs, mtest + 1), (bs,),
                     (np.int32(lmax), np.int32(mtest), np.int32(plan.npair),
                      np.int32(plan.nring), plan.d_moff, plan.d_coef, plan.d_AB,
                      plan.d_mfac, plan.d_powlimit, plan.d_csq, plan.d_cth,
                      plan.d_sth, plan.d_mlim, plan.d_inorth,
                      plan.d_isouth, plan.d_phase))
    cp.cuda.runtime.deviceSynchronize()
    print(f"m in [0,{mtest}]: {time.perf_counter()-t0:.3f}s", flush=True)
