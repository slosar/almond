#!/usr/bin/env python3
"""Instrumented legendre kernel: find exactly which loop spins and for which
(pair, m). Runs m in [256,511] at nside=1024 with hard iteration caps."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cupy as cp
from almond.plan import SynthesisPlan

src_path = Path(__file__).resolve().parents[1] / "alm" / "kernels.cu"
src = src_path.read_text()

# instrumented copy of the legendre kernel
debug_src = src + r"""
extern "C" __global__ void legendre_dbg(const int lmax, const int mmax,
                         const int npair, const int nring,
                         const long long* __restrict__ moff,
                         const double2* __restrict__ coef,
                         const double2* __restrict__ AB,
                         const double* __restrict__ mfac,
                         const double* __restrict__ powlimit,
                         const double* __restrict__ pair_csq,
                         const double* __restrict__ pair_cth,
                         const double* __restrict__ pair_sth,
                         const int* __restrict__ pair_mlim,
                         const int* __restrict__ pair_inorth,
                         const int* __restrict__ pair_isouth,
                         double2* __restrict__ phase,
                         int* __restrict__ flag)   // flag[0]=loop id, flag[1]=p, flag[2]=m
  {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y + 256;
  if (p >= npair || m > mmax) return;
  if (pair_mlim[p] < m) return;

  const double csq = pair_csq[p];
  const long long off = moff[m];
  const int nacc = (lmax - m) / 2 + 1;

  double lam2; int scale;
  mypow_scaled(pair_sth[p], m, powlimit[m], lam2, scale);
  lam2 *= (m & 1) ? -mfac[m] : mfac[m];
  long long g = 0;
  while (fabs(lam2) > FTOL) { lam2 *= FSMALL; ++scale;
    if (++g > 10000) { flag[0]=1; flag[1]=p; flag[2]=m; return; } }
  g = 0;
  if (lam2 != 0.0)
    while (fabs(lam2) < FTOL * FSMALL) { lam2 *= FBIG; --scale;
      if (++g > 10000) { flag[0]=2; flag[1]=p; flag[2]=m; return; } }
  double lam1 = 0.0;

  int il = 0;
  g = 0;
  while (scale < 0)
    {
    if (il >= nacc) return;
    double2 c = coef[off + il];
    double t = (c.x * csq + c.y) * lam2 + lam1;
    lam1 = lam2; lam2 = t;
    ++il;
    if (fabs(lam2) > FTOL) { lam1 *= FSMALL; lam2 *= FSMALL; ++scale; }
    if (++g > 100000) { flag[0]=3; flag[1]=p; flag[2]=m;
                        flag[3]=scale; flag[4]=il; return; }
    }
  g = 0;
  while (scale > 0) { lam1 *= FBIG; lam2 *= FBIG; --scale;
    if (++g > 10000) { flag[0]=4; flag[1]=p; flag[2]=m; return; } }
  // no accumulation in debug variant
  }
"""

nside = 1024
lmax = 3 * nside - 1
plan = SynthesisPlan(nside, lmax)
mod = cp.RawModule(code=debug_src, options=("--std=c++17",))
k = mod.get_function("legendre_dbg")
flag = cp.zeros(8, dtype=cp.int32)
bs = 128
t0 = time.perf_counter()
k(((plan.npair + bs - 1) // bs, 256), (bs,),
  (np.int32(lmax), np.int32(plan.mmax), np.int32(plan.npair),
   np.int32(plan.nring), plan.d_moff, plan.d_coef, plan.d_AB,
   plan.d_mfac, plan.d_powlimit, plan.d_csq, plan.d_cth,
   plan.d_sth, plan.d_mlim, plan.d_inorth, plan.d_isouth,
   plan.d_phase, flag))
cp.cuda.runtime.deviceSynchronize()
print(f"time {time.perf_counter()-t0:.3f}s  flag={cp.asnumpy(flag)}", flush=True)
