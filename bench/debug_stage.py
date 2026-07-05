#!/usr/bin/env python3
"""Per-stage timing breakdown of the GPU synthesis (uses plan's timings hook)."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cupy as cp
from almond.plan import SynthesisPlan

nside = int(sys.argv[1]) if len(sys.argv) > 1 else 512
lmax = 3 * nside - 1
rng = np.random.default_rng(1)
nalm = (lmax + 1) * (lmax + 2) // 2
alm = rng.standard_normal(nalm) + 1j * rng.standard_normal(nalm)

t0 = time.perf_counter()
plan = SynthesisPlan(nside, lmax)
cp.cuda.runtime.deviceSynchronize()
print(f"plan build: {time.perf_counter()-t0:.3f}s")

d_alm = cp.asarray(alm)
plan.synthesis_device(d_alm)  # warmup
for rep in range(3):
    t = {}
    plan.synthesis_device(d_alm, timings=t)
    total = sum(t.values())
    parts = "  ".join(f"{k}={v*1e3:7.2f}ms" for k, v in t.items())
    print(f"total={total*1e3:8.2f}ms  {parts}")
