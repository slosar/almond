# Almond

[![tests](https://github.com/slosar/almond/actions/workflows/tests.yml/badge.svg)](https://github.com/slosar/almond/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-CuPy%2012.x-76b900.svg)](https://cupy.dev/)

**GPU-accelerated spherical harmonic transforms for HEALPix maps** — a
matrix-free, float64, healpy-convention SHT library that reproduces
[ducc0](https://gitlab.mpcdf.mpg.de/mtr/ducc)'s exact algorithm on the GPU and
runs several times faster than 64-thread ducc0 on the CPU.

Almond was built as the SHT backend for
[SiMaster](https://github.com/slosar/SiMaster), a GPU quadratic
maximum-likelihood power spectrum estimator, where nside-2048 transforms
dominate the cost. It works standalone for any HEALPix synthesis / adjoint
workload.

## Highlights

- **Spin-0 and spin-2**, both **synthesis** (`alm → map`) and **exact adjoint
  synthesis** (`map → alm`, i.e. Yᵀ). No dense matrices, no pixel covariance.
- **float64 throughout**, healpy triangular-`alm` conventions, validated
  against ducc0 to ~1e-12 on random `alm` (accuracy gate: max relative error
  < 1e-10).
- **Grid-batched columns** for the many-transform regime (e.g. CG / Monte
  Carlo), plus `AlmondRealSHT`, a drop-in for SiMaster's real-basis cut-sky
  `RealSHT`.
- **Pure CuPy + NVRTC** — no build system; kernels compile at import.

## Performance

Dedicated A100-SXM vs ducc0 on a 64-core EPYC (device-resident, single
transform):

| nside | synth spin-0 | adjoint spin-0 | synth spin-2 | adjoint spin-2 |
|------:|-------------:|---------------:|-------------:|---------------:|
|   512 |  2.0 ms (6.9×) |  2.3 ms (5.9×) |  6.9 ms (4.4×) |  7.1 ms (4.1×) |
|  1024 | 10.2 ms (8.2×) | 11.9 ms (6.5×) | 37.3 ms (4.4×) | 48.3 ms (3.1×) |
|  2048 | 72.6 ms (6.9×) | 80.7 ms (5.7×) | 270.6 ms (3.6×) | 349.8 ms (2.6×) |

Batched (many columns per call) vs ducc0's `ntrans` mode, per column — at
nside 128 (a typical estimator scale) Almond is **~10× faster spin-0 and ~7×
faster spin-2**, in both directions. Numbers in parentheses are the speedup
over ducc0; full tables and methodology are in [`SUMMARY.md`](SUMMARY.md).

## Install

Almond needs an NVIDIA GPU with CUDA 12.x and CuPy. In an environment that
already has a working CuPy (e.g. one set up for JAX/CUDA 12):

```bash
git clone https://github.com/slosar/almond
cd almond
pip install -e .          # pulls cupy-cuda12x + numpy
```

Requirements: Python ≥ 3.10, `numpy`, `cupy-cuda12x`. Running the test suite
additionally needs `ducc0` (the correctness reference) and `pytest`.

## Quickstart

```python
import numpy as np
import almond

# spin-0 synthesis: healpy-layout complex128 alm -> HEALPix RING map
plan = almond.SynthesisPlan(nside=1024, lmax=3071)
m = plan.synthesis(alm)                  # numpy in -> numpy out
a = plan.adjoint(m)                      # exact Y^T (ducc convention)

# stay on the device (no host copies)
m_d = plan.synthesis_device(alm_cupy)

# spin-2: (aE, aB) <-> (Q, U); l < 2 modes must be zero
plan2 = almond.SynthesisPlan(nside=1024, lmax=3071, spin=2)
qu = plan2.synthesis(np.stack([aE, aB]))
eb = plan2.adjoint(qu)

# batched columns (B transforms at once): (B, nalm) -> (B, npix)
maps = plan.synthesis_device_batch(alm_batch)

# drop-in for SiMaster's real-basis cut-sky operator
from almond.simaster import AlmondRealSHT
op = AlmondRealSHT(nside, index, spin=0, obs_pix=obs_pix)
y = op.synth(a_realbasis)
at = op.adjoint(y)
```

## How it works

Synthesis is split into a **Legendre stage** and an **FFT stage**. The
Legendre kernel assigns one thread per `(ring-pair, m)` and runs ducc0's
every-other-`l` recursion in `cos²θ`, with 2^±800 scale tracking to survive
`sinᵐθ` underflow near the poles and ducc's `mlim` cutoff to skip negligible
modes. The resulting `Fₘ(ring)` is aliased onto each ring's FFT bins (folding
is universal at `lmax = 3·nside − 1`); the equatorial belt goes through one
batched cuFFT, and the polar caps use a direct Hermitian DFT / Bluestein path.
The adjoint mirrors this with a block-per-`m`, atomics-free staged reduction.

The NumPy port in [`almond/reference.py`](almond/reference.py) is an
executable, line-by-line spec of the same algorithm, validated against ducc0
to ~1e-14; the GPU kernels are validated against it and against ducc0
directly. The math, design, and memory/flop budget are written up in
[`report/report.pdf`](report/report.pdf).

## Tests & CI

The suite is in [`tests/`](tests/):

| File | Needs | Runs in CI |
|---|---|---|
| `test_reference.py` | numpy, ducc0 (CPU only) | ✅ yes |
| `test_gpu.py` | CuPy + a CUDA GPU | on GPU hardware |
| `test_simaster.py` | CuPy + SiMaster | on GPU hardware |

GitHub Actions has no GPU, so CI runs the **CPU reference suite** — the NumPy
reference validated against ducc0, which is the convention bridge the GPU
kernels are checked against. The GPU tests `importorskip("cupy")` and are run
on CUDA hardware:

```bash
# CPU reference tests (what CI runs)
pytest tests/test_reference.py -v

# GPU tests (require an NVIDIA GPU)
pytest tests/test_gpu.py -v            # fast set (nside ≤ 256)
pytest tests/test_gpu.py -v -m slow    # nside 512 / 1024
```

## Layout

```
almond/
├── almond/
│   ├── geometry.py        native HEALPix RING ring/pair geometry
│   ├── reference.py       pure-NumPy port of ducc0's algorithm (the spec)
│   ├── reference_spin.py  spin-2 NumPy reference
│   ├── kernels.cu         spin-0 CUDA kernels (NVRTC via CuPy RawModule)
│   ├── kernels_spin.cu    spin-2 CUDA kernels
│   ├── plan.py            SynthesisPlan: tables, buffers, kernel orchestration
│   └── simaster.py        AlmondRealSHT drop-in
├── tests/                 reference (CPU) + GPU validation vs ducc0
├── bench/                 accuracy gates + timing vs ducc0
├── report/                LaTeX design/math document (report.pdf)
└── .github/workflows/     CI (reference tests)
```

## Status

v0.4. Spin-0/2 synthesis and exact adjoint are complete and validated;
adjoints run at parity with synthesis and batched throughput is tuned for the
estimator regime. Known ceiling-raisers not yet done: fp64 tensor-core (DMMA)
Legendre tiles, mask-aware ring/pair skipping for cut-sky, and a precomputed-λ
GEMM path for nside ≤ 256. See [`SUMMARY.md`](SUMMARY.md) for the full progress
log and benchmark tables.

## References

- **ducc0** — the CPU SHT engine behind healpy and the correctness reference
  for Almond ([ducc0](https://gitlab.mpcdf.mpg.de/mtr/ducc)).
- **SiMaster** — the QML power spectrum estimator Almond was built for.
- `report/report.pdf` — algorithm derivation, design decisions, and the
  memory/flop budget.
