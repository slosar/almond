# Almond — GPU-accelerated SHT library for HEALPix

## What this is

`Almond` is a prototype GPU (CUDA) spherical-harmonic-transform library for
HEALPix maps, built to replace/beat **ducc0** (the CPU engine behind healpy)
as SiMaster's SHT backend. It implements ducc0's exact algorithm (libsharp2
lineage) on the GPU: float64 only, healpy conventions, matrix-free.

Current scope (v0.4): **spin-0 and spin-2, synthesis and adjoint synthesis**
(exact Yᵀ), grid-batched over columns, plus `AlmondRealSHT`
(almond/simaster.py) — a validated drop-in for `simaster.sht.RealSHT` (real
basis, cut sky, batched columns).  Reference implementation for correctness:
ducc0 (accuracy gate: max abs rel error < 1e-10 on random alm; typically
1e-12).  As of v0.4 the adjoints run at parity with synthesis (the v0.2/0.3
1.6–1.9× adjoint gap is gone) and batched per-column throughput at small
nside is ~4× the loop-of-singles (see SUMMARY 2026-07-04).

- Parent project context: `/global/u1/a/anze/PowerSpec/CLAUDE.md`
  (SiMaster QML estimator; conda env `simaster`).
- Benchmarks that motivated this: `/global/u1/a/anze/PowerSpec/SHT_benchmark/`.
- Math + design document: `report/report.tex` (build with
  `module load texlive; pdflatex report.tex`).
- Progress log / results: `SUMMARY.md`.

## Environment

Same `simaster` conda env as the parent project, plus `cupy-cuda12x`
(installed 2026-07; CuPy reuses the `nvidia-*-cu12` pip wheels that
`jax[cuda12]` brought in — do NOT add another CUDA toolkit):

```bash
source /global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1/etc/profile.d/conda.sh
conda activate simaster
```

The login node has an A100-PCIE-40GB usable for development/tests; clean
benchmark numbers must come from a dedicated node
(`salloc -q debug -C gpu -A m4895 -t 30 -N 1`).

## Layout

```
almond/
├── almond/                 ← python package (import almond)
│   ├── geometry.py      ← native HEALPix RING ring/pair geometry, mlim
│   ├── reference.py     ← pure-NumPy port of ducc0's algorithm (the executable spec)
│   ├── kernels.cu       ← CUDA kernels (NVRTC-compiled via cupy.RawModule)
│   └── plan.py          ← SynthesisPlan: tables + buffers + kernel orchestration
├── tests/
│   ├── test_reference.py  ← NumPy reference vs ducc0
│   └── test_gpu.py        ← GPU vs ducc0 (+ vs reference); -m slow for nside≥512
├── bench/
│   ├── bench_synthesis.py ← accuracy gate + timing vs ducc0 (JSON out)
│   └── run_debug.sh       ← SLURM debug-queue benchmark job
├── results/             ← benchmark JSONs
├── report/              ← LaTeX design/math document
└── external/ducc/       ← ducc0 source clone (reading reference only, not built)
```

## Running

```bash
# tests (login node OK; ~1 min)
python tests/test_reference.py
python tests/test_gpu.py                       # fast set
python -m pytest tests/test_gpu.py -m slow     # nside 512/1024

# benchmark one nside
python bench/bench_synthesis.py --nside 1024

# full benchmark on a dedicated GPU node
cd bench && sbatch run_debug.sh
```

## API

```python
import almond
plan = almond.SynthesisPlan(nside=1024, lmax=3071)   # precomputes tables/buffers
m   = plan.synthesis(alm_np)          # numpy complex128 healpy-layout alm -> numpy map
m_d = plan.synthesis_device(alm_cp)   # cupy in -> cupy out, no host copies
plan.memory_bytes()                    # persistent device buffer footprint
```

## Design in one paragraph

Synthesis is split into a Legendre stage and an FFT stage. The Legendre
kernel assigns one thread per (ring-pair, m); it runs ducc0's
every-other-l recursion in cos²θ (the odd-l coefficients are pre-folded into
the even-l stream by a separate tiny kernel, so the inner loop is 6 FMAs per
2 l's), with 2^±800 scale tracking to survive sin^m θ underflow near the
poles and ducc's mlim cutoff to skip negligible (ring, m) pairs. The result
F_m(ring) is aliased onto each ring's FFT bins (m mod nφ, with exact-rational
sincospi twiddles for e^{imφ0}) because nφ < 2lmax+1 on every HEALPix ring at
lmax = 3nside−1. The equatorial belt (2nside+1 rings, all nφ = 4nside) then
goes through one batched cuFFT inverse transform; the polar caps (nφ = 4i,
all different) use a direct Hermitian DFT kernel. See report/ for the math
and the memory/flop budget.

## API additions (v0.2)

```python
plan = almond.SynthesisPlan(nside, lmax)             # spin 0
a    = plan.adjoint(map_np)                       # exact Y^T (ducc convention)
plan2 = almond.SynthesisPlan(nside, lmax, spin=2)    # spin 2: (2,nalm)<->(2,npix)
qu   = plan2.synthesis(np.stack([aE, aB])); eb = plan2.adjoint(qu)
maps = plan.synthesis_device_batch(alm_B)         # (B,nalm)->(B,npix)

from almond.simaster import AlmondRealSHT               # drop-in for RealSHT
op = AlmondRealSHT(nside, RealAlmIndex(0, lmax), 0, obs_pix)
y = op.synth(a_realbasis); at = op.adjoint(y)     # numpy or cupy, (ncol/nrow, B)
```

## SiMaster integration

`almond` is installed editable in the `simaster` env (`pip install -e almond/`).
SiMaster's `QMLWorkspace`/`CovModel` accept **`backend='almond'`** — the GPU
operators ride the same `pure_callback` path as 'ducc' (patch in
`simaster/covariance.py`). Equivalence validated via `apply_C` (spin-0 +
spin-2 cut-sky, 1e-10) in `tests/test_simaster.py`. val1/val2 reruns with
the new backend are still to be done (deliberately left to a
SiMaster-focused session).

## Developer notes — invariants & hard-won gotchas (read before editing kernels)

**Scaled arithmetic (the #1 bug source).** All near-pole quantities live as
`value * 2^(800*scale)` with |value| kept ≤ 2⁻⁶⁰ (FTOL) until "emergence".
Every normalization MUST be two-sided (clamp under- AND overflow of the
[2⁻⁴⁰⁰, 2⁴⁰⁰] band): two real bugs came from one-sided clamps — mypow
products overflowing to Inf (then an unkillable `while |x|>FTOL x*=2⁻⁸⁰⁰`
spin on Inf), and spin-2 prefac ratios underflowing to exactly 0 at m≳514.
After init normalization scale ∈ {0,1} (true λ ≤ O(1)); the skip loop only
handles scale<0; `isfinite` guards prevent GPU spins. Spin prefac must come
from cumulative √-products with per-step normalization, NOT lgamma
(5e-12 relative error at m~6000).

**Data layouts.**
- coef (a,b): index `moff[m]+il`; per-m table length `(lmax−m)//2+2`; only
  the first `nacc=(lmax−m)//2+1` entries are meaningful (rest garbage).
- AB (prefolded alm): `((off+il)*nchunk + c)*2 + {A,B}`; single path is
  nchunk=1. α is reconstructed as `sign4(il)*sqrt(|a|)`, sign period 4
  (+,+,−,−) via `(il&2)`.
- Adjoint Ĝ half-spectrum: `hstart = cumsum(nphi//2+1)`; belt rows
  contiguous. d_AB is REUSED as ABadj, d_GC as GCadj (synthesis input
  clobbered by adjoint calls — fine, plans are stateless between calls).
- Spin tables: fx (a,b) and walpha=norm_l·α at `soff[m]+(l−m)`, l=m..lmax+1.

**Geometry facts the kernels rely on.**
- pair index runs pole→equator so mlim is nondecreasing → `pstart[m]` via
  searchsorted. Equator ring self-paired: south contribution suppressed
  (Fs=0) in BOTH directions.
- φ0 = π·q/d with small integers; ALL twiddles/chirps use integer mod
  (2d or 2n) then sincospi — exact argument reduction; j·k products fit
  int64 comfortably at nside 4096.
- fold: F_m → bins `m mod n` and `(n−m) mod n` (conj); m=0 adds Re only
  (ducc discards Im a_l0); the k1==k2 collision case is self-consistent
  via the two atomicAdds. At lmax=3nside−1 folding matters on EVERY ring.

**Kernel design decisions (measured, don't re-litigate without data).**
- legendre2: TWO ring pairs per thread — the μ recursion is a serial
  2-FMA chain; one chain/thread is latency-bound (28% peak → 40%+).
  Same trick ported to spin-2 forward in v0.4 (`legendre_spin_2p`, 1.30×):
  the two pairs share the fx/GC loads.
- Chunked batch (CHUNK columns per thread) LOSES to looping singles:
  4·CHUNK accumulator registers (150 @ C=8) + per-column loads at 1:2
  load:FMA (vs 1:4 shared in legendre2). Default chunk=1.
  Grid-DIMENSION column batching (v0.4) is the one that wins: columns on
  blockIdx.z (fwd) / blockIdx.y (adj) with per-column plane strides —
  zero extra registers, fills the GPU at small nside (~4×/col at 128),
  memory-budgeted chunks (env ALMOND_BATCH_MEM, default 4 GiB), automatic
  fallback to looped singles when one column fills the budget.
- Adjoint (v0.4, `legendre_adj2`/`legendre_spin_adj2`): block-per-m owns
  ALL ring pairs → global adds need no atomics.  Per-il warp shuffles were
  >half the instruction stream (40 warp-instr/il) — replaced by staged bulk
  reduction: lanes store chain-summed contributions to per-warp shared
  tiles, bulk-reduced every ADJ2_TR=8 ils (one lane per output), combined
  across warps via a parity-double-buffered tile (no atomics, one
  syncthreads per segment).  Guarded/fast loop split: selects + rescale
  guards drop out once every chain of the lane has surfaced.  ADJ2_PPT=4
  spin-0 / SADJ2_PPT=2 spin-2; blockDim shrinks so blockDim×PPT ≈ npair
  (fixed 256 wastes 4× FMA on zero-chains at nside 128); TR=4/16, bd=512,
  launch_bounds occupancy-forcing all measured worse (GRAVEYARD §9 in
  PowerSpec/GRAVEYARD.md).  The old shuffle kernels are kept as
  `legendre_adj`/`legendre_spin_adj` (env ALMOND_ADJ_IMPL=v1 to A/B).
- Cap rings: i<64 direct Hermitian DFT (incremental rotation, exact
  reseed every RESEED=128 steps); i≥64 Bluestein classes i∈[2^j,2^{j+1})
  sharing M=2^{j+4}, one batched cuFFT pair per class; B̂ built at plan
  time (fwd, chirp sign −1), B̂_adj lazily (sign +1).
- fold/unfold FUSED into Legendre kernels (v0.3): no phase array in
  single-transform paths; only the experimental chunk>1 path has one.
- ducc-ntrans batch mode on 64 cores ≈ its intra-transform threading
  (host memory-BW bound) — the ~6× GPU margin holds either way.
- Timing on the login node is contention-prone: use `min` of several warm
  reps on an idle GPU and re-verify surprising wins (a bd=512 "win" was
  pure contention).

**Conventions.** healpy triangular alm, mstart(m)=m(2lmax+1−m)/2; ducc's
adjoint = transpose under the m>0-doubled inner product (no factor 2 in
the unfold; Im a_l0 dropped both ways). Spin-2: input (aE,aB), output
(Q,U); l<2 modes must be zero. Atomics make results call-to-call
nondeterministic at ~1e-15 (tested bound 1e-13).

**Infrastructure.** NVRTC module cache keyed (device, CHUNK); grid.y limit
65535 caps mmax (fine ≤ 65534); cuFFT plans cached by shape inside cupy;
first call per shape pays plan creation. Python hex-float literals don't
exist (use 2.0**800); nvrtc rejects --use_fast_math=false (use no flag).
Debug trick that found every kernel bug: instrumented kernel variant with
loop-guard counters writing (loop id, thread coords) to a flag buffer;
plus single-mode/m-range bisection against ducc.

## Known limitations / next steps

- **SiMaster callback boundary fixed in v0.5/SiMaster 0.2.** DLPack shares
  JAX/CuPy allocations at solve entry/exit; covariance, preconditioner, and
  batched PCG remain in CuPy between them. At nside=128/lmax=383/B=4 this made
  one covariance apply 14.2x faster and a converged 20-iteration solve 6.43x
  faster. Observed-pixel indexing was already on-device and is now an explicit
  tested API contract.
- `inverse*` is a CGLS synthesis pseudoinverse, distinct from `adjoint*`.
  At nside=128/lmax=191 it agrees with ducc `pseudo_analysis` below 2e-13 and
  is 7.5--8.1x faster. The lmax=3*nside-1 inverse is intrinsically poorly
  conditioned in both libraries; inspect `return_info`/increase `maxiter`.

- fold/unfold are fused into the Legendre kernels (v0.3) — no phase array
  in single-transform paths; the chunked batch path (chunk>1) still uses
  one, and remains slower than looping singles (kept for experiments).
- The restructured adjoints (v0.4) run at ~30–40% of A100 fp64 ideal; the
  next ceiling-raisers would be fp64 tensor cores (DMMA) with λ tiles staged
  in shared memory (applies to synthesis too), mask-aware ring/pair skipping
  for cut-sky SiMaster workloads (~(1−fsky) of both stages, both
  directions), and a precomputed-λ cuBLAS GEMM path for nside ≤ 256.
- Grid batching gains nothing at nside ≥ 256 (GPU already full per
  transform) — per-column cost there is kernel-bound, not occupancy-bound.
- val1/val2 QML validation with backend='almond': NOT yet run.
