# Almond — progress log

Goal: GPU SHT library for HEALPix, faster than ducc0 (the reference
implementation), plugging into SiMaster. Start: spin-0 synthesis only.
See `CLAUDE.md` for orientation, `report/report.tex` for math + design.

## 2026-07-03 — day 1: algorithm study, reference impl, GPU prototype

**Context.** `SHT_benchmark` concluded ducc0-CPU wins at every measured
nside (s2fft-GPU never crosses over; cuHPX fails accuracy). But its ducc
timings are suspicious: identical at 1 and 128 threads (e.g. 0.49 s at
nside=512, 21 s at nside=2048 — consistent with ~1 effective core). The
fair CPU target is therefore substantially faster than the benchmark
table; we re-measure ducc on a dedicated node ourselves.

**Algorithm study.** Cloned ducc0 source (`external/ducc`, github mirror).
Extracted the complete spin-0 synthesis algorithm from
`src/ducc0/sht/sht_inner_loop.h` + `sht.cc`:

- per-m, per-ring-pair Legendre stage with a Δl=2 recursion in cos²θ
  (μ_{i+1} = (a_i cos²θ + b_i) μ_i + μ_{i-1}); odd-l coefficients
  pre-folded into the even stream (A_i, B_i) so one μ sequence yields both
  the symmetric (p1) and antisymmetric (cosθ·p2) parts:
  north = p1 + cosθ p2, south = p1 − cosθ p2. ~3 flops per (pair, m, l).
- Verified numerically the invariant μ_i = λ_{m+2i+1}/(α_i cosθ) and the
  α sign pattern (+,+,−,−) with |α_i| = √|a_i| (lets us drop the α table).
- 2^±800 scale tracking for sin^m θ underflow (mypow + rescale);
  mlim(θ) cutoff rule; aliasing fold of F_m onto ring FFT bins
  (m mod nφ, conjugate at −m mod nφ, e^{imφ0} twiddle); Im(a_l0) ignored.
- Key realization: at lmax = 3nside−1, nφ < 2lmax+1 on *every* ring
  (belt included) → folding is universal, not a cap-only detail.

**NumPy reference (`almond/reference.py`).** Line-by-line port; validated vs
ducc0 first try: max rel err ~1e-14 at nside 8–32 (incl. non-real a_l0 and
lmax≠3nside−1 cases). Geometry (`almond/geometry.py`) computed natively from
the exact HEALPix rationals; matches `Healpix_Base.sht_info()` to 1e-14.

**GPU prototype (`almond/kernels.cu` + `almond/plan.py`).** CuPy RawModule
(NVRTC), no build system. Five kernels: build_coef (plan-time), prefold,
legendre (1 thread per (ring-pair, m)), fold (atomicAdd into FFT bins,
exact-rational sincospi twiddles), cap_dft (direct Hermitian DFT for polar
caps) + batched cuFFT for the belt + belt_finish. Design decisions and
departures from CPU-ducc: report §"Design decisions".

**Validation.** `tests/test_gpu.py` vs ducc0, random alm:
nside 8→256 all pass at <1e-10 on the login-node A100 (first-compile
correct; only harness bugs fixed: hex float literals, nvrtc flag, one
missing scale>0 unscale for m=0). nside 512/1024 pass after the mypow fix
below.

**Bug of the day (mechanism worth remembering).** At nside=1024 the
legendre kernel appeared to hang (GPU 100%, 18+ min). Bisection with an
instrumented kernel (loop-guard flags) pinpointed pair=121, m=398: my CUDA
port of ducc's `mypow` (binary exponentiation of sin^m θ with 2^±800 scale
tracking) only normalized against *underflow* — reasoning "x<1, values only
shrink". Wrong: after an underflow bump v ∈ [1, 2^400), so subsequent
products can exceed 2^800 → Inf; the init loop `while(|lam2|>2^-60)
lam2*=2^-800` then spins forever on Inf. ducc/the NumPy reference normalize
both directions (which is why the reference passed everywhere). Fixes:
two-sided normalization in mypow + an `isfinite` guard so the kernel can
never spin again. Lesson: when porting scaled-arithmetic code, port the
*invariant* (v ∈ [2^-400, 2^400)), not the direction you think values move.
Debug trick that worked: append an instrumented kernel variant with
per-loop guard counters writing (loop id, thread coords) to a flag buffer —
found the exact loop + thread in one 1 ms run.

**Pilot numbers (login-node A100-PCIe, shared/contended — indicative only).**
Accuracy vs ducc0: 8e-13 (nside 512), 5.6e-12 (nside 2048). Timings
(device-resident / incl. host copies vs ducc0@16t on the same busy node):

| nside | Almond dev | Almond host | ducc 16t | ratio (dev) |
|---|---|---|---|---|
| 512  | 4.9 ms | 12 ms  | 41 ms  | 8.5× |
| 2048 | 190 ms | 588 ms | 1.57 s | 8.3× |

Confirms the old SHT_benchmark ducc numbers (21 s at nside 2048) were
effectively single-threaded — real 16-thread ducc is ~1.6 s there.
Stage split at 2048 (first call): legendre 173 ms, cap_dft 110 ms,
fold 5 ms, belt ifft 17 ms → cap_dft is the optimization target as
predicted; legendre runs at ~1.5 Tflop/s (~16% of A100 fp64 peak).
Memory at 2048: plan buffers 2.47 GB, peak pool 3.84 GB.

**Dedicated-node baseline (job 55426428, A100 40GB + 64-core Milan,
exclusive).** Accuracy 1.9e-13 → 5.6e-12 (nside 128 → 2048). ducc0 scales
properly there (2048: 19.2 s @1t → 0.474 s @64t). Almond v1 device-resident
beats *best-threaded* ducc at every nside: 3.3×/4.1×/3.8×/3.2×/2.6× at
nside 128/256/512/1024/2048 (host-inclusive: 1.3–1.9×).

**Optimization round 1 (same day).**
1. *Bluestein caps.* The direct cap DFT was 110 of 179 ms at 2048.
   Replaced (for cap rings i ≥ 64) by Bluestein through batched cuFFT:
   power-of-two size classes i ∈ [2^j, 2^{j+1}) share padded length
   M = 2^{j+4}; chirp FFTs (B̂, 715 MB at 2048) precomputed at plan build;
   chirps exact via integer-reduced sincospi. Cap stage: 110 → 7.6 ms.
2. *Two ring-pairs per thread (legendre2).* The μ recursion is a serial
   2-FMA chain per step; one chain/thread can't hide FMA latency (28% of
   fp64 peak; block-size sweep flat). Two adjacent pairs per thread double
   the independent chains and amortize coefficient loads — ducc's
   SIMD-over-rings, GPU edition. Legendre stage: 95 → 65 ms (66 regs,
   no spills). Solo-advance the earlier-surfacing pair, then fused loop.

Stage split at 2048 after both (login node): legendre 65, bluestein 7.6,
fold 4.9, belt fft 3.3, prefold 0.6, small-caps 0.06 → total ~81 ms.
All 23 tests still pass (1e-12-ish vs ducc everywhere).

**Final v0.1 numbers (job 55427184, dedicated A100-SXM 40GB +
64-core EPYC 7763, exclusive).** Single spin-0 synthesis, B=1,
lmax = 3nside−1, accuracy gate + median of warm calls:

| nside | acc vs ducc | Almond dev | Almond +copies | ducc 1t | ducc best (64t) | speedup dev / host |
|---|---|---|---|---|---|---|
| 128  | 1.9e-13 | 0.3 ms  | 0.6 ms  | 9 ms    | 1.4 ms  | 2.6× / 1.2× |
| 256  | 4.4e-13 | 0.5 ms  | 1.4 ms  | 53 ms   | 2.4 ms  | 4.6× / 1.6× |
| 512  | 8.3e-13 | 2.3 ms  | 6.0 ms  | 353 ms  | 13 ms   | 5.7× / 2.2× |
| 1024 | 2.8e-12 | 13.3 ms | 29 ms   | 2.55 s  | 81 ms   | 6.1× / 2.8× |
| 2048 | 5.6e-12 | 77 ms   | 156 ms  | 19.1 s  | 479 ms  | 6.2× / 3.1× |

Bottom line: **Almond v0.1 beats best-threaded (64-core) ducc0 at every
nside; 6.2× at nside 2048** (250× vs 1-thread), 653 Mpix/s whole-map.
No CPU→GPU crossover to wait for. Including PCIe copies the win is
1.2–3.1× — keep data device-resident (SiMaster's CG does). Peak device
memory 5.0 GB at 2048 (report §Memory). Legendre stage runs at ~4 Tflop/s
(~40% of A100 fp64 peak). Full table + analysis: report/report.pdf §6.

## 2026-07-03 (later) — batched throughput vs ducc's ntrans mode

User correction: ducc0 threads across the *batch* (`alm` shape
(ntrans, 1, nalm)) with near-linear scaling — the fair high-throughput
comparison for QML is per-column time at large B, not single transforms
(where ducc's internal threading saturates ~40× worse than its batch
mode: 426 → 41.6 ms/col with 16 threads at nside 512 on the login node).

**Batched Almond implemented** (`synthesis_device_batch`, (B,nalm)→(B,npix)):
chunked pipeline with a CHUNK-column `legendre_batch` kernel (one μ
recursion feeding 4·CHUNK accumulators), batched fold/belt-FFT/Bluestein.

**Negative result worth keeping (mechanism).** The chunked Legendre
kernel *loses* to looping the single-transform pipeline: at nside 1024,
sequential singles 11.7 ms/col vs chunked 14.4 (C=2) / 16.2 (C=4) /
16.7 (C=8). Two mechanisms: (a) registers — 4·CHUNK accumulators push the
kernel to 150 regs at C=8 (occupancy collapse; 64 regs even at C=2);
(b) L1 load pressure — per il each column needs its own (A,B) loads
(2 loads / 4 FMA = 1:2), whereas legendre2's two ring-pair chains *share*
loads (1:4). The recursion being amortized only ever saved 4 of 12
flops/il, and legendre2 already runs near its latency/load bound, so
there was little to win. Default is now chunk=1 (batch API loops the
single pipeline, GPU stays async); the chunked path is kept for
experiments (`chunk>1`). Batching *does* win at small nside where fixed
overheads dominate... except measured: sequential also wins at 256
(0.56 vs 0.57 ms/col). Ring pairs > column chunks, on GPUs as in SIMD.

Memory hygiene fixes from OOM debugging: batch buffers share the
single-path phase/G (views), remainder buffers lazy, belt/Bluestein
transients del'ed promptly; host OOM on login was the ducc reference at
(16, 1, nalm) — dedicated nodes have 256 GB.

**Final batched numbers (job 55430523, dedicated A100 + 64-core EPYC).**
Per-column, Almond batched (looped single pipeline) vs ducc0 ntrans @64t:

| nside | B | acc | Almond ms/col | +copies | ducc-ntrans-64t | speedup dev / host |
|---|---|---|---|---|---|---|
| 256  | 128 | 6.0e-13 | 0.39  | 1.55  | 2.11  | 5.4× / 1.4× |
| 512  | 64  | 1.3e-12 | 1.98  | 6.52  | 12.0  | 6.1× / 1.8× |
| 1024 | 64  | 3.7e-12 | 11.3  | 31.1  | 68.9  | 6.1× / 2.2× |
| 2048 | 16  | 7.4e-12 | 77.0  | 150   | 461   | 6.0× / 3.1× |

Key insight: on a full 64-core node ducc's ntrans batch mode is barely
faster per column than its intra-transform threading (461 vs 479 ms/col
at 2048) — 64 concurrent CPU transforms hit the host memory-bandwidth
wall, not the core count. So **the ~6× Almond advantage holds unchanged in
the high-throughput batched regime**, uniformly over nside 256–2048.
(ducc's batch mode only shines at low thread counts, where it's ~40×
better than intra-transform threading.)

## 2026-07-03 (v0.2) — adjoint, spin-2, SiMaster backend

**Adjoint synthesis (Yᵀ, map→alm).** Exact transpose in ducc's convention
(Ĝ = forward ring DFT → F'_m = e^{-imφ0}Ĝ_{m mod n} → Legendre adjoint →
postfold gather). NumPy reference validated vs ducc first try; GPU kernels:
rfft belt / Bluestein-adjoint caps (sign-flipped chirps) / unfold /
`legendre_adj` (block-per-m owns all ring pairs → no atomics; warp-shuffle
reduction into shared il-tiles, 4 pairs/lane) / postfold. All ducc-validated
to <1e-10 at nside 8–1024 + transpose identity at 1e-12. Login timings:
4.7 / 21 / 147 ms at nside 512/1024/2048 (~1.5–1.9× synthesis — the
shuffle-reduction overhead).

**Spin-2 (native ducc recursion).** ducc's fast path (spin-2 via spin-0 +
1/sin²θ weights) is inexact near poles (ducc itself falls back for
sθ<0.01), so Almond ports the native sxdata path: two scaled Wigner-d chains,
Δl=1 recursion in cosθ, 8 accumulators, parity-interleaved E/B, half-angle
prefactor powers with 2^800 scaling. Reference matched ducc to 1e-14 first
try at lmax≤95. Two bugs found by systematic bisection at higher lmax:
1. *prefac underflow at m≳514*: my prefac normalization was one-sided;
   the fac-table ratios can underflow to exactly 0 (2^-1100). ducc's
   normalize is two-sided. (Same lesson as the mypow bug: port the
   invariant band, not the direction you assume values move.)
2. *adjoint transpose algebra*: used forward coefficients instead of their
   conjugates on the ±i cross terms (real-linear transpose of complex
   coefficients: y += c·x ⟹ x' += conj(c)·y).
All spin-2 tests pass vs ducc (synthesis + adjoint, nside 8–512, <1e-10).
Login timings spin-2: synth 52/380 ms, adjoint 87/659 ms at 1024/2048.
Stage 2 (fold/FFT) is reused verbatim with the component axis playing the
batch-chunk role.

**SiMaster backend (`almond/simaster.py`).** `AlmondRealSHT` is a drop-in for
`simaster.sht.RealSHT`: same (ncol,B)↔(nrow,B) real-basis interface,
real↔healpy conversion + obs_pix subsetting on the GPU, numpy or cupy
in/out. Validated against RealSHT (spin 0 and 2, cut sky, batched) to
1e-10 + transpose identity. To use in SiMaster: construct fields as usual
and swap RealSHT → AlmondRealSHT where the workspace builds its operators.

**Test suite: 66 tests, all passing** (reference vs ducc; GPU vs ducc
spin-0/2, synth+adjoint, nside 8–1024; batched; SiMaster drop-in).

## 2026-07-03 (v0.3) — fusion, tuning, SiMaster wiring

**Fold/unfold fused into the Legendre kernels.** Forward kernels
(legendre2, legendre_spin) now scatter twiddled F_m straight onto the ring
FFT bins at extract (`fold_scatter` device helper, atomicAdd); adjoint
kernels gather F' from the half-spectrum Ĝ on the fly (`unfold_gather`).
The phase array no longer exists in any single-transform path (−805 MB
spin-0 / −1.6 GB spin-2 at nside 2048, plus one full memory pass each
direction). The chunked batch path (chunk>1, experimental) still uses
phase_b. Timings unchanged (fold was ~6 ms of 77) — this was a memory play.
All 66 tests pass unchanged.

**Spin-2 adjoint tuning.** SADJ_PPT 2→4 (more chains per lane, half the
shuffle reductions): 664→596 ms at 2048 login (168 regs — occupancy-limited;
deeper gains need a different reduction, noted in report roadmap).

**SiMaster wiring.** `almond` is now a proper editable package
(`pip install -e almond/`, version 0.2.0). `simaster/covariance.py` gained
`backend='almond'` — AlmondRealSHT rides the same pure_callback path as 'ducc'.
Equivalence test: CovModel.apply_C('almond') == apply_C('ducc') to 1e-10 on a
cut sky with mixed spin-0+spin-2 fields (almond/tests/test_simaster.py).
SiMaster's own CI suite (42 tests) still passes. **val1/val2 reruns with
backend='almond' intentionally left for the SiMaster-focused agent.**

**Final v0.3 benchmark (job 55449334, dedicated A100-SXM + 64-core EPYC,
after the almond rename).** Almond ms (speedup vs ducc0 @64t),
device-resident; spin-2 accuracy 3e-12 → 2e-11 (nside 512 → 2048):

| nside | synth spin0 | adjoint spin0 | synth spin2 | adjoint spin2 |
|---|---|---|---|---|
| 512  | 2.0 (6.3×)  | 3.7 (3.3×)   | 9.0 (3.3×)   | 14.5 (1.8×) |
| 1024 | 12.1 (6.5×) | 25.0 (3.2×)  | 49.4 (3.0×)  | 80.5 (1.7×) |
| 2048 | 71.9 (6.5×) | 139.7 (3.2×) | 363.5 (2.6×) | 587.4 (1.6×) |

Adjoints trail synthesis because ducc's adjoint ≈ its synthesis cost while
Almond's reductions pay shuffle overhead (the documented tuning target).
Every mode SiMaster's QML filter needs (Y and Yᵀ, spin 0 and 2) is faster
than 64-core ducc0 at every nside. Full table in report §8.

## 2026-07-04 (v0.4) — adjoint restructure, spin-2 two-pair, grid-batched columns

Focus (user priority): **batched per-column throughput** (SiMaster CG regime)
and the adjoint gap.  All numbers below: login-node A100-PCIe, idle, min of
warm calls; dedicated-node job 55476730 re-measures for the record.

**1. Adjoint Legendre restructure (`legendre_adj2` / `legendre_spin_adj2`).**
The v0.2 kernels spent >half their instruction stream on the per-il warp
shuffle reduction (5 butterfly steps × 4 values × (shfl+add) ≈ 40 warp-instr
vs ~32 of work).  v2: each lane stores its chain-summed contributions to a
per-warp shared tile; every TR=8 ils the warp bulk-reduces its own tile (one
lane per output, conflict-free padded rows), writes a parity-double-buffered
cross-warp tile (no atomics), and one `__syncthreads` per segment combines +
flushes to global.  Plus a guarded/fast loop split: once every chain of a
lane has surfaced (scale==0), the select + two-sided-rescale guards drop out
of the inner loop entirely (the underwater phase keeps the exact v1 ducc
semantics).  Ablation that guided this: compute-only 62 ms, +stores 62.5 ms
(stores are free), +reduce 105 ms → the old reduce cost ~40% and it was
barriers/shuffles, not arithmetic.  Sweeps: PPT=4/TR=8/bd=256 best for
spin-0; PPT=2 for spin-2; bd=512 and TR=4/16 lose (GRAVEYARD §9).
blockDim now shrinks so blockDim×PPT covers npair in one pass — at nside 128
the fixed bd=256 wasted 4× FMA work on inactive zero-chains.
Single-transform adjoints (ms): spin-0 4.4/20.9/149 → 2.0/12.5/86 at nside
512/1024/2048; spin-2 13.0/81.7/602 → 7.3/51.3/377.  **Adjoints now at
parity with synthesis in both spins** (was the documented 1.6–1.9× gap).

**2. Spin-2 forward two-pair kernel (`legendre_spin_2p`).**  The legendre2
trick ported to spin 2: two adjacent ring pairs per thread share the fx/GC
loads (3 double2 per l), four independent recursion chains, same guarded/fast
split.  1.30× uniformly: synth2 379→285 ms at 2048, 51→39 at 1024.

**3. Grid-dimension column batching (all four modes).**  Columns ride
blockIdx.z (forward) / blockIdx.y (adjoint) with per-column plane strides on
AB/GC/G/Ghat; prefold/postfold got the same strides; belt FFT, cap DFT and
Bluestein batch across columns through the existing (chunk-)plumbing.
Chunked by a device-memory budget (env `ALMOND_BATCH_MEM`, default 4 GiB);
falls back to looping singles when one column fills the budget (nside 2048).
This is NOT the losing per-thread CHUNK path (which needed 4·CHUNK registers)
— grid batching costs zero registers and simply fills the GPU at small nside
(a single nside-128 transform launches ~49K threads on a 221K-thread A100).
Per-column at nside 128: spin-0 synth 0.244→0.067 ms, adjoint 0.238→0.059 ms
(≈4×); spin-2 0.272→0.198 / 0.284→0.174 ms.  At nside ≥256 batch ≈ loop
(GPU already full per transform), as expected.

**4. `AlmondRealSHT` integration rewrite.**  The real↔healpy conversion is
now a precomputed pure gather (each healpy (l,m≥0) receives exactly one +m
and at most one −m real mode → two-term gather; the old `cp.add.at` scatter
is gone, and results are deterministic).  synth/adjoint feed whole (ncol,B)
batches through the grid-batched plan pipelines, chunked by the same memory
budget; obs-pixel subset/scatter is one indexed assignment per chunk.
Per-column at nside 128, B=64, fsky=0.7 (includes conversions + cut-sky):
spin-0 0.11 ms, spin-2 0.24 ms.

**5. Minor.**  Single-path spin-0 belt FFT now in-place (`overwrite_x`; the
belt region of G is dead after the transform).  `bench/bench_modes.py` +
`bench/bench_batched_modes.py` + `bench/run_v04.sh` added.

Failed variations recorded in `PowerSpec/GRAVEYARD.md` §9 (launch_bounds
occupancy forcing, bd=512, TR=4/16, `#pragma unroll 2`).  Login-node timing
lesson re-learned: an apparent bd=512 win at 1024 was GPU contention;
decisions need `min` of several reps on an idle device, re-checked.

**Tests: 66 fast (incl. 4 new grid-batch roundtrip tests) + 5 slow, all
passing** (ducc accuracy gates unchanged, 1e-13…2e-11); SiMaster CI suite
(42 tests) passes; `apply_C` almond-vs-ducc equivalence holds.  Version
bumped to 0.4.0.

**Final v0.4 numbers (job 55476730, dedicated A100-SXM + 64-core EPYC).**
Single transform, Almond ms (speedup vs ducc0 @64t), device-resident:

| nside | synth spin0 | adjoint spin0 | synth spin2 | adjoint spin2 |
|---|---|---|---|---|
| 512  | 2.0 (6.9×)  | 2.3 (5.9×)   | 6.9 (4.4×)   | 7.1 (4.1×)  |
| 1024 | 10.2 (8.2×) | 11.9 (6.5×)  | 37.3 (4.4×)  | 48.3 (3.1×) |
| 2048 | 72.6 (6.9×) | 80.7 (5.7×)  | 270.6 (3.6×) | 349.8 (2.6×) |

vs v0.3: adjoint spin-0 139.7→80.7 ms (3.2×→5.7× over ducc), spin-2
synthesis 363.5→270.6, spin-2 adjoint 587.4→349.8 (1.6×→2.6×) at 2048.

Batched per-column (grid batch vs ducc0 ntrans @64t), ms/col:

| nside | B | synth spin0 | adjoint spin0 | synth spin2 | adjoint spin2 |
|---|---|---|---|---|---|
| 128  | 64 | 0.07 (9.6×) | 0.07 (10.1×) | 0.20 (6.9×) | 0.20 (6.7×) |
| 256  | 64 | 0.36 (5.7×) | 0.32 (6.3×)  | 0.94 (4.5×) | 1.06 (3.8×) |
| 512  | 32 | 1.75 (6.6×) | 1.87 (5.9×)  | 5.61 (4.2×) | 6.96 (3.1×) |
| 1024 | 16 | 10.4 (6.6×) | 12.1 (5.7×)  | 37.2 (4.0×) | 48.1 (2.7×) |
| 2048 | 8  | 74.2 (6.2×) | 81.9 (5.4×)  | 270.7 (3.5×)| 349.9 (2.6×)|

At nside 128 (val2 scale) the batched win over ducc ntrans is now ~10×
spin-0 / ~7× spin-2, both directions.

### Open items / known costs

- cap_dft is O(nside³) with ~2× the Legendre constant → likely dominates
  at 2048; planned fix: common-length Bluestein through one batched cuFFT.
- fold+phase buffers cost 1.6 GB at nside 2048; planned fix: fuse fold
  into legendre kernel (also removes a full memory pass).
- Plan-time coef table build: trivial. Plan persistent buffers at
  nside=2048: ~2.5 GB (see report §Memory).
