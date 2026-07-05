// CUDA kernels for Almond spin-0 HEALPix synthesis (float64).
//
// Mirrors almond/reference.py one-to-one; see report/ for the math.
// Compiled at runtime with NVRTC through cupy.RawModule.
//
// Stage 0: build_coef   -- recursion coefficient table (a,b) per (m, il)   [plan build]
// Stage 1: prefold      -- fold healpy alm into (A,B) streams per (m, il)  [per call]
//          legendre     -- ring-pair Legendre recursion -> phase[m][ring]  [per call]
// Stage 2: fold         -- alias F_m onto ring FFT bins with e^{i m phi0}  [per call]
//          cap_dft      -- direct Hermitian DFT for polar-cap rings        [per call]
//          (belt rings go through batched cuFFT from Python)

#define FBIG      0x1p+800
#define FSMALL    0x1p-800
#define FBIGHALF  0x1p+400
#define FTOL      0x1p-60

// Number of alm columns processed per kernel launch (compile-time so the
// per-column accumulators live in registers).  The module is compiled once
// per CHUNK value; CHUNK=1 reproduces the single-transform layout exactly.
#ifndef CHUNK
#define CHUNK 1
#endif

extern "C" {

// ---------------------------------------------------------------------------
// eps_l = sqrt((l^2 - m^2) / (4 l^2 - 1))
// ---------------------------------------------------------------------------
__device__ __forceinline__ double eps_lm(double l, double m)
  {
  return sqrt((l * l - m * m) / (4.0 * l * l - 1.0));
  }

// fold-scatter: alias F_m of one ring onto its FFT bins with the e^{i m phi0}
// twiddle (exact integer-reduced sincospi).  Used fused inside the Legendre
// kernels so the phase array never materialises.
__device__ __forceinline__ void fold_scatter(
    const double2 F, const int r, const int m,
    const long long* __restrict__ ringstart,
    const int* __restrict__ nphi,
    const int* __restrict__ phi0_num,
    const int* __restrict__ phi0_den,
    double2* __restrict__ G)
  {
  const int n = nphi[r];
  const long long rs = ringstart[r];
  if (m == 0)
    {
    atomicAdd(&G[rs].x, F.x);   // Im F_0 discarded (ducc convention)
    return;
    }
  const long long num = (long long)m * (long long)phi0_num[r];
  const int den = phi0_den[r];
  double s, c;
  sincospi((double)(num % (2LL * den)) / (double)den, &s, &c);
  const double tr = F.x * c - F.y * s;
  const double ti = F.x * s + F.y * c;
  const int k1 = m % n;
  const int k2 = (n - k1) % n;
  atomicAdd(&G[rs + k1].x, tr);
  atomicAdd(&G[rs + k1].y, ti);
  atomicAdd(&G[rs + k2].x, tr);
  atomicAdd(&G[rs + k2].y, -ti);
  }

// unfold-gather: F'_m of one ring from the half-spectrum Ghat (adjoint
// direction), fused into the Legendre-adjoint kernels.
__device__ __forceinline__ double2 unfold_gather(
    const int r, const int m,
    const double2* __restrict__ Gh,
    const long long* __restrict__ hstart,
    const int* __restrict__ nphi,
    const int* __restrict__ phi0_num,
    const int* __restrict__ phi0_den)
  {
  const int n = nphi[r];
  const int k1 = m % n;
  const long long hs = hstart[r];
  double2 g = (k1 <= n / 2) ? Gh[hs + k1] : Gh[hs + n - k1];
  if (k1 > n / 2) g.y = -g.y;
  if (m == 0) return make_double2(g.x, 0.0);
  const long long num = (long long)m * (long long)phi0_num[r];
  const int den = phi0_den[r];
  double s, c;
  sincospi(-(double)(num % (2LL * den)) / (double)den, &s, &c);
  return make_double2(g.x * c - g.y * s, g.x * s + g.y * c);
  }

// exact Bluestein chirp e^{sgn * i pi t^2 / n}: t^2 reduced mod 2n in integers
__device__ __forceinline__ double2 chirp_w(const long long t, const int n,
                                           const double sgn)
  {
  const long long q = (t * t) % (2LL * n);
  double s, c;
  sincospi(sgn * (double)q / (double)n, &s, &c);
  return make_double2(c, s);
  }

// ---------------------------------------------------------------------------
// build_coef: one thread per m, sequential in il (plan build; not perf critical)
// coef[moff[m] + il] = (a_il, b_il);  l = m + 2 il
// alpha sign pattern is (+ + - -) with period 4 (see report), so the prefold
// kernel can reconstruct alpha_il = sign4(il) * sqrt(|a_il|) from this table.
// ---------------------------------------------------------------------------
__global__ void build_coef(const int lmax, const int mmax,
                           const long long* __restrict__ moff,
                           double2* __restrict__ coef)
  {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m > mmax) return;
  const double md = (double)m;
  const long long off = moff[m];
  const int nil = (int)(moff[m + 1] - off);

  double alpha_prev = 1.0 / eps_lm(md + 1.0, md);            // alpha[0]
  double alpha_cur = (nil > 1)
      ? eps_lm(md + 1.0, md) / (eps_lm(md + 2.0, md) * eps_lm(md + 3.0, md))
      : 0.0;                                                  // alpha[1]
  // coef[0]
    {
    double a = alpha_prev * alpha_prev;                       // sign4(0) = +, il=0 even
    double e1 = eps_lm(md + 1.0, md), e2 = eps_lm(md + 2.0, md);
    coef[off] = make_double2(a, -a * (e2 * e2 + e1 * e1));
    }
  for (int il = 1; il < nil; ++il)
    {
    double l = md + 2.0 * il;
    double sgn = (il & 1) ? -1.0 : 1.0;
    double a = sgn * alpha_cur * alpha_cur;
    double e1 = eps_lm(l + 1.0, md), e2 = eps_lm(l + 2.0, md);
    coef[off + il] = make_double2(a, -a * (e2 * e2 + e1 * e1));
    // advance alpha: alpha[il+1] = sign(il) / (eps(l+2) eps(l+3) alpha[il])
    double alpha_next = sgn / (e2 * eps_lm(l + 3.0, md) * alpha_cur);
    alpha_prev = alpha_cur;
    alpha_cur = alpha_next;
    }
  }

// ---------------------------------------------------------------------------
// prefold: one thread per (m, il); il = threadIdx.x + blockIdx.x*blockDim.x,
// m = blockIdx.y.  Reads healpy-layout alm, writes interleaved (A,B) pairs:
// AB[2*(moff[m]+il)] = A_il, AB[2*(moff[m]+il)+1] = B_il.
// ---------------------------------------------------------------------------
__global__ void prefold(const int lmax, const int mmax,
                        const long long* __restrict__ moff,
                        const double2* __restrict__ coef,
                        const long long* __restrict__ mstart,
                        const long long nalm, const int nchunk,
                        const long long s1, const long long s2,
                        const double2* __restrict__ alm,   // (nchunk, nalm)
                        double2* __restrict__ AB)  // ((off+il)*s1 + c*s2)*2+{0,1}
  {
  // layout: interleaved (chunked batch) s1=nchunk, s2=1;
  //         plane (grid-z batch)        s1=1, s2=ncoef;  single: s1=1, s2=0
  const int m = blockIdx.y;
  const int il = blockIdx.x * blockDim.x + threadIdx.x;
  const long long off = moff[m];
  const int nil = (int)(moff[m + 1] - off);
  if (il >= nil) return;

  const double md = (double)m;
  const int l = m + 2 * il;

  // alpha_il = sign4(il) * sqrt(|a_il|), sign4 pattern (+ + - -)
  double a = coef[off + il].x;
  double alpha = sqrt(fabs(a));
  if ((il & 2)) alpha = -alpha;

  const double e1 = eps_lm((double)l + 1.0, md);
  const double e2 = eps_lm((double)l + 2.0, md);
  const long long ms = mstart[m];

  for (int c = 0; c < nchunk; ++c)
    {
    const double2* almc = alm + (long long)c * nalm;
    double2 al   = (l     <= lmax) ? almc[ms + l]     : make_double2(0.0, 0.0);
    double2 al1  = (l + 1 <= lmax) ? almc[ms + l + 1] : make_double2(0.0, 0.0);
    double2 al2  = (l + 2 <= lmax) ? almc[ms + l + 2] : make_double2(0.0, 0.0);
    const long long base = 2 * ((off + il) * s1 + (long long)c * s2);
    AB[base]     = make_double2(alpha * (e1 * al.x + e2 * al2.x),
                                alpha * (e1 * al.y + e2 * al2.y));
    AB[base + 1] = make_double2(alpha * al1.x, alpha * al1.y);
    }
  }

// ---------------------------------------------------------------------------
// x^n with 2^(800*scale) tracking (x in (0,1]); ducc0's mypow, scalar.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mypow_scaled(double x, int n, double powlimit,
                                             double& res, int& scale)
  {
  if (x >= powlimit)
    {
    double r = 1.0, v = x;
    int nn = n;
    while (nn)
      {
      if (nn & 1) r *= v;
      v *= v;
      nn >>= 1;
      }
    res = r; scale = 0;
    }
  else
    {
    // invariant maintained by the two-sided checks: v, r in [2^-400, 2^400),
    // so any single product lies in (2^-800, 2^800) and ONE bump renormalises.
    // (An underflow bump can push v above 1, after which products may grow:
    // the overflow side is NOT optional even though x < 1.)
    double v = x, r = 1.0;
    int vs = 0, rs = 0;
    int nn = n;
    while (nn)
      {
      if (nn & 1)
        {
        r *= v; rs += vs;
        if (fabs(r) < FBIGHALF * FSMALL && r != 0.0) { r *= FBIG; rs -= 1; }
        else if (fabs(r) > FBIGHALF)                 { r *= FSMALL; rs += 1; }
        }
      v *= v; vs += vs;
      if (fabs(v) < FBIGHALF * FSMALL && v != 0.0) { v *= FBIG; vs -= 1; }
      else if (fabs(v) > FBIGHALF)                 { v *= FSMALL; vs += 1; }
      nn >>= 1;
      }
    res = r; scale = rs;
    }
  }

// ---------------------------------------------------------------------------
// legendre: one thread per (ring pair, m).
//   blockIdx.y = m, threads over pairs (pole -> equator order).
// Writes phase[m * nring + ring] (phase must be zero-initialised).
// ---------------------------------------------------------------------------
__global__ void legendre(const int lmax, const int mmax,
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
                         double2* __restrict__ phase)
  {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  if (p >= npair || m > mmax) return;
  if (pair_mlim[p] < m) return;

  const double csq = pair_csq[p];
  const double cth = pair_cth[p];
  const long long off = moff[m];
  const int nacc = (lmax - m) / 2 + 1;   // number of (A,B) accumulation steps

  // init: lam2 = (-1)^m mfac[m] sth^m, scaled
  double lam2; int scale;
  mypow_scaled(pair_sth[p], m, powlimit[m], lam2, scale);
  lam2 *= (m & 1) ? -mfac[m] : mfac[m];
  if (!isfinite(lam2)) return;   // safety: never spin on Inf/NaN
  while (fabs(lam2) > FTOL) { lam2 *= FSMALL; ++scale; }
  if (lam2 != 0.0)
    while (fabs(lam2) < FTOL * FSMALL) { lam2 *= FBIG; --scale; }
  double lam1 = 0.0;

  // skip phase: advance without accumulating while below IEEE range
  int il = 0;
  while (scale < 0)
    {
    if (il >= nacc) return;   // never surfaced: contribution exactly 0
    double2 c = coef[off + il];
    double t = (c.x * csq + c.y) * lam2 + lam1;
    lam1 = lam2; lam2 = t;
    ++il;
    if (fabs(lam2) > FTOL) { lam1 *= FSMALL; lam2 *= FSMALL; ++scale; }
    }

  // undo any residual positive scale (e.g. m = 0 where lam_mm is O(1) and
  // the init normalisation pushed it below FTOL with scale = 1)
  while (scale > 0) { lam1 *= FBIG; lam2 *= FBIG; --scale; }

  // accumulate phase (scale == 0; lam values are now true lambda / mu)
  double p1r = 0.0, p1i = 0.0, p2r = 0.0, p2i = 0.0;
  // two-step unrolled: roles of lam1/lam2 alternate (no swap needed)
  for (; il + 1 < nacc; il += 2)
    {
    double2 A0 = AB[2 * (off + il)];
    double2 B0 = AB[2 * (off + il) + 1];
    double2 c0 = coef[off + il];
    p1r += lam2 * A0.x; p1i += lam2 * A0.y;
    p2r += lam2 * B0.x; p2i += lam2 * B0.y;
    lam1 = (c0.x * csq + c0.y) * lam2 + lam1;
    double2 A1 = AB[2 * (off + il) + 2];
    double2 B1 = AB[2 * (off + il) + 3];
    double2 c1 = coef[off + il + 1];
    p1r += lam1 * A1.x; p1i += lam1 * A1.y;
    p2r += lam1 * B1.x; p2i += lam1 * B1.y;
    lam2 = (c1.x * csq + c1.y) * lam1 + lam2;
    }
  if (il < nacc)
    {
    double2 A0 = AB[2 * (off + il)];
    double2 B0 = AB[2 * (off + il) + 1];
    p1r += lam2 * A0.x; p1i += lam2 * A0.y;
    p2r += lam2 * B0.x; p2i += lam2 * B0.y;
    }

  const double t2r = cth * p2r, t2i = cth * p2i;
  const int rn = pair_inorth[p], rs = pair_isouth[p];
  phase[(long long)m * nring + rn] = make_double2(p1r + t2r, p1i + t2i);
  if (rs != rn)
    phase[(long long)m * nring + rs] = make_double2(p1r - t2r, p1i - t2i);
  }

// ---------------------------------------------------------------------------
// legendre2: like legendre, but each thread owns TWO adjacent ring pairs.
// The mu recursion is a serial FMA chain (2 dependent FMAs per il); with one
// pair per thread the per-thread ILP is too low to hide the chain latency.
// Two independent chains double the in-flight FMAs and amortise the
// (block-uniform) coefficient loads -- the GPU analogue of ducc's
// SIMD-over-rings vectorisation.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void init_and_skip(
    const double sth, const double csq, const int m,
    const double mfac_m, const double powlimit_m,
    const double2* __restrict__ coef, const long long off, const int nacc,
    double& lam1, double& lam2, int& il)
  {
  lam1 = 0.0; lam2 = 0.0; il = nacc;   // default: no contribution
  double v; int scale;
  mypow_scaled(sth, m, powlimit_m, v, scale);
  v *= (m & 1) ? -mfac_m : mfac_m;
  if (!isfinite(v) || v == 0.0) return;
  while (fabs(v) > FTOL) { v *= FSMALL; ++scale; }
  while (fabs(v) < FTOL * FSMALL) { v *= FBIG; --scale; }
  double l1 = 0.0, l2 = v;
  int i = 0;
  while (scale < 0)
    {
    if (i >= nacc) return;
    double2 c = coef[off + i];
    double t = (c.x * csq + c.y) * l2 + l1;
    l1 = l2; l2 = t;
    ++i;
    if (fabs(l2) > FTOL) { l1 *= FSMALL; l2 *= FSMALL; ++scale; }
    }
  while (scale > 0) { l1 *= FBIG; l2 *= FBIG; --scale; }
  lam1 = l1; lam2 = l2; il = i;
  }

__global__ void legendre2(const int lmax, const int mmax,
                          const int npair, const int nring,
                          const long long* __restrict__ moff,
                          const double2* __restrict__ coef,
                          const double2* __restrict__ AB_,
                          const double* __restrict__ mfac,
                          const double* __restrict__ powlimit,
                          const double* __restrict__ pair_csq,
                          const double* __restrict__ pair_cth,
                          const double* __restrict__ pair_sth,
                          const int* __restrict__ pair_mlim,
                          const int* __restrict__ pair_inorth,
                          const int* __restrict__ pair_isouth,
                          const long long* __restrict__ ringstart,
                          const int* __restrict__ nphi,
                          const int* __restrict__ phi0_num,
                          const int* __restrict__ phi0_den,
                          const long long ab_cstride,   // double2 units
                          const long long g_cstride,    // double2 units
                          double2* __restrict__ G_)
  {
  const int t2 = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  const int pa = 2 * t2, pb = 2 * t2 + 1;
  if (pa >= npair || m > mmax) return;
  // grid-z column batching: per-column planes of AB and G
  const double2* __restrict__ AB = AB_ + (long long)blockIdx.z * ab_cstride;
  double2* __restrict__ G = G_ + (long long)blockIdx.z * g_cstride;

  const long long off = moff[m];
  const int nacc = (lmax - m) / 2 + 1;
  const double mf = mfac[m], pl = powlimit[m];

  // pair a
  double a_l1, a_l2; int a_il = nacc;
  double csqa = 0.0, ctha = 0.0;
  if (pair_mlim[pa] >= m)
    {
    csqa = pair_csq[pa]; ctha = pair_cth[pa];
    init_and_skip(pair_sth[pa], csqa, m, mf, pl, coef, off, nacc,
                  a_l1, a_l2, a_il);
    }
  else { a_l1 = a_l2 = 0.0; }
  // pair b
  double b_l1, b_l2; int b_il = nacc;
  double csqb = 0.0, cthb = 0.0;
  const bool has_b = (pb < npair) && (pair_mlim[pb] >= m);
  if (has_b)
    {
    csqb = pair_csq[pb]; cthb = pair_cth[pb];
    init_and_skip(pair_sth[pb], csqb, m, mf, pl, coef, off, nacc,
                  b_l1, b_l2, b_il);
    }
  else { b_l1 = b_l2 = 0.0; }

  if (a_il >= nacc && b_il >= nacc)
    {
    // no contributions at all; phase is pre-zeroed
    return;
    }

  double a1r = 0.0, a1i = 0.0, a2r = 0.0, a2i = 0.0;
  double b1r = 0.0, b1i = 0.0, b2r = 0.0, b2i = 0.0;

  // solo-advance the earlier-surfacing pair to the later one's start
  int il = min(a_il, b_il);
  const int il_join = min(max(a_il, b_il), nacc);
  if (a_il < b_il)
    for (; il < il_join; ++il)
      {
      double2 A = AB[2 * (off + il)], B = AB[2 * (off + il) + 1];
      double2 c = coef[off + il];
      a1r += a_l2 * A.x; a1i += a_l2 * A.y;
      a2r += a_l2 * B.x; a2i += a_l2 * B.y;
      double t = (c.x * csqa + c.y) * a_l2 + a_l1;
      a_l1 = a_l2; a_l2 = t;
      }
  else if (b_il < a_il)
    for (; il < il_join; ++il)
      {
      double2 A = AB[2 * (off + il)], B = AB[2 * (off + il) + 1];
      double2 c = coef[off + il];
      b1r += b_l2 * A.x; b1i += b_l2 * A.y;
      b2r += b_l2 * B.x; b2i += b_l2 * B.y;
      double t = (c.x * csqb + c.y) * b_l2 + b_l1;
      b_l1 = b_l2; b_l2 = t;
      }
  // if the later pair never surfaces (il_join == nacc), its lam stays 0 and
  // the joint loop below accumulates exact zeros for it -- harmless.

  // joint loop, two independent recursion chains, two-step unrolled
  for (; il + 1 < nacc; il += 2)
    {
    double2 A0 = AB[2 * (off + il)],     B0 = AB[2 * (off + il) + 1];
    double2 c0 = coef[off + il];
    a1r += a_l2 * A0.x; a1i += a_l2 * A0.y;
    a2r += a_l2 * B0.x; a2i += a_l2 * B0.y;
    b1r += b_l2 * A0.x; b1i += b_l2 * A0.y;
    b2r += b_l2 * B0.x; b2i += b_l2 * B0.y;
    a_l1 = (c0.x * csqa + c0.y) * a_l2 + a_l1;
    b_l1 = (c0.x * csqb + c0.y) * b_l2 + b_l1;
    double2 A1 = AB[2 * (off + il) + 2], B1 = AB[2 * (off + il) + 3];
    double2 c1 = coef[off + il + 1];
    a1r += a_l1 * A1.x; a1i += a_l1 * A1.y;
    a2r += a_l1 * B1.x; a2i += a_l1 * B1.y;
    b1r += b_l1 * A1.x; b1i += b_l1 * A1.y;
    b2r += b_l1 * B1.x; b2i += b_l1 * B1.y;
    a_l2 = (c1.x * csqa + c1.y) * a_l1 + a_l2;
    b_l2 = (c1.x * csqb + c1.y) * b_l1 + b_l2;
    }
  if (il < nacc)
    {
    double2 A0 = AB[2 * (off + il)], B0 = AB[2 * (off + il) + 1];
    a1r += a_l2 * A0.x; a1i += a_l2 * A0.y;
    a2r += a_l2 * B0.x; a2i += a_l2 * B0.y;
    b1r += b_l2 * A0.x; b1i += b_l2 * A0.y;
    b2r += b_l2 * B0.x; b2i += b_l2 * B0.y;
    }

  if (a_il < nacc)
    {
    const double t2r = ctha * a2r, t2i = ctha * a2i;
    const int rn = pair_inorth[pa], rs = pair_isouth[pa];
    fold_scatter(make_double2(a1r + t2r, a1i + t2i), rn, m,
                 ringstart, nphi, phi0_num, phi0_den, G);
    if (rs != rn)
      fold_scatter(make_double2(a1r - t2r, a1i - t2i), rs, m,
                   ringstart, nphi, phi0_num, phi0_den, G);
    }
  if (has_b && b_il < nacc)
    {
    const double t2r = cthb * b2r, t2i = cthb * b2i;
    const int rn = pair_inorth[pb], rs = pair_isouth[pb];
    fold_scatter(make_double2(b1r + t2r, b1i + t2i), rn, m,
                 ringstart, nphi, phi0_num, phi0_den, G);
    if (rs != rn)
      fold_scatter(make_double2(b1r - t2r, b1i - t2i), rs, m,
                   ringstart, nphi, phi0_num, phi0_den, G);
    }
  }

// ---------------------------------------------------------------------------
// legendre_batch: one thread per (ring pair, m), CHUNK alm columns at once.
// The mu recursion is column-independent, so ONE chain feeds 4*CHUNK
// accumulator FMAs -- ample ILP (no need for the two-pair trick) and the
// recursion cost is amortised across the batch.  AB layout:
// ((off+il)*CHUNK + c)*2 + {0,1};  phase layout: (CHUNK, mmax+1, nring).
// ---------------------------------------------------------------------------
__global__ void legendre_batch(const int lmax, const int mmax,
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
                               double2* __restrict__ phase)
  {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  if (p >= npair || m > mmax) return;
  if (pair_mlim[p] < m) return;

  const double csq = pair_csq[p];
  const double cth = pair_cth[p];
  const long long off = moff[m];
  const int nacc = (lmax - m) / 2 + 1;

  double lam1, lam2; int il;
  init_and_skip(pair_sth[p], csq, m, mfac[m], powlimit[m], coef, off, nacc,
                lam1, lam2, il);
  if (il >= nacc) return;

  double p1r[CHUNK], p1i[CHUNK], p2r[CHUNK], p2i[CHUNK];
#pragma unroll
  for (int c = 0; c < CHUNK; ++c)
    p1r[c] = p1i[c] = p2r[c] = p2i[c] = 0.0;

  for (; il + 1 < nacc; il += 2)
    {
    const double2 c0 = coef[off + il];
    const double2 c1 = coef[off + il + 1];
    const double2* ab0 = AB + 2 * ((off + il) * (long long)CHUNK);
    const double2* ab1 = ab0 + 2 * CHUNK;
#pragma unroll
    for (int c = 0; c < CHUNK; ++c)
      {
      const double2 A = ab0[2 * c], Bv = ab0[2 * c + 1];
      p1r[c] += lam2 * A.x;  p1i[c] += lam2 * A.y;
      p2r[c] += lam2 * Bv.x; p2i[c] += lam2 * Bv.y;
      }
    lam1 = (c0.x * csq + c0.y) * lam2 + lam1;
#pragma unroll
    for (int c = 0; c < CHUNK; ++c)
      {
      const double2 A = ab1[2 * c], Bv = ab1[2 * c + 1];
      p1r[c] += lam1 * A.x;  p1i[c] += lam1 * A.y;
      p2r[c] += lam1 * Bv.x; p2i[c] += lam1 * Bv.y;
      }
    lam2 = (c1.x * csq + c1.y) * lam1 + lam2;
    }
  if (il < nacc)
    {
    const double2* ab0 = AB + 2 * ((off + il) * (long long)CHUNK);
#pragma unroll
    for (int c = 0; c < CHUNK; ++c)
      {
      const double2 A = ab0[2 * c], Bv = ab0[2 * c + 1];
      p1r[c] += lam2 * A.x;  p1i[c] += lam2 * A.y;
      p2r[c] += lam2 * Bv.x; p2i[c] += lam2 * Bv.y;
      }
    }

  const int rn = pair_inorth[p], rs = pair_isouth[p];
  const long long stride = (long long)(mmax + 1) * nring;
#pragma unroll
  for (int c = 0; c < CHUNK; ++c)
    {
    const double t2r = cth * p2r[c], t2i = cth * p2i[c];
    phase[c * stride + (long long)m * nring + rn] =
        make_double2(p1r[c] + t2r, p1i[c] + t2i);
    if (rs != rn)
      phase[c * stride + (long long)m * nring + rs] =
          make_double2(p1r[c] - t2r, p1i[c] - t2i);
    }
  }

// ---------------------------------------------------------------------------
// fold: one thread per (ring, m); x over rings (coalesced phase reads),
// y over m.  G (complex, map-ragged layout) must be zero-initialised.
// phi0 = pi * phi0_num[r] / phi0_den[r]  -> exact integer argument reduction
// for sincospi.
// ---------------------------------------------------------------------------
__global__ void fold(const int nring, const int mmax, const int nchunk,
                     const long long npix,
                     const double2* __restrict__ phase, // (nchunk, mmax+1, nring)
                     const long long* __restrict__ ringstart,
                     const int* __restrict__ nphi,
                     const int* __restrict__ phi0_num,
                     const int* __restrict__ phi0_den,
                     const int* __restrict__ ring_mlim,
                     double2* __restrict__ G)          // (nchunk, npix)
  {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  if (r >= nring || m > mmax) return;
  if (ring_mlim[r] < m) return;

  const int n = nphi[r];
  const long long rs = ringstart[r];
  const long long pstride = (long long)(mmax + 1) * nring;

  if (m == 0)
    {
    for (int c = 0; c < nchunk; ++c)
      atomicAdd(&G[c * npix + rs].x,
                phase[c * pstride + r].x);  // Im F_0 discarded (ducc)
    return;
    }

  // twiddle e^{i m phi0}: m*phi0/pi = m*num/den, reduce mod 2 in integers
  const long long num = (long long)m * (long long)phi0_num[r];
  const int den = phi0_den[r];
  const double x = (double)(num % (2LL * den)) / (double)den;
  double s, c;
  sincospi(x, &s, &c);

  const int k1 = m % n;
  const int k2 = (n - k1) % n;
  for (int cc = 0; cc < nchunk; ++cc)
    {
    const double2 F = phase[cc * pstride + (long long)m * nring + r];
    const double tr = F.x * c - F.y * s;
    const double ti = F.x * s + F.y * c;
    double2* Gc = G + cc * npix;
    atomicAdd(&Gc[rs + k1].x, tr);
    atomicAdd(&Gc[rs + k1].y, ti);
    atomicAdd(&Gc[rs + k2].x, tr);
    atomicAdd(&Gc[rs + k2].y, -ti);
    }
  }

// ---------------------------------------------------------------------------
// cap_dft: direct Hermitian inverse DFT for the polar-cap rings.
//   f_j = G_0 + (-1)^j G_{n/2} + 2 sum_{k=1}^{n/2-1} Re(G_k e^{2 pi i j k / n})
// blockIdx.y = cap-ring slot (host provides ring index per slot), threads
// over pixels j.  Incremental rotation, re-seeded exactly (sincospi with
// integer-reduced argument) every RESEED steps.
// ---------------------------------------------------------------------------
#define RESEED 128

__global__ void cap_dft(const int ncap, const long long npix,
                        const int* __restrict__ cap_ring,
                        const long long* __restrict__ ringstart,
                        const int* __restrict__ nphi,
                        const double2* __restrict__ G_,   // (gridDim.z, npix)
                        double* __restrict__ out_)        // (gridDim.z, npix)
  {
  const int slot = blockIdx.y;
  if (slot >= ncap) return;
  const double2* G = G_ + (long long)blockIdx.z * npix;
  double* out = out_ + (long long)blockIdx.z * npix;
  const int r = cap_ring[slot];
  const int n = nphi[r];
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  const long long rs = ringstart[r];
  const int nh = n >> 1;

  double acc = G[rs].x + ((j & 1) ? -G[rs + nh].x : G[rs + nh].x);

  // rotation step w = e^{2 pi i j / n}
  double sw, cw;
    {
    const long long q = (2LL * j) % (2LL * n);
    sincospi((double)q / (double)n, &sw, &cw);
    }
  double cr = cw, si = sw;   // current w^k, k = 1
  int since = 0;
  for (int k = 1; k < nh; ++k)
    {
    if (since == RESEED)
      {
      const long long q = (2LL * (long long)j * (long long)k) % (2LL * n);
      sincospi((double)q / (double)n, &si, &cr);
      since = 0;
      }
    double2 g = G[rs + k];
    acc += 2.0 * (g.x * cr - g.y * si);
    double t = cr * cw - si * sw;
    si = cr * sw + si * cw;
    cr = t;
    ++since;
    }
  out[rs + j] = acc;
  }

// ---------------------------------------------------------------------------
// belt_finish: out[i] = n * Re(belt_ifft_result[i]) for the equatorial belt
// (cuFFT ifft is normalised by 1/n; we fold the n back in here).
// ---------------------------------------------------------------------------
__global__ void belt_finish(const long long ntot, const double scale,
                            const double2* __restrict__ z,
                            double* __restrict__ out)
  {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ntot) return;
  out[i] = scale * z[i].x;
  }

// ===========================================================================
// ADJOINT SYNTHESIS (map -> alm), the exact transpose of the pipeline above
// under ducc0's conventions (see almond/reference.py adjoint_* for the math).
//
//   Ghat_k(ring) = sum_j f_j e^{-2 pi i jk/n}     (rfft belt / Bluestein caps
//                                                  / direct DFT small caps;
//                                                  stored half-spectrum,
//                                                  k = 0..n/2, offsets hstart)
//   F'_m(ring)   = e^{-i m phi0} Ghat_{m mod n}    (unfold; Im F'_0 -> 0)
//   A'_i, B'_i   = sum_pairs mu_i (F'_N + F'_S), sum_pairs mu_i cth (F'_N - F'_S)
//   a'_lm        = alpha/eps-weighted gather of A', B'   (postfold)
// ===========================================================================

// direct forward DFT for the small polar-cap rings: one thread per (ring, k)
__global__ void cap_dft_adj(const int ncap, const long long nghalf,
                            const int* __restrict__ cap_ring,
                            const long long* __restrict__ ringstart,
                            const long long* __restrict__ hstart,
                            const int* __restrict__ nphi,
                            const double* __restrict__ map_,  // (gridDim.z, npix)
                            const long long npix,
                            double2* __restrict__ Gh_)        // (gridDim.z, nghalf)
  {
  const int slot = blockIdx.y;
  if (slot >= ncap) return;
  const double* f = map_ + (long long)blockIdx.z * npix;
  double2* Gh = Gh_ + (long long)blockIdx.z * nghalf;
  const int r = cap_ring[slot];
  const int n = nphi[r];
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k > n / 2) return;
  const long long rs = ringstart[r];

  // rotation step w = e^{-2 pi i k / n}, re-seeded exactly every RESEED
  double sw, cw;
    {
    const long long q = (2LL * k) % (2LL * n);
    sincospi(-(double)q / (double)n, &sw, &cw);
    }
  double accr = 0.0, acci = 0.0;
  double cr = 1.0, si = 0.0;   // w^{jk} at j = 0
  int since = 0;
  for (int j = 0; j < n; ++j)
    {
    if (since == RESEED)
      {
      const long long q = (2LL * (long long)j * (long long)k) % (2LL * n);
      sincospi(-(double)q / (double)n, &si, &cr);
      since = 0;
      }
    const double fj = f[rs + j];
    accr += fj * cr;
    acci += fj * si;
    double t = cr * cw - si * sw;
    si = cr * sw + si * cw;
    cr = t;
    ++since;
    }
  Gh[hstart[r] + k] = make_double2(accr, acci);
  }

// Bluestein forward-DFT pre/post chirps (adjoint direction: sgn = -1)
__global__ void bluestein_pre_adj(const int nmem, const int M,
                                  const long long npix,
                                  const int* __restrict__ mem_ring,
                                  const long long* __restrict__ ringstart,
                                  const int* __restrict__ nphi,
                                  const double* __restrict__ map_, // (nchunk,npix)
                                  double2* __restrict__ A)
  {
  const int row = blockIdx.y;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= M) return;
  const int c = row / nmem, mem = row % nmem;
  const int r = mem_ring[mem];
  const int n = nphi[r];
  double2 v = make_double2(0.0, 0.0);
  if (k < n)
    {
    const double f = map_[c * npix + ringstart[r] + k];
    const double2 w = chirp_w(k, n, -1.0);   // e^{-i pi k^2/n}
    v = make_double2(f * w.x, f * w.y);
    }
  A[(long long)row * M + k] = v;
  }

__global__ void bluestein_post_adj(const int nmem, const int M,
                                   const long long nghalf,
                                   const int* __restrict__ mem_ring,
                                   const long long* __restrict__ hstart,
                                   const int* __restrict__ nphi,
                                   const double2* __restrict__ C,
                                   double2* __restrict__ Gh_)  // (nchunk, nghalf)
  {
  const int row = blockIdx.y;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = row / nmem, mem = row % nmem;
  const int r = mem_ring[mem];
  const int n = nphi[r];
  if (k > n / 2) return;
  const double2 v = C[(long long)row * M + k];
  const double2 w = chirp_w(k, n, -1.0);     // e^{-i pi k^2/n}
  Gh_[c * nghalf + hstart[r] + k] =
      make_double2(v.x * w.x - v.y * w.y, v.x * w.y + v.y * w.x);
  }

// unfold: F'_m(ring) = e^{-i m phi0} Ghat_{m mod n}; one thread per (ring, m)
__global__ void unfold(const int nring, const int mmax, const int nchunk,
                       const long long nghalf,
                       const double2* __restrict__ Gh,   // (nchunk, nghalf)
                       const long long* __restrict__ hstart,
                       const int* __restrict__ nphi,
                       const int* __restrict__ phi0_num,
                       const int* __restrict__ phi0_den,
                       const int* __restrict__ ring_mlim,
                       double2* __restrict__ phase)     // (nchunk, mmax+1, nring)
  {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  if (r >= nring || m > mmax) return;
  if (ring_mlim[r] < m) return;
  const int n = nphi[r];
  const int k1 = m % n;
  const long long hs = hstart[r];
  const long long pstride = (long long)(mmax + 1) * nring;

  double s = 0.0, c = 1.0;
  if (m > 0)
    {
    const long long num = (long long)m * (long long)phi0_num[r];
    const int den = phi0_den[r];
    sincospi(-(double)(num % (2LL * den)) / (double)den, &s, &c);
    }
  for (int cc = 0; cc < nchunk; ++cc)
    {
    double2 g = (k1 <= n / 2) ? Gh[cc * nghalf + hs + k1]
                              : Gh[cc * nghalf + hs + n - k1];
    if (k1 > n / 2) g.y = -g.y;          // conjugate mirror
    double2 F = make_double2(g.x * c - g.y * s, g.x * s + g.y * c);
    if (m == 0) F = make_double2(g.x, 0.0);   // adjoint of Im F_0 discard
    phase[cc * pstride + (long long)m * nring + r] = F;
    }
  }

// ---------------------------------------------------------------------------
// legendre_adj: A'_i = sum_pairs mu_i p1', B'_i = sum_pairs mu_i p2'.
// One BLOCK per m owns all ring pairs (chunked over the block), so the
// global accumulation needs no atomics.  Each lane runs ADJ_PPT independent
// recursion chains; contributions are warp-shuffle-reduced each il and
// staged in per-warp shared tiles, flushed to ABadj every ADJ_TILE steps.
// ABadj must be zero-initialised; layout matches AB (A'=2*(off+i), B'=+1).
// ---------------------------------------------------------------------------
#define ADJ_PPT  4
#define ADJ_TILE 128

__global__ void legendre_adj(const int lmax, const int mmax,
                             const int npair, const int nring,
                             const long long* __restrict__ moff,
                             const double2* __restrict__ coef,
                             const double* __restrict__ mfac,
                             const double* __restrict__ powlimit,
                             const double* __restrict__ pair_csq,
                             const double* __restrict__ pair_cth,
                             const double* __restrict__ pair_sth,
                             const int* __restrict__ pair_mlim,
                             const int* __restrict__ pair_inorth,
                             const int* __restrict__ pair_isouth,
                             const int* __restrict__ pstart,
                             const double2* __restrict__ Gh,   // half spectra
                             const long long* __restrict__ hstart,
                             const int* __restrict__ nphi,
                             const int* __restrict__ phi0_num,
                             const int* __restrict__ phi0_den,
                             double2* __restrict__ ABadj)
  {
  const int m = blockIdx.x;
  if (m > mmax) return;
  const long long off = moff[m];
  const int nacc = (lmax - m) / 2 + 1;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int nwarp = blockDim.x >> 5;
  extern __shared__ double tile[];   // [nwarp][ADJ_TILE][4]
  const double mf = mfac[m], pl = powlimit[m];
  const int p0m = pstart[m];

  for (long long base = p0m; base < npair;
       base += (long long)blockDim.x * ADJ_PPT)
    {
    double lam1[ADJ_PPT], lam2[ADJ_PPT], csq[ADJ_PPT];
    double p1r[ADJ_PPT], p1i[ADJ_PPT], p2r[ADJ_PPT], p2i[ADJ_PPT];
    int scale[ADJ_PPT];
#pragma unroll
    for (int q = 0; q < ADJ_PPT; ++q)
      {
      const long long p = base + (long long)q * blockDim.x + tid;
      bool act = (p < npair) && (pair_mlim[p] >= m);
      lam1[q] = lam2[q] = csq[q] = 0.0;
      p1r[q] = p1i[q] = p2r[q] = p2i[q] = 0.0;
      scale[q] = 0;
      if (act)
        {
        csq[q] = pair_csq[p];
        const double cth = pair_cth[p];
        const int rn = pair_inorth[p], rs = pair_isouth[p];
        const double2 Fn = unfold_gather(rn, m, Gh, hstart, nphi,
                                         phi0_num, phi0_den);
        double2 Fs = make_double2(0.0, 0.0);
        if (rs != rn) Fs = unfold_gather(rs, m, Gh, hstart, nphi,
                                         phi0_num, phi0_den);
        p1r[q] = Fn.x + Fs.x;         p1i[q] = Fn.y + Fs.y;
        p2r[q] = cth * (Fn.x - Fs.x); p2i[q] = cth * (Fn.y - Fs.y);
        double v; int s;
        mypow_scaled(pair_sth[p], m, pl, v, s);
        v *= (m & 1) ? -mf : mf;
        if (isfinite(v) && v != 0.0)
          {
          while (fabs(v) > FTOL) { v *= FSMALL; ++s; }
          while (fabs(v) < FTOL * FSMALL) { v *= FBIG; --s; }
          while (s > 0) { v *= FBIG; --s; }
          lam2[q] = v; scale[q] = s;
          }
        }
      }

    for (int seg = 0; seg < nacc; seg += ADJ_TILE)
      {
      const int end = min(seg + ADJ_TILE, nacc);
      const int nsl = (end - seg) * 4;
      for (int i = lane; i < nsl; i += 32)
        tile[(warp * ADJ_TILE) * 4 + i] = 0.0;
      __syncwarp();

      for (int il = seg; il < end; ++il)
        {
        const double2 cc = coef[off + il];
        double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0;
#pragma unroll
        for (int q = 0; q < ADJ_PPT; ++q)
          {
          const double contrib = (scale[q] == 0) ? lam2[q] : 0.0;
          v0 += contrib * p1r[q]; v1 += contrib * p1i[q];
          v2 += contrib * p2r[q]; v3 += contrib * p2i[q];
          const double t = (cc.x * csq[q] + cc.y) * lam2[q] + lam1[q];
          lam1[q] = lam2[q]; lam2[q] = t;
          if (scale[q] < 0 && fabs(lam2[q]) > FTOL)
            { lam1[q] *= FSMALL; lam2[q] *= FSMALL; ++scale[q]; }
          }
#pragma unroll
        for (int o = 16; o; o >>= 1)
          {
          v0 += __shfl_down_sync(0xffffffffu, v0, o);
          v1 += __shfl_down_sync(0xffffffffu, v1, o);
          v2 += __shfl_down_sync(0xffffffffu, v2, o);
          v3 += __shfl_down_sync(0xffffffffu, v3, o);
          }
        if (lane == 0)
          {
          double* slot = &tile[((warp * ADJ_TILE) + (il - seg)) * 4];
          slot[0] += v0; slot[1] += v1; slot[2] += v2; slot[3] += v3;
          }
        }
      __syncthreads();
      for (int i = tid; i < nsl; i += blockDim.x)
        {
        double s = 0.0;
        for (int w = 0; w < nwarp; ++w)
          s += tile[((w * ADJ_TILE) + (i >> 2)) * 4 + (i & 3)];
        double* g = (double*)&ABadj[2 * (off + seg + (i >> 2))];
        g[i & 3] += s;
        }
      __syncthreads();
      }
    }
  }

// ---------------------------------------------------------------------------
// legendre_adj2: like legendre_adj, but the per-il warp-shuffle reduction is
// replaced by staged bulk reduction.  Each lane stores its chain-summed
// contributions (v0..v3) for ADJ2_TR consecutive ils into a per-warp shared
// tile; the warp then reduces its own tile (two lanes per output, one final
// shuffle) and accumulates into a parity-double-buffered block accumulator
// with shared atomics, flushed to global every segment.  This cuts the
// reduction from ~40 warp-instructions per il (5 butterfly steps x 4 values
// x (shfl+add)) to ~13, and decouples the reduction from the serial
// recursion chain.
// ---------------------------------------------------------------------------
#ifndef ADJ2_PPT
#define ADJ2_PPT 4
#endif
#ifndef ADJ2_TR
#define ADJ2_TR  8
#endif
#define ADJ2_W   33   // padded lane-row width (doubles)

__global__ void legendre_adj2(const int lmax, const int mmax,
                              const int npair, const int nring,
                              const long long* __restrict__ moff,
                              const double2* __restrict__ coef,
                              const double* __restrict__ mfac,
                              const double* __restrict__ powlimit,
                              const double* __restrict__ pair_csq,
                              const double* __restrict__ pair_cth,
                              const double* __restrict__ pair_sth,
                              const int* __restrict__ pair_mlim,
                              const int* __restrict__ pair_inorth,
                              const int* __restrict__ pair_isouth,
                              const int* __restrict__ pstart,
                              const double2* __restrict__ Gh_,  // half spectra
                              const long long* __restrict__ hstart,
                              const int* __restrict__ nphi,
                              const int* __restrict__ phi0_num,
                              const int* __restrict__ phi0_den,
                              const long long gh_cstride,   // double2 units
                              const long long ab_cstride,   // double2 units
                              double2* __restrict__ ABadj_)
  {
  const int m = blockIdx.x;
  if (m > mmax) return;
  // grid-y column batching: per-column planes of Ghat and ABadj
  const double2* __restrict__ Gh = Gh_ + (long long)blockIdx.y * gh_cstride;
  double2* __restrict__ ABadj = ABadj_ + (long long)blockIdx.y * ab_cstride;
  const long long off = moff[m];
  const int nacc = (lmax - m) / 2 + 1;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int nwarp = blockDim.x >> 5;
  extern __shared__ double sh[];
  // stage[warp][ilr][comp][ADJ2_W], then xtile[2][nwarp][ADJ2_TR*4+1]
  double* stage = sh;
  double* xtile = sh + (long long)nwarp * ADJ2_TR * 4 * ADJ2_W;
  const double mf = mfac[m], pl = powlimit[m];
  const int p0m = pstart[m];

  for (long long base = p0m; base < npair;
       base += (long long)blockDim.x * ADJ2_PPT)
    {
    double lam1[ADJ2_PPT], lam2[ADJ2_PPT], csq[ADJ2_PPT];
    double p1r[ADJ2_PPT], p1i[ADJ2_PPT], p2r[ADJ2_PPT], p2i[ADJ2_PPT];
    int scale[ADJ2_PPT];
#pragma unroll
    for (int q = 0; q < ADJ2_PPT; ++q)
      {
      const long long p = base + (long long)q * blockDim.x + tid;
      bool act = (p < npair) && (pair_mlim[p] >= m);
      lam1[q] = lam2[q] = csq[q] = 0.0;
      p1r[q] = p1i[q] = p2r[q] = p2i[q] = 0.0;
      scale[q] = 0;
      if (act)
        {
        csq[q] = pair_csq[p];
        const double cth = pair_cth[p];
        const int rn = pair_inorth[p], rs = pair_isouth[p];
        const double2 Fn = unfold_gather(rn, m, Gh, hstart, nphi,
                                         phi0_num, phi0_den);
        double2 Fs = make_double2(0.0, 0.0);
        if (rs != rn) Fs = unfold_gather(rs, m, Gh, hstart, nphi,
                                         phi0_num, phi0_den);
        p1r[q] = Fn.x + Fs.x;         p1i[q] = Fn.y + Fs.y;
        p2r[q] = cth * (Fn.x - Fs.x); p2i[q] = cth * (Fn.y - Fs.y);
        double v; int s;
        mypow_scaled(pair_sth[p], m, pl, v, s);
        v *= (m & 1) ? -mf : mf;
        if (isfinite(v) && v != 0.0)
          {
          while (fabs(v) > FTOL) { v *= FSMALL; ++s; }
          while (fabs(v) < FTOL * FSMALL) { v *= FBIG; --s; }
          while (s > 0) { v *= FBIG; --s; }
          lam2[q] = v; scale[q] = s;
          }
        }
      }

    bool anyneg = false;
#pragma unroll
    for (int q = 0; q < ADJ2_PPT; ++q) anyneg |= (scale[q] < 0);

    int par = 0;
    for (int seg = 0; seg < nacc; seg += ADJ2_TR, par ^= 1)
      {
      const int end = min(seg + ADJ2_TR, nacc);
      int il = seg;
      // guarded loop: some chain still below IEEE range (scale < 0)
      for (; il < end && anyneg; ++il)
        {
        const double2 cc = coef[off + il];
        double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0;
        anyneg = false;
#pragma unroll
        for (int q = 0; q < ADJ2_PPT; ++q)
          {
          const double contrib = (scale[q] == 0) ? lam2[q] : 0.0;
          v0 += contrib * p1r[q]; v1 += contrib * p1i[q];
          v2 += contrib * p2r[q]; v3 += contrib * p2i[q];
          const double t = (cc.x * csq[q] + cc.y) * lam2[q] + lam1[q];
          lam1[q] = lam2[q]; lam2[q] = t;
          if (scale[q] < 0 && fabs(lam2[q]) > FTOL)
            { lam1[q] *= FSMALL; lam2[q] *= FSMALL; ++scale[q]; }
          anyneg |= (scale[q] < 0);
          }
        double* srow = stage
            + ((long long)(warp * ADJ2_TR + (il - seg)) * 4) * ADJ2_W + lane;
        srow[0 * ADJ2_W] = v0; srow[1 * ADJ2_W] = v1;
        srow[2 * ADJ2_W] = v2; srow[3 * ADJ2_W] = v3;
        }
      // fast loop: all chains surfaced -- no guards, no select, two-step
      // unrolled so the lam1/lam2 roles alternate (no register MOVs)
      for (; il + 1 < end; il += 2)
        {
        const double2 c0 = coef[off + il];
        const double2 c1 = coef[off + il + 1];
        double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0;
        double w0 = 0.0, w1 = 0.0, w2 = 0.0, w3 = 0.0;
#pragma unroll
        for (int q = 0; q < ADJ2_PPT; ++q)
          {
          v0 += lam2[q] * p1r[q]; v1 += lam2[q] * p1i[q];
          v2 += lam2[q] * p2r[q]; v3 += lam2[q] * p2i[q];
          lam1[q] = (c0.x * csq[q] + c0.y) * lam2[q] + lam1[q];
          w0 += lam1[q] * p1r[q]; w1 += lam1[q] * p1i[q];
          w2 += lam1[q] * p2r[q]; w3 += lam1[q] * p2i[q];
          lam2[q] = (c1.x * csq[q] + c1.y) * lam1[q] + lam2[q];
          }
        double* srow = stage
            + ((long long)(warp * ADJ2_TR + (il - seg)) * 4) * ADJ2_W + lane;
        srow[0 * ADJ2_W] = v0; srow[1 * ADJ2_W] = v1;
        srow[2 * ADJ2_W] = v2; srow[3 * ADJ2_W] = v3;
        srow[4 * ADJ2_W] = w0; srow[5 * ADJ2_W] = w1;
        srow[6 * ADJ2_W] = w2; srow[7 * ADJ2_W] = w3;
        }
      for (; il < end; ++il)
        {
        const double2 cc = coef[off + il];
        double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0;
#pragma unroll
        for (int q = 0; q < ADJ2_PPT; ++q)
          {
          v0 += lam2[q] * p1r[q]; v1 += lam2[q] * p1i[q];
          v2 += lam2[q] * p2r[q]; v3 += lam2[q] * p2i[q];
          const double t = (cc.x * csq[q] + cc.y) * lam2[q] + lam1[q];
          lam1[q] = lam2[q]; lam2[q] = t;
          }
        double* srow = stage
            + ((long long)(warp * ADJ2_TR + (il - seg)) * 4) * ADJ2_W + lane;
        srow[0 * ADJ2_W] = v0; srow[1 * ADJ2_W] = v1;
        srow[2 * ADJ2_W] = v2; srow[3 * ADJ2_W] = v3;
        }
      __syncwarp();
      // own-warp bulk reduce into the cross-warp tile (parity double-
      // buffered so one barrier per segment suffices, no atomics)
      const int nout = (end - seg) * 4;
#if ADJ2_TR == 8
      // 32 outputs, one lane per output
      double s = 0.0;
      if (lane < nout)
        {
        const double* srow = stage
            + ((long long)(warp * ADJ2_TR) * 4 + lane) * ADJ2_W;
#pragma unroll
        for (int k = 0; k < 32; ++k) s += srow[k];
        }
      if (lane < nout)
        xtile[(par * nwarp + warp) * (ADJ2_TR * 4 + 1) + lane] = s;
#else
      // 16 outputs, two lanes per output (even/odd interleave), 1 shuffle
      const int oid = lane >> 1, h = lane & 1;
      double s = 0.0;
      if (oid < nout)
        {
        const double* srow = stage
            + ((long long)(warp * ADJ2_TR) * 4 + oid) * ADJ2_W;
#pragma unroll
        for (int k = 0; k < 16; ++k) s += srow[2 * k + h];
        }
      s += __shfl_down_sync(0xffffffffu, s, 1);
      if (h == 0 && oid < nout)
        xtile[(par * nwarp + warp) * (ADJ2_TR * 4 + 1) + oid] = s;
#endif
      __syncthreads();
      if (tid < nout)
        {
        double acc = 0.0;
        for (int w = 0; w < nwarp; ++w)
          acc += xtile[(par * nwarp + w) * (ADJ2_TR * 4 + 1) + tid];
        double* g = (double*)&ABadj[2 * (off + seg + (tid >> 2))];
        g[tid & 3] += acc;
        }
      // no second barrier: the next segment writes the other parity; the
      // next same-parity write is separated by that segment's syncthreads
      }
    // base-loop restarts at parity 0: ensure the last combine has read it
    __syncthreads();
    }
  }

// ---------------------------------------------------------------------------
// postfold: gather a'_lm from A', B' (transpose of prefold); thread per (m,l)
// ---------------------------------------------------------------------------
__global__ void postfold(const int lmax, const int mmax,
                         const long long* __restrict__ moff,
                         const double2* __restrict__ coef,
                         const long long* __restrict__ mstart,
                         const long long ab_cstride,    // double2 units
                         const long long alm_cstride,   // double2 units
                         const double2* __restrict__ ABadj_,
                         double2* __restrict__ alm_)
  {
  const int m = blockIdx.y;
  const int dl = blockIdx.x * blockDim.x + threadIdx.x;
  const int l = m + dl;
  if (m > mmax || l > lmax) return;
  const double2* __restrict__ ABadj = ABadj_
      + (long long)blockIdx.z * ab_cstride;
  double2* __restrict__ alm = alm_ + (long long)blockIdx.z * alm_cstride;
  const long long off = moff[m];
  const double md = (double)m;

  double2 v;
  if ((dl & 1) == 0)
    {
    const int il = dl >> 1;
    double a = coef[off + il].x;
    double alpha = sqrt(fabs(a));
    if (il & 2) alpha = -alpha;
    const double w = alpha * eps_lm((double)l + 1.0, md);
    const double2 Ai = ABadj[2 * (off + il)];
    v = make_double2(w * Ai.x, w * Ai.y);
    if (il >= 1)
      {
      double am = coef[off + il - 1].x;
      double alpham = sqrt(fabs(am));
      if ((il - 1) & 2) alpham = -alpham;
      const double w2 = alpham * eps_lm((double)l, md);
      const double2 Am = ABadj[2 * (off + il - 1)];
      v.x += w2 * Am.x; v.y += w2 * Am.y;
      }
    }
  else
    {
    const int il = dl >> 1;
    double a = coef[off + il].x;
    double alpha = sqrt(fabs(a));
    if (il & 2) alpha = -alpha;
    const double2 Bi = ABadj[2 * (off + il) + 1];
    v = make_double2(alpha * Bi.x, alpha * Bi.y);
    }
  alm[mstart[m] + l] = v;
  }

// batched: z is (nchunk * beltpix) contiguous; each chunk's belt segment goes
// to out[c*npix + belt_start ...].
__global__ void belt_finish_b(const long long beltpix, const long long npix,
                              const long long belt_start, const int nchunk,
                              const double scale,
                              const double2* __restrict__ z,
                              double* __restrict__ out)
  {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= beltpix * nchunk) return;
  const long long c = i / beltpix, rem = i % beltpix;
  out[c * npix + belt_start + rem] = scale * z[i].x;
  }

// ---------------------------------------------------------------------------
// Bluestein evaluation of the cap-ring inverse DFTs through batched cuFFT.
//
//   f_j = sum_k G_k e^{2 pi i jk/n}
//       = w_j * sum_k (G_k w_k) * conj(w_{j-k}),     w_t = e^{i pi t^2 / n},
//
// a circular convolution of length M >= 2n-1 (M = power of two, shared by a
// whole class of rings).  Chirps are exact: t^2 mod 2n reduced in integers,
// then sincospi.  bluestein_b builds the (plan-time) kernel rows whose FFT
// is B-hat; bluestein_pre builds a_k = G_k w_k (zero-padded); bluestein_post
// applies the outer chirp and writes Re(...) into the map.
// ---------------------------------------------------------------------------
__global__ void bluestein_b(const int nmem, const int M, const double sgn,
                            const int* __restrict__ mem_ring,
                            const int* __restrict__ nphi,
                            double2* __restrict__ B)
  {
  const int row = blockIdx.y;
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nmem || t >= M) return;
  const int n = nphi[mem_ring[row]];
  double2* Brow = B + (long long)row * M;
  if (t == 0) { Brow[0] = make_double2(1.0, 0.0); return; }
  if (t <= n - 1)
    {
    const double2 w = chirp_w(t, n, sgn);   // e^{sgn * i pi t^2/n}
    Brow[t] = w;
    Brow[M - t] = w;                          // b_{-t} = b_t
    }
  else if (t < M - (n - 1))
    Brow[t] = make_double2(0.0, 0.0);
  // t in (M-n, M): written by the mirror above
  }

// rows run over (chunk, member): row = c*nmem + mem; A is (nchunk*nmem, M);
// the chirp-kernel FFT B-hat (built once by bluestein_b) is broadcast over c.
__global__ void bluestein_pre(const int nmem, const int M, const long long npix,
                              const int* __restrict__ mem_ring,
                              const long long* __restrict__ ringstart,
                              const int* __restrict__ nphi,
                              const double2* __restrict__ G,   // (nchunk, npix)
                              double2* __restrict__ A)
  {
  const int row = blockIdx.y;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= M) return;
  const int c = row / nmem, mem = row % nmem;
  const int r = mem_ring[mem];
  const int n = nphi[r];
  double2 v = make_double2(0.0, 0.0);
  if (k < n)
    {
    const double2 g = G[c * npix + ringstart[r] + k];
    const double2 w = chirp_w(k, n, 1.0);    // e^{+i pi k^2/n}
    v = make_double2(g.x * w.x - g.y * w.y, g.x * w.y + g.y * w.x);
    }
  A[(long long)row * M + k] = v;
  }

__global__ void bluestein_post(const int nmem, const int M, const long long npix,
                               const int* __restrict__ mem_ring,
                               const long long* __restrict__ ringstart,
                               const int* __restrict__ nphi,
                               const double2* __restrict__ C,
                               double* __restrict__ out)  // (nchunk, npix)
  {
  const int row = blockIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = row / nmem, mem = row % nmem;
  const int r = mem_ring[mem];
  const int n = nphi[r];
  if (j >= n) return;
  const double2 v = C[(long long)row * M + j];
  const double2 w = chirp_w(j, n, 1.0);      // e^{+i pi j^2/n}
  out[c * npix + ringstart[r] + j] = v.x * w.x - v.y * w.y;   // Re(w * conv)
  }

}  // extern "C"
