// Spin-2 kernels (appended to kernels.cu by plan.py; float64, healpy (E,B)->(Q,U)).
// Native ducc0 spin recursion: two scaled Wigner-d chains with Delta-l=1
// three-term recursion in cos(theta); see almond/reference_spin.py.

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

extern "C" {

// ---------------------------------------------------------------------------
// build_coef_spin: one thread per m.  Fills, for l = m..lmax+1 at
// soff[m] + (l-m):  fx = (a_l, b_l)  and  walpha = norm_l * alpha_l
// (walpha entries for l < mhi or l > lmax are 0).
// ---------------------------------------------------------------------------
#define SPN 2

__global__ void build_coef_spin(const int lmax, const int mmax,
                                const long long* __restrict__ soff,
                                double2* __restrict__ fx,
                                double* __restrict__ walpha)
  {
  const int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m > mmax) return;
  const long long off = soff[m];
  const int mhi = (m > SPN) ? m : SPN;
  const double md = m;

  for (int l = m; l <= lmax + 1; ++l)
    { fx[off + l - m] = make_double2(0.0, 0.0); walpha[off + l - m] = 0.0; }

  double al_prev = 0.0, al_cur = 1.0;   // alpha[mhi-1] (unused), alpha[mhi]
  for (int l = mhi; l <= lmax; ++l)
    {
    const double el = l;
    const double t = sqrt(1.0 / ((el + md + 1.0) * (el - md + 1.0)
                                 * (el + SPN + 1.0) * (el - SPN + 1.0)));
    const double flp10 = (el + 1.0) * (2.0 * el + 1.0) * t;
    const double flp11 = (l > 0) ? (md * SPN / (el * (el + 1.0))) : 0.0;
    const double t2 = sqrt((el + md) * (el - md) * (el + SPN) * (el - SPN)
                           / ((el + md + 1.0) * (el - md + 1.0)
                              * (el + SPN + 1.0) * (el - SPN + 1.0)));
    const double flp12 = (l > 0) ? (t2 * (el + 1.0) / el) : 0.0;
    const double al_next = (l > mhi) ? (al_prev * flp12) : 1.0;
    fx[off + l + 1 - m] = make_double2(flp10 * al_cur / al_next,
                                       flp11 * flp10 * al_cur / al_next);
    // walpha for this l: norm_l * alpha_l
    const double norm = (l >= SPN) ? (-0.5 * sqrt((2.0 * el + 1.0)
                                                  / (4.0 * M_PI))) : 0.0;
    walpha[off + l - m] = norm * al_cur;
    al_prev = al_cur;
    al_cur = al_next;
    }
  }

// ---------------------------------------------------------------------------
// prefold_spin: GC[2*(soff[m]+l-m)+{0,1}] = (aE, aB)(l,m) * walpha
// ---------------------------------------------------------------------------
__global__ void prefold_spin(const int lmax, const int mmax,
                             const long long* __restrict__ soff,
                             const double* __restrict__ walpha,
                             const long long* __restrict__ mstart,
                             const long long nalm,
                             const long long alm_cstride,  // double2 units
                             const long long gc_cstride,   // double2 units
                             const double2* __restrict__ alm_,  // (2, nalm)
                             double2* __restrict__ GC_)
  {
  const int m = blockIdx.y;
  const int dl = blockIdx.x * blockDim.x + threadIdx.x;
  const int l = m + dl;
  if (m > mmax || l > lmax + 1) return;
  const double2* __restrict__ alm = alm_
      + (long long)blockIdx.z * alm_cstride;
  double2* __restrict__ GC = GC_ + (long long)blockIdx.z * gc_cstride;
  const long long off = soff[m];
  const double w = walpha[off + dl];
  double2 e = make_double2(0.0, 0.0), b = e;
  if (l <= lmax && w != 0.0)
    {
    e = alm[mstart[m] + l];
    b = alm[nalm + mstart[m] + l];
    }
  GC[2 * (off + dl)]     = make_double2(w * e.x, w * e.y);
  GC[2 * (off + dl) + 1] = make_double2(w * b.x, w * b.y);
  }

// ---------------------------------------------------------------------------
// legendre_spin: one thread per (ring pair, m); writes
// phase[(comp)*(mmax+1)*nring + m*nring + ring] for comp = 0 (Q), 1 (U).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void spin_chain_init(
    const double cth, const int m, const int cosPow, const int sinPow,
    const bool pm_p, const bool pm_m, const double prefac, const int prescale,
    const double* __restrict__ powlimit_any,
    double& l2p, int& scp, double& l2m, int& scm)
  {
  const double cth2 = fmax(sqrt((1.0 + cth) * 0.5), 1e-15);
  const double sth2 = fmax(sqrt((1.0 - cth) * 0.5), 1e-15);
  double ccp, ssp, csp, sc2p; int ccps, ssps, csps, scps;
  // powlimit = 2.0 forces the scale-tracked slow path: these powers can
  // underflow while the huge prefac restores a representable product
  mypow_scaled(cth2, cosPow, 2.0, ccp, ccps);
  mypow_scaled(sth2, sinPow, 2.0, ssp, ssps);
  mypow_scaled(cth2, sinPow, 2.0, csp, csps);
  mypow_scaled(sth2, cosPow, 2.0, sc2p, scps);
  l2p = prefac * ccp; scp = prescale + ccps;
  l2m = prefac * csp; scm = prescale + csps;
  // normalize to FBIGHALF band, multiply second factor, then FTOL band
  while (fabs(l2p) > FBIGHALF) { l2p *= FSMALL; ++scp; }
  if (l2p != 0.0) while (fabs(l2p) < FBIGHALF * FSMALL) { l2p *= FBIG; --scp; }
  while (fabs(l2m) > FBIGHALF) { l2m *= FSMALL; ++scm; }
  if (l2m != 0.0) while (fabs(l2m) < FBIGHALF * FSMALL) { l2m *= FBIG; --scm; }
  l2p *= ssp; scp += ssps;
  l2m *= sc2p; scm += scps;
  if (pm_p) l2p = -l2p;
  if (pm_m) l2m = -l2m;
  while (fabs(l2p) > FTOL) { l2p *= FSMALL; ++scp; }
  if (l2p != 0.0) while (fabs(l2p) < FTOL * FSMALL) { l2p *= FBIG; --scp; }
  while (fabs(l2m) > FTOL) { l2m *= FSMALL; ++scm; }
  if (l2m != 0.0) while (fabs(l2m) < FTOL * FSMALL) { l2m *= FBIG; --scm; }
  while (scp > 0) { l2p *= FBIG; --scp; }
  while (scm > 0) { l2m *= FBIG; --scm; }
  }

__global__ void legendre_spin(const int lmax, const int mmax,
                              const int npair, const int nring,
                              const long long* __restrict__ soff,
                              const double2* __restrict__ fx,
                              const double2* __restrict__ GC_,
                              const double* __restrict__ prefac,
                              const int* __restrict__ prescale,
                              const double* __restrict__ pair_cth,
                              const double* __restrict__ pair_sth,
                              const int* __restrict__ pair_mlim,
                              const int* __restrict__ pair_inorth,
                              const int* __restrict__ pair_isouth,
                              const long long* __restrict__ ringstart,
                              const int* __restrict__ nphi,
                              const int* __restrict__ phi0_num,
                              const int* __restrict__ phi0_den,
                              const long long npix,
                              const long long gc_cstride,  // double2 units
                              const long long g_cstride,   // double2 units
                              double2* __restrict__ G_)   // (2, npix)
  {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  if (p >= npair || m > mmax) return;
  if (pair_mlim[p] < m) return;
  const double2* __restrict__ GC = GC_ + (long long)blockIdx.z * gc_cstride;
  double2* __restrict__ G = G_ + (long long)blockIdx.z * g_cstride;

  const double cth = pair_cth[p];
  const long long off = soff[m];
  const int mhi = (m > SPN) ? m : SPN;
  int cosPow, sinPow; bool pm_p, pm_m;
  if (mhi == m) { cosPow = mhi + SPN; sinPow = mhi - SPN;
                  pm_p = pm_m = ((mhi - SPN) & 1); }
  else          { cosPow = mhi + m; sinPow = mhi - m;
                  pm_p = false; pm_m = ((mhi + m) & 1); }

  double l2p, l2m; int scp, scm;
  spin_chain_init(cth, m, cosPow, sinPow, pm_p, pm_m,
                  prefac[m], prescale[m], nullptr, l2p, scp, l2m, scm);
  if (!isfinite(l2p) || !isfinite(l2m)) return;
  double l1p = 0.0, l1m = 0.0;

  double p1pr = 0, p1pi = 0, p1mr = 0, p1mi = 0;
  double p2pr = 0, p2pi = 0, p2mr = 0, p2mi = 0;

  for (int l = mhi; l <= lmax; ++l)
    {
    const int dl = l - m;
    const double2 g = GC[2 * (off + dl)];      // aG (E)
    const double2 c = GC[2 * (off + dl) + 1];  // aC (B)
    const double vp = (scp == 0) ? l2p : 0.0;
    const double vm = (scm == 0) ? l2m : 0.0;
    if (((l - mhi) & 1) == 0)
      {
      p1pr += g.x * vp; p1pi += g.y * vp;
      p1mr += c.x * vp; p1mi += c.y * vp;
      p2pr -= c.y * vm; p2pi += c.x * vm;
      p2mr += g.y * vm; p2mi -= g.x * vm;
      }
    else
      {
      p1pr += c.y * vp; p1pi -= c.x * vp;
      p1mr -= g.y * vp; p1mi += g.x * vp;
      p2pr += g.x * vm; p2pi += g.y * vm;
      p2mr += c.x * vm; p2mi += c.y * vm;
      }
    const double2 f = fx[off + l + 1 - m];
    const double np_ = (cth * f.x - f.y) * l2p - l1p;
    const double nm_ = (cth * f.x + f.y) * l2m - l1m;
    l1p = l2p; l2p = np_;
    l1m = l2m; l2m = nm_;
    if (scp < 0 && fabs(l2p) > FTOL) { l1p *= FSMALL; l2p *= FSMALL; ++scp; }
    if (scm < 0 && fabs(l2m) > FTOL) { l1m *= FSMALL; l2m *= FSMALL; ++scm; }
    }

  const double fct = ((mhi - m + SPN) & 1) ? -1.0 : 1.0;
  // q1p = p1p + i p2m ; q2p = p2p - i p1m ; q1m = p1m - i p2p ; q2m = p2m + i p1p
  const double q1pr = p1pr - p2mi, q1pi = p1pi + p2mr;
  const double q2pr = p2pr + p1mi, q2pi = p2pi - p1mr;
  const double q1mr = p1mr + p2pi, q1mi = p1mi - p2pr;
  const double q2mr = p2mr - p1pi, q2mi = p2mi + p1pr;

  const int rn = pair_inorth[p], rs = pair_isouth[p];
  fold_scatter(make_double2(q1pr + q2pr, q1pi + q2pi), rn, m,
               ringstart, nphi, phi0_num, phi0_den, G);
  fold_scatter(make_double2(q1mr + q2mr, q1mi + q2mi), rn, m,
               ringstart, nphi, phi0_num, phi0_den, G + npix);
  if (rs != rn)
    {
    fold_scatter(make_double2(fct * (q1pr - q2pr), fct * (q1pi - q2pi)),
                 rs, m, ringstart, nphi, phi0_num, phi0_den, G);
    fold_scatter(make_double2(fct * (q1mr - q2mr), fct * (q1mi - q2mi)),
                 rs, m, ringstart, nphi, phi0_num, phi0_den, G + npix);
    }
  }

// ---------------------------------------------------------------------------
// legendre_spin_adj: block per m, transpose of legendre_spin.
// Reduces aG', aC' over ring pairs into GCadj (same layout as GC, pre-zeroed).
// ---------------------------------------------------------------------------
#define SADJ_PPT  4
#define SADJ_TILE 64

__global__ void legendre_spin_adj(const int lmax, const int mmax,
                                  const int npair, const int nring,
                                  const long long* __restrict__ soff,
                                  const double2* __restrict__ fx,
                                  const double* __restrict__ prefac,
                                  const int* __restrict__ prescale,
                                  const double* __restrict__ pair_cth,
                                  const double* __restrict__ pair_sth,
                                  const int* __restrict__ pair_mlim,
                                  const int* __restrict__ pair_inorth,
                                  const int* __restrict__ pair_isouth,
                                  const int* __restrict__ pstart,
                                  const double2* __restrict__ Gh, // (2,nghalf)
                                  const long long* __restrict__ hstart,
                                  const int* __restrict__ nphi,
                                  const int* __restrict__ phi0_num,
                                  const int* __restrict__ phi0_den,
                                  const long long nghalf,
                                  double2* __restrict__ GCadj)
  {
  const int m = blockIdx.x;
  if (m > mmax) return;
  const long long off = soff[m];
  const int mhi = (m > SPN) ? m : SPN;
  const int nl = lmax - mhi + 1;
  if (nl <= 0) return;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int nwarp = blockDim.x >> 5;
  extern __shared__ double tile[];   // [nwarp][SADJ_TILE][4]  (aG', aC')
  const double fct = ((mhi - m + SPN) & 1) ? -1.0 : 1.0;
  int cosPow, sinPow; bool pm_p, pm_m;
  if (mhi == m) { cosPow = mhi + SPN; sinPow = mhi - SPN;
                  pm_p = pm_m = ((mhi - SPN) & 1); }
  else          { cosPow = mhi + m; sinPow = mhi - m;
                  pm_p = false; pm_m = ((mhi + m) & 1); }
  const long long cstride = (long long)(mmax + 1) * nring;
  const long long idx0 = (long long)m * nring;

  for (long long base = pstart[m]; base < npair;
       base += (long long)blockDim.x * SADJ_PPT)
    {
    double l1p[SADJ_PPT], l2p[SADJ_PPT], l1m[SADJ_PPT], l2m[SADJ_PPT];
    double cthq[SADJ_PPT];
    int scp[SADJ_PPT], scm[SADJ_PPT];
    // p-accumulator adjoints per pair (complex as re/im)
    double a1pr[SADJ_PPT], a1pi[SADJ_PPT], a1mr[SADJ_PPT], a1mi[SADJ_PPT];
    double a2pr[SADJ_PPT], a2pi[SADJ_PPT], a2mr[SADJ_PPT], a2mi[SADJ_PPT];
#pragma unroll
    for (int q = 0; q < SADJ_PPT; ++q)
      {
      const long long p = base + (long long)q * blockDim.x + tid;
      const bool act = (p < npair) && (pair_mlim[p] >= m);
      l1p[q] = l2p[q] = l1m[q] = l2m[q] = cthq[q] = 0.0;
      scp[q] = scm[q] = 0;
      a1pr[q] = a1pi[q] = a1mr[q] = a1mi[q] = 0.0;
      a2pr[q] = a2pi[q] = a2mr[q] = a2mi[q] = 0.0;
      if (!act) continue;
      cthq[q] = pair_cth[p];
      const int rn = pair_inorth[p], rs = pair_isouth[p];
      const double2 F0N = unfold_gather(rn, m, Gh, hstart, nphi,
                                        phi0_num, phi0_den);
      const double2 F1N = unfold_gather(rn, m, Gh + nghalf, hstart, nphi,
                                        phi0_num, phi0_den);
      double2 F0S = make_double2(0.0, 0.0), F1S = F0S;
      if (rs != rn)
        {
        F0S = unfold_gather(rs, m, Gh, hstart, nphi, phi0_num, phi0_den);
        F1S = unfold_gather(rs, m, Gh + nghalf, hstart, nphi,
                            phi0_num, phi0_den);
        }
      // q' combinations
      const double q1pr = F0N.x + fct * F0S.x, q1pi = F0N.y + fct * F0S.y;
      const double q2pr = F0N.x - fct * F0S.x, q2pi = F0N.y - fct * F0S.y;
      const double q1mr = F1N.x + fct * F1S.x, q1mi = F1N.y + fct * F1S.y;
      const double q2mr = F1N.x - fct * F1S.x, q2mi = F1N.y - fct * F1S.y;
      // p' = transpose of the q combinations
      a1pr[q] = q1pr + q2mi;  a1pi[q] = q1pi - q2mr;   // p1p' = q1p' - i q2m'
      a2mr[q] = q2mr + q1pi;  a2mi[q] = q2mi - q1pr;   // p2m' = q2m' - i q1p'
      a2pr[q] = q2pr - q1mi;  a2pi[q] = q2pi + q1mr;   // p2p' = q2p' + i q1m'
      a1mr[q] = q1mr - q2pi;  a1mi[q] = q1mi + q2pr;   // p1m' = q1m' + i q2p'
      double vp, vm; int sp, sm;
      spin_chain_init(cthq[q], m, cosPow, sinPow, pm_p, pm_m,
                      prefac[m], prescale[m], nullptr, vp, sp, vm, sm);
      if (isfinite(vp) && isfinite(vm))
        { l2p[q] = vp; scp[q] = sp; l2m[q] = vm; scm[q] = sm; }
      }

    for (int seg = 0; seg < nl; seg += SADJ_TILE)
      {
      const int end = min(seg + SADJ_TILE, nl);
      const int nsl = (end - seg) * 4;
      for (int i = lane; i < nsl; i += 32)
        tile[(warp * SADJ_TILE) * 4 + i] = 0.0;
      __syncwarp();

      for (int il = seg; il < end; ++il)
        {
        const int l = mhi + il;
        const double2 f = fx[off + l + 1 - m];
        double gr = 0, gi = 0, cr = 0, ci = 0;   // aG', aC' contributions
#pragma unroll
        for (int q = 0; q < SADJ_PPT; ++q)
          {
          const double vp = (scp[q] == 0) ? l2p[q] : 0.0;
          const double vm = (scm[q] == 0) ? l2m[q] : 0.0;
          if (((l - mhi) & 1) == 0)
            {
            // aG' += vp p1p' + i vm p2m';  aC' += vp p1m' - i vm p2p'
            gr += vp * a1pr[q] - vm * a2mi[q];
            gi += vp * a1pi[q] + vm * a2mr[q];
            cr += vp * a1mr[q] + vm * a2pi[q];
            ci += vp * a1mi[q] - vm * a2pr[q];
            }
          else
            {
            // aG' += -i vp p1m' + vm p2p';  aC' += i vp p1p' + vm p2m'
            gr +=  vp * a1mi[q] + vm * a2pr[q];
            gi += -vp * a1mr[q] + vm * a2pi[q];
            cr += -vp * a1pi[q] + vm * a2mr[q];
            ci +=  vp * a1pr[q] + vm * a2mi[q];
            }
          const double np_ = (cthq[q] * f.x - f.y) * l2p[q] - l1p[q];
          const double nm_ = (cthq[q] * f.x + f.y) * l2m[q] - l1m[q];
          l1p[q] = l2p[q]; l2p[q] = np_;
          l1m[q] = l2m[q]; l2m[q] = nm_;
          if (scp[q] < 0 && fabs(l2p[q]) > FTOL)
            { l1p[q] *= FSMALL; l2p[q] *= FSMALL; ++scp[q]; }
          if (scm[q] < 0 && fabs(l2m[q]) > FTOL)
            { l1m[q] *= FSMALL; l2m[q] *= FSMALL; ++scm[q]; }
          }
#pragma unroll
        for (int o = 16; o; o >>= 1)
          {
          gr += __shfl_down_sync(0xffffffffu, gr, o);
          gi += __shfl_down_sync(0xffffffffu, gi, o);
          cr += __shfl_down_sync(0xffffffffu, cr, o);
          ci += __shfl_down_sync(0xffffffffu, ci, o);
          }
        if (lane == 0)
          {
          double* slot = &tile[((warp * SADJ_TILE) + (il - seg)) * 4];
          slot[0] += gr; slot[1] += gi; slot[2] += cr; slot[3] += ci;
          }
        }
      __syncthreads();
      for (int i = tid; i < nsl; i += blockDim.x)
        {
        double s = 0.0;
        for (int w = 0; w < nwarp; ++w)
          s += tile[((w * SADJ_TILE) + (i >> 2)) * 4 + (i & 3)];
        const int il = seg + (i >> 2);
        const int dl = mhi + il - m;
        double* g = (double*)&GCadj[2 * (off + dl)];
        g[i & 3] += s;
        }
      __syncthreads();
      }
    }
  }

// ---------------------------------------------------------------------------
// legendre_spin_2p: like legendre_spin, but each thread owns TWO adjacent
// ring pairs (the legendre2 trick): the fx / GC loads are shared between the
// pairs and four independent recursion chains hide the FMA latency.  A
// guarded loop runs while any chain is still scale-tracked; after surfacing
// a fast loop drops the selects and rescale checks (parity branch is
// warp-uniform in l).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void spin_pair_init(
    const bool act, const double cth, const int m,
    const int cosPow, const int sinPow, const bool pm_p, const bool pm_m,
    const double prefac_m, const int prescale_m,
    double& l1p, double& l2p, double& l1m, double& l2m, int& scp, int& scm)
  {
  l1p = l2p = l1m = l2m = 0.0; scp = scm = 0;
  if (!act) return;
  double vp, vm; int sp, sm;
  spin_chain_init(cth, m, cosPow, sinPow, pm_p, pm_m,
                  prefac_m, prescale_m, nullptr, vp, sp, vm, sm);
  if (isfinite(vp) && isfinite(vm))
    { l2p = vp; scp = sp; l2m = vm; scm = sm; }
  }

__global__ void legendre_spin_2p(const int lmax, const int mmax,
                                 const int npair, const int nring,
                                 const long long* __restrict__ soff,
                                 const double2* __restrict__ fx,
                                 const double2* __restrict__ GC_,
                                 const double* __restrict__ prefac,
                                 const int* __restrict__ prescale,
                                 const double* __restrict__ pair_cth,
                                 const double* __restrict__ pair_sth,
                                 const int* __restrict__ pair_mlim,
                                 const int* __restrict__ pair_inorth,
                                 const int* __restrict__ pair_isouth,
                                 const long long* __restrict__ ringstart,
                                 const int* __restrict__ nphi,
                                 const int* __restrict__ phi0_num,
                                 const int* __restrict__ phi0_den,
                                 const long long npix,
                                 const long long gc_cstride,
                                 const long long g_cstride,
                                 double2* __restrict__ G_)
  {
  const int t2 = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  const int pa = 2 * t2, pb = 2 * t2 + 1;
  if (pa >= npair || m > mmax) return;
  const double2* __restrict__ GC = GC_ + (long long)blockIdx.z * gc_cstride;
  double2* __restrict__ G = G_ + (long long)blockIdx.z * g_cstride;

  const long long off = soff[m];
  const int mhi = (m > SPN) ? m : SPN;
  int cosPow, sinPow; bool pm_p, pm_m;
  if (mhi == m) { cosPow = mhi + SPN; sinPow = mhi - SPN;
                  pm_p = pm_m = ((mhi - SPN) & 1); }
  else          { cosPow = mhi + m; sinPow = mhi - m;
                  pm_p = false; pm_m = ((mhi + m) & 1); }
  const double pf = prefac[m]; const int ps = prescale[m];

  const bool acta = pair_mlim[pa] >= m;
  const bool actb = (pb < npair) && (pair_mlim[pb] >= m);
  if (!acta && !actb) return;
  const double ctha = acta ? pair_cth[pa] : 0.0;
  const double cthb = actb ? pair_cth[pb] : 0.0;
  double al1p, al2p, al1m, al2m; int ascp, ascm;
  double bl1p, bl2p, bl1m, bl2m; int bscp, bscm;
  spin_pair_init(acta, ctha, m, cosPow, sinPow, pm_p, pm_m, pf, ps,
                 al1p, al2p, al1m, al2m, ascp, ascm);
  spin_pair_init(actb, cthb, m, cosPow, sinPow, pm_p, pm_m, pf, ps,
                 bl1p, bl2p, bl1m, bl2m, bscp, bscm);

  double ap1pr = 0, ap1pi = 0, ap1mr = 0, ap1mi = 0;
  double ap2pr = 0, ap2pi = 0, ap2mr = 0, ap2mi = 0;
  double bp1pr = 0, bp1pi = 0, bp1mr = 0, bp1mi = 0;
  double bp2pr = 0, bp2pi = 0, bp2mr = 0, bp2mi = 0;

  bool anyneg = (ascp < 0) | (ascm < 0) | (bscp < 0) | (bscm < 0);
  int l = mhi;
  // guarded loop: some chain still below IEEE range
  for (; l <= lmax && anyneg; ++l)
    {
    const int dl = l - m;
    const double2 g = GC[2 * (off + dl)];
    const double2 c = GC[2 * (off + dl) + 1];
    const double2 f = fx[off + l + 1 - m];
    const double avp = (ascp == 0) ? al2p : 0.0;
    const double avm = (ascm == 0) ? al2m : 0.0;
    const double bvp = (bscp == 0) ? bl2p : 0.0;
    const double bvm = (bscm == 0) ? bl2m : 0.0;
    if (((l - mhi) & 1) == 0)
      {
      ap1pr += g.x * avp; ap1pi += g.y * avp;
      ap1mr += c.x * avp; ap1mi += c.y * avp;
      ap2pr -= c.y * avm; ap2pi += c.x * avm;
      ap2mr += g.y * avm; ap2mi -= g.x * avm;
      bp1pr += g.x * bvp; bp1pi += g.y * bvp;
      bp1mr += c.x * bvp; bp1mi += c.y * bvp;
      bp2pr -= c.y * bvm; bp2pi += c.x * bvm;
      bp2mr += g.y * bvm; bp2mi -= g.x * bvm;
      }
    else
      {
      ap1pr += c.y * avp; ap1pi -= c.x * avp;
      ap1mr -= g.y * avp; ap1mi += g.x * avp;
      ap2pr += g.x * avm; ap2pi += g.y * avm;
      ap2mr += c.x * avm; ap2mi += c.y * avm;
      bp1pr += c.y * bvp; bp1pi -= c.x * bvp;
      bp1mr -= g.y * bvp; bp1mi += g.x * bvp;
      bp2pr += g.x * bvm; bp2pi += g.y * bvm;
      bp2mr += c.x * bvm; bp2mi += c.y * bvm;
      }
    double t;
    t = (ctha * f.x - f.y) * al2p - al1p; al1p = al2p; al2p = t;
    t = (ctha * f.x + f.y) * al2m - al1m; al1m = al2m; al2m = t;
    t = (cthb * f.x - f.y) * bl2p - bl1p; bl1p = bl2p; bl2p = t;
    t = (cthb * f.x + f.y) * bl2m - bl1m; bl1m = bl2m; bl2m = t;
    if (ascp < 0 && fabs(al2p) > FTOL) { al1p *= FSMALL; al2p *= FSMALL; ++ascp; }
    if (ascm < 0 && fabs(al2m) > FTOL) { al1m *= FSMALL; al2m *= FSMALL; ++ascm; }
    if (bscp < 0 && fabs(bl2p) > FTOL) { bl1p *= FSMALL; bl2p *= FSMALL; ++bscp; }
    if (bscm < 0 && fabs(bl2m) > FTOL) { bl1m *= FSMALL; bl2m *= FSMALL; ++bscm; }
    anyneg = (ascp < 0) | (ascm < 0) | (bscp < 0) | (bscm < 0);
    }
  // fast loop: all chains surfaced (parity branch is warp-uniform in l)
  for (; l <= lmax; ++l)
    {
    const int dl = l - m;
    const double2 g = GC[2 * (off + dl)];
    const double2 c = GC[2 * (off + dl) + 1];
    const double2 f = fx[off + l + 1 - m];
    if (((l - mhi) & 1) == 0)
      {
      ap1pr += g.x * al2p; ap1pi += g.y * al2p;
      ap1mr += c.x * al2p; ap1mi += c.y * al2p;
      ap2pr -= c.y * al2m; ap2pi += c.x * al2m;
      ap2mr += g.y * al2m; ap2mi -= g.x * al2m;
      bp1pr += g.x * bl2p; bp1pi += g.y * bl2p;
      bp1mr += c.x * bl2p; bp1mi += c.y * bl2p;
      bp2pr -= c.y * bl2m; bp2pi += c.x * bl2m;
      bp2mr += g.y * bl2m; bp2mi -= g.x * bl2m;
      }
    else
      {
      ap1pr += c.y * al2p; ap1pi -= c.x * al2p;
      ap1mr -= g.y * al2p; ap1mi += g.x * al2p;
      ap2pr += g.x * al2m; ap2pi += g.y * al2m;
      ap2mr += c.x * al2m; ap2mi += c.y * al2m;
      bp1pr += c.y * bl2p; bp1pi -= c.x * bl2p;
      bp1mr -= g.y * bl2p; bp1mi += g.x * bl2p;
      bp2pr += g.x * bl2m; bp2pi += g.y * bl2m;
      bp2mr += c.x * bl2m; bp2mi += c.y * bl2m;
      }
    double t;
    t = (ctha * f.x - f.y) * al2p - al1p; al1p = al2p; al2p = t;
    t = (ctha * f.x + f.y) * al2m - al1m; al1m = al2m; al2m = t;
    t = (cthb * f.x - f.y) * bl2p - bl1p; bl1p = bl2p; bl2p = t;
    t = (cthb * f.x + f.y) * bl2m - bl1m; bl1m = bl2m; bl2m = t;
    }

  const double fct = ((mhi - m + SPN) & 1) ? -1.0 : 1.0;
  if (acta)
    {
    const double q1pr = ap1pr - ap2mi, q1pi = ap1pi + ap2mr;
    const double q2pr = ap2pr + ap1mi, q2pi = ap2pi - ap1mr;
    const double q1mr = ap1mr + ap2pi, q1mi = ap1mi - ap2pr;
    const double q2mr = ap2mr - ap1pi, q2mi = ap2mi + ap1pr;
    const int rn = pair_inorth[pa], rs = pair_isouth[pa];
    fold_scatter(make_double2(q1pr + q2pr, q1pi + q2pi), rn, m,
                 ringstart, nphi, phi0_num, phi0_den, G);
    fold_scatter(make_double2(q1mr + q2mr, q1mi + q2mi), rn, m,
                 ringstart, nphi, phi0_num, phi0_den, G + npix);
    if (rs != rn)
      {
      fold_scatter(make_double2(fct * (q1pr - q2pr), fct * (q1pi - q2pi)),
                   rs, m, ringstart, nphi, phi0_num, phi0_den, G);
      fold_scatter(make_double2(fct * (q1mr - q2mr), fct * (q1mi - q2mi)),
                   rs, m, ringstart, nphi, phi0_num, phi0_den, G + npix);
      }
    }
  if (actb)
    {
    const double q1pr = bp1pr - bp2mi, q1pi = bp1pi + bp2mr;
    const double q2pr = bp2pr + bp1mi, q2pi = bp2pi - bp1mr;
    const double q1mr = bp1mr + bp2pi, q1mi = bp1mi - bp2pr;
    const double q2mr = bp2mr - bp1pi, q2mi = bp2mi + bp1pr;
    const int rn = pair_inorth[pb], rs = pair_isouth[pb];
    fold_scatter(make_double2(q1pr + q2pr, q1pi + q2pi), rn, m,
                 ringstart, nphi, phi0_num, phi0_den, G);
    fold_scatter(make_double2(q1mr + q2mr, q1mi + q2mi), rn, m,
                 ringstart, nphi, phi0_num, phi0_den, G + npix);
    if (rs != rn)
      {
      fold_scatter(make_double2(fct * (q1pr - q2pr), fct * (q1pi - q2pi)),
                   rs, m, ringstart, nphi, phi0_num, phi0_den, G);
      fold_scatter(make_double2(fct * (q1mr - q2mr), fct * (q1mi - q2mi)),
                   rs, m, ringstart, nphi, phi0_num, phi0_den, G + npix);
      }
    }
  }

// ---------------------------------------------------------------------------
// legendre_spin_adj2: like legendre_spin_adj, but with the per-il warp
// shuffle reduction replaced by staged bulk reduction (see legendre_adj2),
// and a guarded/fast loop split: once every chain of a lane has surfaced
// (scp == scm == 0) the select and rescale guards drop out of the inner
// loop.  The even/odd-l branch is warp-uniform (il is uniform per warp).
// ---------------------------------------------------------------------------
#ifndef SADJ2_PPT
#define SADJ2_PPT 2
#endif
#ifndef SADJ2_TR
#define SADJ2_TR 8
#endif

__global__ void legendre_spin_adj2(const int lmax, const int mmax,
                                   const int npair, const int nring,
                                   const long long* __restrict__ soff,
                                   const double2* __restrict__ fx,
                                   const double* __restrict__ prefac,
                                   const int* __restrict__ prescale,
                                   const double* __restrict__ pair_cth,
                                   const double* __restrict__ pair_sth,
                                   const int* __restrict__ pair_mlim,
                                   const int* __restrict__ pair_inorth,
                                   const int* __restrict__ pair_isouth,
                                   const int* __restrict__ pstart,
                                   const double2* __restrict__ Gh_, // (2,nghalf)
                                   const long long* __restrict__ hstart,
                                   const int* __restrict__ nphi,
                                   const int* __restrict__ phi0_num,
                                   const int* __restrict__ phi0_den,
                                   const long long nghalf,
                                   const long long gh_cstride,  // double2 units
                                   const long long gc_cstride,  // double2 units
                                   double2* __restrict__ GCadj_)
  {
  const int m = blockIdx.x;
  if (m > mmax) return;
  const double2* __restrict__ Gh = Gh_ + (long long)blockIdx.y * gh_cstride;
  double2* __restrict__ GCadj = GCadj_ + (long long)blockIdx.y * gc_cstride;
  const long long off = soff[m];
  const int mhi = (m > SPN) ? m : SPN;
  const int nl = lmax - mhi + 1;
  if (nl <= 0) return;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int nwarp = blockDim.x >> 5;
  extern __shared__ double sh[];
  double* stage = sh;
  double* xtile = sh + (long long)nwarp * SADJ2_TR * 4 * ADJ2_W;
  const double fct = ((mhi - m + SPN) & 1) ? -1.0 : 1.0;
  int cosPow, sinPow; bool pm_p, pm_m;
  if (mhi == m) { cosPow = mhi + SPN; sinPow = mhi - SPN;
                  pm_p = pm_m = ((mhi - SPN) & 1); }
  else          { cosPow = mhi + m; sinPow = mhi - m;
                  pm_p = false; pm_m = ((mhi + m) & 1); }
  const int dl0 = mhi - m;

  for (long long base = pstart[m]; base < npair;
       base += (long long)blockDim.x * SADJ2_PPT)
    {
    double l1p[SADJ2_PPT], l2p[SADJ2_PPT], l1m[SADJ2_PPT], l2m[SADJ2_PPT];
    double cthq[SADJ2_PPT];
    int scp[SADJ2_PPT], scm[SADJ2_PPT];
    double a1pr[SADJ2_PPT], a1pi[SADJ2_PPT], a1mr[SADJ2_PPT], a1mi[SADJ2_PPT];
    double a2pr[SADJ2_PPT], a2pi[SADJ2_PPT], a2mr[SADJ2_PPT], a2mi[SADJ2_PPT];
#pragma unroll
    for (int q = 0; q < SADJ2_PPT; ++q)
      {
      const long long p = base + (long long)q * blockDim.x + tid;
      const bool act = (p < npair) && (pair_mlim[p] >= m);
      l1p[q] = l2p[q] = l1m[q] = l2m[q] = cthq[q] = 0.0;
      scp[q] = scm[q] = 0;
      a1pr[q] = a1pi[q] = a1mr[q] = a1mi[q] = 0.0;
      a2pr[q] = a2pi[q] = a2mr[q] = a2mi[q] = 0.0;
      if (!act) continue;
      cthq[q] = pair_cth[p];
      const int rn = pair_inorth[p], rs = pair_isouth[p];
      const double2 F0N = unfold_gather(rn, m, Gh, hstart, nphi,
                                        phi0_num, phi0_den);
      const double2 F1N = unfold_gather(rn, m, Gh + nghalf, hstart, nphi,
                                        phi0_num, phi0_den);
      double2 F0S = make_double2(0.0, 0.0), F1S = F0S;
      if (rs != rn)
        {
        F0S = unfold_gather(rs, m, Gh, hstart, nphi, phi0_num, phi0_den);
        F1S = unfold_gather(rs, m, Gh + nghalf, hstart, nphi,
                            phi0_num, phi0_den);
        }
      const double q1pr = F0N.x + fct * F0S.x, q1pi = F0N.y + fct * F0S.y;
      const double q2pr = F0N.x - fct * F0S.x, q2pi = F0N.y - fct * F0S.y;
      const double q1mr = F1N.x + fct * F1S.x, q1mi = F1N.y + fct * F1S.y;
      const double q2mr = F1N.x - fct * F1S.x, q2mi = F1N.y - fct * F1S.y;
      a1pr[q] = q1pr + q2mi;  a1pi[q] = q1pi - q2mr;   // p1p' = q1p' - i q2m'
      a2mr[q] = q2mr + q1pi;  a2mi[q] = q2mi - q1pr;   // p2m' = q2m' - i q1p'
      a2pr[q] = q2pr - q1mi;  a2pi[q] = q2pi + q1mr;   // p2p' = q2p' + i q1m'
      a1mr[q] = q1mr - q2pi;  a1mi[q] = q1mi + q2pr;   // p1m' = q1m' + i q2p'
      double vp, vm; int sp, sm;
      spin_chain_init(cthq[q], m, cosPow, sinPow, pm_p, pm_m,
                      prefac[m], prescale[m], nullptr, vp, sp, vm, sm);
      if (isfinite(vp) && isfinite(vm))
        { l2p[q] = vp; scp[q] = sp; l2m[q] = vm; scm[q] = sm; }
      }
    bool anyneg = false;
#pragma unroll
    for (int q = 0; q < SADJ2_PPT; ++q)
      anyneg |= (scp[q] < 0) | (scm[q] < 0);

    int par = 0;
    for (int seg = 0; seg < nl; seg += SADJ2_TR, par ^= 1)
      {
      const int end = min(seg + SADJ2_TR, nl);
      int il = seg;
      // guarded loop
      for (; il < end && anyneg; ++il)
        {
        const int l = mhi + il;
        const double2 f = fx[off + l + 1 - m];
        double gr = 0, gi = 0, cr = 0, ci = 0;
        anyneg = false;
#pragma unroll
        for (int q = 0; q < SADJ2_PPT; ++q)
          {
          const double vp = (scp[q] == 0) ? l2p[q] : 0.0;
          const double vm = (scm[q] == 0) ? l2m[q] : 0.0;
          if ((il & 1) == 0)
            {
            gr += vp * a1pr[q] - vm * a2mi[q];
            gi += vp * a1pi[q] + vm * a2mr[q];
            cr += vp * a1mr[q] + vm * a2pi[q];
            ci += vp * a1mi[q] - vm * a2pr[q];
            }
          else
            {
            gr +=  vp * a1mi[q] + vm * a2pr[q];
            gi += -vp * a1mr[q] + vm * a2pi[q];
            cr += -vp * a1pi[q] + vm * a2mr[q];
            ci +=  vp * a1pr[q] + vm * a2mi[q];
            }
          const double np_ = (cthq[q] * f.x - f.y) * l2p[q] - l1p[q];
          const double nm_ = (cthq[q] * f.x + f.y) * l2m[q] - l1m[q];
          l1p[q] = l2p[q]; l2p[q] = np_;
          l1m[q] = l2m[q]; l2m[q] = nm_;
          if (scp[q] < 0 && fabs(l2p[q]) > FTOL)
            { l1p[q] *= FSMALL; l2p[q] *= FSMALL; ++scp[q]; }
          if (scm[q] < 0 && fabs(l2m[q]) > FTOL)
            { l1m[q] *= FSMALL; l2m[q] *= FSMALL; ++scm[q]; }
          anyneg |= (scp[q] < 0) | (scm[q] < 0);
          }
        double* srow = stage
            + ((long long)(warp * SADJ2_TR + (il - seg)) * 4) * ADJ2_W + lane;
        srow[0 * ADJ2_W] = gr; srow[1 * ADJ2_W] = gi;
        srow[2 * ADJ2_W] = cr; srow[3 * ADJ2_W] = ci;
        }
      // fast loop: no selects, no rescale guards (il uniform per warp so
      // the parity branch does not diverge)
      for (; il < end; ++il)
        {
        const int l = mhi + il;
        const double2 f = fx[off + l + 1 - m];
        double gr = 0, gi = 0, cr = 0, ci = 0;
        if ((il & 1) == 0)
          {
#pragma unroll
          for (int q = 0; q < SADJ2_PPT; ++q)
            {
            gr += l2p[q] * a1pr[q] - l2m[q] * a2mi[q];
            gi += l2p[q] * a1pi[q] + l2m[q] * a2mr[q];
            cr += l2p[q] * a1mr[q] + l2m[q] * a2pi[q];
            ci += l2p[q] * a1mi[q] - l2m[q] * a2pr[q];
            const double np_ = (cthq[q] * f.x - f.y) * l2p[q] - l1p[q];
            const double nm_ = (cthq[q] * f.x + f.y) * l2m[q] - l1m[q];
            l1p[q] = l2p[q]; l2p[q] = np_;
            l1m[q] = l2m[q]; l2m[q] = nm_;
            }
          }
        else
          {
#pragma unroll
          for (int q = 0; q < SADJ2_PPT; ++q)
            {
            gr +=  l2p[q] * a1mi[q] + l2m[q] * a2pr[q];
            gi += -l2p[q] * a1mr[q] + l2m[q] * a2pi[q];
            cr += -l2p[q] * a1pi[q] + l2m[q] * a2mr[q];
            ci +=  l2p[q] * a1pr[q] + l2m[q] * a2mi[q];
            const double np_ = (cthq[q] * f.x - f.y) * l2p[q] - l1p[q];
            const double nm_ = (cthq[q] * f.x + f.y) * l2m[q] - l1m[q];
            l1p[q] = l2p[q]; l2p[q] = np_;
            l1m[q] = l2m[q]; l2m[q] = nm_;
            }
          }
        double* srow = stage
            + ((long long)(warp * SADJ2_TR + (il - seg)) * 4) * ADJ2_W + lane;
        srow[0 * ADJ2_W] = gr; srow[1 * ADJ2_W] = gi;
        srow[2 * ADJ2_W] = cr; srow[3 * ADJ2_W] = ci;
        }
      __syncwarp();
      const int nout = (end - seg) * 4;
#if SADJ2_TR == 8
      double s = 0.0;
      if (lane < nout)
        {
        const double* srow = stage
            + ((long long)(warp * SADJ2_TR) * 4 + lane) * ADJ2_W;
#pragma unroll
        for (int k = 0; k < 32; ++k) s += srow[k];
        }
      if (lane < nout)
        xtile[(par * nwarp + warp) * (SADJ2_TR * 4 + 1) + lane] = s;
#else
      const int oid = lane >> 1, h = lane & 1;
      double s = 0.0;
      if (oid < nout)
        {
        const double* srow = stage
            + ((long long)(warp * SADJ2_TR) * 4 + oid) * ADJ2_W;
#pragma unroll
        for (int k = 0; k < 16; ++k) s += srow[2 * k + h];
        }
      s += __shfl_down_sync(0xffffffffu, s, 1);
      if (h == 0 && oid < nout)
        xtile[(par * nwarp + warp) * (SADJ2_TR * 4 + 1) + oid] = s;
#endif
      __syncthreads();
      if (tid < nout)
        {
        double acc = 0.0;
        for (int w = 0; w < nwarp; ++w)
          acc += xtile[(par * nwarp + w) * (SADJ2_TR * 4 + 1) + tid];
        const int dl = dl0 + seg + (tid >> 2);
        double* g = (double*)&GCadj[2 * (off + dl)];
        g[tid & 3] += acc;
        }
      }
    __syncthreads();
    }
  }

// ---------------------------------------------------------------------------
// postfold_spin: alm_out(E/B)(m,l) = GCadj * walpha   (transpose of prefold)
// ---------------------------------------------------------------------------
__global__ void postfold_spin(const int lmax, const int mmax,
                              const long long* __restrict__ soff,
                              const double* __restrict__ walpha,
                              const long long* __restrict__ mstart,
                              const long long nalm,
                              const long long gc_cstride,   // double2 units
                              const long long alm_cstride,  // double2 units
                              const double2* __restrict__ GCadj_,
                              double2* __restrict__ alm_)  // (2, nalm)
  {
  const int m = blockIdx.y;
  const int dl = blockIdx.x * blockDim.x + threadIdx.x;
  const int l = m + dl;
  if (m > mmax || l > lmax) return;
  const double2* __restrict__ GCadj = GCadj_
      + (long long)blockIdx.z * gc_cstride;
  double2* __restrict__ alm = alm_ + (long long)blockIdx.z * alm_cstride;
  const long long off = soff[m];
  const double w = walpha[off + dl];
  const double2 g = GCadj[2 * (off + dl)];
  const double2 c = GCadj[2 * (off + dl) + 1];
  alm[mstart[m] + l] = make_double2(w * g.x, w * g.y);
  alm[nalm + mstart[m] + l] = make_double2(w * c.x, w * c.y);
  }

}  // extern "C"
