"""Validate the GPU synthesis against ducc0 (and the NumPy reference)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cupy = pytest.importorskip("cupy")

import ducc0

from almond.plan import SynthesisPlan
from almond import reference


def ducc_synthesis(alm, nside, lmax, nthreads=8):
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    m = np.arange(lmax + 1)
    mstart = (m * (2 * lmax + 1 - m) // 2).astype(np.uint64)
    return ducc0.sht.experimental.synthesis(
        alm=alm.reshape(1, -1), lmax=lmax, spin=0, mstart=mstart,
        nthreads=nthreads, **geom)[0]


def random_alm(lmax, seed):
    rng = np.random.default_rng(seed)
    nalm = (lmax + 1) * (lmax + 2) // 2
    alm = rng.standard_normal(nalm) + 1j * rng.standard_normal(nalm)
    alm[: lmax + 1] = alm[: lmax + 1].real
    return alm


def test_coef_table_matches_reference():
    nside, lmax = 16, 47
    plan = SynthesisPlan(nside, lmax)
    coef = cupy.asnumpy(plan.d_coef)
    moff = cupy.asnumpy(plan.d_moff)
    for m in [0, 1, 2, 17, 46, 47]:
        alpha, ca, cb = reference.recursion_coeffs(m, lmax)
        got = coef[moff[m]: moff[m + 1]]
        # only the first (lmax-m)//2+1 entries are used by the synthesis;
        # trailing table entries are padding and may differ
        n = (lmax - m) // 2 + 1
        np.testing.assert_allclose(got.real[:n], ca[:n], rtol=1e-14, atol=0)
        np.testing.assert_allclose(got.imag[:n], cb[:n], rtol=1e-14, atol=1e-300)
        # alpha reconstruction used by the prefold kernel
        sign4 = np.where((np.arange(n) & 2).astype(bool), -1.0, 1.0)
        np.testing.assert_allclose(sign4 * np.sqrt(np.abs(ca[:n])), alpha[:n],
                                   rtol=1e-14)


@pytest.mark.parametrize("nside,lmax", [
    (8, 23), (16, 47), (32, 95), (64, 191), (16, 16), (32, 40),
    (128, 383), (256, 767),
])
def test_gpu_synthesis_vs_ducc(nside, lmax):
    alm = random_alm(lmax, seed=nside + lmax)
    plan = SynthesisPlan(nside, lmax)
    got = plan.synthesis(alm)
    ref = ducc_synthesis(alm, nside, lmax)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    assert err < 1e-10, f"nside={nside} lmax={lmax}: rel err {err:.3e}"


@pytest.mark.slow
@pytest.mark.parametrize("nside", [512, 1024])
def test_gpu_synthesis_vs_ducc_large(nside):
    lmax = 3 * nside - 1
    alm = random_alm(lmax, seed=nside)
    plan = SynthesisPlan(nside, lmax)
    got = plan.synthesis(alm)
    ref = ducc_synthesis(alm, nside, lmax, nthreads=32)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    assert err < 1e-10, f"nside={nside}: rel err {err:.3e}"


@pytest.mark.parametrize("nside,B,chunk", [
    (32, 1, 4), (32, 4, 4), (32, 5, 4), (64, 11, 4), (128, 8, 4),
    (64, 5, None),   # default chunk=1: sequential loop path
])
def test_batched_synthesis(nside, B, chunk):
    """Batched path must agree with per-column single transforms and ducc."""
    lmax = 3 * nside - 1
    rng = np.random.default_rng(nside + B)
    nalm = (lmax + 1) * (lmax + 2) // 2
    alm = rng.standard_normal((B, nalm)) + 1j * rng.standard_normal((B, nalm))
    plan = SynthesisPlan(nside, lmax, chunk=chunk)
    got = cupy.asnumpy(plan.synthesis_device_batch(cupy.asarray(alm)))
    assert got.shape == (B, 12 * nside * nside)
    for b in range(B):
        single = plan.synthesis(alm[b])
        err = np.abs(got[b] - single).max() / np.abs(single).max()
        assert err < 1e-12, f"col {b} vs single: {err:.2e}"
    ref = ducc_synthesis(alm[B - 1], nside, lmax)
    err = np.abs(got[B - 1] - ref).max() / np.abs(ref).max()
    assert err < 1e-10, f"vs ducc: {err:.2e}"


@pytest.mark.parametrize("spin,nside,B", [
    (0, 32, 5), (0, 64, 7), (2, 32, 5), (2, 64, 3),
])
def test_batched_gridz_roundtrip(spin, nside, B):
    """Grid-z batched synth+adjoint (both spins) vs per-column singles."""
    lmax = 3 * nside - 1
    rng = np.random.default_rng(10 * nside + B + spin)
    nalm = (lmax + 1) * (lmax + 2) // 2
    shape = (B, nalm) if spin == 0 else (B, 2, nalm)
    alm = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    if spin == 2:
        for m in (0, 1):
            ms = m * (2 * lmax + 1 - m) // 2
            alm[..., ms + m: ms + 2] = 0
    plan = SynthesisPlan(nside, lmax, spin=spin)
    d_alm = cupy.asarray(alm)
    maps = plan.synthesis_device_batch(d_alm)
    almT = plan.adjoint_device_batch(maps)
    maps_h, almT_h = cupy.asnumpy(maps), cupy.asnumpy(almT)
    for b in range(B):
        ms = cupy.asnumpy(plan.synthesis_device(d_alm[b]))
        err = np.abs(maps_h[b] - ms).max() / np.abs(ms).max()
        assert err < 1e-12, f"synth col {b}: {err:.2e}"
        at = cupy.asnumpy(plan.adjoint_device(maps[b]))
        err = np.abs(almT_h[b] - at).max() / np.abs(at).max()
        assert err < 1e-12, f"adjoint col {b}: {err:.2e}"


def ducc_adjoint(maps, nside, lmax, nthreads=8):
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    m = np.arange(lmax + 1)
    mstart = (m * (2 * lmax + 1 - m) // 2).astype(np.uint64)
    return ducc0.sht.experimental.adjoint_synthesis(
        map=maps.reshape(1, -1), lmax=lmax, spin=0, mstart=mstart,
        nthreads=nthreads, **geom)[0]


@pytest.mark.parametrize("nside,lmax", [
    (8, 23), (16, 47), (32, 95), (64, 191), (32, 40), (128, 383), (256, 767),
])
def test_gpu_adjoint_vs_ducc(nside, lmax):
    rng = np.random.default_rng(nside + 2 * lmax)
    maps = rng.standard_normal(12 * nside * nside)
    plan = SynthesisPlan(nside, lmax)
    got = plan.adjoint(maps)
    ref = ducc_adjoint(maps, nside, lmax)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    assert err < 1e-10, f"nside={nside} lmax={lmax}: rel err {err:.3e}"


@pytest.mark.slow
@pytest.mark.parametrize("nside", [512, 1024])
def test_gpu_adjoint_vs_ducc_large(nside):
    lmax = 3 * nside - 1
    rng = np.random.default_rng(nside)
    maps = rng.standard_normal(12 * nside * nside)
    plan = SynthesisPlan(nside, lmax)
    got = plan.adjoint(maps)
    ref = ducc_adjoint(maps, nside, lmax, nthreads=32)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    assert err < 1e-10, f"nside={nside}: rel err {err:.3e}"


def test_gpu_adjoint_identity():
    """<synth(a), f> == <a, adjoint(f)> with m>0 doubling, on the GPU pair."""
    nside, lmax = 64, 191
    rng = np.random.default_rng(5)
    nalm = (lmax + 1) * (lmax + 2) // 2
    a = rng.standard_normal(nalm) + 1j * rng.standard_normal(nalm)
    a[: lmax + 1] = a[: lmax + 1].real
    f = rng.standard_normal(12 * nside * nside)
    plan = SynthesisPlan(nside, lmax)
    lhs = float(plan.synthesis(a) @ f)
    ad = plan.adjoint(f)
    rhs = float(np.sum(ad[: lmax + 1].real * a[: lmax + 1].real)
                + 2 * np.sum((ad[lmax + 1:] * np.conj(a[lmax + 1:])).real))
    assert abs(lhs - rhs) < 1e-12 * abs(lhs), f"{lhs} vs {rhs}"


def test_gpu_adjoint_batched():
    nside, lmax, B = 64, 191, 5
    rng = np.random.default_rng(6)
    maps = rng.standard_normal((B, 12 * nside * nside))
    plan = SynthesisPlan(nside, lmax)
    got = cupy.asnumpy(plan.adjoint_device_batch(cupy.asarray(maps)))
    for b in range(B):
        single = plan.adjoint(maps[b])
        assert np.abs(got[b] - single).max() <= 1e-13 * np.abs(single).max()


@pytest.mark.parametrize("spin", [0, 2])
def test_gpu_inverse_recovers_bandlimited_coefficients(spin):
    """The inverse is analysis/pseudoinverse, not merely the adjoint."""
    nside, lmax, B = 16, 32, 3
    nalm = (lmax + 1) * (lmax + 2) // 2
    rng = np.random.default_rng(701 + spin)
    shape = (B, nalm) if spin == 0 else (B, 2, nalm)
    alm = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    alm[..., :lmax + 1] = alm[..., :lmax + 1].real
    if spin == 2:
        alm[..., :2] = 0.0
        alm[..., lmax + 1] = 0.0
    plan = SynthesisPlan(nside, lmax, spin=spin)
    d_alm = cupy.asarray(alm)
    maps = plan.synthesis_device_batch(d_alm)
    got, info = plan.inverse_device_batch(
        maps, epsilon=1e-11, maxiter=30, return_info=True)
    got = cupy.asnumpy(got)
    rel = np.linalg.norm((got - alm).ravel()) / np.linalg.norm(alm.ravel())
    assert rel < 2e-9
    assert bool(cupy.all(info["converged"]))
    assert float(cupy.max(info["relative_map_residual"])) < 2e-10


def test_dlpack_jax_cupy_zero_copy_roundtrip():
    """JAX and CuPy view the same allocation, with no host staging."""
    import jax
    import jax.numpy as jnp
    from almond.interop import as_cupy, as_jax

    if jax.default_backend() != "gpu":
        pytest.skip("requires JAX CUDA backend")
    jax.config.update("jax_enable_x64", True)
    x = jnp.arange(32, dtype=jnp.float64)
    c = as_cupy(x)
    assert c.data.ptr == x.unsafe_buffer_pointer()
    y = as_jax(c)
    assert y.unsafe_buffer_pointer() == c.data.ptr
    np.testing.assert_array_equal(np.asarray(y), np.arange(32))


def test_inverse_reports_nonconvergence():
    nside, lmax = 8, 12
    plan = SynthesisPlan(nside, lmax)
    alm = cupy.asarray(random_alm(lmax, 909))
    maps = plan.synthesis_device(alm)
    with pytest.raises(RuntimeError, match="did not converge"):
        plan.inverse_device(maps, epsilon=1e-30, maxiter=0)


def ducc_spin2(alm2, nside, lmax, adjoint=False, maps=None, nthreads=8):
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    m = np.arange(lmax + 1)
    mstart = (m * (2 * lmax + 1 - m) // 2).astype(np.uint64)
    if adjoint:
        return ducc0.sht.experimental.adjoint_synthesis(
            map=maps, lmax=lmax, spin=2, mstart=mstart, nthreads=nthreads,
            **geom)
    return ducc0.sht.experimental.synthesis(
        alm=alm2, lmax=lmax, spin=2, mstart=mstart, nthreads=nthreads, **geom)


def random_alm2(lmax, seed):
    rng = np.random.default_rng(seed)
    nalm = (lmax + 1) * (lmax + 2) // 2
    a = rng.standard_normal((2, nalm)) + 1j * rng.standard_normal((2, nalm))
    a[:, : lmax + 1] = a[:, : lmax + 1].real
    a[:, :2] = 0        # no l < 2 for spin 2
    a[:, lmax + 1] = 0  # (l=1, m=1)
    return a


@pytest.mark.parametrize("nside,lmax", [
    (8, 23), (16, 47), (32, 95), (64, 191), (32, 40), (128, 383), (256, 767),
])
def test_gpu_spin2_synthesis_vs_ducc(nside, lmax):
    alm2 = random_alm2(lmax, seed=nside)
    plan = SynthesisPlan(nside, lmax, spin=2)
    got = plan.synthesis(alm2)
    ref = ducc_spin2(alm2, nside, lmax)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    assert err < 1e-10, f"nside={nside} lmax={lmax}: rel err {err:.3e}"


@pytest.mark.parametrize("nside,lmax", [(16, 47), (64, 191), (128, 383)])
def test_gpu_spin2_adjoint_vs_ducc(nside, lmax):
    rng = np.random.default_rng(nside + 1)
    maps = rng.standard_normal((2, 12 * nside * nside))
    plan = SynthesisPlan(nside, lmax, spin=2)
    got = plan.adjoint(maps)
    ref = ducc_spin2(None, nside, lmax, adjoint=True, maps=maps)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    assert err < 1e-10, f"nside={nside} lmax={lmax}: rel err {err:.3e}"


@pytest.mark.slow
@pytest.mark.parametrize("nside", [512])
def test_gpu_spin2_large(nside):
    lmax = 3 * nside - 1
    alm2 = random_alm2(lmax, seed=2)
    plan = SynthesisPlan(nside, lmax, spin=2)
    got = plan.synthesis(alm2)
    ref = ducc_spin2(alm2, nside, lmax, nthreads=32)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    assert err < 1e-10, f"rel err {err:.3e}"
    rng = np.random.default_rng(3)
    maps = rng.standard_normal((2, 12 * nside * nside))
    gota = plan.adjoint(maps)
    refa = ducc_spin2(None, nside, lmax, adjoint=True, maps=maps, nthreads=32)
    erra = np.abs(gota - refa).max() / np.abs(refa).max()
    assert erra < 1e-10, f"adjoint rel err {erra:.3e}"


def test_repeated_calls_are_deterministic():
    nside, lmax = 32, 95
    alm = random_alm(lmax, seed=3)
    plan = SynthesisPlan(nside, lmax)
    a = plan.synthesis(alm)
    b = plan.synthesis(alm)
    # atomics in the fold stage may reorder additions between calls;
    # results must still agree to full double precision
    assert np.abs(a - b).max() <= 1e-13 * np.abs(a).max()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-m", "not slow"]))
