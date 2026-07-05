"""Validate the NumPy reference implementation against ducc0."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ducc0

from almond.geometry import ring_geometry
from almond import reference


def ducc_synthesis(alm, nside, lmax):
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    m = np.arange(lmax + 1)
    mstart = (m * (2 * lmax + 1 - m) // 2).astype(np.uint64)
    return ducc0.sht.experimental.synthesis(
        alm=alm.reshape(1, -1), lmax=lmax, spin=0, mstart=mstart, nthreads=8, **geom
    )[0]


def random_alm(lmax, seed, real_m0=True):
    rng = np.random.default_rng(seed)
    nalm = (lmax + 1) * (lmax + 2) // 2
    alm = rng.standard_normal(nalm) + 1j * rng.standard_normal(nalm)
    if real_m0:
        alm[: lmax + 1] = alm[: lmax + 1].real
    return alm


@pytest.mark.parametrize("nside,lmax", [(8, 23), (16, 47), (32, 95), (16, 16), (32, 40)])
def test_geometry_matches_ducc(nside, lmax):
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    info = base.sht_info()
    g = ring_geometry(nside)
    assert np.allclose(np.cos(info["theta"]), g.cth, atol=1e-14, rtol=0)
    assert np.array_equal(info["nphi"].astype(np.int64), g.nphi)
    assert np.allclose(info["phi0"], g.phi0, atol=1e-14, rtol=0)
    assert np.array_equal(info["ringstart"].astype(np.int64), g.ringstart)


@pytest.mark.parametrize("nside,lmax", [(8, 23), (16, 47), (32, 95), (16, 16), (32, 40)])
def test_synthesis_vs_ducc(nside, lmax):
    alm = random_alm(lmax, seed=nside * 1000 + lmax)
    ref = ducc_synthesis(alm, nside, lmax)
    got = reference.synthesis(alm, nside, lmax)
    scale = np.abs(ref).max()
    err = np.abs(got - ref).max() / scale
    assert err < 1e-12, f"rel err {err:.3e}"


def ducc_adjoint(maps, nside, lmax):
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    m = np.arange(lmax + 1)
    mstart = (m * (2 * lmax + 1 - m) // 2).astype(np.uint64)
    return ducc0.sht.experimental.adjoint_synthesis(
        map=maps.reshape(1, -1), lmax=lmax, spin=0, mstart=mstart,
        nthreads=8, **geom)[0]


@pytest.mark.parametrize("nside,lmax", [(8, 23), (16, 47), (32, 95), (16, 16)])
def test_adjoint_vs_ducc(nside, lmax):
    rng = np.random.default_rng(nside)
    maps = rng.standard_normal(12 * nside * nside)
    ref = ducc_adjoint(maps, nside, lmax)
    got = reference.adjoint_synthesis(maps, nside, lmax)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    assert err < 1e-12, f"rel err {err:.3e}"


@pytest.mark.parametrize("nside,lmax", [(16, 47), (32, 95)])
def test_adjoint_identity(nside, lmax):
    """<synth(a), f>_pix == <a, adjoint(f)> with m>0 doubling."""
    rng = np.random.default_rng(1)
    nalm = (lmax + 1) * (lmax + 2) // 2
    a = rng.standard_normal(nalm) + 1j * rng.standard_normal(nalm)
    a[: lmax + 1] = a[: lmax + 1].real
    f = rng.standard_normal(12 * nside * nside)
    lhs = float(reference.synthesis(a, nside, lmax) @ f)
    ad = reference.adjoint_synthesis(f, nside, lmax)
    rhs = float(np.sum(ad[: lmax + 1].real * a[: lmax + 1].real)
                + 2 * np.sum((ad[lmax + 1:] * np.conj(a[lmax + 1:])).real))
    assert abs(lhs - rhs) < 1e-11 * abs(lhs), f"{lhs} vs {rhs}"


def test_synthesis_complex_m0():
    """ducc discards Im(a_l0); make sure we reproduce that too."""
    nside, lmax = 16, 47
    alm = random_alm(lmax, seed=7, real_m0=False)
    ref = ducc_synthesis(alm, nside, lmax)
    got = reference.synthesis(alm, nside, lmax)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    assert err < 1e-12, f"rel err {err:.3e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
