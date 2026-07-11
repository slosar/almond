"""AlmondRealSHT vs SiMaster's ducc-based RealSHT (drop-in check)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# In the coordinated NG checkout, prefer the sibling SiMaster branch. In a
# standalone Almond checkout, use whichever simaster is installed/PYTHONPATH.
_sibling_simaster = Path(__file__).resolve().parents[2] / "SiMaster"
if _sibling_simaster.exists():
    sys.path.insert(0, str(_sibling_simaster))

cupy = pytest.importorskip("cupy")
simaster = pytest.importorskip("simaster")

from simaster.sht import RealSHT
from simaster.utils import RealAlmIndex

from almond.simaster import AlmondRealSHT


@pytest.mark.parametrize("spin,nside,lmax", [(0, 32, 95), (2, 32, 95),
                                             (0, 64, 150), (2, 64, 191)])
def test_vs_simaster_realsht(spin, nside, lmax):
    rng = np.random.default_rng(spin * 100 + nside)
    index = RealAlmIndex(0 if spin == 0 else 2, lmax)
    npix = 12 * nside * nside
    obs_pix = np.sort(rng.choice(npix, size=npix // 3, replace=False))
    ref = RealSHT(nside, index, spin, obs_pix)
    gpu = AlmondRealSHT(nside, index, spin, obs_pix)
    assert (gpu.ncol, gpu.nrow) == (ref.ncol, ref.nrow)

    B = 3
    a = rng.standard_normal((ref.ncol, B))
    y_ref = ref.synth(a)
    y_gpu = gpu.synth(a)
    err = np.abs(y_gpu - y_ref).max() / np.abs(y_ref).max()
    assert err < 1e-10, f"synth: {err:.2e}"

    m = rng.standard_normal((ref.nrow, B))
    at_ref = ref.adjoint(m)
    at_gpu = gpu.adjoint(m)
    err = np.abs(at_gpu - at_ref).max() / np.abs(at_ref).max()
    assert err < 1e-10, f"adjoint: {err:.2e}"

    # transpose identity in the real basis
    ip1 = float(y_gpu[:, 0] @ m[:, 0])
    ip2 = float(a[:, 0] @ at_gpu[:, 0])
    assert abs(ip1 - ip2) < 1e-11 * abs(ip1)


def test_covmodel_backend_alm_vs_ducc():
    """QML covariance operator C.x must agree between backends."""
    import simaster as sm
    from simaster.covariance import CovModel
    from simaster.utils import RealAlmIndex

    nside, lmax = 16, 47
    npix = 12 * nside * nside
    rng = np.random.default_rng(0)
    mask = (rng.uniform(size=npix) > 0.3).astype(float)
    ivar = np.where(mask > 0, 1.0 + rng.uniform(size=npix), 0.0)
    f0 = sm.Field(mask, [rng.standard_normal(npix)], ivar=ivar)
    f2 = sm.Field(mask, [rng.standard_normal(npix),
                         rng.standard_normal(npix)], spin=2, ivar=ivar)
    index = RealAlmIndex(2, lmax)
    ncomp = 3
    l = np.arange(lmax + 1)
    clmat = np.zeros((ncomp, ncomp, lmax + 1))
    base = 1.0 / (1.0 + l) ** 2
    clmat[0, 0] = base
    clmat[1, 1] = 0.3 * base
    clmat[2, 2] = 0.1 * base
    clmat[0, 1] = clmat[1, 0] = 0.1 * base

    cov_d = CovModel([f0, f2], clmat, index, backend="ducc")
    cov_a = CovModel([f0, f2], clmat, index, backend="almond")
    x = rng.standard_normal((cov_d.nrow, 2))
    y_d = np.asarray(cov_d.apply_C(x))
    y_a = np.asarray(cov_a.apply_C(x))
    err = np.abs(y_a - y_d).max() / np.abs(y_d).max()
    assert err < 1e-10, f"apply_C backend mismatch: {err:.2e}"


def test_observed_pixel_ops_remain_device_resident():
    """Cut-sky gather/scatter accept and return CuPy without host arrays."""
    nside, lmax, B = 8, 20, 3
    npix = 12 * nside ** 2
    obs = np.sort(np.random.default_rng(12).choice(npix, npix // 2,
                                                   replace=False))
    index = RealAlmIndex(2, lmax)
    op = AlmondRealSHT(nside, index, 0, obs)
    a = cupy.asarray(np.random.default_rng(13).standard_normal((op.ncol, B)))
    m = op.synth_device(a)
    at = op.adjoint_device(m)
    assert isinstance(m, cupy.ndarray) and isinstance(at, cupy.ndarray)
    assert m.shape == (op.nrow, B) and at.shape == (op.ncol, B)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
