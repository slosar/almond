"""HEALPix RING-scheme ring geometry, computed natively (no healpy needed).

Conventions match ducc0's ``Healpix_Base(nside, "RING").sht_info()``:
rings are indexed ``i = 1 .. 4*nside-1`` from the north pole; each ring has
``nphi`` equidistant pixels at ``phi_j = phi0 + 2*pi*j/nphi``.

For the spherical-harmonic transform we work with *ring pairs*
``(i, 4*nside-i)`` that share ``|cos(theta)|``; the equator ring
``i = 2*nside`` is self-paired.  ``cos(theta)`` is an exact rational in the
HEALPix definition, so we compute ``cth``/``sth`` directly from it instead of
going through ``theta = arccos(z)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RingGeometry:
    """Full ring description of a HEALPix RING grid (north to south)."""

    nside: int
    nring: int          # 4*nside - 1
    npix: int           # 12*nside^2
    nphi: np.ndarray    # (nring,) int64   pixels per ring
    phi0: np.ndarray    # (nring,) float64 azimuth of first pixel
    ringstart: np.ndarray  # (nring,) int64 index of first pixel in RING map
    cth: np.ndarray     # (nring,) float64 cos(theta)
    sth: np.ndarray     # (nring,) float64 sin(theta)
    theta: np.ndarray   # (nring,) float64 colatitude


def ring_geometry(nside: int) -> RingGeometry:
    nring = 4 * nside - 1
    i = np.arange(1, nring + 1)          # 1 .. 4*nside-1, north to south
    northcap = i < nside
    southcap = i > 3 * nside
    ir = np.where(southcap, 4 * nside - i, i)   # mirrored ring index

    nphi = np.where(northcap | southcap, 4 * ir, 4 * nside).astype(np.int64)

    # z = cos(theta): cap 1 - ir^2/(3 nside^2), belt 4/3 - 2i/(3 nside)
    z = np.where(
        northcap | southcap,
        1.0 - ir.astype(np.float64) ** 2 / (3.0 * nside**2),
        4.0 / 3.0 - 2.0 * i.astype(np.float64) / (3.0 * nside),
    )
    z = np.where(southcap, -z, z)
    # 1 - |z| exactly, for an accurate sth near the poles
    one_minus_absz = np.where(
        northcap | southcap,
        ir.astype(np.float64) ** 2 / (3.0 * nside**2),
        np.abs(4.0 / 3.0 - 2.0 * i.astype(np.float64) / (3.0 * nside) - 1.0) * 0
        + (1.0 - np.abs(4.0 / 3.0 - 2.0 * i.astype(np.float64) / (3.0 * nside))),
    )
    sth = np.sqrt(one_minus_absz * (1.0 + np.abs(z)))
    theta = np.arctan2(sth, z)

    # phi0: caps phi_j = pi/(2 ir) (j + 1/2); belt phi_j = pi/(2 nside) (j + s/2),
    # s = (i - nside + 1) mod 2
    s_belt = ((i - nside + 1) % 2).astype(np.float64)
    phi0 = np.where(
        northcap | southcap,
        np.pi / (4.0 * ir),
        s_belt * np.pi / (4.0 * nside),
    )

    ringstart = np.concatenate([[0], np.cumsum(nphi)[:-1]]).astype(np.int64)

    return RingGeometry(
        nside=nside,
        nring=nring,
        npix=12 * nside * nside,
        nphi=nphi,
        phi0=phi0,
        ringstart=ringstart,
        cth=z,
        sth=sth,
        theta=theta,
    )


def get_mlim(lmax: int, spin: int, sth: np.ndarray, cth: np.ndarray) -> np.ndarray:
    """Highest m with non-negligible Legendre functions on a ring (ducc0's rule)."""
    ofs = max(0.01 * lmax, 100.0)
    b = -2.0 * spin * np.abs(cth)
    t1 = lmax * sth + ofs
    c = float(spin) ** 2 - t1 * t1
    discr = b * b - 4.0 * c
    res = np.where(discr <= 0, float(lmax), np.minimum((-b + np.sqrt(np.maximum(discr, 0.0))) / 2.0, float(lmax)))
    return (res + 0.5).astype(np.int64)


@dataclass
class PairGeometry:
    """Ring-pair view used by the Legendre stage.

    Pair ``p`` couples northern ring ``p`` (index into the ring arrays) with
    southern ring ``nring-1-p``; the equator pair ``p = 2*nside-1`` is
    self-paired.  ``cth``/``sth`` are those of the *northern* member
    (``cth >= 0``).
    """

    npair: int
    cth: np.ndarray     # (npair,)
    sth: np.ndarray     # (npair,)
    csq: np.ndarray     # (npair,)  cos^2, computed accurately near poles
    inorth: np.ndarray  # (npair,) ring index of northern member
    isouth: np.ndarray  # (npair,) ring index of southern member (== inorth at equator)
    mlim: np.ndarray    # (npair,) int64


def pair_geometry(geom: RingGeometry, lmax: int, spin: int = 0) -> PairGeometry:
    npair = 2 * geom.nside
    inorth = np.arange(npair, dtype=np.int64)
    isouth = geom.nring - 1 - inorth
    cth = geom.cth[inorth]
    sth = geom.sth[inorth]
    csq = np.where(np.abs(cth) > 0.99, (1.0 - sth) * (1.0 + sth), cth * cth)
    mlim = get_mlim(lmax, spin, sth, cth)
    return PairGeometry(npair=npair, cth=cth, sth=sth, csq=csq,
                        inorth=inorth, isouth=isouth, mlim=mlim)
