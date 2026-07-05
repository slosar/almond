"""Almond — a GPU-accelerated spherical harmonic transform library for HEALPix.

Minimal, exact (float64), healpy-compatible conventions.  Reference
implementation: ducc0.  Prototype scope: spin-0 synthesis.
"""

__version__ = "0.4.0"

from .geometry import ring_geometry, pair_geometry
from . import reference


def __getattr__(name):
    # lazy import so CPU-only hosts can still use almond.reference
    if name == "SynthesisPlan":
        from .plan import SynthesisPlan
        return SynthesisPlan
    raise AttributeError(name)


__all__ = ["ring_geometry", "pair_geometry", "reference", "SynthesisPlan",
           "__version__"]
