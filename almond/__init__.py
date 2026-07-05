# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Anze Slosar and the PowerSpec/SiMaster project
#
# Almond is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. See the LICENSE file for the full text.
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
