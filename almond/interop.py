"""Zero-copy CUDA-array interoperability helpers.

Almond's kernels use CuPy, while consumers such as SiMaster keep their
linear-algebra state in JAX.  Both libraries implement the Python DLPack
protocol, allowing ownership-safe views of an existing device allocation
without staging through NumPy or host memory.
"""

from __future__ import annotations

import numpy as np


def as_cupy(x, *, dtype=None):
    """Return *x* as a CuPy array, using DLPack for foreign GPU arrays.

    NumPy inputs are copied to the device. CuPy inputs are returned directly
    (subject to an optional dtype conversion). Objects implementing
    ``__dlpack__``—including JAX CUDA arrays—are imported without a copy.
    """
    import cupy as cp

    if isinstance(x, cp.ndarray):
        out = x
    elif isinstance(x, np.ndarray):
        out = cp.asarray(x)
    elif hasattr(x, "__dlpack__"):
        out = cp.from_dlpack(x)
    else:
        out = cp.asarray(x)
    return out.astype(dtype, copy=False) if dtype is not None else out


def as_jax(x):
    """Return a zero-copy JAX view of a CuPy CUDA array via DLPack."""
    import jax

    # New JAX releases consume any __dlpack__ provider directly. Keeping the
    # legacy fallback makes Almond usable with the older JAX versions still
    # deployed on several HPC systems.
    try:
        return jax.dlpack.from_dlpack(x)
    except TypeError:  # pragma: no cover - legacy JAX only
        return jax.dlpack.from_dlpack(x.toDlpack())


def is_host_array(x) -> bool:
    """Whether the public convenience API should return a NumPy result."""
    return isinstance(x, np.ndarray)
