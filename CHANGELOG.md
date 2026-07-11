# Changelog

## 0.5.0

- Added `SynthesisPlan.inverse()` / `inverse_device()` and batched variants.
  These perform iterative CGLS least-squares analysis (the pseudoinverse of
  synthesis), distinct from the existing exact transpose `adjoint()`.
- Added `almond.as_cupy()` and `almond.as_jax()` DLPack helpers. JAX and CuPy
  can now share CUDA allocations without NumPy staging or device copies.
- Added explicit `AlmondRealSHT.synth_device()` and `adjoint_device()` entry
  points. Their real-basis conversion and observed-pixel gather/scatter stay
  on the GPU and accept JAX/CuPy/DLPack-compatible input.
- Added inverse-recovery, pointer-identity, and cut-sky device-residency tests.
- Bumped the package version from 0.4.0 to 0.5.0.

NERSC A100 validation: 43 non-slow GPU tests, 17 CPU reference tests, and six
cut-sky SiMaster integration tests passed. At nside 128/lmax 191/batch 4 the
inverse is 7.46x (spin 0) and 8.10x (spin 2) faster than 64-thread ducc0, with
coefficient differences below 2e-13 after both converge in five iterations.

The inverse is poorly conditioned at the aggressive HEALPix bandlimit
`lmax=3*nside-1`; Almond and ducc0 can both require hundreds of iterations.
Almond raises on non-convergence unless `return_info=True` is requested.
