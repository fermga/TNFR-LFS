# `tnfr_lfs.analysis.contextual_delta` module
LFS-facing helpers for contextual ΔNFR adjustments.

The module re-exports the contextual helpers from :mod:`tnfr_core` while
installing a context-matrix loader backed by the packaged
``context_factors.toml`` payload.  Importing the module ensures that
:mod:`tnfr_core` is ready to compute contextual deltas without requiring manual
configuration.

## Functions
- `ensure_context_loader() -> None`
  - Register the default calibration loader with :mod:`tnfr_core`.
