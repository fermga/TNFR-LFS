# `tnfr_core.utils` module
Shared helper utilities for TNFR × LFS core modules.

## Functions
- `normalised_entropy(values: Iterable[float]) -> float`
  - Return the Shannon entropy of ``values`` normalised to ``[0, 1]``.

The helper gracefully handles non-numeric and non-positive entries by
ignoring them.  The computation automatically normalises the input weights
prior to calculating the entropy so that callers may supply either raw
weights or probability distributions.

