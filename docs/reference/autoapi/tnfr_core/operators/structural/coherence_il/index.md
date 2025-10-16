# `tnfr_core.operators.structural.coherence_il` module
Ideal line coherence smoothing utilities.

## Functions
- `coherence_operator_il(series: Sequence[float], window: int = 5) -> List[float]`
  - Smooth a numeric series with a mean-preserving moving average.

The operator validates that ``window`` is a positive odd integer, handles edge
segments with the available neighbours, and debiases the output so the original
mean is preserved exactly up to floating-point precision.
