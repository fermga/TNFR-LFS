# `tnfr_core.equations.coherence` module
Utilities for analysing ΔNFR coherence and entropy penalties.

## Functions
- `compute_node_delta_nfr(node: str, delta_nfr: float, feature_map: Mapping[str, float], *, prefix: bool = True) -> NodeDeltaMap`
  - Distribute a node-level ΔNFR among its internal sub-features.

Parameters
----------
node:
    Name of the node whose signals are being analysed.  The node name is
    used to optionally prefix the resulting keys so that flattened maps do
    not collide when multiple subsystems expose the same signal name.
delta_nfr:
    ΔNFR magnitude attributed to the node.  The sign of ``delta_nfr`` is
    preserved in the output while the magnitude is scaled according to the
    relative weights in ``feature_map``.
feature_map:
    Mapping of sub-feature identifiers to their measured intensity.  The
    function uses absolute values for weighting, meaning the caller is
    responsible for carrying the directional information in
    ``delta_nfr``.
prefix:
    When ``True`` (the default) each key is prefixed with ``"node."`` to
    avoid collisions across subsystems.
- `sense_index(delta_nfr: float, deltas_by_node: Mapping[str, float], baseline_nfr: float, *, nu_f_by_node: Mapping[str, float], active_phase: str, w_phase: Mapping[str, Mapping[str, float] | float] | Mapping[str, float], nu_f_targets: Mapping[str, Mapping[str, float] | float] | Mapping[str, float] | float | None = None, entropy_lambda: float = 0.1) -> float`
  - Compute the entropy-penalised sense index for a ΔNFR distribution.

The metric follows the expression ``1 / (1 + Σ w · |ΔNFR| · g(ν_f) · g*(ν_f^*))
- λ·H`` combining phase-dependent weights ``w``, the magnitude of the ΔNFR
contributions for every subsystem, the natural frequency gain ``g`` derived
from the measured natural frequency ``ν_f`` and an optional goal gain
``g*`` driven by the target natural frequency ``ν_f^*``.  The entropy term
``H`` is derived from the distribution of the node deltas.
``λ`` is exposed as ``entropy_lambda`` for fine tuning while remaining
backward compatible with previous heuristics.

## Attributes
- `NodeDeltaMap = Dict[str, float]`

