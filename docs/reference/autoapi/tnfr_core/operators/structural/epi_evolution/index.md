# `tnfr_core.operators.structural.epi_evolution` module
Structural evolution utilities for EPI computations.

## Classes
### `NodalEvolution` (dict[str, tuple[float, float]])
Dictionary mapping nodes to ``(integral, derivative)`` tuples with metadata.

## Functions
- `evolve_epi(prev_epi: float, delta_map: Mapping[str, float], dt: float, nu_f_by_node: Mapping[str, float]) -> tuple[float, float, Dict[str, tuple[float, float]]]`
  - Integrate the Event Performance Index using explicit Euler steps.

The integrator validates ``dt >= 0``, derives per-node weights from the supplied
ν_f map (or targets embedded in ``delta_map``), and clamps phase modifiers to
keep their influence bounded.  The returned :class:`NodalEvolution` instance
includes optional metadata such as phase weights and θ identifiers so downstream
consumers can inspect contextual adjustments without affecting the value tuples.
