# TNFR × LFS Toolkit

**TNFR × LFS** packages the Fractal-Resonant Nature Theory into a race-ready
telemetry and recommendation workflow for Live for Speed.

## Documentation

- [Full documentation index](docs/index.md) – complete feature catalogue,
  onboarding narrative, and API reference.
- [Beginner quickstart](docs/tutorials.md) – step-by-step installation and
  sample session walkthrough.
- [Examples gallery](docs/examples.md) – automation scripts for common
  engineering tasks.

## TNFR alignment
The operational theory defines thirteen structural operators; each one is
addressed through a short code that `canonical_operator_label` expands into its
canonical tag:

| Code | Canonical label          | Summary |
| ---- | ------------------------ | ------- |
| `AL`   | Support                | Tunes structural supports and reinforcements. |
| `EN`   | Reception              | Normalises incoming EPI packets. |
| `IL`   | Coherence              | Smooths series without altering their mean. |
| `NAV`  | Transition             | Keeps phase transitions in sync. |
| `NUL`  | Contraction            | Manages structural contraction. |
| `OZ`   | Dissonance             | Isolates useful ΔNFR deviations from parasitic noise. |
| `RA`   | Propagation            | Propagates nodal adjustments to preserve resonance. |
| `REMESH` | Remeshing            | Re-meshes the nodal graph to detect discontinuities. |
| `SILENCE` | Structural silence  | Marks windows with no structural excitation. |
| `THOL` | Auto-organisation      | Observes self-organisation during regime changes. |
| `UM`   | Coupling               | Evaluates coupling between nodes. |
| `VAL`  | Amplification          | Highlights where to amplify structural supports. |
| `ZHIR` | Transformation         | Traces deep archetype mutations. |

The extended nodal equation evaluates the entropy-penalised Sense Index by
incorporating the ΔNFR gradient and each node’s natural frequency:

```
Sense Index = 1 / (1 + Σ w_i · |ΔNFR_i| · g(ν_f_i)) - λ · H
```

* `w_i` weights each node according to its structural role.
* `g(ν_f_i)` adjusts the response using the observed natural frequency.
* `H` captures the entropy of the nodal ΔNFR distribution.

New structural metrics exposed by the public interface maintain the terminology
documented in `docs/DESIGN.md` and the API reference:

- `support_effective`, `load_support_ratio`.
- `structural_expansion_longitudinal` / `structural_contraction_longitudinal`.
- `structural_expansion_lateral` / `structural_contraction_lateral`.
- `delta_nfr_entropy`, `node_entropy`, `phase_delta_nfr_entropy`, `phase_node_entropy`.
- `thermal_load`, `style_index`, `network_memory`.
- `aero_balance_drift`, `slide_catch_budget`, `ackermann_parallel_index`.

Generate the theoretical coverage report with:

```
poetry run python tools/tnfr_theory_audit.py --core --tests --output tests/_report/theory_impl_matrix.md
```

Once generated, review it in
[`tests/_report/theory_impl_matrix.md`](tests/_report/theory_impl_matrix.md).

## TNFR parity verification

Use the parity verifier to compare the local TNFR engine against the canonical
extension. The script ingests a telemetry series, runs both engines, and
emits a JSON report with absolute values and relative differences:

```
poetry run python tools/verify_tnfr_parity.py tests/data/synthetic_stint.csv --output parity_report.json
```

The same workflow is available via the `make verify-tnfr-parity` target, which
enables strict mode and exits with a non-zero status when any metric exceeds the
configured tolerances.
