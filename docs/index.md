# TNFR × LFS Documentation

Treat this index as the canonical home for the extended
introduction, feature descriptions, onboarding narratives, checklists and API
map that the project README now summarises.

## Feature overview

TNFR × LFS bundles the following core capabilities for race engineers and data
scientists:

- Live ΔNFR HUD and capture pipeline wired directly to Live for Speed
  OutSim/OutGauge streams.
- Automated baselines, analysis, suggestions and reporting from a single CLI.
- Extensible exporters (JSONL, Markdown, HTML) and configuration packs for
  repeatable engineering workflows.
- Benchmark suite and reproducible examples for regression testing new ideas.
- Ingestion, analytics, recommendation and export subsystems that map the TNFR
  framework to actionable setup advice:
  1. **Ingest** – capture telemetry samples from OutSim-compatible streams via
     :class:`tnfr_lfs.telemetry.live.OutSimClient`.
  2. **Core analytics** – extract Event Performance Indicators (EPI) and
     compute ΔFz/ΔSi deltas through :class:`tnfr_core.epi.EPIExtractor`.
  3. **Recommendation engine** – map metrics to actionable setup advice using
     the rule engine in :mod:`tnfr_lfs.recommender.rules`.
  4. **Exporters** – serialise analysis results to JSON or CSV with the
     functions in :mod:`tnfr_lfs.exporters`.

The project ships with a CLI (`tnfr_lfs`) as well as examples and unit tests to
illustrate the workflow. The ingestion pipeline regression suite in
`tests/test_ingestion.py` exercises the OutSim/OutGauge UDP clients and fusion
helpers end-to-end.【F:tests/test_ingestion.py†L1-L376】

## Quickstart

!!! tip "Ready to run the toolkit?"
    Follow the [beginner quickstart tutorial](tutorials.md) for the step-by-step
    installation, sample scenario, telemetry configuration and artefact review
    walkthrough that onboards you from a fresh environment to a race-ready
    workflow.

TNFR × LFS targets Python 3.10+ environments and Live for Speed telemetry
streams. The quickstart covers dependency setup, the baseline scenario, the
simulator configuration and the report validation process without duplicating
those instructions here.

## Operational checklist

Pre-stint reviews rely on a compact checklist that validates four
quantitative objectives:

1. **Average Sense Index ≥ 0.75** – the baseline target for the operator
   profile, used by the rule engine and persisted in
   `RuleProfileObjectives`.【F:tnfr_lfs/recommender/rules.py†L604-L615】
2. **ΔNFR density ≤ 6.00 kN·s⁻¹** – the default reference used to compute
   the absolute ΔNFR integral while scoring objectives.【F:tnfr_lfs/recommender/search.py†L210-L236】
3. **Brake headroom ≥ 0.40** – the minimum margin before brake-bias or
   cooling interventions become mandatory.【F:tnfr_lfs/recommender/rules.py†L604-L615】
4. **Aerodynamic Δμ ≤ 0.12** – the tolerance applied when normalising
   aero-mechanical coherence and drift alerts.【F:src/tnfr_core/metrics/metrics.py†L2558-L2575】

The HUD/CLI summarises these objectives under a “Checklist” line that marks
completed goals with ✅ and highlights pending work with ⚠️ using the
real-time metrics (Si average, ΔNFR integral, brake headroom, and
aerodynamic imbalance).【F:tnfr_lfs/cli/osd.py†L1477-L1567】

## Branding and terminology

- **TNFR theory** is the conceptual framework that formalises the EPI and
  ΔNFR/ΔSi metrics.
- **TNFR × LFS toolkit** identifies the software implementation that automates
  those principles inside Live for Speed.
- The CLI (``tnfr_lfs``), Python package (``tnfr_lfs``) and configuration
  tables (the ``[tool.tnfr_lfs]`` block in ``pyproject.toml``) intentionally
  retain their respective names for compatibility with existing workflows.

## TNFR alignment

The TNFR structural operators span thirteen archetypes. Their codes come from
the detectors `AL`, `EN`, `IL`, `NAV`, `NUL`, `OZ`, `RA`, `REMESH`, `SILENCE`,
`THOL`, `UM`, `VAL`, and `ZHIR`, which `canonical_operator_label` maps to their
canonical labels (Support, Reception, Coherence, Transition, Contraction,
Dissonance, Propagation, Remeshing, Structural silence, Auto-organisation,
Coupling, Amplification, and Transformation).【F:src/tnfr_core/operators/operator_detection.py†L43-L68】

The structural detectors are exposed directly by
``tnfr_core.operators.operator_detection``. Alongside `detect_al`, `detect_oz`,
`detect_il`, `detect_silence`, and `detect_nav`, the core provides
``detect_en`` (Reception), ``detect_um`` (Coupling), ``detect_ra`` (Propagation),
``detect_val`` (Amplification), ``detect_nul`` (Contraction), ``detect_thol``
(Auto-organisation), ``detect_zhir`` (Transformation), and ``detect_remesh``
(Remeshing). Each function follows the same sliding-window scan pattern and
exposes explicit thresholds:

* ``detect_en`` integrates the ψ flux across a sliding ``window``; it requires
  the integral to exceed ``psi_threshold`` while the |EPI| norm remains below
  ``epi_norm_max`` without decaying.
* ``detect_um`` cross-checks ``mu_delta_threshold``, ``load_ratio_threshold``,
  and ``suspension_delta_threshold`` to measure nodal coupling.
* ``detect_ra`` evaluates ΔNFR diffusion with ``nfr_rate_threshold``,
  ``si_span_threshold``, and ``speed_threshold``.
* ``detect_val`` inspects ``window`` samples to confirm that the |EPI| growth
  rate exceeds ``epi_growth_min`` while the number of support nodes increases by
  at least ``active_nodes_delta_min`` under loads above
  ``active_node_load_min``.
* ``detect_nul`` reviews ``window`` records to confirm that the active-node
  count drops to ``active_nodes_delta_max`` (negative value) while the EPI
  concentration exceeds ``epi_concentration_min`` over nodes with minimum load
  ``active_node_load_min``.
* ``detect_thol`` monitors |EPI| accelerations above ``epi_accel_min`` that
  persist for ``stability_window`` seconds within the
  ``stability_tolerance`` margin on the first derivative.
* ``detect_zhir`` uses a bidirectional ``window`` to detect phase jumps of at
  least ``phase_jump_min`` while the |EPI| derivative exceeds ``xi_min`` for
  ``min_persistence`` seconds.
* ``detect_remesh`` evaluates repeated patterns across ``window`` samples and
  requires at least ``min_repeats`` delays from ``tau_candidates`` to reach an
  autocorrelation of ``acf_min``.

The extended nodal equation driving the Sense Index weights each node by its
ΔNFR contribution, natural frequency, and structural entropy:

```
Sense Index = 1 / (1 + Σ w_i · |ΔNFR_i| · g(ν_f_i)) - λ · H
```

The weighted term `Σ w_i · |ΔNFR_i| · g(ν_f_i)` absorbs the nodal dynamics, the
`g(ν_f_i)` function corrects for natural frequency, and `H` represents the
Shannon entropy computed over the ΔNFR distribution.【F:docs/DESIGN.md†L41-L88】

The new metrics exposed by the core aggregate structural expansions and
entropies (`support_effective`, `load_support_ratio`,
`structural_expansion_longitudinal`, `structural_contraction_longitudinal`,
`structural_expansion_lateral`, `structural_contraction_lateral`,
`delta_nfr_entropy`, `node_entropy`, `phase_delta_nfr_entropy`,
`phase_node_entropy`, `thermal_load`, `style_index`, `network_memory`, along
with `aero_balance_drift`, `slide_catch_budget`, and
`ackermann_parallel_index`). These series feed the `recursivity_operator` and
`mutation_operator` to preserve network memory and archetype transitions.【F:docs/DESIGN.md†L65-L89】【F:tools/tnfr_theory_audit.py†L13-L103】

The `tools/tnfr_theory_audit.py` script produces an up-to-date theoretical
alignment report. Run it with `poetry run python tools/tnfr_theory_audit.py
--core --tests --output tests/_report/theory_impl_matrix.md` and review the
output in `tests/_report/theory_impl_matrix.md` to validate the coverage of the
exposed operators and metrics.【F:tools/tnfr_theory_audit.py†L1-L67】

## Resources

- [Documentation index](index.md) – MkDocs entry point for the full manual.
- [Beginner quickstart](tutorials.md) – installation walkthrough and dataset
  primer.
- [Advanced workflows](advanced_workflows.md) – Pareto sweeps, robustness
  checks and A/B comparisons.
- [API reference](api_reference.md) – module-level documentation.
- [Operational checklist](#operational-checklist) – quantitative targets for
  stint reviews and automated rules.
- [Examples gallery](examples.md) – automation scripts under ``examples/``.
- [CLI guide](cli.md) – command-line usage and configuration templates.
- [Setup equivalences](setup_equivalences.md) – map TNFR metrics to setup
  adjustments.
- [Preset workflow](presets.md) – shareable configurations and HUD layouts.
- [Brake thermal proxy](brake_thermal_proxy.md) – details of the brake fade
  model.
- [Design notes](DESIGN.md) – textual summary of the TNFR operations manual.
