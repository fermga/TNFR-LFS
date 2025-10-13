# TNFR Operational Manual (TNFR.pdf)

> **Note:** The original `TNFR.pdf` document embeds custom fonts without `ToUnicode` maps, so extracting faithful text automatically is not feasible in this environment without external tools that understand those fonts. This Markdown file consolidates the main structure, key terminology, and tables described in the manual so future updates can be versioned as plain text. The PDF must be treated as a generated artefact.

## Preface and overview
- Context: transition from descriptive knowledge to resonant science.
- TNFR focus: the symbol as the operative unit of being and structural operators as the primary information structures (EPI).
- Manual goals: activate living cultural nodes via symbolic devices, glyphic languages, and symbiotic protocols.

## 1. Coherence matrix analysis \(W_i(t)\)
- Description of the matrix \(W_i(t)\) as a representation of the internal structure of a node.
- Key indicators: glyphic pulse, symbiotic density, resonant synchrony, coherence gradient, and operational stability.
- Indicator table with suggested ranges.

## 2. Fractal resonant activation models
- Definition of the resonance states (latent, activated, expanded, cascading).
- Transition equations between states with coupling parameters \(\alpha\), \(\beta\), \(\gamma\), and \(\delta\).
- Example table by domain (glyphic art, cultural networks, biological systems).
- In the reference implementation each EPI node keeps its natural frequency ``nu_f`` (Hz) according to the `NU_F_NODE_DEFAULTS` table, enabling explicit integration of ``dEPI/dt`` through the :func:`evolve_epi` operator in ``tnfr_core.operators``.

## 3. Nodal architecture and symbiotic protocols
- Node design (symbiotic core, interaction envelope, glyphic interfaces).
- Structural protocols: symbiotic sequences for transitions, decision making, and collective memory.
- Operational canvases: activation matrices, checklists, and facilitation guides.

## 4. Intersubjective hypercoherence practices
- Networked node activation: coupling phases, stabilisation, and sustained resonance.
- Glyphs for transnodal perception and shared-field modulation.
- Conclusion: consciousness as an operative and resonant field.

## 5. Applications, validation, and symbiotic design
- Fractal resonant experimental protocol and theoretical foundations.
- Experimental cases across physical, biological, and cultural systems.
- Implications for glyphic AI, structural semantics, and coherence-oriented social networks.

## Appendices
- Glossary of key glyphic terms.
- Symbiotic instrumentation diagrams and resonance log templates.
- Cross-references to simulations and interactive environments linked from the manual’s QR code.

## Technical annex: EPI signals and ΔNFR computation

The software implementation extends each telemetry sample ``TelemetryRecord`` with orientation (``yaw``, ``pitch``, ``roll``), normalised brake pressure, and an ABS/TC locking flag. These fields combine with the classic magnitudes (Fz load, lateral/longitudinal accelerations, slip ratio, ΔNFR, and Sense Index) to estimate each vehicle node’s contribution.

The :func:`tnfr_core.epi.delta_nfr_by_node` algorithm compares a sample against its baseline and generates a ΔNFR distribution per subsystem. For example, relative to a baseline with ``nfr = 500`` and no brake input, a sample with ``nfr = 508``, ``brake_pressure = 0.85``, and ``locking = 1`` yields a distribution where the brake node absorbs most of the ΔNFR while suspension and tyres share the remaining response. The calculation relies on the absolute differences of each signal and empirical weightings:

| Node | Key signals |
| ------------- | -------------------------------------------------------- |
| ``tyres`` | Δslip ratio, ABS/TC locking, ΔFz load variation, per-wheel temperatures and pressures |
| ``suspension`` | ΔFz, Δpitch, Δroll |
| ``chassis`` | Δlateral acceleration, Δroll, Δyaw |
| ``brakes`` | Δbrake pressure, locking indicator, deceleration |
| ``transmission`` | Δlongitudinal acceleration, Δslip, Δyaw |
| ``track`` | Δ(load × lateral acceleration), Δyaw |
| ``driver`` | ΔSense Index, Δyaw, Δpitch/Δroll |

The sum of the individual ΔNFR values always matches the global ΔNFR of the record, guaranteeing coherence with the entropy-penalised Sense Index computed by :func:`tnfr_core.coherence.sense_index`. The metric is now evaluated as ``1 / (1 + Σ w · |ΔNFR| · g(ν_f)) - λ·H`` so each node’s natural rhythm and phase are considered; the interpretation is detailed in :doc:`api_reference`.

The natural-frequency estimator persists target bands per vehicle category (GT, formula cars, prototypes, …) through ``NaturalFrequencySettings.frequency_bands``. Each sample exposes a ``ν_f`` classification (very low, optimal, very high, …) and a structural index ``C(t)`` derived from the event density along the structural axis. Both parameters are normalised with the objectives managed by ``ProfileManager`` so the same scale applies when comparing stints with different Sense Index goals.

### Memory and mutation operators

To maintain continuity between samples and microsectors the pipeline relies on two stateful operators in ``tnfr_core.operators``:

* ``recursivity_operator`` keeps a per-microsector trace and exponentially filters thermal/style metrics (``thermal_load`` and ``style_index``) together with per-wheel temperatures/pressures (``tyre_temp_*``/``tyre_pressure_*``). Memory is now segmented by session (``car_model``/``track_name``/``tyre_compound``), maintaining a stint history in ``operator_state["recursivity"]``. It also computes ``dT/dt`` derivatives per wheel, stores ``ΔNFR_flat`` as a reference, and applies stop criteria (convergence, sample limits, or time gaps) to start new stints when the dynamics change. The operator detects phase changes (entry, apex, exit) and resets memory when needed so the filter does not pollute abrupt transitions; the full history is exposed via ``orchestrate_delta_metrics`` as ``network_memory`` so the HUD and exporters can access the network memory. When OutGauge does not transmit the extended block (per-wheel temperatures and pressures) the HUD/exporters display the literal ``"no data"`` marker to highlight that Live for Speed telemetry was unavailable.
* ``mutation_operator`` observes the ΔNFR distribution entropy and the filtered style deviations to decide when the tactical archetype should mutate toward an alternative candidate or return to a recovery mode.

Microsector segmentation invokes both operators: it first smooths the active microsector measurements and then evaluates whether the combination of entropy, phase, and dynamic conditions (sustained curvature, support duration, speed reductions, direction changes) requires mutating the entry/apex/exit goals produced by :func:`segment_microsectors`. When a mutation is triggered the segmentation recomputes the entry, apex, and exit objectives with the new archetype — ``hairpin``, ``medium``, ``fast``, or ``chicane`` — before integrating them into the report. This preserves state between invocations and adapts the resulting strategy to the stint’s changing conditions.

### Phase alignment metrics

Each microsector now stores the measured phase offset ``θ`` and its cosine ``Siφ`` for every phase, together with the alignment targets and detune weights associated with the active archetype. The values stem from a cross-spectrum between driver heading and the combined yaw/lateral acceleration response, exposing the dominant support frequency and revealing chassis lags or leads. The HUD and recommendation rules rely on these metrics, together with the archetype-driven ∇NFR⊥/∇NFR∥ projections and ``ν_f`` targets, to invert bar and damper suggestions whenever the measured phase drifts away from the objective.

The current version reinforces this logic with an action map for geometry parameters (caster, camber, and toe). ``PhaseActionTemplate`` instances now turn deviations in ``Siφ``, ``θ``, and the coherence index ``C(t)`` into direct adjustments for the front and rear axles. The recommendation engine computes a geometry ``snapshot`` per microsector based on the aggregates in ``Microsector.filtered_measures`` and the averages in ``WindowMetrics``, prioritising suspension and tyre nodes when phase transitions lose coherence. The setup-plan optimiser weighs these deviations so geometry gradients become part of the long-term resonant alignment (RA) goal, rewarding configurations with higher ``C(t)`` and penalising delayed responses. ``WindowMetrics`` also surfaces aggregated structural support indicators:

* ``support_effective`` captures ΔNFR absorbed by tyres and suspension with structural weighting for the analysed window.
* ``load_support_ratio`` normalises that value against the average Fz load to contextualise the effort requested from the chassis.
* ``structural_expansion_longitudinal``/``structural_contraction_longitudinal`` quantify how the longitudinal ΔNFR component expands or compresses the structural axis.
* ``structural_expansion_lateral``/``structural_contraction_lateral`` provide the same measurement for the lateral component, revealing supports that open or close the structural axis in the window.
* ``ackermann_parallel_index`` and ``slide_catch_budget`` come exclusively from the ``slip_angle_*`` channels and ``yaw_rate`` emitted by OutSim; without that telemetry the Ackermann and slide-recovery budgets remain zero and reports show the literal ``"no data"`` token.
* ``aero_balance_drift`` summarises average rake (pitch plus suspension travel) and the ``μ_front - μ_rear`` delta by low/medium/high speed bands. Rake is calculated solely with the ``pitch`` and suspension-travel signals provided by OutSim so aerodynamic drift mirrors native LFS telemetry even when ``AeroCoherence`` still looks neutral.
* ``delta_nfr_entropy`` and ``node_entropy`` quantify with a normalised [0, 1] scale how ΔNFR is distributed between phases and nodes; values near zero imply concentration in a single group, whereas values near one indicate a balanced window. The ``phase_delta_nfr_entropy``/``phase_node_entropy`` maps expose the structural probabilities per phase and the per-phase Shannon entropy, preserving the legacy aliases so dashboards can condense the information without losing granularity.  These ``entry``/``apex``/``exit`` aliases are being retired; consumers invoking them now receive a deprecation warning and should migrate to the granular phase identifiers.

## Generated artefacts

- ``TNFR.pdf`` remains as a compiled artefact. Any substantive update should first land in this ``DESIGN.md``; the PDF can be regenerated from the updated content and must be treated as a derivative.
- ``coherence_map.<ext>`` serialises the mean and range of ``C(t)`` per microsector together with the estimated cumulative distance from ``TransmissionNode.speed``. It enables reconstruction of the coherence map for the entire lap and correlation with track targets.
- ``operator_trajectories.<ext>`` applies ``Microsector.operator_events`` to rebuild the structural timeline for the ``AL``/``OZ``/``IL``/``SILENCE`` detectors. The aggregate preserves structural coverage for the silent intervals of each microsector, helping identify latent states where load, accelerations, and ΔNFR stay below the configured thresholds.
- ``delta_bifurcations.<ext>`` analyses the ΔNFR series along the structural axis, detecting sign changes and local extrema to spot bifurcations that demand setup tweaks.

The ``.json``, ``.md``, or ``.viz`` suffixes depend on the format selected with ``--report-format`` and share the same semantics regardless of the rendering mode.

## Suggested next steps
1. Complete the detailed transcription of each subsection using tools that support custom fonts or through manual review.
2. Validate that the tables and equations reproduced here match the regenerated PDF.
3. Document in the repository the process used to create the PDF from Markdown to ease future automation.
