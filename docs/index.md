# TNFR × LFS Documentation

TNFR × LFS (Fractal-Resonant Telemetry Analytics for Live for Speed) is
the lightweight Python toolkit that operationalises the canonical
Telemetría/Teoría de la Naturaleza Fractal-Resonante (TNFR) framework
alongside the Load Flow Synthesis methodology. The theory establishes
how to reason about Event Performance Indicators (EPI), ΔFz/ΔSi deltas
and Sense Index signals; the toolkit provides production-ready layers to
apply those ideas in racing telemetry pipelines:

1. **Acquisition** – ingest telemetry samples from OutSim-compatible
   streams via :class:`tnfr_lfs.acquisition.outsim_client.OutSimClient`.
2. **Core analytics** – extract Event Performance Indicators (EPI) and
   compute ΔFz/ΔSi deltas through :class:`tnfr_lfs.core.epi.EPIExtractor`.
3. **Recommendation engine** – map metrics to actionable setup advice
   using the rule engine in :mod:`tnfr_lfs.recommender.rules`.
4. **Exporters** – serialise analysis results to JSON or CSV with the
   functions in :mod:`tnfr_lfs.exporters`.

The project ships with a CLI (`tnfr-lfs`) as well as examples and unit
tests to illustrate the workflow.

## Installation

Install the toolkit (including its scientific dependencies
`numpy>=1.24,<2.0` and `pandas>=1.5,<3.0`) with:

```bash
pip install .[dev]
```

If your environment cannot consume pre-built wheels, ensure build tools for
NumPy and pandas are available before running the verification pipeline.

## Telemetry requirements

All TNFR metrics (`ΔNFR`, the nodal projections `∇NFR∥`/`∇NFR⊥`, `ν_f`, `C(t)` and related
indicators) are derived from the Live for Speed OutSim/OutGauge telemetry
streams; the toolkit does not synthesise additional inputs. Enable both
UDP broadcasters and extend OutSim with `OutSim Opts ff` in `cfg.txt`
(or run `/outsim Opts ff` before `/outsim 1 …`) so the UDP packets
include the player identifier, driver inputs and the four-wheel block
that feeds the telemetry fusion layer.【F:tnfr_lfs/acquisition/fusion.py†L93-L200】
Likewise, enable the extended OutGauge payload (via `OutGauge Opts …` in
`cfg.txt` or `/outgauge Opts …`) so the simulator transmits tyre
temperatures, their three-layer profile and pressures; set at least the
`OG_EXT_TYRE_TEMP`, `OG_EXT_TYRE_PRESS` and `OG_EXT_BRAKE_TEMP` flags so the
20-float block (inner/middle/outer layers, pressure ring and brake discs)
is broadcast. Otherwise the HUD and CLI will surface those entries as “sin
datos”.【F:tnfr_lfs/acquisition/fusion.py†L594-L657】
The CSV reader mirrors that philosophy by preserving optional columns as
`math.nan` when OutSim leaves them out, preventing artificial estimates
from leaking into the metrics pipeline.【F:tnfr_lfs/acquisition/outsim_client.py†L87-L155】
When the wheel payload is disabled the toolkit now surfaces tyre loads,
slip ratios and suspension metrics as “sin datos” rather than
fabricating zeroed values, making it obvious that the telemetry stream
is incomplete.【F:tnfr_lfs/acquisition/fusion.py†L93-L200】【F:tests/test_acquisition.py†L229-L288】

### Metric field checklist

- **ΔNFR (gradiente nodal) y ∇NFR∥/∇NFR⊥ (proyecciones del gradiente)** – consumen
  las cargas Fz por rueda, sus ΔFz, las fuerzas longitudinales/laterales y las
  deflexiones de suspensión reportadas por OutSim junto con el régimen del
  motor, pedales y banderas ABS/TC que aporta OutGauge para determinar el
  gradiente nodal. Las proyecciones ∇NFR∥/∇NFR⊥ son componentes del gradiente,
  no sustituyen a los canales de carga reales; cruza siempre las recomendaciones
  con los registros `Fz`/`ΔFz` cuando necesites cuantificar fuerzas absolutas.
  【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】
- **ν_f (natural frequency)** – requires load split, slip ratios/angles
  and yaw rate/velocity from OutSim, plus driver style signals (throttle,
  gear) resolved via OutGauge to tailor node categories and spectral
  windows.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L648-L710】
- **C(t) (structural coherence)** – builds on the ΔNFR distribution and
  ν_f bands, leveraging the same OutSim data, the derived `mu_eff_*`
  coefficients and the ABS/TC flags that OutGauge exposes.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】【F:tnfr_lfs/core/coherence.py†L65-L125】
- **Ackermann / Slide Catch budgets** – use only the `slip_angle_*`
  channels and `yaw_rate` broadcast by OutSim to measure parallel steer
  deltas and slide recovery headroom; when these signals are absent the
  toolkit surfaces `"sin datos"` instead of synthetic values.
- **Aero balance drift** – derives rake trends exclusively from OutSim
  `pitch` plus front/rear suspension travel so the drift guidance mirrors
  native LFS telemetry even if `AeroCoherence` appears neutral.【F:tnfr_lfs/core/metrics.py†L1650-L1735】
- **Tyre temperatures/pressures** – TNFR × LFS now consumes the values
  emitted by the OutGauge extended payload when they are finite and
  positive; when the block is disabled the fusion keeps the historical
  sample or `"sin datos"` so downstream tooling does not fabricate
  temperatures.【F:tnfr_lfs/acquisition/fusion.py†L594-L657】

## Checklist operativo

La revisión previa a una tanda se apoya en un checklist compacto que
evalúa cuatro objetivos cuantitativos:

1. **Sense Index medio ≥ 0.75** – objetivo base de perfil para el
   operador, usado por el motor de reglas y persistido en
   `RuleProfileObjectives`.【F:tnfr_lfs/recommender/rules.py†L604-L615】
2. **Densidad ΔNFR ≤ 6.00 kN·s⁻¹** – referencia por defecto utilizada
   para calcular la integral absoluta de ΔNFR al evaluar la puntuación
   de objetivos.【F:tnfr_lfs/recommender/search.py†L210-L236】
3. **Headroom de freno ≥ 0.40** – margen mínimo recomendado antes de que
   las intervenciones en reparto o ventilación sean obligadas.【F:tnfr_lfs/recommender/rules.py†L604-L615】
4. **Δμ aerodinámico ≤ 0.12** – tolerancia con la que se normaliza la
   coherencia aero-mecánica y las alertas de drift.【F:tnfr_lfs/core/metrics.py†L2558-L2575】

El HUD/CLI resume estos objetivos en una línea “Checklist” que marca con
✅ los objetivos cumplidos y con ⚠️ los que requieran atención, usando
las métricas en tiempo real (media de Si, integral ΔNFR, headroom y
imbalance aerodinámico).【F:tnfr_lfs/cli/osd.py†L1477-L1567】

## Branding and terminology

- **TNFR theory** describes the conceptual framework and vocabulary.
- **TNFR × LFS toolkit** is the software package documented in this
  site; it contains the CLI, Python modules and exporter pipelines.
- Identifiers that keep the hyphen (e.g. `tnfr-lfs`, `tnfr-lfs.toml`)
  remain for compatibility with existing scripts while representing the
  same TNFR × LFS toolkit.

## Resources

- [API Reference](api_reference.md)
- [Command Line Interface](cli.md)
- [Tutorials](tutorials.md)
- [Setup equivalences](setup_equivalences.md)
- [Preset workflow](presets.md)
