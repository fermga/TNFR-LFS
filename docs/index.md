# TNFR × LFS Documentation

TNFR × LFS (Tyre Normal Force × Load Flow Synthesis) is the lightweight
Python toolkit that operationalises the core insights of the
Telemetry/Teoría de la Naturaleza Fractal-Resonante (TNFR) framework.
The theory establishes how to reason about Event Performance Indicators
(EPI), ΔNFR deltas and Sense Index signals; the toolkit provides
production-ready layers to apply those ideas in racing telemetry
pipelines:

1. **Acquisition** – ingest telemetry samples from OutSim-compatible
   streams via :class:`tnfr_lfs.acquisition.outsim_client.OutSimClient`.
2. **Core analytics** – extract Event Performance Indicators (EPI) and
   compute ΔNFR/ΔSi deltas through :class:`tnfr_lfs.core.epi.EPIExtractor`.
3. **Recommendation engine** – map metrics to actionable setup advice
   using the rule engine in :mod:`tnfr_lfs.recommender.rules`.
4. **Exporters** – serialise analysis results to JSON or CSV with the
   functions in :mod:`tnfr_lfs.exporters`.

The project ships with a CLI (`tnfr-lfs`) as well as examples and unit
tests to illustrate the workflow.

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
