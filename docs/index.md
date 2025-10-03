# TNFR-LFS Documentation

TNFR-LFS (Tyre Normal Force & Load Flow Synthesis) is a lightweight
Python toolkit that replicates the most important analytical steps of
the Telemetry Normal Force Reconstruction workflow.  The package
consists of four layers:

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


## Resources

- [API Reference](api_reference.md)
- [Command Line Interface](cli.md)
- [Tutorials](tutorials.md)
