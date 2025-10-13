# API Reference overview

The TNFR × LFS toolkit publishes its Python API through
[AutoAPI-generated documentation](reference/autoapi/index.md).  AutoAPI renders
live signatures from the source tree so the reference pages always match the
installed package.  This overview explains how those pages are organised and
provides quick links to the modules that teams consult most often.

## How the reference is organised

1. **Package index** – Start at the [AutoAPI index](reference/autoapi/index.md)
   to browse the `tnfr_lfs` package tree.  Every submodule is grouped by
   package name, mirroring the layout under `src/tnfr_lfs/`.
2. **Module details** – Each module page lists public classes, functions and
   constants with short excerpts from the in-code documentation.  Use the
   sidebar or page table of contents to jump to individual symbols.
3. **Search** – The MkDocs search box searches the AutoAPI pages too, making it
   easy to jump directly to a class or function when you already know its
   name.

## Frequently used entry points

These links point straight to the most common modules in the AutoAPI tree:

- [`tnfr_lfs.telemetry`](reference/autoapi/tnfr_lfs/ingestion/index.md) covers
  live telemetry capture, file replay helpers and session configuration
  loaders.
- [`tnfr_core.epi`](reference/autoapi/tnfr_core/epi/index.md) documents
  the event processing interfaces consumed by downstream analytics.
- [`tnfr_lfs.recommender`](reference/autoapi/tnfr_lfs/recommender/index.md)
  describes the recommendation engines that pair telemetry with setup
  adjustments.
- [`tnfr_lfs.exporters`](reference/autoapi/tnfr_lfs/exporters/index.md) lists
  data exporters for dashboards, CSV pipelines and HUD integrations.
- [`tnfr_lfs.visualization`](reference/autoapi/tnfr_lfs/visualization/index.md)
  summarises reusable plotting and rendering helpers for reporting.

## Related guides

The narrative guides under the Reference section complement the API pages:

- [Design manual](DESIGN.md) describes architectural decisions that shape the
  module layout.
- [Development workflow](DEVELOPMENT.md) explains coding conventions and
  release practices for contributors.
- [Setup plan workflow](setup_plan.md) and [preset workflow](presets.md) walk
  through end-to-end configuration examples before you dive into the API.

Refer back to this page whenever you need to re-orient yourself before diving
into the detailed API reference.
