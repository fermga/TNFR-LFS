# Detection configuration loader plan

## Current module overview

The ``tnfr_core.config.loader`` module resolves detection overrides by merging
hierarchical YAML tables and loading them from multiple locations. The
``get_params`` entry point walks defaults, track sections, car sections, and
compound overrides, using helper functions such as ``_lookup_section`` and
``_deep_merge`` to normalise identifiers and perform recursive merges.
The ``load_detection_config`` helper then materialises the raw detection table by
respecting explicit file paths, caller-provided search directories, pack roots,
and the packaged ``detection.yaml`` fallback.

## Extension strategy

All upcoming work explicitly reuses ``tnfr_core.config.loader`` and expands the
logic inside that module rather than introducing alternative entry points. The
new behaviours will be expressed by extending the existing helper functions and
entry points: ``get_params`` will continue to orchestrate the merge order and
will receive the additional hierarchy hooks, while ``load_detection_config``
will host any new validation passes that operate on the fully merged mapping.
Supporting helpers such as ``merge_section`` and ``_merge_compounds`` will be
augmented in-place so the override precedence stays aligned with the current
implementation while keeping the helper surface area unchanged. Validation
steps will also live in this module, leveraging the already merged payload
before it is frozen via ``MappingProxyType``.

### Upcoming class hierarchy support

The next increment will add explicit ``classes`` handling directly inside
``get_params`` so that shared overrides can be defined per car class before
being specialised at the individual car level. The update will also consider a
top-level ``compounds`` table so that global compound defaults can be reused
outside the track/car/class sections.

To integrate classes cleanly we will:

* Extend ``get_params`` so the merge order slots classes between the existing
  track and car resolutions. The precedence will therefore be ``defaults`` →
  top-level compounds → top-level tracks → class defaults/tracks/compounds →
  car defaults/tracks/compounds. Car-specific entries will continue to override
  class-level data, while class-level overrides will in turn override
  track/global selections.
* Reuse ``_normalise_identifier`` across the new sections so that class names
  and compound identifiers remain case-insensitive and resilient to formatting
  differences. This keeps lookups aligned with current car/track behaviour.
* Update ``get_params`` to ensure class overrides compose with car and track
  overrides. Class-specific ``tracks`` blocks will be resolved before car-level
  ``tracks`` so that car overrides stay authoritative while still inheriting
  defaults from their classes. Global/top-level ``compounds`` will be merged
  first, followed by class-level compounds and finally car-level compounds for
  the selected tyre.
* Add an opt-in validation pass inside ``load_detection_config`` to confirm
  that class references and compound selections resolved by ``get_params`` map
  to existing identifiers before returning the frozen mapping.

### Test coverage expansion

``tests/test_config_loader_core.py`` will gain fixtures that exercise the full
cascade (global defaults/compounds, tracks, classes, car overrides) so that the
updated ``get_params`` and ``load_detection_config`` flows are verified end to
end and precedence remains stable when the new sections are introduced.

## Search path compatibility

Maintaining compatibility with the current search order is critical for existing
deployments. Any additional logic must preserve the candidate resolution used by
``load_detection_config``:

1. Honour explicit ``path`` arguments when supplied.
2. Iterate caller-provided ``search_paths`` directories or files, resolving
   directories against ``detection.yaml``.
3. Inspect ``pack_root / "config" / "detection.yaml"`` and
   ``pack_root / "detection.yaml"`` when a pack root is given.
4. Fall back to the packaged resource in ``tnfr_lfs.resources.config``.

Additions such as post-load validation or layered overrides will execute after
this candidate list is resolved so ``pack_root``, ``search_paths``, and the
packaged defaults continue to work exactly as they do today.

### Calibrator hand-off

The calibration tooling that generates bespoke detection heuristics must emit
its overrides inside the same hierarchy inspected by ``load_detection_config``.
When the calibrator materialises a pack, it will write the merged payload to
``<pack_root>/config/detection.yaml`` so that the runtime can discover the file
without any extra flags. If the tool needs to output multiple variants, it will
surface the concrete destination and expose that path back to the runtime as an
explicit override. This keeps the calibration workflow aligned with the search
order above while allowing specialised deployments to point the loader at the
exact artefact produced by the tool.

## Deliverables

Calibration runs hand over a detection bundle that mirrors the pack layout. The
generated YAML must live under ``config/detection.yaml`` at the root of the pack
that will be deployed (either a staging directory referenced via
``--pack-root`` or the static bundle shipped with the release). Deployments that
partition overrides per customer or environment should replicate the same
layout, placing the calibrated payload under ``<pack_root>/config``. With this
structure in place the loader will pick up the new thresholds automatically,
falling back to caller-specified paths only when the calibrator exposes an
alternative location.
