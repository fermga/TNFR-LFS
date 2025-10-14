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

Future enhancements that introduce new override levels or validation stages will
extend this existing module instead of creating parallel loaders. New logic will
hook into the current merge flow (for example by augmenting ``merge_section`` or
``_merge_compounds``) so that the override precedence stays aligned with the
current implementation while keeping the helper surface area unchanged.
Validation steps will also live in this module, leveraging the already merged
payload before it is frozen via ``MappingProxyType``.

### Upcoming class hierarchy support

The next increment will add explicit ``classes`` handling to ``get_params`` so
that shared overrides can be defined per car class before being specialised at
the individual car level. The update will also consider a top-level
``compounds`` table so that global compound defaults can be reused outside the
track/car/class sections.

To integrate classes cleanly we will:

* Extend the merge order to slot classes between the existing track and car
  resolutions. The precedence will therefore be ``defaults`` → top-level
  compounds → top-level tracks → class defaults/tracks/compounds → car
  defaults/tracks/compounds. Car-specific entries will continue to override
  class-level data, while class-level overrides will in turn override
  track/global selections.
* Reuse ``_normalise_identifier`` across the new sections so that class names
  and compound identifiers remain case-insensitive and resilient to formatting
  differences. This keeps lookups aligned with current car/track behaviour.
* Ensure class overrides compose with car and track overrides. Class-specific
  ``tracks`` blocks will be resolved before car-level ``tracks`` so that car
  overrides stay authoritative while still inheriting defaults from their
  classes. Global/top-level ``compounds`` will be merged first, followed by
  class-level compounds and finally car-level compounds for the selected tyre.

### Test coverage expansion

``tests/test_config_loader_core.py`` will gain fixtures that exercise the full
cascade (global defaults/compounds, tracks, classes, car overrides) to ensure
that each merge level behaves as documented and that precedence remains stable
when the new sections are introduced.

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
