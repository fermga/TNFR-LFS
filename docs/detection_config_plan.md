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
