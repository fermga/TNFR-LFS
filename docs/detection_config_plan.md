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

Global defaults remain meaningful even when operator-specific sections are
absent. The detection layer now falls back to the shared ``defaults``/``tracks``/
``classes``/``cars``/``compounds`` hierarchy materialised by
``load_detection_config`` so detectors such as ``detect_il`` inherit the bundled
thresholds without needing bespoke sections in ``detection.yaml``.【F:src/tnfr_core/operators/operator_detection.py†L312-L330】

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

## Conflict resolution order

``get_params`` already resolves overrides by walking the hierarchy in a strict
sequence: top-level ``defaults`` → top-level ``compounds`` → ``tracks`` →
``classes`` → ``cars``. Each layer reuses ``merge_section`` so that the
sub-section ``defaults`` are applied first, followed by inline keys, compound
overrides (including ``__default__``/``*`` fallbacks and the requested tyre),
and finally nested ``tracks`` entries within that layer.【F:src/tnfr_core/config/loader.py†L37-L101】 This means that:

* Track-wide tweaks override global defaults and compounds before class or car
  logic runs.
* Class-level data starts from the merged track payload and can be
  refined per-compound or per-track before handing control to the car layer.
* Car entries inherit everything resolved so far and can still specialise per
  compound or track as needed.

When writing ``best_params.yaml`` (or any other detection profile), structure
the mapping to mirror that order. The snippet below shows the expected layout
with explicit comments to highlight how overrides compose:

```yaml
defaults:
  pressure: 1.0
  brake_bias: 52

compounds:
  __default__:
    pressure: 1.05        # Applies to every tyre before track/class/car logic
  r2:
    brake_bias: 53        # Global R2 tweak before track/class/car overrides

tracks:
  __default__:
    defaults:
      pressure: 1.1       # Baseline for every track
    compounds:
      __default__:
        brake_bias: 54    # Track-level tweak applied before classes/cars
  bl1:
    defaults:
      pressure: 1.2

classes:
  __default__:
    defaults:
      camber:
        front: -2.0
  tcr:
    defaults:
      pressure: 1.3
    compounds:
      r2:
        brake_bias: 56    # Class + compound override
    tracks:
      bl1:
        defaults:
          pressure: 1.32  # Class + track override

cars:
  __default__:
    defaults:
      toe:
        front: -0.05
  xfg:
    defaults:
      pressure: 1.4
    compounds:
      r2:
        pressure: 1.5     # Final car-specific compound tweak
    tracks:
      bl1:
        defaults:
          toe:
            front: -0.08  # Car + track override (runs last)
```

### Changing the order intentionally

If a different precedence is required, adjust the call order inside
``get_params``—for example, moving class resolution ahead of track resolution
would mean reordering the ``merge_section`` invocations for ``tracks_table``
and ``classes_table`` and updating any nested ``tracks`` handling to match the
new precedence.【F:src/tnfr_core/config/loader.py†L73-L101】 After doing so,
refresh the expectations in ``tests/test_config_loader_core.py`` so the
assertions that cover compound/track/class precedence continue to reflect the
runtime behaviour.【F:tests/test_config_loader_core.py†L6-L107】 Adjust or add
fixtures to match the intended cascade before touching production data.

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

The calibration CLI accepts one or more structural identifiers via
``--operators`` and uses the packaged search spaces for each detector. Provide a
custom sweep file with ``--operator-grid`` when bespoke ranges are required, for
example:

```bash
python tools/calibrate_detectors.py \
  --raf-root /data/raf \
  --labels labels.jsonl \
  --out output/calibration \
  --operators NAV EN SILENCE \
  --operator-grid grids.yaml
```

Omitting ``--operator-grid`` still calibrates the listed detectors using the
bundled defaults, which is convenient for quick explorations or smoke tests.

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
