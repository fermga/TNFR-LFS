# `tnfr_lfs.core.phases` module
Shared definitions for TNFR phase nomenclature.

## Functions
- `normalise_phase_key(phase: str) -> str`
  - Return the canonical lower-case identifier for ``phase``.
- `expand_phase_alias(phase: str) -> Tuple[str, ...]`
  - Return all concrete phases associated with ``phase``.
- `phase_family(phase: str) -> str`
  - Return the legacy family identifier for ``phase``.
- `replicate_phase_aliases(payload: Mapping[str, T], *, combine: Callable[[Sequence[T]], T] | None = None) -> dict[str, T]`
  - Return ``payload`` enriched with legacy phase aliases.

Parameters
----------
payload:
    Mapping keyed by phase identifiers. Keys are normalised to lower case in
    the resulting dictionary. Values are copied verbatim unless ``combine``
    is provided.
combine:
    Optional reducer applied when populating a legacy alias from multiple
    concrete phase values. When omitted the last available concrete phase
    value is reused for the alias.

## Attributes
- `PHASE_SEQUENCE = ('entry1', 'entry2', 'apex3a', 'apex3b', 'exit4')`
- `LEGACY_PHASE_MAP = {'entry': ('entry1', 'entry2'), 'apex': ('apex3a', 'apex3b'), 'exit': ('exit4',)}`
- `T = TypeVar('T')`

