# `tnfr_lfs.plugins.registry` module
Metadata registry for TNFR × LFS plugin dependencies.

## Classes
### `PluginMetadataError` (ValueError)
Raised when invalid plugin metadata is supplied to the registry.

## Functions
- `available_operator_identifiers() -> frozenset[OperatorName]`
  - Return the canonical set of operator identifiers available to plugins.
- `register_plugin_metadata(plugin_cls: Type[TNFRPlugin], *, operators: Sequence[OperatorName] | None) -> Type[TNFRPlugin]`
  - Register metadata describing the operator requirements of ``plugin_cls``.
- `plugin_metadata(*, operators: Sequence[OperatorName] | None) -> type`
  - Decorator used by plugins to declare their operator dependencies.
- `get_plugin_operator_requirements(plugin_cls: Type[TNFRPlugin]) -> Tuple[OperatorName, ...]`
  - Return the operator identifiers registered for ``plugin_cls``.
- `iter_plugin_operator_requirements() -> Iterator[tuple[type[TNFRPlugin], Tuple[OperatorName, ...]]]`
  - Iterate over registered plugin classes and their operator requirements.

## Attributes
- `OperatorName = str`

