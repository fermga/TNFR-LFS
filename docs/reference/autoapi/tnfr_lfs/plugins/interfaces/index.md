# `tnfr_lfs.plugins.interfaces` module
Public interfaces describing TNFR × LFS plugin metadata contracts.

## Classes
### `PluginMetadata`
Immutable description of a TNFR × LFS plugin contract.

#### Methods
- `from_plugin(cls, plugin_cls: Type[TNFRPlugin], *, identifier: str, version: str, description: str = '', optional_dependencies: Sequence[str] | None = None) -> 'PluginMetadata'`
  - Build metadata using the registry information for ``plugin_cls``.

### `PluginContract`
Binding between a plugin class and its associated metadata.

#### Methods
- `from_plugin(cls, plugin_cls: Type[TNFRPlugin], *, identifier: str, version: str, description: str = '', optional_dependencies: Sequence[str] | None = None) -> 'PluginContract'`
  - Create a contract ensuring metadata matches the registry state.

## Attributes
- `OperatorIdentifier = str`

