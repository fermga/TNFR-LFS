"""Public interfaces describing TNFR × LFS plugin metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence, Tuple, Type

from .base import TNFRPlugin
from .registry import get_plugin_operator_requirements

OperatorIdentifier = str

__all__ = [
    "OperatorIdentifier",
    "PluginMetadata",
    "PluginContract",
]


def _ensure_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalised = value.strip()
    if not normalised:
        raise ValueError(f"{field_name} must not be empty")
    return normalised


def _normalise_identifiers(
    identifiers: Iterable[OperatorIdentifier],
    *,
    field_name: str,
) -> Tuple[OperatorIdentifier, ...]:
    if identifiers is None:
        raise TypeError(f"{field_name} must be provided")

    seen: set[OperatorIdentifier] = set()
    ordered: list[OperatorIdentifier] = []

    for name in identifiers:
        if not isinstance(name, str):
            raise TypeError(f"{field_name} entries must be strings")
        normalised = name.strip()
        if not normalised:
            raise ValueError(f"{field_name} entries must be non-empty strings")
        if normalised in seen:
            continue
        seen.add(normalised)
        ordered.append(normalised)

    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class PluginMetadata:
    """Immutable description of a TNFR × LFS plugin contract."""

    identifier: str
    version: str
    description: str = ""
    required_operators: Tuple[OperatorIdentifier, ...] = field(default_factory=tuple)
    optional_dependencies: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "identifier", _ensure_string(self.identifier, field_name="identifier"))
        object.__setattr__(self, "version", _ensure_string(self.version, field_name="version"))
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")

        object.__setattr__(
            self,
            "required_operators",
            _normalise_identifiers(
                self.required_operators,
                field_name="required_operators",
            ),
        )
        object.__setattr__(
            self,
            "optional_dependencies",
            _normalise_identifiers(
                self.optional_dependencies,
                field_name="optional_dependencies",
            ),
        )

    @classmethod
    def from_plugin(
        cls,
        plugin_cls: Type[TNFRPlugin],
        *,
        identifier: str,
        version: str,
        description: str = "",
        optional_dependencies: Sequence[str] | None = None,
    ) -> "PluginMetadata":
        """Build metadata using the registry information for ``plugin_cls``."""

        requirements = get_plugin_operator_requirements(plugin_cls)
        dependencies: Sequence[str] = optional_dependencies or ()
        return cls(
            identifier=identifier,
            version=version,
            description=description,
            required_operators=requirements,
            optional_dependencies=tuple(dependencies),
        )


@dataclass(frozen=True, slots=True)
class PluginContract:
    """Binding between a plugin class and its associated metadata."""

    plugin_cls: Type[TNFRPlugin]
    metadata: PluginMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.plugin_cls, type) or not issubclass(self.plugin_cls, TNFRPlugin):
            raise TypeError("plugin_cls must be a TNFRPlugin subclass")

        try:
            registered = get_plugin_operator_requirements(self.plugin_cls)
        except LookupError as exc:  # pragma: no cover - defensive guard
            raise ValueError(
                f"plugin {self.plugin_cls.__name__!s} has not been registered in the metadata registry"
            ) from exc

        if registered != self.metadata.required_operators:
            raise ValueError(
                "metadata required_operators must match registered operator requirements",
            )

    @classmethod
    def from_plugin(
        cls,
        plugin_cls: Type[TNFRPlugin],
        *,
        identifier: str,
        version: str,
        description: str = "",
        optional_dependencies: Sequence[str] | None = None,
    ) -> "PluginContract":
        """Create a contract ensuring metadata matches the registry state."""

        metadata = PluginMetadata.from_plugin(
            plugin_cls,
            identifier=identifier,
            version=version,
            description=description,
            optional_dependencies=optional_dependencies,
        )
        return cls(plugin_cls=plugin_cls, metadata=metadata)
