"""Plugin manager responsible for discovery and lifecycle operations."""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .base import TNFRPlugin
from . import registry


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredPlugin:
    """Metadata describing a discovered plugin class."""

    cls: type[TNFRPlugin]
    module_name: str
    operators: tuple[str, ...]


class PluginManager:
    """Manage discovery, loading and execution of TNFR × LFS plugins."""

    def __init__(self) -> None:
        self.plugins: Dict[str, TNFRPlugin] = {}
        self.plugin_registry: Dict[str, RegisteredPlugin] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover_plugins(self, plugin_dir: str | Path) -> Dict[str, RegisteredPlugin]:
        """Discover plugins located in ``plugin_dir``.

        Parameters
        ----------
        plugin_dir:
            Directory containing plugin modules or packages.

        Returns
        -------
        Dict[str, RegisteredPlugin]
            Mapping of qualified plugin names to registration metadata.
        """

        path = Path(plugin_dir)
        discovered: Dict[str, RegisteredPlugin] = {}

        if not path.exists():
            logger.warning("Plugin directory '%s' does not exist", path)
            return discovered

        for module_path in self._iter_plugin_module_paths(path):
            module_name = self._unique_module_name(module_path)
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                logger.warning("Unable to create import spec for plugin at '%s'", module_path)
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)  # type: ignore[union-attr]
            except Exception:  # pragma: no cover - exercised in tests
                logger.exception("Failed to import plugin module '%s'", module_path)
                sys.modules.pop(module_name, None)
                continue

            for qualified_name, registration in self._register_module_plugins(module).items():
                discovered[qualified_name] = registration

        return discovered

    def _iter_plugin_module_paths(self, root: Path) -> Iterable[Path]:
        """Yield importable module paths contained in ``root``."""

        for entry in sorted(root.iterdir()):
            if entry.name.startswith("__"):
                continue

            if entry.is_file() and entry.suffix == ".py":
                yield entry
            elif entry.is_dir():
                init_file = entry / "__init__.py"
                if init_file.exists():
                    yield init_file

    def _unique_module_name(self, module_path: Path) -> str:
        """Return a deterministic but unique module name for ``module_path``."""

        token = uuid.uuid5(uuid.NAMESPACE_URL, str(module_path.resolve()))
        return f"tnfr_lfs.plugins.dynamic_{token.hex}"

    def _register_module_plugins(self, module: types.ModuleType) -> Dict[str, RegisteredPlugin]:
        """Register plugin classes defined in ``module``."""

        registered: Dict[str, RegisteredPlugin] = {}

        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if (
                isinstance(attribute, type)
                and issubclass(attribute, TNFRPlugin)
                and attribute is not TNFRPlugin
            ):
                qualified_name = f"{attribute.__module__}.{attribute.__name__}"
                try:
                    operators = self._resolve_plugin_dependencies(attribute)
                except Exception:
                    logger.exception(
                        "Skipping plugin '%s' due to dependency resolution failure",
                        qualified_name,
                    )
                    continue

                registration = RegisteredPlugin(
                    cls=attribute,
                    module_name=module.__name__,
                    operators=operators,
                )
                self.plugin_registry[qualified_name] = registration
                registered[qualified_name] = registration
                logger.info("Registered plugin '%s' from module '%s'", qualified_name, module.__name__)

        return registered

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------
    def load_plugin(self, plugin_name: str, *args: Any, **kwargs: Any) -> TNFRPlugin:
        """Load and instantiate the plugin identified by ``plugin_name``."""

        with self._lock:
            if plugin_name in self.plugins:
                return self.plugins[plugin_name]

            registration = self.plugin_registry.get(plugin_name)
            if registration is None:
                raise LookupError(f"Plugin '{plugin_name}' is not registered")

            self._resolve_plugin_dependencies(registration.cls)
            try:
                instance = registration.cls(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - error path tested
                logger.exception("Failed to instantiate plugin '%s'", plugin_name)
                raise RuntimeError(f"Unable to instantiate plugin '{plugin_name}'") from exc

            self.plugins[plugin_name] = instance
            logger.info("Loaded plugin '%s'", plugin_name)
            return instance

    def unload_plugin(self, plugin_name: str) -> None:
        """Unload the plugin instance identified by ``plugin_name``."""

        with self._lock:
            plugin = self.plugins.pop(plugin_name, None)
            if plugin is None:
                raise LookupError(f"Plugin '{plugin_name}' is not loaded")

            try:
                plugin.reset_state()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.exception("Error while resetting plugin '%s' during unload", plugin_name)

            logger.info("Unloaded plugin '%s'", plugin_name)

    # ------------------------------------------------------------------
    # Execution utilities
    # ------------------------------------------------------------------
    def execute_analysis(self, payload: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        """Execute ``analyze`` on all loaded plugins returning results and errors."""

        results: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        for plugin_name, plugin in list(self.plugins.items()):
            try:
                analysis = plugin.analyze(payload)  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - error path tested
                logger.exception("Plugin '%s' failed during analyze", plugin_name)
                errors[plugin_name] = str(exc)
            else:
                results[plugin_name] = analysis

        return {"results": results, "errors": errors}

    def get_plugin_health(self) -> Dict[str, Dict[str, Any]]:
        """Return health information for registered plugins."""

        health: Dict[str, Dict[str, Any]] = {}
        available_operators = registry.available_operator_identifiers()

        with self._lock:
            for plugin_name, registration in self.plugin_registry.items():
                instance = self.plugins.get(plugin_name)
                operators = registration.operators
                missing = tuple(op for op in operators if op not in available_operators)

                info: Dict[str, Any] = {
                    "module": registration.module_name,
                    "loaded": instance is not None,
                    "required_operators": operators,
                    "missing_operators": missing,
                }

                if instance is not None:
                    info.update(
                        {
                            "identifier": getattr(instance, "identifier", plugin_name),
                            "display_name": getattr(instance, "display_name", instance.__class__.__name__),
                            "version": getattr(instance, "version", "unknown"),
                            "nu_f_nodes": len(getattr(instance, "nu_f", {})),
                            "coherence_index": getattr(instance, "coherence_index", 0.0),
                        }
                    )

                health[plugin_name] = info

        return health

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_plugin_dependencies(self, plugin_cls: type[TNFRPlugin]) -> tuple[str, ...]:
        """Return the operator requirements for ``plugin_cls`` ensuring availability."""

        try:
            operators = registry.get_plugin_operator_requirements(plugin_cls)
        except LookupError as exc:
            raise RuntimeError(
                f"Plugin '{plugin_cls.__name__}' has no registered operator metadata"
            ) from exc

        available = registry.available_operator_identifiers()
        missing = [operator for operator in operators if operator not in available]
        if missing:
            raise RuntimeError(
                f"Plugin '{plugin_cls.__name__}' requires unknown operators: {missing}"
            )

        return operators
