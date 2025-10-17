"""Shared helpers for operator modules."""

from __future__ import annotations

from importlib.util import find_spec


def is_module_available(module_name: str) -> bool:
    """Return ``True`` when ``module_name`` can be imported."""

    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


_HAS_JAX = is_module_available("jax.numpy")

if _HAS_JAX:  # pragma: no cover - exercised only when JAX is installed
    import jax.numpy as jnp  # type: ignore[import-not-found]
else:  # pragma: no cover - exercised when JAX is unavailable
    jnp = None


__all__ = ["_HAS_JAX", "jnp", "is_module_available"]

