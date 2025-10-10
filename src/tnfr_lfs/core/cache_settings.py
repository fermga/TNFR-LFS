"""Runtime cache configuration models."""

from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_RECOMMENDER_CACHE_SIZE = 32
DEFAULT_DYNAMIC_CACHE_SIZE = 256
LEGACY_TELEMETRY_CACHE_KEY = "telemetry_cache_size"


def resolve_recommender_cache_size(cache_size: int | None) -> int:
    """Normalise cache sizes used by recommendation helpers."""

    if cache_size is None:
        return DEFAULT_RECOMMENDER_CACHE_SIZE
    return max(0, int(cache_size))


@dataclass(frozen=True, slots=True)
class CacheOptions:
    """Immutable cache configuration parsed from TOML sources."""

    enable_delta_cache: bool = True
    nu_f_cache_size: int = DEFAULT_DYNAMIC_CACHE_SIZE
    telemetry_cache_size: int = DEFAULT_DYNAMIC_CACHE_SIZE
    recommender_cache_size: int = DEFAULT_RECOMMENDER_CACHE_SIZE

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | None = None
    ) -> "CacheOptions":
        """Coerce raw configuration mappings into cache options.

        The helper accepts mappings produced either by the new
        ``[performance]`` table or legacy ``[cache]`` payloads and normalises
        them into a :class:`CacheOptions` instance.  Unknown values fall back to
        the existing defaults, and booleans/integers are coerced to sensible
        values mirroring the previous parsing logic.
        """

        def _as_mapping(value: Any) -> Mapping[str, Any]:
            if isinstance(value, ABCMapping):
                return value
            return {}

        def _coerce_bool(value: Any, fallback: bool) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes", "on"}:
                    return True
                if lowered in {"0", "false", "no", "off"}:
                    return False
            return fallback

        def _coerce_int(value: Any, fallback: int) -> int:
            try:
                numeric = int(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return fallback
            if numeric < 0:
                return 0
            return numeric

        performance_cfg = _as_mapping(config.get("performance")) if config else {}
        legacy_cache_cfg = _as_mapping(config.get("cache")) if config else {}
        legacy_telemetry_cfg = _as_mapping(legacy_cache_cfg.get("telemetry"))

        # Coerce cache_enabled first using [performance] overrides when
        # available, falling back to legacy hints and ultimately to True.
        legacy_enabled = _coerce_bool(
            legacy_cache_cfg.get("cache_enabled"),
            _coerce_bool(legacy_cache_cfg.get("enable_delta_cache"), True),
        )
        enabled = _coerce_bool(
            performance_cfg.get("cache_enabled"),
            _coerce_bool(performance_cfg.get("enable_delta_cache"), legacy_enabled),
        )

        # Resolve the shared cache size honouring the same precedence rules as
        # the old CLI migration helpers.  ``max_cache_size`` takes priority,
        # falling back to other known keys and finally the dynamic default.
        shared_size = DEFAULT_DYNAMIC_CACHE_SIZE

        legacy_shared_candidate: Any = legacy_cache_cfg.get("max_cache_size")
        if legacy_shared_candidate is None:
            for fallback_key in ("nu_f_cache_size", "recommender_cache_size"):
                if fallback_key in legacy_cache_cfg:
                    legacy_shared_candidate = legacy_cache_cfg.get(fallback_key)
                    break
        if legacy_shared_candidate is None and legacy_telemetry_cfg:
            legacy_shared_candidate = legacy_telemetry_cfg.get(LEGACY_TELEMETRY_CACHE_KEY)

        if legacy_shared_candidate is not None:
            shared_size = _coerce_int(legacy_shared_candidate, shared_size)

        shared_candidate: Any = performance_cfg.get("max_cache_size")
        if shared_candidate is None:
            for fallback_key in (
                "nu_f_cache_size",
                "recommender_cache_size",
                LEGACY_TELEMETRY_CACHE_KEY,
            ):
                if fallback_key in performance_cfg:
                    shared_candidate = performance_cfg.get(fallback_key)
                    break

        cache_size = _coerce_int(shared_candidate, shared_size)

        nu_f_size = cache_size
        nu_f_override = performance_cfg.get("nu_f_cache_size")
        if nu_f_override is not None:
            nu_f_size = _coerce_int(nu_f_override, nu_f_size)
        else:
            legacy_nu_f_override = legacy_cache_cfg.get("nu_f_cache_size")
            if legacy_nu_f_override is not None:
                nu_f_size = _coerce_int(legacy_nu_f_override, nu_f_size)

        recommender_size = cache_size
        recommender_override = performance_cfg.get("recommender_cache_size")
        if recommender_override is not None:
            recommender_size = _coerce_int(recommender_override, recommender_size)
        else:
            legacy_recommender_override = legacy_cache_cfg.get("recommender_cache_size")
            if legacy_recommender_override is not None:
                recommender_size = _coerce_int(legacy_recommender_override, recommender_size)

        telemetry_size = cache_size
        telemetry_override = performance_cfg.get(LEGACY_TELEMETRY_CACHE_KEY)
        if telemetry_override is None:
            telemetry_override = legacy_cache_cfg.get(LEGACY_TELEMETRY_CACHE_KEY)
        if telemetry_override is None and legacy_telemetry_cfg:
            telemetry_override = legacy_telemetry_cfg.get(LEGACY_TELEMETRY_CACHE_KEY)
        if telemetry_override is not None:
            telemetry_size = _coerce_int(telemetry_override, telemetry_size)

        if not enabled:
            nu_f_size = telemetry_size = recommender_size = 0

        options = cls(
            enable_delta_cache=enabled,
            nu_f_cache_size=nu_f_size,
            telemetry_cache_size=telemetry_size,
            recommender_cache_size=recommender_size,
        )
        return options.with_defaults()

    def with_defaults(self) -> "CacheOptions":
        """Return an instance with normalised field values."""

        return CacheOptions(
            enable_delta_cache=bool(self.enable_delta_cache),
            nu_f_cache_size=max(0, int(self.nu_f_cache_size)),
            telemetry_cache_size=max(0, int(self.telemetry_cache_size)),
            recommender_cache_size=max(0, int(self.recommender_cache_size)),
        )

    @property
    def cache_enabled(self) -> bool:
        """Backward compatible alias describing whether caches are active."""

        return self.enable_delta_cache and self.nu_f_cache_size > 0

    @property
    def max_cache_size(self) -> int:
        """Largest cache size configured for runtime helpers."""

        return max(self.nu_f_cache_size, self.telemetry_cache_size, self.recommender_cache_size)


__all__ = [
    "CacheOptions",
    "DEFAULT_DYNAMIC_CACHE_SIZE",
    "DEFAULT_RECOMMENDER_CACHE_SIZE",
    "resolve_recommender_cache_size",
]
