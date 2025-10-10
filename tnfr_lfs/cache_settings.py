"""Runtime cache configuration models."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_RECOMMENDER_CACHE_SIZE = 32


@dataclass(frozen=True, slots=True)
class CacheOptions:
    """Immutable cache configuration parsed from TOML sources."""

    enable_delta_cache: bool = True
    nu_f_cache_size: int = 256
    telemetry_cache_size: int = 1
    recommender_cache_size: int = DEFAULT_RECOMMENDER_CACHE_SIZE

    def with_defaults(self) -> "CacheOptions":
        """Return an instance with normalised field values."""

        return CacheOptions(
            enable_delta_cache=bool(self.enable_delta_cache),
            nu_f_cache_size=max(0, int(self.nu_f_cache_size)),
            telemetry_cache_size=max(0, int(self.telemetry_cache_size)),
            recommender_cache_size=max(0, int(self.recommender_cache_size)),
        )


__all__ = ["CacheOptions", "DEFAULT_RECOMMENDER_CACHE_SIZE"]
