"""Primitives shared between the equations and operators layers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from importlib.util import find_spec
import threading
from typing import (
    Any,
    Callable,
    Generic,
    Hashable,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    runtime_checkable,
)


__all__ = [
    "DEFAULT_RECOMMENDER_CACHE_SIZE",
    "DEFAULT_DYNAMIC_CACHE_SIZE",
    "LEGACY_TELEMETRY_CACHE_KEY",
    "CacheOptions",
    "resolve_recommender_cache_size",
    "LRUCache",
    "cached_delta_nfr_map",
    "invalidate_delta_record",
    "clear_delta_cache",
    "cached_dynamic_multipliers",
    "invalidate_dynamic_record",
    "clear_dynamic_cache",
    "configure_cache",
    "configure_cache_from_options",
    "should_use_delta_cache",
    "should_use_dynamic_cache",
    "delta_cache_enabled",
    "dynamic_cache_enabled",
    "is_module_available",
    "_HAS_JAX",
    "jnp",
    "SupportsTelemetrySample",
    "SupportsEPINode",
    "SupportsTyresNode",
    "SupportsSuspensionNode",
    "SupportsChassisNode",
    "SupportsBrakesNode",
    "SupportsTransmissionNode",
    "SupportsTrackNode",
    "SupportsDriverNode",
    "SupportsEPIBundle",
    "SupportsContextRecord",
    "SupportsContextBundle",
    "SupportsContextTyres",
    "SupportsContextChassis",
    "SupportsContextTransmission",
    "SupportsGoal",
    "SupportsMicrosector",
]


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


DEFAULT_RECOMMENDER_CACHE_SIZE = 32
DEFAULT_DYNAMIC_CACHE_SIZE = 256
LEGACY_TELEMETRY_CACHE_KEY = "telemetry_cache_size"


def resolve_recommender_cache_size(cache_size: int | None) -> int:
    """Normalise cache sizes used by recommendation helpers."""

    if cache_size is None:
        return DEFAULT_RECOMMENDER_CACHE_SIZE
    return max(0, int(cache_size))

_T = TypeVar("_T")
_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")


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
        """Coerce raw configuration mappings into cache options."""

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

        legacy_enabled = _coerce_bool(
            legacy_cache_cfg.get("cache_enabled"),
            _coerce_bool(legacy_cache_cfg.get("enable_delta_cache"), True),
        )
        enabled = _coerce_bool(
            performance_cfg.get("cache_enabled"),
            _coerce_bool(performance_cfg.get("enable_delta_cache"), legacy_enabled),
        )

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

    def to_performance_config(self) -> dict[str, int | bool]:
        """Serialise the cache options into a ``[performance]`` mapping."""

        normalised = self.with_defaults()
        return {
            "cache_enabled": normalised.enable_delta_cache,
            "max_cache_size": normalised.max_cache_size,
            "nu_f_cache_size": normalised.nu_f_cache_size,
            "telemetry_cache_size": normalised.telemetry_cache_size,
            "recommender_cache_size": normalised.recommender_cache_size,
        }

    @property
    def cache_enabled(self) -> bool:
        """Backward compatible alias describing whether caches are active."""

        return self.enable_delta_cache and self.nu_f_cache_size > 0

    @property
    def max_cache_size(self) -> int:
        """Largest cache size configured for runtime helpers."""

        return max(
            self.nu_f_cache_size,
            self.telemetry_cache_size,
            self.recommender_cache_size,
        )


class _LRUCache(Generic[_K, _V]):
    """Minimal LRU cache with targeted invalidation support."""

    __slots__ = ("_maxsize", "_data", "_lock")

    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self._maxsize = int(maxsize)
        self._data: "OrderedDict[_K, _V]" = OrderedDict()
        self._lock = threading.RLock()

    def get_or_create(self, key: _K, factory: Callable[[], _V]) -> _V:
        with self._lock:
            try:
                value = self._data.pop(key)
            except KeyError:
                value = factory()
            else:
                self._data[key] = value
                return value
            self._data[key] = value
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)
            return value

    def invalidate(self, predicate: Callable[[_K], bool]) -> None:
        with self._lock:
            for candidate in list(self._data.keys()):
                if predicate(candidate):
                    self._data.pop(candidate, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class LRUCache(Generic[_K, _V]):
    """Public wrapper around :class:`_LRUCache` supporting ``maxsize >= 0``."""

    __slots__ = ("_maxsize", "_cache")

    def __init__(self, *, maxsize: int) -> None:
        size = int(maxsize)
        if size < 0:
            raise ValueError("maxsize must be >= 0")
        self._maxsize = size
        self._cache: _LRUCache[_K, _V] | None = None
        if size > 0:
            self._cache = _LRUCache(maxsize=size)

    @property
    def maxsize(self) -> int:
        """Return the configured capacity for the cache."""

        return self._maxsize

    def get_or_create(self, key: _K, factory: Callable[[], _V]) -> _V:
        """Return cached value for ``key`` or materialise it via ``factory``."""

        cache = self._cache
        if cache is None:
            return factory()
        return cache.get_or_create(key, factory)

    def invalidate(self, predicate: Callable[[_K], bool]) -> None:
        """Remove cached entries matching ``predicate`` when caching is active."""

        cache = self._cache
        if cache is None:
            return
        cache.invalidate(predicate)

    def clear(self) -> None:
        """Drop all cached entries when caching is active."""

        cache = self._cache
        if cache is None:
            return
        cache.clear()


_DEFAULT_DELTA_CACHE_SIZE = 1024
_ENABLE_DELTA_CACHE = True
_DELTA_NFR_CACHE: _LRUCache[Tuple[int, int | None], Mapping[str, float]] | None = _LRUCache(
    maxsize=_DEFAULT_DELTA_CACHE_SIZE
)
_DYNAMIC_MULTIPLIER_CACHE: _LRUCache[
    Tuple[str | None, int, float | None], Tuple[Mapping[str, float], float]
] | None = _LRUCache(maxsize=DEFAULT_DYNAMIC_CACHE_SIZE)


def cached_delta_nfr_map(record: _T, factory: Callable[[], Mapping[str, float]]) -> Mapping[str, float]:
    """Return a cached ΔNFR-by-node map for *record*."""

    if not delta_cache_enabled():
        return dict(factory())

    reference = getattr(record, "reference", None)
    reference_id: int | None = id(reference) if reference is not None else None
    key = (id(record), reference_id)
    assert _DELTA_NFR_CACHE is not None
    value = _DELTA_NFR_CACHE.get_or_create(key, lambda: dict(factory()))
    return dict(value)


def invalidate_delta_record(record: _T) -> None:
    """Invalidate cached ΔNFR entries that reference *record*."""

    if _DELTA_NFR_CACHE is None:
        return
    record_id = id(record)

    def _should_remove(key: Tuple[int, int | None]) -> bool:
        current_id, reference_id = key
        return current_id == record_id or reference_id == record_id

    _DELTA_NFR_CACHE.invalidate(_should_remove)


def clear_delta_cache() -> None:
    """Clear all cached ΔNFR entries."""

    if _DELTA_NFR_CACHE is not None:
        _DELTA_NFR_CACHE.clear()


def cached_dynamic_multipliers(
    car_model: str | None,
    history: Sequence[_T],
    factory: Callable[[], Tuple[Mapping[str, float], float]],
) -> Tuple[Mapping[str, float], float]:
    """Return cached dynamic ν_f multipliers for the supplied *history*."""

    timestamp = getattr(history[-1], "timestamp", None) if history else None
    key = (car_model, len(history), timestamp)
    cache = _DYNAMIC_MULTIPLIER_CACHE
    if cache is None:
        return _prepare_dynamic_entry(factory)
    value = cache.get_or_create(
        key,
        lambda: _prepare_dynamic_entry(factory),
    )
    multipliers, frequency = value
    return dict(multipliers), frequency


def _prepare_dynamic_entry(
    factory: Callable[[], Tuple[Mapping[str, float], float]]
) -> Tuple[Mapping[str, float], float]:
    multipliers, frequency = factory()
    return dict(multipliers), frequency


def invalidate_dynamic_record(record: _T) -> None:
    """Drop cached ν_f multipliers that are older than *record*."""

    if _DYNAMIC_MULTIPLIER_CACHE is None:
        return
    threshold = getattr(record, "timestamp", None)

    def _should_remove(key: Tuple[str | None, int, float | None]) -> bool:
        _, _, cached_timestamp = key
        if cached_timestamp is None or threshold is None:
            return False
        return cached_timestamp <= threshold

    _DYNAMIC_MULTIPLIER_CACHE.invalidate(_should_remove)


def clear_dynamic_cache() -> None:
    """Clear the dynamic ν_f multiplier cache."""

    if _DYNAMIC_MULTIPLIER_CACHE is not None:
        _DYNAMIC_MULTIPLIER_CACHE.clear()


def configure_cache(*, enable_delta_cache: bool | None = None, nu_f_cache_size: int | None = None) -> None:
    """Configure cache toggles and capacities used by EPI helpers."""

    global _ENABLE_DELTA_CACHE, _DELTA_NFR_CACHE, _DYNAMIC_MULTIPLIER_CACHE

    if enable_delta_cache is not None:
        enable = bool(enable_delta_cache)
        cache = _DELTA_NFR_CACHE
        if not enable:
            if cache is not None:
                with cache._lock:
                    cache.clear()
                    _DELTA_NFR_CACHE = None
        elif cache is None:
            _DELTA_NFR_CACHE = _LRUCache(maxsize=_DEFAULT_DELTA_CACHE_SIZE)
        _ENABLE_DELTA_CACHE = enable

    if nu_f_cache_size is not None:
        size = int(nu_f_cache_size)
        if size <= 0:
            cache = _DYNAMIC_MULTIPLIER_CACHE
            if cache is not None:
                with cache._lock:
                    cache.clear()
                    _DYNAMIC_MULTIPLIER_CACHE = None
        else:
            cache = _DYNAMIC_MULTIPLIER_CACHE
            if cache is None or cache._maxsize != size:
                _DYNAMIC_MULTIPLIER_CACHE = _LRUCache(maxsize=size)
    elif _DYNAMIC_MULTIPLIER_CACHE is None:
        _DYNAMIC_MULTIPLIER_CACHE = _LRUCache(maxsize=DEFAULT_DYNAMIC_CACHE_SIZE)


def configure_cache_from_options(options: CacheOptions) -> None:
    """Normalise and apply cache settings declared via :class:`CacheOptions`."""

    normalised = options.with_defaults()
    configure_cache(
        enable_delta_cache=normalised.enable_delta_cache,
        nu_f_cache_size=normalised.nu_f_cache_size,
    )


def should_use_delta_cache(cache_options: CacheOptions | None) -> bool:
    """Return ``True`` when ΔNFR caching should be used."""

    if cache_options is not None:
        return bool(cache_options.enable_delta_cache)
    return delta_cache_enabled()


def should_use_dynamic_cache(cache_options: CacheOptions | None) -> bool:
    """Return ``True`` when ν_f multiplier caching should be used."""

    if cache_options is not None:
        return cache_options.nu_f_cache_size > 0
    return dynamic_cache_enabled()


def delta_cache_enabled() -> bool:
    """Return ``True`` when ΔNFR caching is active."""

    return _ENABLE_DELTA_CACHE and _DELTA_NFR_CACHE is not None


def dynamic_cache_enabled() -> bool:
    """Return ``True`` when ν_f multipliers caching is active."""

    return _DYNAMIC_MULTIPLIER_CACHE is not None


@runtime_checkable
class SupportsTelemetrySample(Protocol):
    """Telemetry payload exposing analytics fields and optional metadata."""

    timestamp: float
    structural_timestamp: float | None

    lateral_accel: float
    longitudinal_accel: float
    vertical_load: float
    vertical_load_front: float
    vertical_load_rear: float

    speed: float
    yaw: float
    pitch: float
    roll: float
    yaw_rate: float
    steer: float

    suspension_velocity_front: float
    suspension_velocity_rear: float
    suspension_travel_front: float
    suspension_travel_rear: float

    brake_pressure: float
    locking: float
    throttle: float
    gear: int

    nfr: float
    si: float
    slip_ratio: float
    slip_angle: float

    mu_eff_front: float
    mu_eff_rear: float
    mu_eff_front_lateral: float
    mu_eff_front_longitudinal: float
    mu_eff_rear_lateral: float
    mu_eff_rear_longitudinal: float

    slip_ratio_fl: float
    slip_ratio_fr: float
    slip_ratio_rl: float
    slip_ratio_rr: float

    slip_angle_fl: float
    slip_angle_fr: float
    slip_angle_rl: float
    slip_angle_rr: float

    wheel_load_fl: float
    wheel_load_fr: float
    wheel_load_rl: float
    wheel_load_rr: float

    wheel_lateral_force_fl: float
    wheel_lateral_force_fr: float
    wheel_lateral_force_rl: float
    wheel_lateral_force_rr: float

    wheel_longitudinal_force_fl: float
    wheel_longitudinal_force_fr: float
    wheel_longitudinal_force_rl: float
    wheel_longitudinal_force_rr: float

    tyre_temp_fl: float
    tyre_temp_fr: float
    tyre_temp_rl: float
    tyre_temp_rr: float
    tyre_temp_fl_inner: float
    tyre_temp_fr_inner: float
    tyre_temp_rl_inner: float
    tyre_temp_rr_inner: float
    tyre_temp_fl_middle: float
    tyre_temp_fr_middle: float
    tyre_temp_rl_middle: float
    tyre_temp_rr_middle: float
    tyre_temp_fl_outer: float
    tyre_temp_fr_outer: float
    tyre_temp_rl_outer: float
    tyre_temp_rr_outer: float

    tyre_pressure_fl: float
    tyre_pressure_fr: float
    tyre_pressure_rl: float
    tyre_pressure_rr: float

    brake_temp_fl: float
    brake_temp_fr: float
    brake_temp_rl: float
    brake_temp_rr: float

    rpm: float
    line_deviation: float

    reference: "SupportsTelemetrySample" | None
    car_model: str | None
    track_name: str | None
    tyre_compound: str | None


@runtime_checkable
class SupportsEPINode(Protocol):
    """Subsystem node payload used by EPI analytics."""

    delta_nfr: float
    sense_index: float
    nu_f: float
    dEPI_dt: float
    integrated_epi: float


@runtime_checkable
class SupportsTyresNode(SupportsEPINode, Protocol):
    """Tyre subsystem payload consumed by analytics and operators."""

    load: float
    slip_ratio: float
    mu_eff_front: float
    mu_eff_rear: float
    mu_eff_front_lateral: float
    mu_eff_front_longitudinal: float
    mu_eff_rear_lateral: float
    mu_eff_rear_longitudinal: float
    tyre_temp_fl: float
    tyre_temp_fr: float
    tyre_temp_rl: float
    tyre_temp_rr: float
    tyre_temp_fl_inner: float
    tyre_temp_fr_inner: float
    tyre_temp_rl_inner: float
    tyre_temp_rr_inner: float
    tyre_temp_fl_middle: float
    tyre_temp_fr_middle: float
    tyre_temp_rl_middle: float
    tyre_temp_rr_middle: float
    tyre_temp_fl_outer: float
    tyre_temp_fr_outer: float
    tyre_temp_rl_outer: float
    tyre_temp_rr_outer: float
    tyre_pressure_fl: float
    tyre_pressure_fr: float
    tyre_pressure_rl: float
    tyre_pressure_rr: float


@runtime_checkable
class SupportsSuspensionNode(SupportsEPINode, Protocol):
    """Suspension subsystem payload consumed by analytics and operators."""

    travel_front: float
    travel_rear: float
    velocity_front: float
    velocity_rear: float


@runtime_checkable
class SupportsChassisNode(SupportsEPINode, Protocol):
    """Chassis subsystem payload consumed by analytics and operators."""

    yaw: float
    pitch: float
    roll: float
    yaw_rate: float
    lateral_accel: float
    longitudinal_accel: float


@runtime_checkable
class SupportsBrakesNode(SupportsEPINode, Protocol):
    """Brake subsystem payload consumed by analytics and operators."""

    brake_pressure: float
    locking: float
    brake_temp_fl: float
    brake_temp_fr: float
    brake_temp_rl: float
    brake_temp_rr: float
    brake_temp_peak: float
    brake_temp_mean: float


@runtime_checkable
class SupportsTransmissionNode(SupportsEPINode, Protocol):
    """Transmission subsystem payload consumed by analytics and operators."""

    throttle: float
    gear: int
    speed: float
    longitudinal_accel: float
    rpm: float
    line_deviation: float


@runtime_checkable
class SupportsTrackNode(SupportsEPINode, Protocol):
    """Track condition payload consumed by analytics and operators."""

    axle_load_balance: float
    axle_velocity_balance: float
    yaw: float
    lateral_accel: float
    gradient: float


@runtime_checkable
class SupportsDriverNode(SupportsEPINode, Protocol):
    """Driver payload consumed by analytics and operators."""

    steer: float
    throttle: float
    style_index: float


@runtime_checkable
class SupportsEPIBundle(Protocol):
    """Aggregated telemetry insights required by EPI consumers."""

    timestamp: float
    epi: float
    delta_nfr: float
    sense_index: float
    delta_breakdown: Mapping[str, Mapping[str, float]]
    node_evolution: Mapping[str, tuple[float, float]]
    structural_timestamp: float | None
    dEPI_dt: float
    integrated_epi: float
    delta_nfr_proj_longitudinal: float
    delta_nfr_proj_lateral: float
    nu_f_classification: str
    nu_f_category: str
    nu_f_label: str
    nu_f_dominant: float
    coherence_index: float
    ackermann_parallel_index: float

    tyres: SupportsTyresNode
    suspension: SupportsSuspensionNode
    chassis: SupportsChassisNode
    brakes: SupportsBrakesNode
    transmission: SupportsTransmissionNode
    track: SupportsTrackNode
    driver: SupportsDriverNode


@runtime_checkable
class SupportsContextRecord(Protocol):
    """Telemetry-like payload exposing the fields required for context factors."""

    lateral_accel: float
    vertical_load: float
    longitudinal_accel: float


@runtime_checkable
class SupportsContextTyres(Protocol):
    """Tyre subsystem metrics required to derive contextual surface ratios."""

    load: float


@runtime_checkable
class SupportsContextChassis(Protocol):
    """Chassis subsystem metrics required to derive curvature and traffic cues."""

    lateral_accel: float
    longitudinal_accel: float


@runtime_checkable
class SupportsContextTransmission(Protocol):
    """Transmission subsystem metrics used as a fallback for traffic cues."""

    longitudinal_accel: float


@runtime_checkable
class SupportsContextBundle(Protocol):
    """Aggregate bundle exposing the nodes required by contextual helpers."""

    tyres: SupportsContextTyres | None
    chassis: SupportsContextChassis | None
    transmission: SupportsContextTransmission | None


@runtime_checkable
class SupportsGoal(Protocol):
    """Goal specification produced by the segmentation heuristics."""

    phase: str
    archetype: str
    description: str
    target_delta_nfr: float
    target_sense_index: float
    nu_f_target: float
    nu_exc_target: float
    rho_target: float
    target_phase_lag: float
    target_phase_alignment: float
    measured_phase_lag: float
    measured_phase_alignment: float
    target_phase_synchrony: float
    measured_phase_synchrony: float
    slip_lat_window: Tuple[float, float]
    slip_long_window: Tuple[float, float]
    yaw_rate_window: Tuple[float, float]
    dominant_nodes: Tuple[str, ...]
    target_delta_nfr_long: float
    target_delta_nfr_lat: float
    delta_axis_weights: Mapping[str, float]
    archetype_delta_nfr_long_target: float
    archetype_delta_nfr_lat_target: float
    archetype_nu_f_target: float
    archetype_si_phi_target: float
    detune_ratio_weights: Mapping[str, float]
    track_gradient: float


@runtime_checkable
class SupportsMicrosector(Protocol):
    """Microsector abstraction consumed by operator orchestration."""

    index: int
    start_time: float
    end_time: float
    curvature: float
    brake_event: bool
    support_event: bool
    delta_nfr_signature: float
    goals: Sequence[SupportsGoal]
    phase_boundaries: Mapping[str, Tuple[int, int]]
    phase_samples: Mapping[str, Tuple[int, ...]]
    active_phase: str
    dominant_nodes: Mapping[str, Tuple[str, ...]]
    phase_weights: Mapping[str, Mapping[str, float] | float]
    grip_rel: float
    phase_lag: Mapping[str, float]
    phase_alignment: Mapping[str, float]
    phase_synchrony: Mapping[str, float]
    phase_motor_latency: Mapping[str, float]
    motor_latency_ms: float
    filtered_measures: Mapping[str, object]
    recursivity_trace: Sequence[Mapping[str, float | str | None]]
    last_mutation: Mapping[str, object] | None
    window_occupancy: Mapping[str, Mapping[str, float]]
    delta_nfr_std: float
    nodal_delta_nfr_std: float
    phase_delta_nfr_std: Mapping[str, float]
    phase_nodal_delta_nfr_std: Mapping[str, float]
    delta_nfr_entropy: float
    node_entropy: float
    phase_delta_nfr_entropy: Mapping[str, float]
    phase_node_entropy: Mapping[str, float]
    phase_axis_targets: Mapping[str, Mapping[str, float]]
    phase_axis_weights: Mapping[str, Mapping[str, float]]
    context_factors: Mapping[str, float]
    sample_context_factors: Mapping[int, Mapping[str, float]]
    operator_events: Mapping[str, Sequence[Mapping[str, object]]]
