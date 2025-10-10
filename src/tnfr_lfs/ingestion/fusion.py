"""Fusion utilities for OutSim and OutGauge telemetry."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Dict, List, Mapping, Tuple, cast

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

from .._pack_resources import pack_root
from ..analysis.brake_thermal import (
    BrakeThermalConfig,
    BrakeThermalEstimator,
    merge_brake_config,
)
from ..core.epi import EPIExtractor, EPIBundle, TelemetryRecord
from ..utils.numeric import _safe_float
from .outsim_udp import OutSimPacket, OutSimWheelState
from .outgauge_udp import OutGaugePacket

__all__ = ["TelemetryFusion"]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))



@dataclass(frozen=True)
class _WheelTelemetry:
    """Intermediate representation of per-wheel telemetry values."""

    wheels: tuple[OutSimWheelState, OutSimWheelState, OutSimWheelState, OutSimWheelState]
    slip_ratios: tuple[float, float, float, float]
    slip_angles: tuple[float, float, float, float]
    lateral_forces: tuple[float, float, float, float]
    longitudinal_forces: tuple[float, float, float, float]
    loads: tuple[float, float, float, float]
    deflections: tuple[float, float, float, float]
    data_present: bool


@dataclass(frozen=True)
class FusionCalibration:
    """Parameter set describing how telemetry signals should be scaled."""

    load_scale: float = 1000.0
    load_bias: float = 0.0
    slip_ratio_gain: float = 1.0
    slip_ratio_bias: float = 0.0
    steer_ratio: float = 1.0
    steer_offset: float = 0.0
    wheelbase: float = 2.6
    network_latency: float = 0.06
    tyre_radius: float = 0.33
    mu_lateral_gain: float = 1.0
    mu_longitudinal_gain: float = 1.0
    axle_weight_front: float = 0.5
    axle_weight_rear: float = 0.5
    suspension_window: int = 5


@dataclass
class TelemetryFusion:
    """Combine OutSim and OutGauge packets into :class:`TelemetryRecord` objects."""

    load_scale: float = 1000.0
    extractor: EPIExtractor = field(default_factory=EPIExtractor)
    _last_record: TelemetryRecord | None = field(default=None, init=False, repr=False)
    _calibration_table: Mapping[str, object] = field(default_factory=dict, init=False, repr=False)
    _calibration_cache: Dict[Tuple[str, str], FusionCalibration] = field(
        default_factory=dict, init=False, repr=False
    )
    _vertical_history: List[float] = field(default_factory=list, init=False, repr=False)
    _line_history: List[Tuple[float, float]] = field(default_factory=list, init=False, repr=False)
    _brake_thermal_defaults: BrakeThermalConfig = field(init=False, repr=False)
    _brake_thermal_mode: str = field(init=False, repr=False)
    _brake_thermal_car_profiles: Dict[str, Tuple[BrakeThermalConfig, str | None]] = field(
        default_factory=dict, init=False, repr=False
    )
    _brake_thermal_estimator: BrakeThermalEstimator | None = field(
        default=None, init=False, repr=False
    )
    _brake_thermal_active_config: BrakeThermalConfig | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._calibration_table = self._load_calibration_table()
        self._calibration_cache = {}
        self._vertical_history = []
        self._line_history = []
        self._last_record = None
        (
            self._brake_thermal_defaults,
            self._brake_thermal_mode,
            self._brake_thermal_car_profiles,
        ) = self._load_brake_thermal_profiles()
        self._brake_thermal_estimator = BrakeThermalEstimator(
            self._brake_thermal_defaults
        )
        self._brake_thermal_active_config = self._brake_thermal_defaults

    def reset(self) -> None:
        """Clear the internal telemetry history."""

        self.extractor.reset()
        self._vertical_history.clear()
        self._line_history.clear()
        self._last_record = None
        if self._brake_thermal_estimator is not None:
            self._brake_thermal_estimator.reset()
        self.__dict__.pop("_records", None)

    def fuse(self, outsim: OutSimPacket, outgauge: OutGaugePacket) -> TelemetryRecord:
        """Return a :class:`TelemetryRecord` derived from both UDP sources."""

        timestamp = outsim.time / 1000.0
        previous = self._last_record
        dt = timestamp - previous.timestamp if previous else 0.0

        calibration = self._select_calibration(outgauge)
        self._append_vertical_accel(outsim.accel_z, calibration)

        wheel_data = self._preprocess_wheels(outsim)
        aggregated_slip_ratio, aggregated_slip_angle = self._aggregate_wheel_slip(
            wheel_data
        )

        inputs = getattr(outsim, "inputs", None)
        throttle = _clamp(
            _safe_float(getattr(inputs, "throttle", outgauge.throttle)), 0.0, 1.0
        )
        brake_input = _clamp(
            _safe_float(getattr(inputs, "brake", outgauge.brake)), 0.0, 1.0
        )
        clutch_input = _clamp(
            _safe_float(getattr(inputs, "clutch", outgauge.clutch)), 0.0, 1.0
        )
        handbrake_input = _clamp(_safe_float(getattr(inputs, "handbrake", 0.0)), 0.0, 1.0)
        steer_input = _clamp(_safe_float(getattr(inputs, "steer", 0.0)), -1.0, 1.0)

        vertical_load = self._compute_vertical_load(outsim, calibration)
        speed = self._compute_speed(outsim, outgauge)
        yaw = self._normalise_heading(outsim.heading)
        yaw_rate = self._compute_yaw_rate(timestamp, yaw, outsim, previous, dt)
        if math.isfinite(aggregated_slip_ratio):
            slip_ratio = aggregated_slip_ratio
        else:
            slip_ratio = self._compute_slip_ratio(outsim, outgauge, calibration)
        if math.isfinite(aggregated_slip_angle):
            slip_angle = aggregated_slip_angle
        else:
            slip_angle = self._compute_slip_angle(
                yaw_rate, speed, slip_ratio, outsim, calibration, previous, dt
            )
        steer = self._compute_steer(yaw_rate, speed, calibration)
        front_share, rear_share = self._estimate_axle_distribution(
            outsim, speed, calibration
        )
        front_load = vertical_load * front_share
        rear_load = vertical_load * rear_share

        if all(math.isfinite(load) for load in wheel_data.loads):
            total_wheel_load = sum(wheel_data.loads)
        else:
            total_wheel_load = math.nan
        if math.isfinite(total_wheel_load) and total_wheel_load > 1e-6:
            front_load = wheel_data.loads[0] + wheel_data.loads[1]
            rear_load = wheel_data.loads[2] + wheel_data.loads[3]
            vertical_load = front_load + rear_load
            if vertical_load > 1e-6:
                front_share = _clamp(front_load / vertical_load, 0.0, 1.0)
                rear_share = 1.0 - front_share
            else:
                front_share = rear_share = 0.5

        (
            travel_front,
            travel_rear,
            vel_front,
            vel_rear,
        ) = self._calculate_suspension(
            wheel_data,
            front_share,
            rear_share,
            previous,
            dt,
            calibration,
        )

        mu_front, mu_front_lat, mu_front_long = self._compute_mu_eff(
            outsim.accel_y, outsim.accel_x, front_share, calibration
        )
        mu_rear, mu_rear_lat, mu_rear_long = self._compute_mu_eff(
            outsim.accel_y, outsim.accel_x, rear_share, calibration
        )

        deceleration = max(0.0, -outsim.accel_x)
        (
            tyre_temp_layers,
            tyre_temps,
            tyre_pressures,
            brake_temps,
        ) = self._estimate_thermal(
            outgauge,
            previous,
            dt,
            speed,
            deceleration,
            brake_input,
            wheel_data.loads,
        )
        line_deviation = self._compute_line_deviation(outsim)

        record = self._construct_record(
            timestamp=timestamp,
            outsim=outsim,
            outgauge=outgauge,
            wheel_data=wheel_data,
            slip_ratio=slip_ratio,
            slip_angle=slip_angle,
            throttle=throttle,
            brake_input=brake_input,
            clutch_input=clutch_input,
            handbrake_input=handbrake_input,
            steer_input=steer_input,
            speed=speed,
            yaw=yaw,
            yaw_rate=yaw_rate,
            steer=steer,
            vertical_load=vertical_load,
            front_share=front_share,
            rear_share=rear_share,
            front_load=front_load,
            rear_load=rear_load,
            mu_front=mu_front,
            mu_front_lat=mu_front_lat,
            mu_front_long=mu_front_long,
            mu_rear=mu_rear,
            mu_rear_lat=mu_rear_lat,
            mu_rear_long=mu_rear_long,
            suspension_travel_front=travel_front,
            suspension_travel_rear=travel_rear,
            suspension_velocity_front=vel_front,
            suspension_velocity_rear=vel_rear,
            tyre_temp_layers=tyre_temp_layers,
            tyre_temps=tyre_temps,
            tyre_pressures=tyre_pressures,
            brake_temps=brake_temps,
            line_deviation=line_deviation,
        )
        self._last_record = record
        self.__dict__["_records"] = [record]
        return record

    def fuse_to_bundle(self, outsim: OutSimPacket, outgauge: OutGaugePacket) -> EPIBundle:
        """Return an :class:`EPIBundle` for the latest fused sample."""

        record = self.fuse(outsim, outgauge)
        try:
            return self.extractor.update(
                record,
                car_model=getattr(record, "car_model", None) or outgauge.car,
                track_name=getattr(record, "track_name", None) or outgauge.track,
                player_name=outgauge.player_name,
            )
        finally:
            self.__dict__.pop("_records", None)

    # ------------------------------------------------------------------
    # Fusion orchestration helpers
    # ------------------------------------------------------------------
    def _preprocess_wheels(self, outsim: OutSimPacket) -> _WheelTelemetry:
        wheels_raw = tuple(getattr(outsim, "wheels", ()))
        if len(wheels_raw) < 4:
            wheels_raw = wheels_raw + tuple(
                OutSimWheelState() for _ in range(4 - len(wheels_raw))
            )
        wheels = wheels_raw[:4]

        wheel_slip_ratios = []
        wheel_slip_angles = []
        wheel_lateral_forces = []
        wheel_longitudinal_forces = []
        wheel_loads = []
        wheel_deflections = []

        for wheel in wheels:
            if wheel.decoded:
                slip_ratio = _safe_float(wheel.slip_ratio, default=math.nan)
                if math.isfinite(slip_ratio):
                    slip_ratio = _clamp(slip_ratio, -1.0, 1.0)
                slip_angle = _safe_float(wheel.slip_angle, default=math.nan)
                lateral_force = _safe_float(wheel.lateral_force, default=math.nan)
                longitudinal_force = _safe_float(
                    wheel.longitudinal_force, default=math.nan
                )
                load = _safe_float(wheel.load, default=math.nan)
                if math.isfinite(load):
                    load = max(0.0, load)
                deflection = _safe_float(
                    wheel.suspension_deflection, default=math.nan
                )
            else:
                slip_ratio = math.nan
                slip_angle = math.nan
                lateral_force = math.nan
                longitudinal_force = math.nan
                load = math.nan
                deflection = math.nan

            wheel_slip_ratios.append(slip_ratio)
            wheel_slip_angles.append(slip_angle)
            wheel_lateral_forces.append(lateral_force)
            wheel_longitudinal_forces.append(longitudinal_force)
            wheel_loads.append(load)
            wheel_deflections.append(deflection)

        wheel_slip_ratios = tuple(wheel_slip_ratios)
        wheel_slip_angles = tuple(wheel_slip_angles)
        wheel_lateral_forces = tuple(wheel_lateral_forces)
        wheel_longitudinal_forces = tuple(wheel_longitudinal_forces)
        wheel_loads = tuple(wheel_loads)
        wheel_deflections = tuple(wheel_deflections)
        data_present = any(math.isfinite(deflection) for deflection in wheel_deflections)

        return _WheelTelemetry(
            wheels=wheels,
            slip_ratios=wheel_slip_ratios,
            slip_angles=wheel_slip_angles,
            lateral_forces=wheel_lateral_forces,
            longitudinal_forces=wheel_longitudinal_forces,
            loads=wheel_loads,
            deflections=wheel_deflections,
            data_present=data_present,
        )

    def _aggregate_wheel_slip(self, wheel_data: _WheelTelemetry) -> tuple[float, float]:
        angle_weight_sum = 0.0
        angle_total_weight = 0.0
        angle_sum = 0.0
        angle_count = 0
        for angle, load in zip(wheel_data.slip_angles, wheel_data.loads):
            if not math.isfinite(angle):
                continue
            angle_sum += angle
            angle_count += 1
            if math.isfinite(load) and load > 1e-3:
                weight = load
            else:
                weight = 1.0
            angle_weight_sum += angle * weight
            angle_total_weight += weight
        if angle_count:
            if angle_total_weight > 1e-6:
                aggregated_slip_angle = angle_weight_sum / angle_total_weight
            else:
                aggregated_slip_angle = angle_sum / angle_count
        else:
            aggregated_slip_angle = math.nan

        finite_slip_ratios = [
            ratio for ratio in wheel_data.slip_ratios if math.isfinite(ratio)
        ]
        if finite_slip_ratios:
            aggregated_slip_ratio = _clamp(
                sum(finite_slip_ratios) / len(finite_slip_ratios), -1.0, 1.0
            )
        else:
            aggregated_slip_ratio = math.nan

        return aggregated_slip_ratio, aggregated_slip_angle

    def _calculate_suspension(
        self,
        wheel_data: _WheelTelemetry,
        front_share: float,
        rear_share: float,
        previous: TelemetryRecord | None,
        dt: float,
        calibration: FusionCalibration,
    ) -> tuple[float, float, float, float]:
        def _mean_deflection(indices: tuple[int, int]) -> float:
            values = [wheel_data.deflections[index] for index in indices]
            finite = [value for value in values if math.isfinite(value)]
            if not finite:
                return math.nan
            return sum(finite) / len(finite)

        def _resolve_velocity(travel: float, previous_value: float | None) -> float:
            if not math.isfinite(travel):
                return math.nan
            if previous_value is None or not math.isfinite(previous_value) or dt <= 1e-6:
                return 0.0
            return _clamp((travel - previous_value) / dt, -5.0, 5.0)

        if wheel_data.data_present:
            travel_front = _mean_deflection((0, 1))
            travel_rear = _mean_deflection((2, 3))
            previous_front = previous.suspension_travel_front if previous else None
            previous_rear = previous.suspension_travel_rear if previous else None
            vel_front = _resolve_velocity(travel_front, previous_front)
            vel_rear = _resolve_velocity(travel_rear, previous_rear)
        else:
            travel_front, vel_front = self._compute_suspension_velocity(
                "front",
                front_share,
                previous.suspension_travel_front if previous else None,
                dt,
                calibration,
            )
            travel_rear, vel_rear = self._compute_suspension_velocity(
                "rear",
                rear_share,
                previous.suspension_travel_rear if previous else None,
                dt,
                calibration,
            )

        return travel_front, travel_rear, vel_front, vel_rear

    def _estimate_thermal(
        self,
        outgauge: OutGaugePacket,
        previous: TelemetryRecord | None,
        dt: float,
        speed: float,
        deceleration: float,
        brake_input: float,
        wheel_loads: tuple[float, float, float, float],
    ) -> tuple[
        tuple[tuple[float, float, float, float], ...],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]:
        tyre_temp_layers = self._resolve_wheel_temperature_layers(outgauge, previous)
        tyre_temps = self._resolve_wheel_temperatures(
            outgauge, previous, layers=tyre_temp_layers
        )
        tyre_pressures = self._resolve_wheel_pressures(outgauge, previous)
        brake_temps = self._resolve_brake_temperatures(
            dt,
            speed,
            deceleration,
            brake_input,
            wheel_loads,
            outgauge,
            previous,
        )
        return tyre_temp_layers, tyre_temps, tyre_pressures, brake_temps

    def _construct_record(
        self,
        *,
        timestamp: float,
        outsim: OutSimPacket,
        outgauge: OutGaugePacket,
        wheel_data: _WheelTelemetry,
        slip_ratio: float,
        slip_angle: float,
        throttle: float,
        brake_input: float,
        clutch_input: float,
        handbrake_input: float,
        steer_input: float,
        speed: float,
        yaw: float,
        yaw_rate: float,
        steer: float,
        vertical_load: float,
        front_share: float,
        rear_share: float,
        front_load: float,
        rear_load: float,
        mu_front: float,
        mu_front_lat: float,
        mu_front_long: float,
        mu_rear: float,
        mu_rear_lat: float,
        mu_rear_long: float,
        suspension_travel_front: float,
        suspension_travel_rear: float,
        suspension_velocity_front: float,
        suspension_velocity_rear: float,
        tyre_temp_layers: tuple[tuple[float, float, float, float], ...],
        tyre_temps: tuple[float, float, float, float],
        tyre_pressures: tuple[float, float, float, float],
        brake_temps: tuple[float, float, float, float],
        line_deviation: float,
    ) -> TelemetryRecord:
        return TelemetryRecord(
            timestamp=timestamp,
            vertical_load=vertical_load,
            slip_ratio=slip_ratio,
            lateral_accel=outsim.accel_y,
            longitudinal_accel=outsim.accel_x,
            yaw=yaw,
            pitch=outsim.pitch,
            roll=outsim.roll,
            brake_pressure=_clamp(outgauge.brake, 0.0, 1.0),
            locking=self._compute_locking(outgauge),
            nfr=self._compute_nfr(outgauge),
            si=self._compute_sense_index(outgauge),
            speed=speed,
            yaw_rate=yaw_rate,
            slip_angle=slip_angle,
            steer=steer,
            throttle=throttle,
            gear=int(outgauge.gear),
            vertical_load_front=front_load,
            vertical_load_rear=rear_load,
            mu_eff_front=mu_front,
            mu_eff_rear=mu_rear,
            mu_eff_front_lateral=mu_front_lat,
            mu_eff_front_longitudinal=mu_front_long,
            mu_eff_rear_lateral=mu_rear_lat,
            mu_eff_rear_longitudinal=mu_rear_long,
            suspension_travel_front=suspension_travel_front,
            suspension_travel_rear=suspension_travel_rear,
            suspension_velocity_front=suspension_velocity_front,
            suspension_velocity_rear=suspension_velocity_rear,
            tyre_temp_fl=tyre_temps[0],
            tyre_temp_fr=tyre_temps[1],
            tyre_temp_rl=tyre_temps[2],
            tyre_temp_rr=tyre_temps[3],
            tyre_temp_fl_inner=tyre_temp_layers[0][0],
            tyre_temp_fr_inner=tyre_temp_layers[0][1],
            tyre_temp_rl_inner=tyre_temp_layers[0][2],
            tyre_temp_rr_inner=tyre_temp_layers[0][3],
            tyre_temp_fl_middle=tyre_temp_layers[1][0],
            tyre_temp_fr_middle=tyre_temp_layers[1][1],
            tyre_temp_rl_middle=tyre_temp_layers[1][2],
            tyre_temp_rr_middle=tyre_temp_layers[1][3],
            tyre_temp_fl_outer=tyre_temp_layers[2][0],
            tyre_temp_fr_outer=tyre_temp_layers[2][1],
            tyre_temp_rl_outer=tyre_temp_layers[2][2],
            tyre_temp_rr_outer=tyre_temp_layers[2][3],
            tyre_pressure_fl=tyre_pressures[0],
            tyre_pressure_fr=tyre_pressures[1],
            tyre_pressure_rl=tyre_pressures[2],
            tyre_pressure_rr=tyre_pressures[3],
            brake_temp_fl=brake_temps[0],
            brake_temp_fr=brake_temps[1],
            brake_temp_rl=brake_temps[2],
            brake_temp_rr=brake_temps[3],
            rpm=float(outgauge.rpm),
            line_deviation=line_deviation,
            slip_ratio_fl=wheel_data.slip_ratios[0],
            slip_ratio_fr=wheel_data.slip_ratios[1],
            slip_ratio_rl=wheel_data.slip_ratios[2],
            slip_ratio_rr=wheel_data.slip_ratios[3],
            slip_angle_fl=wheel_data.slip_angles[0],
            slip_angle_fr=wheel_data.slip_angles[1],
            slip_angle_rl=wheel_data.slip_angles[2],
            slip_angle_rr=wheel_data.slip_angles[3],
            brake_input=brake_input,
            clutch_input=clutch_input,
            handbrake_input=handbrake_input,
            steer_input=steer_input,
            wheel_load_fl=wheel_data.loads[0],
            wheel_load_fr=wheel_data.loads[1],
            wheel_load_rl=wheel_data.loads[2],
            wheel_load_rr=wheel_data.loads[3],
            wheel_lateral_force_fl=wheel_data.lateral_forces[0],
            wheel_lateral_force_fr=wheel_data.lateral_forces[1],
            wheel_lateral_force_rl=wheel_data.lateral_forces[2],
            wheel_lateral_force_rr=wheel_data.lateral_forces[3],
            wheel_longitudinal_force_fl=wheel_data.longitudinal_forces[0],
            wheel_longitudinal_force_fr=wheel_data.longitudinal_forces[1],
            wheel_longitudinal_force_rl=wheel_data.longitudinal_forces[2],
            wheel_longitudinal_force_rr=wheel_data.longitudinal_forces[3],
            suspension_deflection_fl=wheel_data.deflections[0],
            suspension_deflection_fr=wheel_data.deflections[1],
            suspension_deflection_rl=wheel_data.deflections[2],
            suspension_deflection_rr=wheel_data.deflections[3],
            car_model=getattr(outgauge, "car", getattr(outgauge, "car_model", "")),
            track_name=(
                getattr(outgauge, "track", None)
                or getattr(outgauge, "layout", "")
            ),
        )

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------
    def _load_calibration_table(self) -> Mapping[str, object]:
        try:
            with resources.open_binary("tnfr_lfs.data", "fusion_calibration.toml") as handle:
                return tomllib.load(handle)
        except FileNotFoundError:  # pragma: no cover - optional resource
            return {"defaults": {}}

    def _resolve_brake_pack_root(self) -> Path | None:
        candidates: list[Path] = []
        env_root = os.environ.get("TNFR_LFS_PACK_ROOT")
        if env_root:
            candidates.append(Path(env_root).expanduser())
        cwd = Path.cwd()
        candidates.append(cwd)
        package_root = Path(__file__).resolve().parents[2]
        if package_root not in candidates:
            candidates.append(package_root)

        installed_pack_root = pack_root()
        if installed_pack_root not in candidates:
            candidates.append(installed_pack_root)

        for root in candidates:
            config_path = root / "config" / "global.toml"
            cars_dir = root / "data" / "cars"
            if config_path.exists() or cars_dir.exists():
                return root
        return None

    def _load_toml_mapping(self, path: Path) -> Mapping[str, object] | None:
        try:
            with path.open("rb") as buffer:
                payload = tomllib.load(buffer)
        except FileNotFoundError:
            return None
        except (OSError, tomllib.TOMLDecodeError):  # pragma: no cover - invalid file
            return None
        return payload if isinstance(payload, Mapping) else None

    def _extract_brake_section(
        self, payload: Mapping[str, object]
    ) -> Mapping[str, object] | None:
        thermal = payload.get("thermal")
        if not isinstance(thermal, Mapping):
            return None
        brakes = thermal.get("brakes")
        if not isinstance(brakes, Mapping):
            return None
        return brakes

    def _normalise_brake_mode(self, value: object, fallback: str | None = None) -> str | None:
        if not isinstance(value, str):
            return fallback
        candidate = value.strip().lower()
        if candidate in {"auto", "off", "force"}:
            return candidate
        return fallback

    def _load_brake_thermal_profiles(
        self,
    ) -> tuple[
        BrakeThermalConfig,
        str,
        Dict[str, Tuple[BrakeThermalConfig, str | None]],
    ]:
        defaults = BrakeThermalConfig()
        mode = "auto"
        overrides: Dict[str, Tuple[BrakeThermalConfig, str | None]] = {}

        pack_root = self._resolve_brake_pack_root()
        if pack_root is None:
            return defaults, mode, overrides

        global_payload = self._load_toml_mapping(pack_root / "config" / "global.toml")
        if global_payload:
            section = self._extract_brake_section(global_payload)
            if section:
                defaults = merge_brake_config(defaults, section)
                mode_override = self._normalise_brake_mode(section.get("mode"), None)
                if mode_override:
                    mode = mode_override

        cars_dir = pack_root / "data" / "cars"
        if cars_dir.exists():
            for manifest in sorted(cars_dir.glob("*.toml")):
                payload = self._load_toml_mapping(manifest)
                if not payload:
                    continue
                car_abbrev = payload.get("abbrev")
                if not isinstance(car_abbrev, str) or not car_abbrev:
                    continue
                section = self._extract_brake_section(payload)
                if not section:
                    continue
                config = merge_brake_config(defaults, section)
                mode_override = self._normalise_brake_mode(section.get("mode"), None)
                overrides[car_abbrev] = (config, mode_override)

        return defaults, mode, overrides

    def _select_brake_thermal_profile(
        self, car: str | None
    ) -> tuple[BrakeThermalConfig, str]:
        config = self._brake_thermal_defaults
        mode = self._brake_thermal_mode
        if car:
            profile = self._brake_thermal_car_profiles.get(car)
            if profile:
                config = profile[0]
                if profile[1]:
                    mode = profile[1]
        env_override = self._normalise_brake_mode(
            os.environ.get("TNFR_LFS_BRAKE_THERMAL"), None
        )
        if env_override:
            mode = env_override
        return config, mode

    def _ensure_brake_estimator(
        self, config: BrakeThermalConfig, previous: TelemetryRecord | None
    ) -> BrakeThermalEstimator:
        estimator = self._brake_thermal_estimator
        if estimator is None or self._brake_thermal_active_config != config:
            estimator = BrakeThermalEstimator(config)
            if previous is not None:
                estimator.seed(
                    (
                        previous.brake_temp_fl,
                        previous.brake_temp_fr,
                        previous.brake_temp_rl,
                        previous.brake_temp_rr,
                    )
                )
            self._brake_thermal_estimator = estimator
            self._brake_thermal_active_config = config
        return estimator

    def _select_calibration(self, outgauge: OutGaugePacket) -> FusionCalibration:
        car = outgauge.car or "__default__"
        track = outgauge.track or "__default__"
        key = (car, track)
        cached = self._calibration_cache.get(key)
        if cached is not None:
            return cached

        merged: Dict[str, object] = {}

        def merge(source: Mapping[str, object] | None) -> None:
            if not source:
                return
            for k, v in source.items():
                if isinstance(v, Mapping):
                    existing = merged.get(k)
                    if isinstance(existing, Mapping):
                        combined = dict(existing)
                        combined.update(v)
                        merged[k] = combined
                    else:
                        merged[k] = dict(v)
                else:
                    merged[k] = v

        table = self._calibration_table
        merge(table.get("defaults"))

        tracks_table = table.get("tracks")
        if isinstance(tracks_table, Mapping):
            merge(tracks_table.get(track))

        cars_table = table.get("cars")
        car_table = cars_table.get(car) if isinstance(cars_table, Mapping) else None
        if isinstance(car_table, Mapping):
            merge(car_table.get("defaults"))
            track_overrides = car_table.get("tracks")
            if isinstance(track_overrides, Mapping):
                merge(track_overrides.get(track))

        calibration = self._build_calibration(merged)
        self._calibration_cache[key] = calibration
        return calibration

    def _build_calibration(self, raw: Mapping[str, object]) -> FusionCalibration:
        slip = raw.get("slip_ratio") if isinstance(raw.get("slip_ratio"), Mapping) else {}
        steer = raw.get("steer") if isinstance(raw.get("steer"), Mapping) else {}
        mu = raw.get("mu") if isinstance(raw.get("mu"), Mapping) else {}
        suspension = raw.get("suspension") if isinstance(raw.get("suspension"), Mapping) else {}
        tyre = raw.get("tyre") if isinstance(raw.get("tyre"), Mapping) else {}
        latency = raw.get("latency") if isinstance(raw.get("latency"), Mapping) else {}

        load_scale = float(raw.get("load_scale", self.load_scale))
        load_bias = float(raw.get("load_bias", 0.0))
        slip_gain = float(slip.get("gain", 1.0))
        slip_bias = float(slip.get("bias", 0.0))
        steer_ratio = float(steer.get("ratio", 1.0))
        steer_offset = float(steer.get("offset", 0.0))
        wheelbase = float(steer.get("wheelbase", raw.get("wheelbase", 2.6)))
        latency_value = float(latency.get("network", latency.get("value", 0.06)))
        tyre_radius = float(tyre.get("effective_radius", tyre.get("radius", 0.33)))
        mu_lat = float(mu.get("lateral_gain", mu.get("lateral", 1.0)))
        mu_long = float(mu.get("longitudinal_gain", mu.get("longitudinal", 1.0)))
        front_weight = float(
            suspension.get(
                "axle_weighting_front",
                suspension.get("front_weight", suspension.get("front", 0.5)),
            )
        )
        rear_weight = float(
            suspension.get(
                "axle_weighting_rear",
                suspension.get("rear_weight", suspension.get("rear", 0.5)),
            )
        )
        total_weight = front_weight + rear_weight
        if total_weight <= 0.0:
            front_weight = rear_weight = 0.5
        else:
            front_weight /= total_weight
            rear_weight /= total_weight
        window = int(suspension.get("window", suspension.get("samples", 5)) or 5)
        window = max(3, window)

        return FusionCalibration(
            load_scale=load_scale,
            load_bias=load_bias,
            slip_ratio_gain=slip_gain,
            slip_ratio_bias=slip_bias,
            steer_ratio=steer_ratio,
            steer_offset=steer_offset,
            wheelbase=wheelbase,
            network_latency=latency_value,
            tyre_radius=tyre_radius,
            mu_lateral_gain=mu_lat,
            mu_longitudinal_gain=mu_long,
            axle_weight_front=front_weight,
            axle_weight_rear=rear_weight,
            suspension_window=window,
        )

    def _append_vertical_accel(self, accel_z: float, calibration: FusionCalibration) -> None:
        self._vertical_history.append(accel_z)
        max_samples = max(32, calibration.suspension_window * 4)
        excess = len(self._vertical_history) - max_samples
        if excess > 0:
            del self._vertical_history[0:excess]

    def _filtered_vertical_accel(self, window: int) -> float:
        if not self._vertical_history:
            return 0.0
        window = max(1, min(window, len(self._vertical_history)))
        samples = self._vertical_history[-window:]
        if window == 1:
            return samples[0]
        weights = [0.5 - 0.5 * math.cos(2.0 * math.pi * i / (window - 1)) for i in range(window)]
        weight_sum = sum(weights) or float(window)
        return sum(sample * weight for sample, weight in zip(samples, weights)) / weight_sum

    def _compute_vertical_load(self, outsim: OutSimPacket, calibration: FusionCalibration) -> float:
        g_force = outsim.accel_z + 9.81
        load = (g_force * calibration.load_scale) + calibration.load_bias
        return max(0.0, load)

    def _compute_slip_ratio(
        self, outsim: OutSimPacket, outgauge: OutGaugePacket, calibration: FusionCalibration
    ) -> float:
        reference_speed = max(abs(outgauge.speed), 1e-6)
        slip = (outsim.vel_x - outgauge.speed) / reference_speed
        slip = (slip * calibration.slip_ratio_gain) + calibration.slip_ratio_bias
        return _clamp(slip, -1.0, 1.0)

    def _normalise_heading(self, heading: float) -> float:
        if not math.isfinite(heading):
            return 0.0
        wrapped = (heading + math.pi) % (2.0 * math.pi)
        return wrapped - math.pi

    def _compute_speed(self, outsim: OutSimPacket, outgauge: OutGaugePacket) -> float:
        sim_speed = math.hypot(outsim.vel_x, outsim.vel_y)
        gauge_speed = abs(outgauge.speed)
        return max(sim_speed, gauge_speed)

    def _compute_slip_angle(
        self,
        yaw_rate: float,
        speed: float,
        slip_ratio: float,
        outsim: OutSimPacket,
        calibration: FusionCalibration,
        previous: TelemetryRecord | None,
        dt: float,
    ) -> float:
        if speed <= 1e-4:
            return 0.0
        yaw_rate_derivative = 0.0
        if previous and dt > 1e-6:
            yaw_rate_derivative = (yaw_rate - previous.yaw_rate) / dt
        compensated_yaw = yaw_rate + yaw_rate_derivative * calibration.network_latency
        effective_speed = max(speed, 1e-3)
        tyre_slip_velocity = slip_ratio * effective_speed
        tyre_slip_velocity += abs(slip_ratio) * calibration.tyre_radius * abs(compensated_yaw)
        predictive_lateral = compensated_yaw * calibration.wheelbase * 0.5
        measured_lateral = outsim.vel_y
        blended_lateral = (0.6 * measured_lateral) + (0.4 * predictive_lateral)
        longitudinal_correction = effective_speed + tyre_slip_velocity
        return math.atan2(blended_lateral, max(longitudinal_correction, 1e-3))

    def _compute_yaw_rate(
        self,
        timestamp: float,
        yaw: float,
        outsim: OutSimPacket,
        previous: TelemetryRecord | None,
        dt: float,
    ) -> float:
        if previous and dt > 1e-6:
            delta = self._angle_delta(yaw, previous.yaw)
            rate = delta / dt
            return _clamp(rate, -5.0, 5.0)
        if math.isfinite(outsim.ang_vel_z):
            return _clamp(outsim.ang_vel_z, -5.0, 5.0)
        return 0.0

    def _estimate_axle_distribution(
        self, outsim: OutSimPacket, speed: float, calibration: FusionCalibration
    ) -> tuple[float, float]:
        front_base = calibration.axle_weight_front
        rear_base = calibration.axle_weight_rear
        pitch_component = _clamp(-outsim.pitch / 0.15, -0.25, 0.25)
        accel_component = _clamp(-outsim.accel_x / 9.81 * 0.2, -0.2, 0.2)
        curvature = 0.0
        if speed > 1e-3:
            curvature = _clamp(outsim.accel_y / max(speed * speed, 1e-3) * 0.1, -0.05, 0.05)
        front_share = _clamp(front_base + pitch_component + accel_component + curvature, 0.1, 0.9)
        rear_share = 1.0 - front_share
        return front_share, rear_share

    def _compute_suspension_velocity(
        self,
        axle: str,
        share: float,
        previous: float | None,
        dt: float,
        calibration: FusionCalibration,
    ) -> tuple[float, float]:
        return math.nan, math.nan

    def _compute_mu_eff(
        self,
        lateral_accel: float,
        longitudinal_accel: float,
        share: float,
        calibration: FusionCalibration,
    ) -> tuple[float, float, float]:
        if share <= 1e-4:
            return 0.0, 0.0, 0.0
        lateral_g = abs(lateral_accel) / 9.81
        longitudinal_g = abs(longitudinal_accel) / 9.81
        lateral_mu = _clamp((lateral_g / share) * calibration.mu_lateral_gain, 0.0, 3.0)
        longitudinal_mu = _clamp((longitudinal_g / share) * calibration.mu_longitudinal_gain, 0.0, 3.0)
        combined = _clamp((lateral_mu + longitudinal_mu) * 0.5, 0.0, 3.0)
        return combined, lateral_mu, longitudinal_mu

    def _resolve_wheel_temperature_layers(
        self, outgauge: OutGaugePacket, previous: TelemetryRecord | None
    ) -> tuple[tuple[float, float, float, float], ...]:
        def _layer_from_previous(
            *,
            inner: bool = False,
            middle: bool = False,
            outer: bool = False,
        ) -> tuple[float, float, float, float]:
            if not previous:
                return (
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                )
            if inner:
                return (
                    previous.tyre_temp_fl_inner,
                    previous.tyre_temp_fr_inner,
                    previous.tyre_temp_rl_inner,
                    previous.tyre_temp_rr_inner,
                )
            if middle:
                return (
                    previous.tyre_temp_fl_middle,
                    previous.tyre_temp_fr_middle,
                    previous.tyre_temp_rl_middle,
                    previous.tyre_temp_rr_middle,
                )
            if outer:
                return (
                    previous.tyre_temp_fl_outer,
                    previous.tyre_temp_fr_outer,
                    previous.tyre_temp_rl_outer,
                    previous.tyre_temp_rr_outer,
                )
            return (
                previous.tyre_temp_fl,
                previous.tyre_temp_fr,
                previous.tyre_temp_rl,
                previous.tyre_temp_rr,
            )

        def _resolve_layer(
            candidate: tuple[float, float, float, float] | object,
            fallback: tuple[float, float, float, float],
        ) -> tuple[float, float, float, float]:
            if not isinstance(candidate, tuple) or len(candidate) != 4:
                values = (0.0, 0.0, 0.0, 0.0)
            else:
                values = candidate
            resolved: list[float] = []
            for value, default in zip(values, fallback):
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = math.nan
                if math.isfinite(numeric) and numeric > 0.0:
                    resolved_value = numeric
                else:
                    resolved_value = default
                if not math.isfinite(resolved_value) or resolved_value <= 0.0:
                    resolved_value = float("nan")
                resolved.append(float(resolved_value))
            # `values` and `fallback` are normalised to four wheels, so `resolved`
            # always contains exactly four entries matching the return type.
            return cast(tuple[float, float, float, float], tuple(resolved))

        inner = _resolve_layer(
            getattr(outgauge, "tyre_temps_inner", (0.0, 0.0, 0.0, 0.0)),
            _layer_from_previous(inner=True),
        )
        middle = _resolve_layer(
            getattr(outgauge, "tyre_temps_middle", (0.0, 0.0, 0.0, 0.0)),
            _layer_from_previous(middle=True),
        )
        outer = _resolve_layer(
            getattr(outgauge, "tyre_temps_outer", (0.0, 0.0, 0.0, 0.0)),
            _layer_from_previous(outer=True),
        )
        return inner, middle, outer

    def _resolve_wheel_temperatures(
        self,
        outgauge: OutGaugePacket,
        previous: TelemetryRecord | None,
        *,
        layers: tuple[tuple[float, float, float, float], ...] | None = None,
    ) -> tuple[float, float, float, float]:
        fallback = (
            previous.tyre_temp_fl if previous else float("nan"),
            previous.tyre_temp_fr if previous else float("nan"),
            previous.tyre_temp_rl if previous else float("nan"),
            previous.tyre_temp_rr if previous else float("nan"),
        )

        candidate = getattr(outgauge, "tyre_temps", (0.0, 0.0, 0.0, 0.0))
        if not isinstance(candidate, tuple) or len(candidate) != 4:
            candidate = (0.0, 0.0, 0.0, 0.0)

        candidate_values: list[float] = []
        for value in candidate:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = math.nan
            candidate_values.append(numeric)

        def _layer_average(index: int) -> float:
            if layers is None:
                return float("nan")
            aggregates: list[float] = []
            for layer in layers:
                if not isinstance(layer, tuple) or len(layer) != 4:
                    continue
                try:
                    numeric = float(layer[index])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric) and numeric > 0.0:
                    aggregates.append(numeric)
            if not aggregates:
                return float("nan")
            return sum(aggregates) / len(aggregates)

        layer_fallback = tuple(_layer_average(index) for index in range(4))

        resolved: list[float] = []
        for index, default in enumerate(fallback):
            numeric = candidate_values[index]
            if math.isfinite(numeric) and numeric > 0.0:
                resolved_value = numeric
            else:
                layer_value = layer_fallback[index]
                if math.isfinite(layer_value) and layer_value > 0.0:
                    resolved_value = layer_value
                else:
                    resolved_value = default
            if not math.isfinite(resolved_value) or resolved_value <= 0.0:
                resolved_value = float("nan")
            resolved.append(float(resolved_value))

        # `candidate` and `fallback` are normalised to four wheels, so `resolved`
        # always contains exactly four entries matching the declared tuple type.
        return cast(tuple[float, float, float, float], tuple(resolved))

    def _resolve_wheel_pressures(
        self, outgauge: OutGaugePacket, previous: TelemetryRecord | None
    ) -> tuple[float, float, float, float]:
        fallback = (
            previous.tyre_pressure_fl if previous else float("nan"),
            previous.tyre_pressure_fr if previous else float("nan"),
            previous.tyre_pressure_rl if previous else float("nan"),
            previous.tyre_pressure_rr if previous else float("nan"),
        )

        candidate = getattr(outgauge, "tyre_pressures", (0.0, 0.0, 0.0, 0.0))
        if not isinstance(candidate, tuple) or len(candidate) != 4:
            candidate = (0.0, 0.0, 0.0, 0.0)
        candidate_values: list[float] = []
        for value in candidate:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = math.nan
            candidate_values.append(numeric)

        resolved: list[float] = []
        for numeric, default in zip(candidate_values, fallback):
            if math.isfinite(numeric) and numeric > 0.0:
                resolved_value = numeric
            else:
                resolved_value = default
            if not math.isfinite(resolved_value) or resolved_value <= 0.0:
                resolved_value = float("nan")
            resolved.append(float(resolved_value))

        # `candidate_values` and `fallback` always provide one entry per wheel, so
        # `resolved` contains four elements by construction.
        return cast(tuple[float, float, float, float], tuple(resolved))

    # OutGauge brake temperatures are preferred when it supplies plausible
    # readings, even if they rely on non-standard vendor scales; the estimator
    # proxy only steps in according to the configured mode (`auto` falls back to
    # the proxy, `off` sticks to raw OutGauge data, `force` ignores it). The
    # `TNFR_LFS_BRAKE_THERMAL` environment variable can override the mode for
    # expected scenarios such as forcing the proxy on hardware without sensors
    # or disabling it entirely during calibration captures.
    def _resolve_brake_temperatures(
        self,
        dt: float,
        speed: float,
        deceleration: float,
        brake_input: float,
        wheel_loads: tuple[float, float, float, float],
        outgauge: OutGaugePacket,
        previous: TelemetryRecord | None,
    ) -> tuple[float, float, float, float]:
        config, mode = self._select_brake_thermal_profile(outgauge.car)
        estimator = self._ensure_brake_estimator(config, previous)
        estimated = estimator.step(dt, speed, deceleration, brake_input, wheel_loads)

        candidate = getattr(outgauge, "brake_temps", (0.0, 0.0, 0.0, 0.0))
        if not isinstance(candidate, tuple) or len(candidate) != 4:
            candidate = (0.0, 0.0, 0.0, 0.0)
        fallback = (
            previous.brake_temp_fl if previous else math.nan,
            previous.brake_temp_fr if previous else math.nan,
            previous.brake_temp_rl if previous else math.nan,
            previous.brake_temp_rr if previous else math.nan,
        )

        gauge_values: list[float] = []
        gauge_valid = False
        for raw, default in zip(candidate, fallback):
            try:
                numeric = float(raw)
            except (TypeError, ValueError):
                numeric = math.nan
            if math.isfinite(numeric) and numeric > 0.0:
                gauge_valid = True
                gauge_values.append(float(numeric))
            else:
                replacement = default if math.isfinite(default) and default > 0.0 else math.nan
                gauge_values.append(float(replacement))

        if gauge_valid:
            estimator.observe(gauge_values)
            if mode != "force":
                # The estimator stores one temperature per wheel and exposes a
                # four-element tuple via its property.
                return estimator.temperatures

        env_mode = mode
        if env_mode == "off":
            # `gauge_values` captures four brake readings, one per wheel.
            return cast(tuple[float, float, float, float], tuple(gauge_values))
        if env_mode == "force":
            return estimated

        if gauge_valid:
            # The estimator stores one temperature per wheel and exposes a
            # four-element tuple via its property.
            return estimator.temperatures

        return estimated

    def _compute_line_deviation(self, outsim: OutSimPacket, window: int = 25) -> float:
        position = (float(outsim.pos_x), float(outsim.pos_y))
        self._line_history.append(position)
        max_length = max(window * 2, 5)
        if len(self._line_history) > max_length:
            del self._line_history[: len(self._line_history) - max_length]

        history = self._line_history[-window:]
        if len(history) < 2:
            return 0.0

        mean_x = sum(point[0] for point in history) / len(history)
        mean_y = sum(point[1] for point in history) / len(history)
        centred = [(x - mean_x, y - mean_y) for x, y in history]
        var_x = sum(dx * dx for dx, _ in centred)
        var_y = sum(dy * dy for _, dy in centred)
        cov_xy = sum(dx * dy for dx, dy in centred)

        if var_x <= 1e-9 and var_y <= 1e-9:
            return 0.0

        trace = var_x + var_y
        determinant = (var_x * var_y) - (cov_xy * cov_xy)
        discriminant = max(0.0, (trace * trace) - (4.0 * determinant))
        eigenvalue = 0.5 * (trace + math.sqrt(discriminant))

        if eigenvalue <= 1e-12:
            direction = (1.0, 0.0) if var_x >= var_y else (0.0, 1.0)
        else:
            if abs(cov_xy) > 1e-12:
                vx = cov_xy
                vy = eigenvalue - var_x
            else:
                vx, vy = (1.0, 0.0) if var_x >= var_y else (0.0, 1.0)
            norm = math.hypot(vx, vy)
            if norm <= 1e-12:
                direction = (1.0, 0.0)
            else:
                direction = (vx / norm, vy / norm)

        delta_x = position[0] - mean_x
        delta_y = position[1] - mean_y
        projection = (delta_x * direction[0]) + (delta_y * direction[1])
        proj_x = direction[0] * projection
        proj_y = direction[1] * projection
        perp_x = delta_x - proj_x
        perp_y = delta_y - proj_y
        deviation = math.hypot(perp_x, perp_y)
        cross = (direction[0] * delta_y) - (direction[1] * delta_x)
        sign = 1.0 if cross >= 0.0 else -1.0
        return deviation * sign

    def _compute_steer(
        self, yaw_rate: float, speed: float, calibration: FusionCalibration
    ) -> float:
        if speed <= 1e-3:
            return 0.0
        curvature = yaw_rate / speed
        steer_ratio = curvature * calibration.wheelbase * calibration.steer_ratio
        steer_ratio += calibration.steer_offset
        return _clamp(steer_ratio, -1.5, 1.5)

    def _angle_delta(self, value: float, reference: float) -> float:
        delta = value - reference
        wrapped = (delta + math.pi) % (2.0 * math.pi)
        return wrapped - math.pi

    def _compute_nfr(self, outgauge: OutGaugePacket) -> float:
        return outgauge.rpm / 10.0

    def _compute_sense_index(self, outgauge: OutGaugePacket) -> float:
        return _clamp(outgauge.throttle, 0.0, 1.0)

    def _compute_locking(self, outgauge: OutGaugePacket) -> float:
        # The ABS dash light signals wheel locking mitigation.  We use it as
        # a binary proxy for locking activity which is smoothed downstream by
        # :func:`delta_nfr_by_node`.
        abs_active = bool(outgauge.dash_lights & 0x20)
        tc_active = bool(outgauge.dash_lights & 0x10)
        return 1.0 if abs_active or tc_active else 0.0

