"""Fusion utilities for OutSim and OutGauge telemetry."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from ..core.epi import EPIExtractor, EPIBundle, TelemetryRecord
from .outsim_udp import OutSimPacket
from .outgauge_udp import OutGaugePacket

__all__ = ["TelemetryFusion"]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


@dataclass
class TelemetryFusion:
    """Combine OutSim and OutGauge packets into :class:`TelemetryRecord` objects."""

    load_scale: float = 1000.0
    extractor: EPIExtractor = field(default_factory=EPIExtractor)
    _records: List[TelemetryRecord] = field(default_factory=list, init=False)

    def reset(self) -> None:
        """Clear the internal telemetry history."""

        self._records.clear()

    def fuse(self, outsim: OutSimPacket, outgauge: OutGaugePacket) -> TelemetryRecord:
        """Return a :class:`TelemetryRecord` derived from both UDP sources."""

        timestamp = outsim.time / 1000.0
        previous = self._records[-1] if self._records else None
        dt = timestamp - previous.timestamp if previous else 0.0

        vertical_load = self._compute_vertical_load(outsim)
        speed = self._compute_speed(outsim, outgauge)
        yaw = self._normalise_heading(outsim.heading)
        yaw_rate = self._compute_yaw_rate(timestamp, yaw, outsim, previous, dt)
        slip_angle = self._compute_slip_angle(outsim)
        slip_ratio = self._compute_slip_ratio(outsim, outgauge)
        throttle = _clamp(outgauge.throttle, 0.0, 1.0)
        steer = self._compute_steer(yaw_rate, speed)
        front_share, rear_share = self._estimate_axle_distribution(outsim, speed)
        front_load = vertical_load * front_share
        rear_load = vertical_load * rear_share
        travel_front = front_share
        travel_rear = rear_share
        vel_front = self._compute_suspension_velocity(
            travel_front, previous.suspension_travel_front if previous else None, dt
        )
        vel_rear = self._compute_suspension_velocity(
            travel_rear, previous.suspension_travel_rear if previous else None, dt
        )

        record = TelemetryRecord(
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
            mu_eff_front=self._compute_mu_eff(outsim.accel_y, front_share),
            mu_eff_rear=self._compute_mu_eff(outsim.accel_y, rear_share),
            suspension_travel_front=travel_front,
            suspension_travel_rear=travel_rear,
            suspension_velocity_front=vel_front,
            suspension_velocity_rear=vel_rear,
        )
        self._records.append(record)
        return record

    def fuse_to_bundle(self, outsim: OutSimPacket, outgauge: OutGaugePacket) -> EPIBundle:
        """Return an :class:`EPIBundle` for the latest fused sample."""

        self.fuse(outsim, outgauge)
        bundles = self.extractor.extract(self._records)
        return bundles[-1]

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------
    def _compute_vertical_load(self, outsim: OutSimPacket) -> float:
        g_force = outsim.accel_z + 9.81
        return max(0.0, g_force * self.load_scale)

    def _compute_slip_ratio(self, outsim: OutSimPacket, outgauge: OutGaugePacket) -> float:
        reference_speed = max(abs(outgauge.speed), 1e-6)
        slip = (outsim.vel_x - outgauge.speed) / reference_speed
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

    def _compute_slip_angle(self, outsim: OutSimPacket) -> float:
        return math.atan2(outsim.vel_y, max(1e-6, abs(outsim.vel_x)))

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
        self, outsim: OutSimPacket, speed: float
    ) -> tuple[float, float]:
        pitch_component = _clamp(-outsim.pitch / 0.15, -0.3, 0.3)
        accel_component = _clamp(-outsim.accel_x / 9.81 * 0.2, -0.2, 0.2)
        curvature = 0.0
        if speed > 1e-3:
            curvature = _clamp(outsim.accel_y / max(speed * speed, 1e-3) * 0.1, -0.05, 0.05)
        front_share = _clamp(0.5 + pitch_component + accel_component + curvature, 0.25, 0.75)
        rear_share = 1.0 - front_share
        return front_share, rear_share

    def _compute_suspension_velocity(
        self, current: float, previous: float | None, dt: float
    ) -> float:
        if previous is None or dt <= 1e-6:
            return 0.0
        derivative = (current - previous) / dt
        return _clamp(derivative, -5.0, 5.0)

    def _compute_mu_eff(self, lateral_accel: float, share: float) -> float:
        if share <= 1e-4:
            return 0.0
        lateral_g = abs(lateral_accel) / 9.81
        mu = lateral_g / max(share, 1e-4)
        return _clamp(mu, 0.0, 3.0)

    def _compute_steer(self, yaw_rate: float, speed: float) -> float:
        if speed <= 1e-3:
            return 0.0
        wheelbase = 2.6
        curvature = yaw_rate / speed
        steer_ratio = curvature * wheelbase
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
