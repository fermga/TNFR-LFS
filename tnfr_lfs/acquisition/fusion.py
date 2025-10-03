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

        record = TelemetryRecord(
            timestamp=outsim.time / 1000.0,
            vertical_load=self._compute_vertical_load(outsim),
            slip_ratio=self._compute_slip_ratio(outsim, outgauge),
            lateral_accel=outsim.accel_y,
            longitudinal_accel=outsim.accel_x,
            yaw=self._normalise_heading(outsim.heading),
            pitch=outsim.pitch,
            roll=outsim.roll,
            brake_pressure=_clamp(outgauge.brake, 0.0, 1.0),
            locking=self._compute_locking(outgauge),
            nfr=self._compute_nfr(outgauge),
            si=self._compute_sense_index(outgauge),
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
        speed = max(abs(outgauge.speed), 1e-6)
        slip = (outsim.vel_x - outgauge.speed) / speed
        return _clamp(slip, -1.0, 1.0)

    def _normalise_heading(self, heading: float) -> float:
        if not math.isfinite(heading):
            return 0.0
        wrapped = (heading + math.pi) % (2.0 * math.pi)
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
