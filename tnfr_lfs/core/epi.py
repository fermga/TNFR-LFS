"""EPI extraction and ΔNFR/ΔSi computations."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class TelemetryRecord:
    """Single telemetry sample emitted by the acquisition backend."""

    timestamp: float
    vertical_load: float
    slip_ratio: float
    lateral_accel: float
    longitudinal_accel: float
    nfr: float
    si: float


@dataclass(frozen=True)
class EPIResult:
    """Computed Event Performance Indicator for a single record."""

    timestamp: float
    epi: float
    delta_nfr: float
    delta_si: float


class EPIExtractor:
    """Compute EPI values for a stream of telemetry records.

    The algorithm implemented here is deliberately simple: the EPI is
    the weighted sum of load transfer and slip ratio, normalised to a
    0..1 scale assuming realistic race telemetry ranges.  This provides
    a deterministic implementation that can be unit-tested and used in
    examples.
    """

    def __init__(self, load_weight: float = 0.6, slip_weight: float = 0.4) -> None:
        if not 0 <= load_weight <= 1:
            raise ValueError("load_weight must be in the 0..1 range")
        if not 0 <= slip_weight <= 1:
            raise ValueError("slip_weight must be in the 0..1 range")
        self.load_weight = load_weight
        self.slip_weight = slip_weight

    def extract(self, records: Sequence[TelemetryRecord]) -> List[EPIResult]:
        if not records:
            return []
        baseline = DeltaCalculator.derive_baseline(records)
        results: List[EPIResult] = []
        for record in records:
            epi_value = self._compute_epi(record)
            delta = DeltaCalculator.compute(record, baseline)
            results.append(
                EPIResult(
                    timestamp=record.timestamp,
                    epi=epi_value,
                    delta_nfr=delta.delta_nfr,
                    delta_si=delta.delta_si,
                )
            )
        return results

    def _compute_epi(self, record: TelemetryRecord) -> float:
        # Normalise vertical load between 0 and 10 kN which is a typical
        # race car range.  Slip ratio is expected in -1..1.
        load_component = min(max(record.vertical_load / 10000.0, 0.0), 1.0)
        slip_component = min(max((record.slip_ratio + 1.0) / 2.0, 0.0), 1.0)
        return (load_component * self.load_weight) + (slip_component * self.slip_weight)


@dataclass(frozen=True)
class DeltaMetrics:
    """ΔNFR and ΔSi values relative to a baseline."""

    delta_nfr: float
    delta_si: float


class DeltaCalculator:
    """Compute delta metrics relative to a baseline."""

    @staticmethod
    def derive_baseline(records: Sequence[TelemetryRecord]) -> TelemetryRecord:
        """Return a synthetic baseline record representing the average state."""

        return TelemetryRecord(
            timestamp=records[0].timestamp,
            vertical_load=mean(record.vertical_load for record in records),
            slip_ratio=mean(record.slip_ratio for record in records),
            lateral_accel=mean(record.lateral_accel for record in records),
            longitudinal_accel=mean(record.longitudinal_accel for record in records),
            nfr=mean(record.nfr for record in records),
            si=mean(record.si for record in records),
        )

    @staticmethod
    def compute(record: TelemetryRecord, baseline: TelemetryRecord) -> DeltaMetrics:
        return DeltaMetrics(
            delta_nfr=record.nfr - baseline.nfr,
            delta_si=record.si - baseline.si,
        )


def compute_coherence(results: Iterable[EPIResult]) -> float:
    """Compute a coherence score for a sequence of EPI results.

    The coherence is defined as the inverse of the coefficient of
    variation (standard deviation divided by the mean) of the EPI
    values.  A high coherence indicates consistent performance across
    the analysed telemetry segment.
    """

    epi_values = [result.epi for result in results]
    if len(epi_values) < 2:
        return 1.0
    avg = mean(epi_values)
    if avg == 0:
        return 0.0
    variance = mean((value - avg) ** 2 for value in epi_values)
    stddev = variance ** 0.5
    return max(0.0, min(1.0, 1.0 - (stddev / avg)))
