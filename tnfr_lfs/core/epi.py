"""EPI extraction and ΔNFR/ΔSi computations."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Mapping, Sequence

from .coherence import compute_node_delta_nfr, sense_index
from .epi_models import (
    BrakesNode,
    ChassisNode,
    DriverNode,
    EPIBundle,
    SuspensionNode,
    TrackNode,
    TransmissionNode,
    TyresNode,
)


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

class EPIExtractor:
    """Compute EPI bundles for a stream of telemetry records."""

    def __init__(self, load_weight: float = 0.6, slip_weight: float = 0.4) -> None:
        if not 0 <= load_weight <= 1:
            raise ValueError("load_weight must be in the 0..1 range")
        if not 0 <= slip_weight <= 1:
            raise ValueError("slip_weight must be in the 0..1 range")
        self.load_weight = load_weight
        self.slip_weight = slip_weight

    def extract(self, records: Sequence[TelemetryRecord]) -> List[EPIBundle]:
        if not records:
            return []
        baseline = DeltaCalculator.derive_baseline(records)
        results: List[EPIBundle] = []
        for record in records:
            epi_value = self._compute_epi(record)
            results.append(DeltaCalculator.compute_bundle(record, baseline, epi_value))
        return results

    def _compute_epi(self, record: TelemetryRecord) -> float:
        # Normalise vertical load between 0 and 10 kN which is a typical
        # race car range.  Slip ratio is expected in -1..1.
        load_component = min(max(record.vertical_load / 10000.0, 0.0), 1.0)
        slip_component = min(max((record.slip_ratio + 1.0) / 2.0, 0.0), 1.0)
        return (load_component * self.load_weight) + (slip_component * self.slip_weight)


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
    def compute_bundle(
        record: TelemetryRecord, baseline: TelemetryRecord, epi_value: float
    ) -> EPIBundle:
        delta_nfr = record.nfr - baseline.nfr
        feature_map = DeltaCalculator._feature_map(record, baseline)
        node_deltas = compute_node_delta_nfr(delta_nfr, feature_map)
        global_si = sense_index(delta_nfr, node_deltas, baseline.nfr)
        nodes = DeltaCalculator._build_nodes(node_deltas, delta_nfr)
        return EPIBundle(
            timestamp=record.timestamp,
            epi=epi_value,
            delta_nfr=delta_nfr,
            sense_index=global_si,
            tyres=nodes["tyres"],
            suspension=nodes["suspension"],
            chassis=nodes["chassis"],
            brakes=nodes["brakes"],
            transmission=nodes["transmission"],
            track=nodes["track"],
            driver=nodes["driver"],
        )

    @staticmethod
    def _feature_map(
        record: TelemetryRecord, baseline: TelemetryRecord
    ) -> Mapping[str, float]:
        return {
            "tyres": abs(record.slip_ratio - baseline.slip_ratio),
            "suspension": abs(record.vertical_load - baseline.vertical_load),
            "chassis": abs(record.lateral_accel - baseline.lateral_accel),
            "brakes": abs(record.longitudinal_accel - baseline.longitudinal_accel),
            "transmission": abs(
                (record.longitudinal_accel + record.slip_ratio)
                - (baseline.longitudinal_accel + baseline.slip_ratio)
            ),
            "track": abs(
                (record.vertical_load * record.lateral_accel)
                - (baseline.vertical_load * baseline.lateral_accel)
            ),
            "driver": abs(record.si - baseline.si),
        }

    @staticmethod
    def _build_nodes(node_deltas: Mapping[str, float], delta_nfr: float) -> Dict[str, object]:
        def node_si(node_delta: float) -> float:
            if abs(delta_nfr) < 1e-9:
                return 1.0
            ratio = min(1.0, abs(node_delta) / (abs(delta_nfr) + 1e-9))
            return max(0.0, min(1.0, 1.0 - ratio))

        return {
            "tyres": TyresNode(
                delta_nfr=node_deltas.get("tyres", 0.0),
                sense_index=node_si(node_deltas.get("tyres", 0.0)),
            ),
            "suspension": SuspensionNode(
                delta_nfr=node_deltas.get("suspension", 0.0),
                sense_index=node_si(node_deltas.get("suspension", 0.0)),
            ),
            "chassis": ChassisNode(
                delta_nfr=node_deltas.get("chassis", 0.0),
                sense_index=node_si(node_deltas.get("chassis", 0.0)),
            ),
            "brakes": BrakesNode(
                delta_nfr=node_deltas.get("brakes", 0.0),
                sense_index=node_si(node_deltas.get("brakes", 0.0)),
            ),
            "transmission": TransmissionNode(
                delta_nfr=node_deltas.get("transmission", 0.0),
                sense_index=node_si(node_deltas.get("transmission", 0.0)),
            ),
            "track": TrackNode(
                delta_nfr=node_deltas.get("track", 0.0),
                sense_index=node_si(node_deltas.get("track", 0.0)),
            ),
            "driver": DriverNode(
                delta_nfr=node_deltas.get("driver", 0.0),
                sense_index=node_si(node_deltas.get("driver", 0.0)),
            ),
        }
