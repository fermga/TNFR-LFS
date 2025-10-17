"""Reception stage helpers for the ΔNFR×Si pipeline."""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Sequence, Tuple

from tnfr_core.equations.epi import TelemetryRecord
from tnfr_core.runtime.shared import SupportsEPIBundle

ReceptionOperator = Callable[[Sequence[TelemetryRecord]], Sequence[SupportsEPIBundle]]


def _stage_reception(
    telemetry_segments: Sequence[Sequence[TelemetryRecord]],
    *,
    reception_fn: ReceptionOperator,
) -> tuple[Dict[str, object], List[TelemetryRecord]]:
    """Return the reception stage payload and the flattened telemetry records."""

    bundles: List[SupportsEPIBundle] = []
    lap_indices: List[int] = []
    lap_metadata: List[Dict[str, object]] = []
    flattened_records: List[TelemetryRecord] = []

    for lap_index, segment in enumerate(telemetry_segments):
        segment_records = list(segment)
        flattened_records.extend(segment_records)
        label_value = next(
            (
                record.lap
                for record in segment_records
                if getattr(record, "lap", None) is not None
            ),
            None,
        )
        explicit = label_value is not None
        label = str(label_value) if explicit else f"Lap {lap_index + 1}"
        lap_metadata.append(
            {
                "index": lap_index,
                "label": label,
                "value": label_value,
                "explicit": explicit,
            }
        )
        segment_bundles = reception_fn(segment_records)
        lap_indices.extend([lap_index] * len(segment_bundles))
        bundles.extend(segment_bundles)

    stage_payload: Dict[str, object] = {
        "bundles": bundles,
        "lap_indices": lap_indices,
        "lap_sequence": lap_metadata,
        "sample_count": len(bundles),
    }
    return stage_payload, flattened_records
