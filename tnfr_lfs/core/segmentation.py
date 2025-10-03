"""Microsector segmentation utilities.

This module analyses the stream of telemetry samples together with the
corresponding :class:`~tnfr_lfs.core.epi_models.EPIBundle` instances to
derive microsectors and their tactical goals.  The segmentation is
performed using three simple heuristics inspired by motorsport
engineering practice:

* **Curvature detection** – sustained lateral acceleration indicates a
  cornering event which is the base unit for microsectors.
* **Support events** – sharp vertical load increases signal the moment
  where the chassis "leans" on the tyres and generates grip.
* **ΔNFR signatures** – the average ΔNFR within the microsector is used
  to classify the underlying archetype which in turn drives the goals
  for the entry, apex and exit phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Mapping, Sequence, Tuple

from .epi import TelemetryRecord
from .epi_models import EPIBundle

# Thresholds derived from typical race car dynamics.  They can be tuned in
# the future without affecting the public API of the segmentation module.
CURVATURE_THRESHOLD = 1.2  # g units of lateral acceleration
MIN_SEGMENT_LENGTH = 3
BRAKE_THRESHOLD = -0.35  # g units of longitudinal deceleration
SUPPORT_THRESHOLD = 350.0  # Newtons of vertical load delta


PhaseLiteral = str


@dataclass(frozen=True)
class Goal:
    """Operational goal associated with a microsector phase."""

    phase: PhaseLiteral
    archetype: str
    description: str
    target_delta_nfr: float
    target_sense_index: float


@dataclass(frozen=True)
class Microsector:
    """Segment of the lap grouped by curvature and ΔNFR behaviour."""

    index: int
    start_time: float
    end_time: float
    curvature: float
    brake_event: bool
    support_event: bool
    delta_nfr_signature: float
    goals: Tuple[Goal, Goal, Goal]
    phase_boundaries: Mapping[PhaseLiteral, Tuple[int, int]]

    def phase_indices(self, phase: PhaseLiteral) -> range:
        """Return the range of sample indices assigned to ``phase``."""

        start, stop = self.phase_boundaries[phase]
        return range(start, stop)


def segment_microsectors(
    records: Sequence[TelemetryRecord], bundles: Sequence[EPIBundle]
) -> List[Microsector]:
    """Derive microsectors from telemetry and ΔNFR signatures.

    Parameters
    ----------
    records:
        Telemetry samples in chronological order.
    bundles:
        Computed :class:`~tnfr_lfs.core.epi_models.EPIBundle` for the same
        timestamps as ``records``.

    Returns
    -------
    list of :class:`Microsector`
        Each microsector contains phase objectives bound to an archetype.
    """

    if len(records) != len(bundles):
        raise ValueError("records and bundles must have the same length")

    segments = _identify_corner_segments(records)
    microsectors: List[Microsector] = []
    for index, (start, end) in enumerate(segments):
        phase_boundaries = _compute_phase_boundaries(records, start, end)
        curvature = mean(abs(records[i].lateral_accel) for i in range(start, end + 1))
        brake_event = any(records[i].longitudinal_accel <= BRAKE_THRESHOLD for i in range(start, end + 1))
        support_event = _detect_support_event(records[start : end + 1])
        delta_signature = mean(b.delta_nfr for b in bundles[start : end + 1])
        avg_si = mean(b.sense_index for b in bundles[start : end + 1])
        archetype = _classify_archetype(delta_signature, avg_si, brake_event, support_event)
        goals = _build_goals(archetype, bundles, phase_boundaries)
        microsectors.append(
            Microsector(
                index=index,
                start_time=records[start].timestamp,
                end_time=records[end].timestamp,
                curvature=curvature,
                brake_event=brake_event,
                support_event=support_event,
                delta_nfr_signature=delta_signature,
                goals=goals,
                phase_boundaries=phase_boundaries,
            )
        )
    return microsectors


def _identify_corner_segments(records: Sequence[TelemetryRecord]) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start_index: int | None = None
    for idx, record in enumerate(records):
        if abs(record.lateral_accel) >= CURVATURE_THRESHOLD:
            if start_index is None:
                start_index = idx
        else:
            if start_index is not None and idx - start_index >= MIN_SEGMENT_LENGTH:
                segments.append((start_index, idx - 1))
            start_index = None
    if start_index is not None and len(records) - start_index >= MIN_SEGMENT_LENGTH:
        segments.append((start_index, len(records) - 1))
    return segments


def _compute_phase_boundaries(
    records: Sequence[TelemetryRecord], start: int, end: int
) -> Dict[PhaseLiteral, Tuple[int, int]]:
    idx_range = range(start, end + 1)
    entry_idx = min(idx_range, key=lambda i: records[i].longitudinal_accel)
    apex_idx = max(idx_range, key=lambda i: abs(records[i].lateral_accel))
    exit_idx = max(idx_range, key=lambda i: records[i].longitudinal_accel)

    entry_idx = max(start, min(entry_idx, end))
    apex_idx = max(start, min(apex_idx, end))
    exit_idx = max(apex_idx, min(exit_idx, end))

    entry_end = max(start, min(entry_idx, apex_idx)) + 1
    apex_start = entry_end
    apex_end = max(apex_idx + 1, apex_start + 1 if apex_start <= end else end + 1)
    exit_start = min(apex_end, end + 1)
    exit_end = end + 1

    # Guard rails to ensure every phase includes at least one sample and the
    # spans remain ordered and non-overlapping.
    entry_span = (start, min(entry_end, end + 1))
    if entry_span[0] >= entry_span[1]:
        entry_span = (start, min(start + 1, end + 1))

    apex_span = (max(entry_span[1], apex_start), min(apex_end, end + 1))
    if apex_span[0] >= apex_span[1]:
        apex_span = (entry_span[1], min(entry_span[1] + 1, end + 1))

    exit_span = (max(apex_span[1], exit_start), exit_end)
    if exit_span[0] >= exit_span[1]:
        exit_span = (apex_span[1], min(apex_span[1] + 1, end + 1))

    return {
        "entry": entry_span,
        "apex": apex_span,
        "exit": exit_span,
    }


def _detect_support_event(records: Sequence[TelemetryRecord]) -> bool:
    loads = [record.vertical_load for record in records]
    return max(loads) - min(loads) >= SUPPORT_THRESHOLD


def _classify_archetype(
    delta_signature: float,
    sense_index_value: float,
    brake_event: bool,
    support_event: bool,
) -> str:
    if sense_index_value < 0.55:
        return "recuperacion"
    if delta_signature > 5.0 and support_event:
        return "apoyo"
    if delta_signature < -5.0 and brake_event:
        return "liberacion"
    return "equilibrio"


def _build_goals(
    archetype: str, bundles: Sequence[EPIBundle], boundaries: Mapping[PhaseLiteral, Tuple[int, int]]
) -> Tuple[Goal, Goal, Goal]:
    descriptions = _goal_descriptions(archetype)
    goals: List[Goal] = []
    for phase in ("entry", "apex", "exit"):
        start, stop = boundaries[phase]
        segment = bundles[start:stop]
        if segment:
            avg_delta = mean(bundle.delta_nfr for bundle in segment)
            avg_si = mean(bundle.sense_index for bundle in segment)
        else:
            avg_delta = 0.0
            avg_si = 1.0
        goals.append(
            Goal(
                phase=phase,
                archetype=archetype,
                description=descriptions[phase],
                target_delta_nfr=avg_delta,
                target_sense_index=avg_si,
            )
        )
    return goals[0], goals[1], goals[2]


def _goal_descriptions(archetype: str) -> Mapping[str, str]:
    base = {
        "entry": "Modular la transferencia para consolidar el arquetipo de {archetype} en entrada.",
        "apex": "Alinear el vértice con el patrón de {archetype} manteniendo ΔNFR estable.",
        "exit": "Liberar energía siguiendo el arquetipo de {archetype} hacia la salida.",
    }
    if archetype == "apoyo":
        base.update(
            {
                "entry": "Generar deceleración progresiva para preparar el arquetipo de apoyo.",
                "apex": "Sostener la carga máxima en el vértice para reforzar el apoyo.",
                "exit": "Transferir carga manteniendo el apoyo en la tracción inicial.",
            }
        )
    elif archetype == "liberacion":
        base.update(
            {
                "entry": "Extender la frenada para inducir liberación controlada.",
                "apex": "Permitir la rotación asociada al arquetipo de liberación.",
                "exit": "Reequilibrar al salir para cerrar la fase de liberación.",
            }
        )
    elif archetype == "recuperacion":
        base.update(
            {
                "entry": "Recuperar estabilidad en la aproximación al vértice.",
                "apex": "Maximizar el índice de sentido para restablecer la recuperación.",
                "exit": "Asegurar la tracción evitando pérdidas posteriores.",
            }
        )
    return {phase: message.format(archetype=archetype) for phase, message in base.items()}

