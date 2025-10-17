"""Operator event aggregation helpers for the ΔNFR×Si pipeline."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from typing import Dict, List, Mapping, Sequence, Tuple

from tnfr_core.runtime.shared import SupportsMicrosector
from tnfr_core.operators.operator_labels import (
    normalize_structural_operator_identifier,
    silence_event_payloads,
)


def _aggregate_operator_events(
    microsectors: Sequence[SupportsMicrosector] | None,
) -> Dict[str, object]:
    aggregated: Dict[str, List[Mapping[str, object]]] = {}
    latent_states: Dict[str, Dict[int, Dict[str, float]]] = {}
    if not microsectors:
        return {"events": aggregated, "latent_states": latent_states}
    for microsector in microsectors:
        raw_events = getattr(microsector, "operator_events", {}) or {}
        events: Dict[str, Tuple[Mapping[str, object], ...]] = {}
        for name, payload in raw_events.items():
            normalized_name = normalize_structural_operator_identifier(name)
            if isinstance(payload, SequenceABC) and not isinstance(payload, MappingABC):
                entries = tuple(payload)
            elif payload is None:
                entries = ()
            else:
                entries = (payload,)
            if not entries:
                continue
            if normalized_name in events:
                events[normalized_name] = events[normalized_name] + entries
            else:
                events[normalized_name] = entries
        micro_duration = max(
            0.0,
            float(getattr(microsector, "end_time", 0.0))
            - float(getattr(microsector, "start_time", 0.0)),
        )
        silent_duration = 0.0
        silent_density = 0.0
        silent_events = 0
        for name, payload in events.items():
            bucket = aggregated.setdefault(name, [])
            for entry in payload:
                event_payload = dict(entry)
                event_payload.setdefault("microsector", microsector.index)
                bucket.append(event_payload)
        silence_entries = silence_event_payloads(events)
        silent_events = len(silence_entries)
        for event_payload in silence_entries:
            duration = float(event_payload.get("duration", 0.0) or 0.0)
            silent_duration += max(0.0, duration)
            density_value = float(
                event_payload.get("structural_density_mean", 0.0) or 0.0
            )
            silent_density += max(0.0, density_value)
        if silent_events:
            coverage = 0.0
            if micro_duration > 1e-9:
                coverage = min(1.0, silent_duration / micro_duration)
            state_entry = latent_states.setdefault("SILENCE", {})
            state_entry[microsector.index] = {
                "coverage": float(coverage),
                "duration": float(silent_duration),
                "count": float(silent_events),
            }
            if silent_events > 0:
                state_entry[microsector.index]["mean_density"] = float(
                    silent_density / silent_events
                )
    return {"events": aggregated, "latent_states": latent_states}
