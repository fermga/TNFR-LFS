"""Recursivity operator implementation and supporting utilities."""

from __future__ import annotations

from collections import deque
from collections.abc import (
    Mapping as MappingABC,
    MutableMapping as MutableMappingABC,
    Sequence as SequenceABC,
)
from typing import (
    Deque,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    Tuple,
    TypedDict,
    cast,
)


class RecursivityMicroState(TypedDict, total=False):
    """In-memory representation of a microsector recursivity state."""

    filtered: Dict[str, float]
    phase: str | None
    samples: int
    trace: Deque[Mapping[str, float | str | None]]
    last_measures: Dict[str, float | str | None]
    _rolling: MutableMapping[str, Deque[float]]
    converged: bool
    last_timestamp: float | None


class RecursivityMicroStateSnapshot(TypedDict, total=False):
    """Serialisable snapshot of a microsector recursivity state."""

    phase: str | None
    samples: int
    filtered: Dict[str, float]
    trace: Tuple[Mapping[str, float | str | None], ...]
    converged: bool
    last_measures: Dict[str, float | str | None]


class RecursivityHistoryEntry(TypedDict):
    """Historical record for a completed stint."""

    stint: int
    ended_at: float | None
    reason: str
    samples: int
    microsectors: Dict[str, RecursivityMicroStateSnapshot]


class RecursivitySessionState(TypedDict, total=False):
    """State container for an entire recursivity session."""

    active: MutableMapping[str, RecursivityMicroState]
    history: List[RecursivityHistoryEntry]
    samples: int
    stint_index: int
    last_timestamp: float | None
    components: tuple[str, str, str]


class RecursivityStateRoot(TypedDict, total=False):
    """Root object storing sessions keyed by identifier."""

    sessions: MutableMapping[str, RecursivitySessionState]
    active_session: str | None


class RecursivityOperatorResult(TypedDict):
    """Return payload from :func:`recursivity_operator`."""

    session: str
    session_components: tuple[str, str, str]
    stint: int
    microsector_id: str
    phase: str | None
    filtered: Dict[str, float]
    samples: int
    phase_changed: bool
    trace: Tuple[Mapping[str, float | str | None], ...]
    state: RecursivityMicroStateSnapshot
    converged: bool
    stint_completed: str | None
    timestamp: float | None


class RecursivityNetworkHistoryEntry(TypedDict):
    """Snapshot of a historical stint for network payloads."""

    stint: int
    ended_at: float | None
    reason: str
    samples: int
    microsectors: Dict[str, RecursivityMicroStateSnapshot]


class RecursivityNetworkSession(TypedDict):
    """Session snapshot serialised for network transmission."""

    components: tuple[str, str, str]
    stint: int
    samples: int
    active: Dict[str, RecursivityMicroStateSnapshot]
    history: Tuple[RecursivityNetworkHistoryEntry, ...]


class RecursivityNetworkMemory(TypedDict, total=False):
    """Root payload returned by :func:`extract_network_memory`."""

    active_session: str | None
    sessions: Dict[str, RecursivityNetworkSession]


WHEEL_TEMPERATURE_KEYS = (
    "tyre_temp_fl",
    "tyre_temp_fr",
    "tyre_temp_rl",
    "tyre_temp_rr",
)


def _normalise_session_identifier(
    session_id: object,
) -> tuple[str, tuple[str, str, str]]:
    """Return a canonical session key and the associated components."""

    defaults = ("generic", "unknown", "default")

    def _normalise(value: object, fallback: str) -> str:
        if value is None:
            return fallback
        token = str(value).strip()
        return token or fallback

    if isinstance(session_id, MappingABC):
        components = (
            _normalise(session_id.get("car_model") or session_id.get("car"), defaults[0]),
            _normalise(session_id.get("track_name") or session_id.get("track"), defaults[1]),
            _normalise(
                session_id.get("tyre_compound") or session_id.get("compound"),
                defaults[2],
            ),
        )
        return "|".join(components), components

    if isinstance(session_id, SequenceABC) and not isinstance(session_id, (str, bytes)):
        values = list(session_id)[:3]
        padded = list(defaults)
        for index, value in enumerate(values):
            padded[index] = _normalise(value, defaults[index])
        components = tuple(padded)
        return "|".join(components), components

    if isinstance(session_id, str):
        parts = [part.strip() for part in session_id.split("|")]
        padded = list(defaults)
        for index, value in enumerate(parts[:3]):
            padded[index] = _normalise(value, defaults[index])
        components = tuple(padded)
        return "|".join(components), components

    components = (
        _normalise(session_id, defaults[0]),
        defaults[1],
        defaults[2],
    )
    return "|".join(components), components


def _ensure_recursivity_sessions(
    state: RecursivityStateRoot,
) -> MutableMapping[str, RecursivitySessionState]:
    sessions_obj = state.get("sessions")
    if not isinstance(sessions_obj, MutableMappingABC):
        sessions: MutableMapping[str, RecursivitySessionState] = {}
        legacy_candidates = [
            key
            for key, value in list(state.items())
            if key not in {"sessions", "active_session"} and isinstance(value, MappingABC)
        ]
        if legacy_candidates:
            legacy_active: MutableMapping[str, RecursivityMicroState] = {}
            for key in legacy_candidates:
                value = state.get(key)
                if isinstance(value, MappingABC):
                    legacy_active[str(key)] = cast(RecursivityMicroState, dict(value))
            for key in legacy_candidates:
                state.pop(key, None)
            legacy_session: RecursivitySessionState = {
                "active": legacy_active,
                "history": [],
                "samples": sum(
                    int(cast(Mapping[str, object], entry).get("samples", 0))
                    for entry in legacy_active.values()
                    if isinstance(entry, MappingABC)
                ),
                "stint_index": 0,
                "components": ("legacy", "unknown", "default"),
            }
            sessions["legacy|unknown|default"] = legacy_session
            state["active_session"] = "legacy|unknown|default"
        state["sessions"] = sessions
    else:
        sessions = cast(MutableMapping[str, RecursivitySessionState], sessions_obj)
    state.setdefault("active_session", None)
    return sessions


def _initialise_session_entry(
    entry: RecursivitySessionState,
    components: tuple[str, str, str],
) -> RecursivitySessionState:
    entry.setdefault("active", cast(MutableMapping[str, RecursivityMicroState], {}))
    entry.setdefault("history", [])
    entry.setdefault("samples", 0)
    entry.setdefault("stint_index", 0)
    entry.setdefault("last_timestamp", None)
    entry.setdefault("components", components)
    return entry


def _serialise_mapping(mapping: Mapping[str, object]) -> Dict[str, object]:
    serialised: Dict[str, object] = {}
    for key, value in mapping.items():
        if isinstance(value, (int, float)):
            serialised[key] = float(value)
        else:
            serialised[key] = value
    return serialised


def _snapshot_micro_state(
    micro_state: RecursivityMicroState,
) -> RecursivityMicroStateSnapshot:
    filtered = cast(Dict[str, float], micro_state.get("filtered", {}) or {})
    trace_entries = micro_state.get("trace", []) or []
    trace_snapshot = []
    for entry in trace_entries:
        if isinstance(entry, MappingABC):
            trace_snapshot.append(_serialise_mapping(entry))
    snapshot: RecursivityMicroStateSnapshot = {
        "phase": micro_state.get("phase"),
        "samples": int(micro_state.get("samples", 0)),
        "filtered": {
            key: float(value)
            for key, value in filtered.items()
            if isinstance(value, (int, float))
        },
        "trace": tuple(trace_snapshot),
        "converged": bool(micro_state.get("converged", False)),
        "last_measures": _serialise_mapping(
            micro_state.get("last_measures", {}) or {}
        ),
    }
    return snapshot


def _finalise_session_state(
    session_state: RecursivitySessionState,
    reason: str,
    timestamp: float | None,
) -> None:
    active = session_state.get("active")
    if not isinstance(active, MappingABC) or not active:
        session_state["samples"] = 0
        session_state["last_timestamp"] = timestamp if timestamp is not None else None
        return
    history_entry: RecursivityHistoryEntry = {
        "stint": int(session_state.get("stint_index", 0)),
        "ended_at": float(timestamp) if timestamp is not None else None,
        "reason": str(reason or ""),
        "samples": int(session_state.get("samples", 0)),
        "microsectors": {
            micro_id: _snapshot_micro_state(cast(RecursivityMicroState, micro_state))
            for micro_id, micro_state in active.items()
            if isinstance(micro_state, MappingABC)
        },
    }
    history = session_state.setdefault(
        "history", cast(List[RecursivityHistoryEntry], [])
    )
    if isinstance(history, list):
        history.append(history_entry)
    session_state["active"] = {}
    session_state["samples"] = 0
    session_state["stint_index"] = int(session_state.get("stint_index", 0)) + 1
    session_state["last_timestamp"] = timestamp if timestamp is not None else None


def _ensure_trace_log(
    micro_state: RecursivityMicroState,
    history: int,
) -> Deque[Mapping[str, float | str | None]]:
    trace_store = micro_state.get("trace")
    if isinstance(trace_store, deque):
        if trace_store.maxlen != history:
            trace_log: Deque[Mapping[str, float | str | None]] = deque(trace_store, maxlen=history)
            micro_state["trace"] = trace_log
            return trace_log
        return trace_store
    if isinstance(trace_store, SequenceABC) and not isinstance(trace_store, (str, bytes, bytearray)):
        trace_log = deque(trace_store, maxlen=history)
        micro_state["trace"] = trace_log
        return trace_log
    trace_log = deque(maxlen=history)
    micro_state["trace"] = trace_log
    return trace_log


def _update_rolling_buffers(
    micro_state: RecursivityMicroState,
    numeric_keys: Sequence[str],
    filtered_values: Mapping[str, float],
    *,
    convergence_window: int,
    convergence_threshold: float,
) -> bool:
    rolling_buffers = cast(
        MutableMapping[str, Deque[float]],
        micro_state.setdefault("_rolling", {}),
    )
    if numeric_keys:
        for key in numeric_keys:
            buffer = rolling_buffers.setdefault(key, deque(maxlen=convergence_window))
            buffer.append(filtered_values[key])
    candidate_buffers = [
        buffer for buffer in rolling_buffers.values() if isinstance(buffer, deque)
    ]
    micro_converged = False
    for buffer in candidate_buffers:
        if len(buffer) >= convergence_window:
            if max(buffer) - min(buffer) <= convergence_threshold:
                micro_converged = True
                break
    if int(micro_state.get("samples", 0)) < convergence_window:
        micro_converged = False
    micro_state["converged"] = micro_converged
    return micro_converged


def recursivity_operator(
    state: RecursivityStateRoot,
    session_id: Mapping[str, object] | Sequence[object] | str | None,
    microsector_id: str,
    measures: Mapping[str, float | str],
    *,
    decay: float = 0.4,
    history: int = 20,
    max_samples: int = 600,
    max_time_gap: float = 60.0,
    convergence_window: int = 5,
    convergence_threshold: float = 0.02,
) -> RecursivityOperatorResult:
    """Maintain recursive thermal/style state per session and microsector."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in the [0, 1) interval")
    if not microsector_id:
        raise ValueError("microsector_id must be a non-empty string")
    if history <= 0:
        raise ValueError("history must be a positive integer")
    if max_samples <= 0:
        raise ValueError("max_samples must be a positive integer")
    if convergence_window <= 0:
        raise ValueError("convergence_window must be a positive integer")
    if max_time_gap < 0.0:
        raise ValueError("max_time_gap must be non-negative")

    session_key, components = _normalise_session_identifier(session_id)
    sessions = _ensure_recursivity_sessions(state)
    session_entry = sessions.setdefault(
        session_key, cast(RecursivitySessionState, {})
    )
    session_state = _initialise_session_entry(session_entry, components)
    state["active_session"] = session_key

    timestamp_raw = measures.get("timestamp")
    timestamp = float(timestamp_raw) if isinstance(timestamp_raw, (int, float)) else None

    last_timestamp = session_state.get("last_timestamp")
    if (
        timestamp is not None
        and isinstance(last_timestamp, (int, float))
        and session_state.get("active")
        and max_time_gap > 0.0
    ):
        gap = timestamp - float(last_timestamp)
        if gap >= max_time_gap:
            _finalise_session_state(session_state, "time_gap", float(last_timestamp))

    active_map = session_state.setdefault(
        "active", cast(MutableMapping[str, RecursivityMicroState], {})
    )
    micro_state = cast(
        RecursivityMicroState,
        active_map.setdefault(
            microsector_id,
            cast(
                RecursivityMicroState,
                {
                    "filtered": {},
                    "phase": None,
                    "samples": 0,
                    "trace": deque(maxlen=history),
                    "last_measures": {},
                    "_rolling": {},
                    "converged": False,
                    "last_timestamp": None,
                },
            ),
        ),
    )

    phase = measures.get("phase") if isinstance(measures.get("phase"), str) else None
    previous_phase = micro_state.get("phase")
    phase_changed = (
        phase is not None and previous_phase is not None and phase != previous_phase
    )
    if phase is not None:
        micro_state["phase"] = phase

    filtered_store = cast(Dict[str, float], micro_state.setdefault("filtered", {}))
    filtered_values: Dict[str, float] = {}
    numeric_keys = [
        key
        for key, value in measures.items()
        if key not in {"phase", "timestamp"} and isinstance(value, (int, float))
    ]
    previous_filtered_snapshot: Dict[str, float | None] = {}
    for key in numeric_keys:
        value = float(measures[key])
        previous = filtered_store.get(key)
        if isinstance(previous, (int, float)):
            previous_filtered_snapshot[key] = float(previous)
        else:
            previous_filtered_snapshot[key] = None
        if previous is None or phase_changed:
            filtered = value
        else:
            filtered = (decay * float(previous)) + ((1.0 - decay) * value)
        filtered_store[key] = filtered
        filtered_values[key] = filtered

    if timestamp is not None:
        last_timestamp = micro_state.get("last_timestamp")
        dt = None
        if isinstance(last_timestamp, (int, float)):
            dt = timestamp - float(last_timestamp)
        if dt is not None and dt > 1e-9 and not phase_changed:
            for key in WHEEL_TEMPERATURE_KEYS:
                if key not in filtered_values:
                    continue
                previous_value = previous_filtered_snapshot.get(key)
                if previous_value is None:
                    continue
                derivative = (filtered_values[key] - previous_value) / dt
                derivative_key = f"{key}_dt"
                filtered_store[derivative_key] = derivative
                filtered_values[derivative_key] = derivative
        micro_state["last_timestamp"] = timestamp

    micro_state["samples"] = int(micro_state.get("samples", 0)) + 1
    trace_log = _ensure_trace_log(micro_state, history)
    trace_log.append({"phase": micro_state.get("phase"), **filtered_values})

    _update_rolling_buffers(
        micro_state,
        numeric_keys,
        filtered_values,
        convergence_window=convergence_window,
        convergence_threshold=convergence_threshold,
    )

    micro_state["last_measures"] = _serialise_mapping(dict(measures))

    session_state["samples"] = int(session_state.get("samples", 0)) + 1
    if timestamp is not None:
        session_state["last_timestamp"] = timestamp

    state_snapshot = _snapshot_micro_state(micro_state)

    rollover_reason: str | None = None
    if session_state["samples"] >= max_samples:
        rollover_reason = "max_samples"
    elif (
        session_state.get("active")
        and all(
            isinstance(entry, MappingABC) and entry.get("converged")
            for entry in session_state["active"].values()
        )
    ):
        rollover_reason = "convergence"

    if rollover_reason:
        _finalise_session_state(session_state, rollover_reason, timestamp)

    session_components = cast(
        tuple[str, str, str],
        session_state.get("components", components),
    )
    result: RecursivityOperatorResult = {
        "session": session_key,
        "session_components": session_components,
        "stint": int(session_state.get("stint_index", 0)),
        "microsector_id": microsector_id,
        "phase": state_snapshot.get("phase"),
        "filtered": state_snapshot["filtered"],
        "samples": state_snapshot["samples"],
        "phase_changed": phase_changed,
        "trace": state_snapshot["trace"],
        "state": state_snapshot,
        "converged": bool(state_snapshot.get("converged", False)),
        "stint_completed": rollover_reason,
        "timestamp": timestamp,
    }
    return result


def extract_network_memory(
    operator_state: Mapping[str, Mapping[str, object]] | None,
) -> RecursivityNetworkMemory:
    """Build a serialisable snapshot of the recursivity session memory."""

    if not operator_state:
        return {}
    rec_state_raw = operator_state.get("recursivity")
    if not isinstance(rec_state_raw, MappingABC):
        return {}
    rec_state = cast(RecursivityStateRoot, rec_state_raw)
    sessions = rec_state.get("sessions")
    if not isinstance(sessions, MappingABC):
        return {}

    payload_sessions: Dict[str, RecursivityNetworkSession] = {}
    for session_key, session_entry_raw in sessions.items():
        if not isinstance(session_entry_raw, MappingABC):
            continue
        session_entry = cast(RecursivitySessionState, session_entry_raw)
        active_state = session_entry.get("active", {})
        history_state = session_entry.get("history", [])
        components = cast(
            tuple[str, str, str],
            tuple(session_entry.get("components", ())),
        )
        session_payload: RecursivityNetworkSession = {
            "components": components,
            "stint": int(session_entry.get("stint_index", 0)),
            "samples": int(session_entry.get("samples", 0)),
            "active": {},
            "history": (),
        }
        active_payload: Dict[str, RecursivityMicroStateSnapshot] = {}
        if isinstance(active_state, MappingABC):
            for micro_id, micro_state in active_state.items():
                if not isinstance(micro_state, MappingABC):
                    continue
                snapshot = _snapshot_micro_state(
                    cast(RecursivityMicroState, micro_state)
                )
                active_payload[str(micro_id)] = snapshot
        session_payload["active"] = active_payload

        history_payload: List[RecursivityNetworkHistoryEntry] = []
        if isinstance(history_state, SequenceABC):
            for history_entry in history_state:
                if not isinstance(history_entry, MappingABC):
                    continue
                micro_map = history_entry.get("microsectors", {})
                micro_payload: Dict[str, RecursivityMicroStateSnapshot] = {}
                if isinstance(micro_map, MappingABC):
                    for micro_id, micro_state in micro_map.items():
                        if isinstance(micro_state, MappingABC):
                            micro_payload[str(micro_id)] = cast(
                                RecursivityMicroStateSnapshot,
                                {
                                    key: value for key, value in micro_state.items()
                                },
                            )
                history_payload.append(
                    {
                        "stint": int(history_entry.get("stint", 0)),
                        "ended_at": (
                            float(history_entry.get("ended_at"))
                            if isinstance(history_entry.get("ended_at"), (int, float))
                            else None
                        ),
                        "reason": str(history_entry.get("reason", "")),
                        "samples": int(history_entry.get("samples", 0)),
                        "microsectors": micro_payload,
                    }
                )
        session_payload["history"] = tuple(history_payload)
        payload_sessions[str(session_key)] = session_payload

    return {
        "active_session": rec_state.get("active_session"),
        "sessions": payload_sessions,
    }


__all__ = [
    "RecursivityMicroState",
    "RecursivityMicroStateSnapshot",
    "RecursivityHistoryEntry",
    "RecursivitySessionState",
    "RecursivityStateRoot",
    "RecursivityOperatorResult",
    "RecursivityNetworkHistoryEntry",
    "RecursivityNetworkSession",
    "RecursivityNetworkMemory",
    "recursivity_operator",
    "extract_network_memory",
]
