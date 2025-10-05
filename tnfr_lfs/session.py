"""Helpers to expose session weight and hint summaries."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple


def _format_weight_profile(profile: Any) -> str | None:
    if isinstance(profile, Mapping):
        entries: list[str] = []
        default = profile.get("__default__")
        if isinstance(default, (int, float)):
            entries.append(f"base×{float(default):.2f}")
        for key in sorted(profile):
            if key == "__default__":
                continue
            value = profile[key]
            try:
                entries.append(f"{key}×{float(value):.2f}")
            except (TypeError, ValueError):
                continue
        if entries:
            return " ".join(entries[:3])
        return None
    try:
        value = float(profile)
    except (TypeError, ValueError):
        return None
    return f"base×{value:.2f}"


def _format_weights(weights: Mapping[str, Any] | None) -> str | None:
    if not isinstance(weights, Mapping):
        return None
    segments: list[str] = []
    for phase in ("entry", "apex", "exit"):
        profile = weights.get(phase)
        formatted = _format_weight_profile(profile)
        if formatted:
            segments.append(f"{phase}: {formatted}")
    for phase, profile in weights.items():
        if phase in {"entry", "apex", "exit"}:
            continue
        formatted = _format_weight_profile(profile)
        if formatted:
            segments.append(f"{phase}: {formatted}")
    if not segments:
        return None
    return " · ".join(segments[:3])


def _format_hints(hints: Mapping[str, Any] | None) -> str | None:
    if not isinstance(hints, Mapping):
        return None
    entries: list[str] = []
    for key, value in hints.items():
        if isinstance(value, Mapping):
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            subset = [str(item) for item in list(value)[:3] if item is not None]
            if subset:
                entries.append(f"{key}={', '.join(subset)}")
            continue
        entries.append(f"{key}={value}")
    if not entries:
        return None
    return " · ".join(entries[:4])


def format_session_messages(session: Mapping[str, Any] | None) -> Tuple[str, ...]:
    """Return short messages describing the active session weights and hints."""

    if not session:
        return ()
    track_profile = session.get("track_profile") if isinstance(session, Mapping) else None
    if isinstance(track_profile, str) and track_profile.strip():
        title = f"Perfil {track_profile.strip()}"
    else:
        title = "Perfil sesión"
    messages: list[str] = []
    weights_line = _format_weights(session.get("weights")) if isinstance(session, Mapping) else None
    if weights_line:
        messages.append(f"{title} pesos → {weights_line}")
    hints_line = _format_hints(session.get("hints")) if isinstance(session, Mapping) else None
    if hints_line:
        messages.append(f"{title} hints → {hints_line}")
    return tuple(messages)


__all__ = ["format_session_messages"]
