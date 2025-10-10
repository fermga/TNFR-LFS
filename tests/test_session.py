"""Tests for session summary helpers."""

from tnfr_lfs.cli import format_session_messages


def test_format_session_messages_uses_english_defaults() -> None:
    session = {
        "weights": {"entry": {"__default__": 1.0}},
        "hints": {"foco": "estabilidad"},
    }
    messages = format_session_messages(session)
    assert messages
    assert messages[0].startswith("Session profile weights → ")
    assert any(message.startswith("Session profile hints → ") for message in messages)


def test_format_session_messages_respects_track_profile() -> None:
    session = {
        "track_profile": "BL1",
        "weights": {"entry": {"__default__": 1.0}},
    }
    messages = format_session_messages(session)
    assert messages
    assert messages[0].startswith("Profile BL1 weights → ")
