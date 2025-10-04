"""Tests for :mod:`tnfr_lfs.track_loader`."""

from __future__ import annotations

import math
from types import MappingProxyType

from tnfr_lfs.track_loader import (
    TrackConfig,
    assemble_session_weights,
    load_modifiers,
    load_track,
    load_track_profiles,
)


def test_load_track_parses_configs() -> None:
    track = load_track("SO")

    assert track.slug == "SO"
    assert isinstance(track.configs, MappingProxyType)
    assert "SO1" in track.configs

    layout = track.configs["SO1"]
    assert isinstance(layout, TrackConfig)
    assert layout.name == "South City Classic"
    assert layout.length_km is not None and math.isclose(layout.length_km, 2.0)
    assert layout.surface == "asphalt"
    assert layout.track_profile == "p_technical_short"
    assert dict(layout.extras) == {}


def test_load_track_profiles_normalises_weights() -> None:
    profiles = load_track_profiles()

    assert "p_balanced_medium" in profiles
    profile = profiles["p_balanced_medium"]

    weights = profile["weights"]
    assert isinstance(weights, MappingProxyType)
    assert math.isclose(weights["entry"]["__default__"], 1.0)
    assert math.isclose(weights["entry"]["brakes"], 1.05)

    hints = profile["hints"]
    assert hints["microsector_span"] == "standard"


def test_load_modifiers_indexed_by_pair() -> None:
    modifiers = load_modifiers()

    key = ("gtr_mid", "p_balanced_medium")
    assert key in modifiers

    modifier = modifiers[key]
    weights = modifier["scale"]["weights"]
    assert math.isclose(weights["entry"]["__default__"], 1.02)
    assert math.isclose(weights["entry"]["brakes"], 1.08)
    assert modifier["hints"]["slip_ratio_bias"] == "neutral"


def test_assemble_session_weights_applies_modifier() -> None:
    profiles = load_track_profiles()
    modifiers = load_modifiers()

    session = assemble_session_weights(
        "gtr_mid",
        "p_balanced_medium",
        track_profiles=profiles,
        modifiers=modifiers,
    )

    weights = session["weights"]
    assert math.isclose(weights["entry"]["__default__"], 1.02)
    assert math.isclose(weights["entry"]["brakes"], 1.05 * 1.08)
    assert math.isclose(weights["exit"]["__default__"], 1.03)

    hints = session["hints"]
    assert hints["microsector_span"] == "standard"
    assert hints["slip_ratio_bias"] == "neutral"


def test_assemble_session_weights_without_modifier() -> None:
    profiles = load_track_profiles()

    session = assemble_session_weights(
        "unknown",
        "p_balanced_medium",
        track_profiles=profiles,
        modifiers={},
    )

    weights = session["weights"]
    assert math.isclose(weights["entry"]["__default__"], 1.0)
    assert math.isclose(weights["exit"]["differential"], 1.1)

