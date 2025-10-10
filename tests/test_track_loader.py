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


def test_load_track_parses_configs(mini_track_pack: MiniTrackPack) -> None:
    track = load_track(mini_track_pack.track_slug, tracks_dir=mini_track_pack.tracks_dir)

    assert track.slug == mini_track_pack.track_slug
    assert isinstance(track.configs, MappingProxyType)
    assert mini_track_pack.layout_code in track.configs

    layout = track.configs[mini_track_pack.layout_code]
    assert isinstance(layout, TrackConfig)
    assert layout.name == "Mini Aston Historic"
    assert layout.length_km is not None and math.isclose(layout.length_km, 5.2)
    assert layout.surface == "asphalt"
    assert layout.track_profile == mini_track_pack.track_profile
    assert layout.extras["pit_boxes"] == 32
    assert layout.extras["notes"] == ("tight chicane", "long straight")


def test_load_track_resolves_alias_layout(mini_track_pack: MiniTrackPack) -> None:
    track = load_track(mini_track_pack.track_slug, tracks_dir=mini_track_pack.tracks_dir)

    assert "AS3R" in track.configs

    base = track.configs["AS3"]
    alias = track.configs["AS3R"]

    assert alias.name == "Mini Aston Historic Reverse"
    assert alias.length_km == base.length_km
    assert alias.surface == base.surface
    assert alias.track_profile == base.track_profile
    assert alias.extras == base.extras
    assert "alias_of" not in alias.extras


def test_load_track_profiles_normalises_weights(mini_track_pack: MiniTrackPack) -> None:
    profiles = load_track_profiles(mini_track_pack.track_profiles_dir)

    assert mini_track_pack.track_profile in profiles
    profile = profiles[mini_track_pack.track_profile]

    weights = profile["weights"]
    assert isinstance(weights, MappingProxyType)
    assert math.isclose(weights["entry"]["__default__"], 1.0)
    assert math.isclose(weights["entry"]["brakes"], 1.05)
    assert math.isclose(weights["apex"]["anti_roll"], 1.1)

    hints = profile["hints"]
    assert hints["microsector_span"] == "compact"
    assert hints["surface_bias"]["entry"] == 0.2
    assert hints["notes"] == ("aggressive hairpins", "short braking")


def test_load_modifiers_indexed_by_pair(mini_track_pack: MiniTrackPack) -> None:
    modifiers = load_modifiers(mini_track_pack.modifiers_dir)

    key = (mini_track_pack.car_profile, mini_track_pack.track_profile)
    assert key in modifiers

    modifier = modifiers[key]
    weights = modifier["scale"]["weights"]
    assert math.isclose(weights["__default__"]["__default__"], 1.1)
    assert math.isclose(weights["entry"]["brakes"], 1.4)
    assert math.isclose(weights["exit"]["__default__"], 0.95)
    assert modifier["hints"]["slip_ratio_bias"] == "aggressive"


def test_assemble_session_weights_applies_modifier(mini_track_pack: MiniTrackPack) -> None:
    profiles = load_track_profiles(mini_track_pack.track_profiles_dir)
    modifiers = load_modifiers(mini_track_pack.modifiers_dir)

    session = assemble_session_weights(
        mini_track_pack.car_profile,
        mini_track_pack.track_profile,
        track_profiles=profiles,
        modifiers=modifiers,
    )

    weights = session["weights"]
    assert math.isclose(weights["entry"]["__default__"], 1.0 * 1.25)
    assert math.isclose(weights["entry"]["brakes"], 1.05 * 1.4)
    assert math.isclose(weights["apex"]["__default__"], 0.95 * 1.1)
    assert math.isclose(weights["exit"]["__default__"], 0.9 * 0.95)
    assert math.isclose(weights["exit"]["differential"], 1.2 * 1.3)

    hints = session["hints"]
    assert hints["microsector_span"] == "compact"
    assert hints["slip_ratio_bias"] == "aggressive"
    assert hints["surface"] == "asphalt"


def test_assemble_session_weights_without_modifier(mini_track_pack: MiniTrackPack) -> None:
    profiles = load_track_profiles(mini_track_pack.track_profiles_dir)

    session = assemble_session_weights(
        "unknown",
        mini_track_pack.track_profile,
        track_profiles=profiles,
        modifiers={},
    )

    weights = session["weights"]
    assert math.isclose(weights["entry"]["__default__"], 1.0)
    assert math.isclose(weights["exit"]["differential"], 1.2)

