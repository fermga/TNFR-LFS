from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from tnfr_lfs.config_loader import (
    Car,
    example_pipeline,
    load_cars,
    load_lfs_class_overrides,
    load_profiles,
    resolve_targets,
    _deep_merge,
)


@pytest.fixture()
def config_pack(tmp_path: Path) -> Path:
    pack_root = tmp_path / "pack"
    cars_dir = pack_root / "cars"
    profiles_dir = pack_root / "profiles"
    cars_dir.mkdir(parents=True)
    profiles_dir.mkdir()

    cars_dir.joinpath("ABC.toml").write_text(
        dedent(
            """
            abbrev = "ABC"
            name = "Alpha"
            license = "demo"
            engine_layout = "front"
            drive = "RWD"
            weight_kg = 900
            wheel_rotation_group_deg = 30
            profile = "custom-profile"
            lfs_class = "STD"
            """
        )
    )

    cars_dir.joinpath("DEF.toml").write_text(
        dedent(
            """
            abbrev = "DEF"
            name = "Delta"
            license = "s3"
            engine_layout = "mid"
            drive = "AWD"
            weight_kg = 1000
            wheel_rotation_group_deg = 28
            profile = "missing-profile"
            lfs_class = "Unclassified"
            """
        )
    )

    cars_dir.joinpath("GHI.toml").write_text(
        dedent(
            """
            abbrev = "GHI"
            name = "Gamma"
            license = "demo"
            engine_layout = "rear"
            drive = "RWD"
            weight_kg = 950
            wheel_rotation_group_deg = 32
            profile = "fallback"
            lfs_class = "GT"
            """
        )
    )

    profiles_dir.joinpath("custom.toml").write_text(
        dedent(
            """
            [meta]
            id = "custom-profile"
            category = "road"

            [targets.balance]
            delta_nfr = 0.1

            [policy.steering]
            aggressiveness = 0.5

            [recommender.steering]
            kp = 1.0
            """
        )
    )

    profiles_dir.joinpath("fallback.toml").write_text(
        dedent(
            """
            [meta]
            category = "race"

            [targets.balance]
            delta_nfr = 0.2

            [policy.steering]
            aggressiveness = 0.6

            [recommender.steering]
            kp = 1.2
            """
        )
    )

    pack_root.joinpath("lfs_class_overrides.toml").write_text(
        dedent(
            """
            ["STD".overrides.targets.balance]
            delta_nfr = 0.3

            ["STD".overrides.policy.steering]
            aggressiveness = 0.75

            ["STD".overrides.recommender.steering]
            kp = 2.5
            """
        )
    )

    return pack_root


def test_load_cars_indexed_by_abbrev(config_pack: Path) -> None:
    cars = load_cars(config_pack / "cars")

    assert set(cars) == {"ABC", "DEF", "GHI"}
    assert isinstance(cars["ABC"], Car)
    assert cars["ABC"].lfs_class == "STD"
    assert cars["DEF"].lfs_class == "Unclassified"
    assert cars["GHI"].lfs_class == "GT"


def test_load_profiles_prefers_meta_id(config_pack: Path) -> None:
    profiles = load_profiles(config_pack / "profiles")

    assert "custom-profile" in profiles
    assert "fallback" in profiles


def test_resolve_targets_missing_profile(config_pack: Path) -> None:
    cars = load_cars(config_pack / "cars")
    profiles = load_profiles(config_pack / "profiles")

    with pytest.raises(KeyError):
        resolve_targets("DEF", cars, profiles)


def test_resolve_targets_returns_expected_sections(config_pack: Path) -> None:
    cars = load_cars(config_pack / "cars")
    profiles = load_profiles(config_pack / "profiles")

    resolved = resolve_targets("ABC", cars, profiles)

    assert set(resolved.keys()) == {"meta", "targets", "policy", "recommender"}
    assert resolved["meta"]["category"] == "road"


def test_resolve_targets_applies_class_overrides(config_pack: Path) -> None:
    cars = load_cars(config_pack / "cars")
    profiles = load_profiles(config_pack / "profiles")
    overrides = load_lfs_class_overrides(config_pack / "lfs_class_overrides.toml")

    resolved = resolve_targets("ABC", cars, profiles, overrides=overrides)

    assert resolved["targets"]["balance"]["delta_nfr"] == pytest.approx(0.3)
    assert resolved["policy"]["steering"]["aggressiveness"] == pytest.approx(0.75)
    assert resolved["recommender"]["steering"]["kp"] == pytest.approx(2.5)


def test_example_pipeline_accepts_custom_data_root(config_pack: Path) -> None:
    resolved = example_pipeline("ABC", data_root=config_pack)

    assert resolved["targets"]["balance"]["delta_nfr"] == pytest.approx(0.3)
