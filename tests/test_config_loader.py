from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from tnfr_lfs.config_loader import (
    Car,
    example_pipeline,
    load_cars,
    load_profiles,
    resolve_targets,
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

    return pack_root


def test_load_cars_indexed_by_abbrev(config_pack: Path) -> None:
    cars = load_cars(config_pack / "cars")

    assert set(cars) == {"ABC", "DEF"}
    assert isinstance(cars["ABC"], Car)


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


def test_example_pipeline_accepts_custom_data_root(config_pack: Path) -> None:
    resolved = example_pipeline("ABC", data_root=config_pack)

    assert resolved["targets"]["balance"]["delta_nfr"] == 0.1
