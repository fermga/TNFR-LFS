from __future__ import annotations

import importlib
from pathlib import Path
from textwrap import dedent

import pytest

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

from tnfr_lfs.core.cache_settings import DEFAULT_DYNAMIC_CACHE_SIZE
from tnfr_lfs.ingestion.config_loader import (
    Car,
    CacheOptions,
    example_pipeline,
    load_cars,
    load_lfs_class_overrides,
    load_profiles,
    parse_cache_options,
    resolve_targets,
    _deep_merge,
)
from tnfr_lfs._pack_resources import data_root


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


def test_parse_cache_options_defaults() -> None:
    options = parse_cache_options({})
    assert isinstance(options, CacheOptions)
    assert options.enable_delta_cache is True
    assert options.nu_f_cache_size == DEFAULT_DYNAMIC_CACHE_SIZE
    assert options.recommender_cache_size == DEFAULT_DYNAMIC_CACHE_SIZE
    assert options.telemetry_cache_size == DEFAULT_DYNAMIC_CACHE_SIZE


def test_parse_cache_options_overrides() -> None:
    raw = {"performance": {"cache_enabled": False, "max_cache_size": 16}}
    options = parse_cache_options(raw)
    assert options.enable_delta_cache is False
    assert options.nu_f_cache_size == 0
    assert options.recommender_cache_size == 0
    assert options.telemetry_cache_size == 0


def test_parse_cache_options_supports_legacy_cache_section() -> None:
    raw = {
        "cache": {
            "cache_enabled": "true",
            "nu_f_cache_size": "64",
            "telemetry": {"telemetry_cache_size": "12"},
        }
    }

    options = parse_cache_options(raw)

    assert options.enable_delta_cache is True
    assert options.nu_f_cache_size == 64
    assert options.recommender_cache_size == 64
    assert options.telemetry_cache_size == 12


def test_parse_cache_options_uses_pack_defaults(tmp_path: Path) -> None:
    from tnfr_lfs import _pack_resources

    pack_root = tmp_path / "pack"
    (pack_root / "config").mkdir(parents=True)
    (pack_root / "data").mkdir()
    pack_root.joinpath("config", "global.toml").write_text("", encoding="utf8")

    _pack_resources.set_pack_root_override(pack_root)
    config_loader_module = importlib.import_module("tnfr_lfs.ingestion.config_loader")
    try:
        config_loader = importlib.reload(config_loader_module)
        options = config_loader.parse_cache_options({})
        assert options.enable_delta_cache is True
        assert options.nu_f_cache_size == DEFAULT_DYNAMIC_CACHE_SIZE
        assert options.recommender_cache_size == DEFAULT_DYNAMIC_CACHE_SIZE
        assert options.telemetry_cache_size == DEFAULT_DYNAMIC_CACHE_SIZE
    finally:
        _pack_resources.set_pack_root_override(None)
        importlib.reload(config_loader_module)


def test_load_cars_uses_packaged_resources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tnfr_lfs import _pack_resources

    pack_root = tmp_path / "pack"
    cars_dir = pack_root / "data" / "cars"
    cars_dir.mkdir(parents=True)
    cars_dir.joinpath("AAA.toml").write_text(
        dedent(
            """
            abbrev = "AAA"
            name = "Alpha"
            license = "demo"
            engine_layout = "front"
            drive = "RWD"
            weight_kg = 900
            wheel_rotation_group_deg = 30
            profile = "default"
            """
        )
    )

    _pack_resources.set_pack_root_override(pack_root)
    config_loader_module = importlib.import_module("tnfr_lfs.ingestion.config_loader")
    config_loader = importlib.reload(config_loader_module)

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    monkeypatch.chdir(workspace)
    try:
        cars = config_loader.load_cars()
    finally:
        _pack_resources.set_pack_root_override(None)
        importlib.reload(config_loader_module)

    assert set(cars) == {"AAA"}
