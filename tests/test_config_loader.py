from __future__ import annotations

import importlib
from pathlib import Path

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
from tests.helpers import MINIMAL_DATA_CAR, create_config_pack, pack_builder


def test_load_cars_indexed_by_abbrev(tmp_path: Path) -> None:
    pack_root = create_config_pack(tmp_path / "pack")
    cars = load_cars(pack_root / "cars")

    assert set(cars) == {"ABC", "DEF", "GHI"}
    assert isinstance(cars["ABC"], Car)
    assert cars["ABC"].lfs_class == "STD"
    assert cars["DEF"].lfs_class == "Unclassified"
    assert cars["GHI"].lfs_class == "GT"


def test_load_profiles_prefers_meta_id(tmp_path: Path) -> None:
    pack_root = create_config_pack(tmp_path / "pack")
    profiles = load_profiles(pack_root / "profiles")

    assert "custom-profile" in profiles
    assert "fallback" in profiles


def test_resolve_targets_missing_profile(tmp_path: Path) -> None:
    pack_root = create_config_pack(tmp_path / "pack")
    cars = load_cars(pack_root / "cars")
    profiles = load_profiles(pack_root / "profiles")

    with pytest.raises(KeyError):
        resolve_targets("DEF", cars, profiles)


def test_resolve_targets_returns_expected_sections(tmp_path: Path) -> None:
    pack_root = create_config_pack(tmp_path / "pack")
    cars = load_cars(pack_root / "cars")
    profiles = load_profiles(pack_root / "profiles")

    resolved = resolve_targets("ABC", cars, profiles)

    assert set(resolved.keys()) == {"meta", "targets", "policy", "recommender"}
    assert resolved["meta"]["category"] == "road"


def test_resolve_targets_applies_class_overrides(tmp_path: Path) -> None:
    pack_root = create_config_pack(tmp_path / "pack")
    cars = load_cars(pack_root / "cars")
    profiles = load_profiles(pack_root / "profiles")
    overrides = load_lfs_class_overrides(pack_root / "lfs_class_overrides.toml")

    resolved = resolve_targets("ABC", cars, profiles, overrides=overrides)

    assert resolved["targets"]["balance"]["delta_nfr"] == pytest.approx(0.3)
    assert resolved["policy"]["steering"]["aggressiveness"] == pytest.approx(0.75)
    assert resolved["recommender"]["steering"]["kp"] == pytest.approx(2.5)


def test_example_pipeline_accepts_custom_data_root(tmp_path: Path) -> None:
    pack_root = create_config_pack(tmp_path / "pack")
    resolved = example_pipeline("ABC", data_root=pack_root)

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

    builder = pack_builder(tmp_path / "pack")
    builder.add_config("")
    builder.ensure_dir("data")

    _pack_resources.set_pack_root_override(builder.root)
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

    builder = pack_builder(tmp_path / "pack")
    builder.add_car("AAA", MINIMAL_DATA_CAR, in_data=True)

    _pack_resources.set_pack_root_override(builder.root)
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
