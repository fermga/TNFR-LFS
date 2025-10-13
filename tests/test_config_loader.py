from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import pytest

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib  # type: ignore

from tnfr_core.cache_settings import DEFAULT_DYNAMIC_CACHE_SIZE
from tnfr_lfs.telemetry.config_loader import (
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
from tnfr_lfs.resources import data_root
from tests.helpers import MINIMAL_DATA_CAR, create_config_pack, pack_builder


@dataclass(frozen=True)
class LoadedConfigPack:
    """Container for config pack artifacts shared across tests."""

    root: Path
    cars: Mapping[str, Car]
    profiles: Mapping[str, Mapping[str, object]]
    overrides: Mapping[str, object]


@pytest.fixture(scope="module")
def config_pack(tmp_path_factory: pytest.TempPathFactory) -> LoadedConfigPack:
    pack_root = create_config_pack(tmp_path_factory.mktemp("pack"))
    cars = load_cars(pack_root / "cars")
    profiles = load_profiles(pack_root / "profiles")
    overrides = load_lfs_class_overrides(pack_root / "lfs_class_overrides.toml")
    return LoadedConfigPack(
        root=pack_root,
        cars=cars,
        profiles=profiles,
        overrides=overrides,
    )


def test_load_cars_indexed_by_abbrev(config_pack: LoadedConfigPack) -> None:
    cars = load_cars(config_pack.root / "cars")

    assert set(cars) == {"ABC", "DEF", "GHI"}
    assert isinstance(cars["ABC"], Car)
    assert cars["ABC"].lfs_class == "STD"
    assert cars["DEF"].lfs_class == "Unclassified"
    assert cars["GHI"].lfs_class == "GT"


def test_load_profiles_prefers_meta_id(config_pack: LoadedConfigPack) -> None:
    profiles = load_profiles(config_pack.root / "profiles")

    assert "custom-profile" in profiles
    assert "fallback" in profiles


def _assert_expected_sections(resolved: Mapping[str, object]) -> None:
    assert set(resolved.keys()) == {"meta", "targets", "policy", "recommender"}
    meta = resolved["meta"]
    assert isinstance(meta, Mapping)
    assert meta["category"] == "road"


def _assert_class_overrides(resolved: Mapping[str, object]) -> None:
    _assert_expected_sections(resolved)
    targets = resolved["targets"]
    policy = resolved["policy"]
    recommender = resolved["recommender"]

    assert isinstance(targets, Mapping)
    assert isinstance(policy, Mapping)
    assert isinstance(recommender, Mapping)

    balance = targets["balance"]
    assert isinstance(balance, Mapping)
    assert balance["delta_nfr"] == pytest.approx(0.3)

    steering_policy = policy["steering"]
    assert isinstance(steering_policy, Mapping)
    assert steering_policy["aggressiveness"] == pytest.approx(0.75)

    steering_recommender = recommender["steering"]
    assert isinstance(steering_recommender, Mapping)
    assert steering_recommender["kp"] == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("car_abbrev", "use_overrides", "expectation", "assertion"),
    [
        pytest.param(
            "DEF",
            False,
            pytest.raises(KeyError),
            None,
            id="missing-profile",
        ),
        pytest.param(
            "ABC",
            False,
            nullcontext(),
            _assert_expected_sections,
            id="expected-sections",
        ),
        pytest.param(
            "ABC",
            True,
            nullcontext(),
            _assert_class_overrides,
            id="class-overrides",
        ),
    ],
)
def test_resolve_targets_variants(
    config_pack: LoadedConfigPack,
    car_abbrev: str,
    use_overrides: bool,
    expectation,
    assertion: Callable[[Mapping[str, object]], None] | None,
) -> None:
    overrides = config_pack.overrides if use_overrides else None
    resolved: Mapping[str, object] | None = None

    with expectation:
        resolved = resolve_targets(
            car_abbrev,
            config_pack.cars,
            config_pack.profiles,
            overrides=overrides,
        )

    if assertion is not None:
        assert resolved is not None
        assertion(resolved)


def test_example_pipeline_accepts_custom_data_root(
    config_pack: LoadedConfigPack,
) -> None:
    resolved = example_pipeline("ABC", data_root=config_pack.root)

    assert resolved["targets"]["balance"]["delta_nfr"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("raw_config", "expected"),
    [
        (
            {},
            CacheOptions(
                enable_delta_cache=True,
                nu_f_cache_size=DEFAULT_DYNAMIC_CACHE_SIZE,
                telemetry_cache_size=DEFAULT_DYNAMIC_CACHE_SIZE,
                recommender_cache_size=DEFAULT_DYNAMIC_CACHE_SIZE,
            ),
        ),
        (
            {"performance": {"cache_enabled": False, "max_cache_size": 16}},
            CacheOptions(
                enable_delta_cache=False,
                nu_f_cache_size=0,
                telemetry_cache_size=0,
                recommender_cache_size=0,
            ),
        ),
        (
            {
                "cache": {
                    "cache_enabled": "true",
                    "nu_f_cache_size": "64",
                    "telemetry": {"telemetry_cache_size": "12"},
                }
            },
            CacheOptions(
                enable_delta_cache=True,
                nu_f_cache_size=64,
                telemetry_cache_size=12,
                recommender_cache_size=64,
            ),
        ),
    ],
    ids=["defaults", "performance-overrides", "legacy-cache-section"],
)
def test_parse_cache_options_variants(
    raw_config: Mapping[str, object], expected: CacheOptions
) -> None:
    options = parse_cache_options(raw_config)

    assert isinstance(options, CacheOptions)
    assert options == expected


def test_parse_cache_options_uses_pack_defaults(tmp_path: Path) -> None:
    from tnfr_lfs import resources

    builder = pack_builder(tmp_path / "pack")
    builder.add_config("")
    builder.ensure_dir("data")

    resources.set_pack_root_override(builder.root)
    config_loader_module = importlib.import_module("tnfr_lfs.telemetry.config_loader")
    try:
        config_loader = importlib.reload(config_loader_module)
        options = config_loader.parse_cache_options({})
        assert options.enable_delta_cache is True
        assert options.nu_f_cache_size == DEFAULT_DYNAMIC_CACHE_SIZE
        assert options.recommender_cache_size == DEFAULT_DYNAMIC_CACHE_SIZE
        assert options.telemetry_cache_size == DEFAULT_DYNAMIC_CACHE_SIZE
    finally:
        resources.set_pack_root_override(None)
        importlib.reload(config_loader_module)


def test_load_cars_uses_packaged_resources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tnfr_lfs import resources

    builder = pack_builder(tmp_path / "pack")
    builder.add_car("AAA", MINIMAL_DATA_CAR, in_data=True)

    resources.set_pack_root_override(builder.root)
    config_loader_module = importlib.import_module("tnfr_lfs.telemetry.config_loader")
    config_loader = importlib.reload(config_loader_module)

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    monkeypatch.chdir(workspace)
    try:
        cars = config_loader.load_cars()
    finally:
        resources.set_pack_root_override(None)
        importlib.reload(config_loader_module)

    assert set(cars) == {"AAA"}
