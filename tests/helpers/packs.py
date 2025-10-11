from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

__all__ = [
    "PackBuilder",
    "create_cli_config_pack",
    "create_config_pack",
    "create_brake_thermal_pack",
    "pack_builder",
    "MINIMAL_DATA_CAR",
]


def _toml(text: str) -> str:
    """Normalize TOML payload indentation for filesystem writes."""

    return dedent(text).strip() + "\n"


@dataclass(frozen=True)
class PackBuilder:
    """Utility for assembling temporary pack directory structures."""

    root: Path

    def __post_init__(self) -> None:  # pragma: no cover - simple filesystem guard
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, relative_path: str | Path, contents: str, *, encoding: str = "utf8") -> Path:
        destination = self.root / Path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents, encoding=encoding)
        return destination

    def ensure_dir(self, relative_path: str | Path) -> Path:
        directory = self.root / Path(relative_path)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def add_config(self, contents: str, filename: str = "global.toml") -> Path:
        return self.write(Path("config") / filename, contents)

    def add_car(self, name: str, contents: str, *, in_data: bool = True) -> Path:
        base = Path("data") if in_data else Path()
        return self.write(base / "cars" / f"{name}.toml", contents)

    def add_profile(self, name: str, contents: str, *, in_data: bool = True) -> Path:
        base = Path("data") if in_data else Path()
        return self.write(base / "profiles" / f"{name}.toml", contents)

    def add_lfs_class_overrides(self, contents: str, filename: str = "lfs_class_overrides.toml") -> Path:
        return self.write(filename, contents)


def pack_builder(root: Path) -> PackBuilder:
    """Instantiate a :class:`PackBuilder` for the given root directory."""

    return PackBuilder(root=root)


# Shared TOML payloads -----------------------------------------------------
CLI_CAR_ALPHA = _toml(
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

CONFIG_CAR_ALPHA = _toml(
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

CONFIG_CAR_DELTA = _toml(
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

CONFIG_CAR_GAMMA = _toml(
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

CLI_PROFILE_CUSTOM = _toml(
    """
    [meta]
    id = "custom-profile"
    category = "road"

    [targets.balance]
    delta_nfr = 0.42
    sense_index = 0.83

    [policy.steering]
    aggressiveness = 0.5

    [recommender.steering]
    kp = 1.0
    """
)

CONFIG_PROFILE_CUSTOM = _toml(
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

CONFIG_PROFILE_FALLBACK = _toml(
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

CONFIG_LFS_CLASS_OVERRIDES = _toml(
    """
    ["STD".overrides.targets.balance]
    delta_nfr = 0.3

    ["STD".overrides.policy.steering]
    aggressiveness = 0.75

    ["STD".overrides.recommender.steering]
    kp = 2.5
    """
)

CLI_GLOBAL_CONFIG = _toml(
    """
    [analyze]
    car_model = "ABC"

    [suggest]
    car_model = "ABC"
    track = "AS5"

    [write_set]
    car_model = "ABC"

    [osd]
    car_model = "ABC"
    track = "AS5"
    """
)

THERMAL_GLOBAL_CONFIG = _toml(
    """
    [thermal.brakes]
    mode = "auto"
    """
)

THERMAL_CAR_ALPHA = _toml(
    """
    abbrev = "AAA"
    name = "Alpha"
    license = "demo"
    engine_layout = "front"
    drive = "RWD"
    weight_kg = 900
    wheel_rotation_group_deg = 30
    profile = "default"

    [thermal.brakes]
    mode = "auto"
    """
)

MINIMAL_DATA_CAR = _toml(
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


# Factory helpers ----------------------------------------------------------
def create_cli_config_pack(root: Path) -> Path:
    builder = pack_builder(root)
    builder.add_car("ABC", CLI_CAR_ALPHA, in_data=True)
    builder.add_profile("custom", CLI_PROFILE_CUSTOM, in_data=True)
    builder.add_config(CLI_GLOBAL_CONFIG)
    return builder.root


def create_config_pack(root: Path) -> Path:
    builder = pack_builder(root)
    builder.add_car("ABC", CONFIG_CAR_ALPHA, in_data=False)
    builder.add_car("DEF", CONFIG_CAR_DELTA, in_data=False)
    builder.add_car("GHI", CONFIG_CAR_GAMMA, in_data=False)
    builder.add_profile("custom", CONFIG_PROFILE_CUSTOM, in_data=False)
    builder.add_profile("fallback", CONFIG_PROFILE_FALLBACK, in_data=False)
    builder.add_lfs_class_overrides(CONFIG_LFS_CLASS_OVERRIDES)
    return builder.root


def create_brake_thermal_pack(root: Path) -> Path:
    builder = pack_builder(root)
    builder.add_config(THERMAL_GLOBAL_CONFIG)
    builder.add_car("AAA", THERMAL_CAR_ALPHA, in_data=True)
    return builder.root
