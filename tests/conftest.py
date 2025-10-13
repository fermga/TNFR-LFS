from __future__ import annotations

import csv
import importlib.util
import json
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Awaitable, Callable, Dict, Iterable, List, cast

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def write_pyproject(directory: Path, contents: str) -> Path:
    """Persist a ``pyproject.toml`` under ``directory`` and return its path."""

    payload = dedent(contents).lstrip()
    target = directory / "pyproject.toml"
    target.write_text(payload, encoding="utf8")
    return target

if importlib.util.find_spec("pytest_cov") is None:

    def pytest_addoption(parser: pytest.Parser) -> None:
        """Register stub coverage options when pytest-cov is unavailable."""

        parser.addoption(
            "--cov",
            action="append",
            default=[],
            metavar="MODULE",
            help="Stub option provided when pytest-cov is not installed.",
        )
        parser.addoption(
            "--cov-report",
            action="append",
            default=[],
            metavar="TYPE",
            help="Stub option provided when pytest-cov is not installed.",
        )
        parser.addoption(
            "--cov-config",
            action="store",
            default=None,
            metavar="PATH",
            help="Stub option provided when pytest-cov is not installed.",
        )

    def pytest_configure(config: pytest.Config) -> None:
        """Inform users that coverage collection is skipped without pytest-cov."""

        cov_requested = bool(config.getoption("--cov")) or bool(
            config.getoption("--cov-report")
        ) or bool(config.getoption("--cov-config"))
        if cov_requested:
            warnings.warn(
                "pytest-cov is not installed; coverage options will be ignored.",
                RuntimeWarning,
                stacklevel=2,
            )

from tnfr_lfs.resources import data_root
from tnfr_lfs.examples import quickstart_dataset

from tnfr_lfs.ingestion import outgauge_udp as outgauge_module
from tnfr_lfs.ingestion import outsim_udp as outsim_module
from tnfr_lfs.ingestion.outgauge_udp import AsyncOutGaugeUDPClient, OutGaugeUDPClient
from tnfr_lfs.ingestion.outsim_udp import AsyncOutSimUDPClient, OutSimUDPClient

from tests.helpers import build_outgauge_payload, build_outsim_payload, run_cli_in_tmp

from tnfr_lfs.core.cache_settings import CacheOptions

from tnfr_lfs.core.epi import EPIExtractor, TelemetryRecord
from tnfr_lfs.core.epi_models import (
    BrakesNode,
    ChassisNode,
    DriverNode,
    EPIBundle,
    SuspensionNode,
    TrackNode,
    TransmissionNode,
    TyresNode,
)
from tnfr_lfs.core.segmentation import Goal, Microsector, segment_microsectors
from tnfr_lfs.recommender.rules import ThresholdProfile


@pytest.fixture(scope="session")
def quickstart_dataset_path() -> Path:
    """Baseline dataset referenced by the quickstart flow."""

    return quickstart_dataset.dataset_path()


@pytest.fixture(params=("outgauge", "outsim"), name="udp_client_spec")
def udp_client_spec_fixture(request: pytest.FixtureRequest) -> Dict[str, Any]:
    """Provide parameterized specifications for UDP client tests."""

    specs: Dict[str, Dict[str, Any]] = {
        "outgauge": {
            "name": "outgauge",
            "module": outgauge_module,
            "sync_constructor": OutGaugeUDPClient,
            "async_constructor": cast(
                Callable[..., Awaitable[Any]], AsyncOutGaugeUDPClient.create
            ),
            "payload_factory": build_outgauge_payload,
            "value_extractor": lambda packet: packet.packet_id,
            "expected_attribute": "packet_id",
            "default_address": ("127.0.0.1", 3000),
            "host_payload_spec": {"packet_id": 7, "time_value": 70, "layout": "LYT"},
            "host_expected_value": 7,
            "batch_specs": [
                {"packet_id": 5, "time_value": 50, "layout": "LYT"},
                {"packet_id": 6, "time_value": 60, "layout": "LYT"},
                {"packet_id": 7, "time_value": 70, "layout": "LYT"},
            ],
            "batch_expected_values": [5, 6, 7],
            "pending_first": {"packet_id": 5, "time_value": 50, "layout": "LYT"},
            "pending_successor": {"packet_id": 6, "time_value": 60, "layout": "LYT"},
            "pending_appended": {"packet_id": 7, "time_value": 70, "layout": "LYT"},
            "pending_expected_values": (5, 6),
            "isolated_first": {"packet_id": 7, "time_value": 70, "layout": "LYT"},
            "isolated_second": {"packet_id": 8, "time_value": 90, "layout": "LYT"},
            "isolated_expected_values": (7, 8),
            "async_send_specs": [
                {"packet_id": 5, "time_value": 50, "layout": "LYT"},
                {"packet_id": 7, "time_value": 70, "layout": "LYT"},
                {"packet_id": 6, "time_value": 60, "layout": "LYT"},
            ],
            "async_expected_values": [5, 6, 7],
            "release_packets": True,
        },
        "outsim": {
            "name": "outsim",
            "module": outsim_module,
            "sync_constructor": OutSimUDPClient,
            "async_constructor": cast(
                Callable[..., Awaitable[Any]], AsyncOutSimUDPClient.create
            ),
            "payload_factory": build_outsim_payload,
            "value_extractor": lambda packet: packet.time,
            "expected_attribute": "time",
            "default_address": ("127.0.0.1", 4123),
            "host_payload_spec": 200,
            "host_expected_value": 200,
            "batch_specs": [100, 120, 140],
            "batch_expected_values": [100, 120, 140],
            "pending_first": 100,
            "pending_successor": 120,
            "pending_appended": 140,
            "pending_expected_values": (100, 120),
            "isolated_first": 100,
            "isolated_second": 120,
            "isolated_expected_values": (100, 120),
            "async_send_specs": [100, 140, 120],
            "async_expected_values": [100, 120, 140],
            "release_packets": False,
        },
    }

    return specs[request.param]


@pytest.fixture(scope="session")
def raf_sample_path() -> Path:
    """Location of the bundled RAF telemetry sample."""

    sample = data_root() / "test1.raf"
    if not sample.exists():  # pragma: no cover - defensive guard for local runs
        raise FileNotFoundError(sample)
    return sample


@pytest.fixture(scope="session")
def synthetic_stint_path() -> Path:
    """Location of the bundled synthetic telemetry stint."""

    return Path(__file__).with_name("data") / "synthetic_stint.csv"


@pytest.fixture
def baseline_cli_runner(
    tmp_path: Path,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """Invoke the baseline CLI within ``tmp_path`` and report its output path."""

    def _extract_positional(args: list[str]) -> tuple[Path | None, list[str]]:
        remaining = list(args)
        positional: Path | None = None
        if remaining and not remaining[0].startswith("--"):
            raw_positional = Path(remaining.pop(0))
            positional = (
                raw_positional
                if raw_positional.is_absolute()
                else tmp_path / raw_positional
            )
        return positional, remaining

    def _infer_destination(positional: Path | None, options: list[str]) -> Path:
        if positional is not None:
            return positional

        output_dir = tmp_path / "runs"
        output_name: str | None = None
        iterator = iter(options)
        for token in iterator:
            if token == "--output-dir":
                value = next(iterator, None)
                if value is None:
                    raise ValueError("--output-dir requires a value")
                path = Path(value)
                output_dir = path if path.is_absolute() else tmp_path / path
            elif token == "--output":
                value = next(iterator, None)
                if value is None:
                    raise ValueError("--output requires a value")
                output_name = value
            else:
                # Skip positional values for unrelated options.
                continue

        if output_name is not None:
            return output_dir / output_name

        return output_dir

    def _run(cli_args: Sequence[str]) -> tuple[str, pytest.CaptureResult[str], Path]:
        args_list = [str(arg) for arg in cli_args]
        positional, remaining = _extract_positional(args_list)

        invocation = ["baseline", *args_list]
        if "--simulate" not in args_list:
            invocation.extend(["--simulate", str(synthetic_stint_path)])

        result, captured = run_cli_in_tmp(
            invocation,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            capsys=capsys,
            capture_output=True,
        )

        destination = _infer_destination(positional, remaining)
        return result, captured, destination

    return _run


@pytest.fixture
def baseline_path(
    tmp_path: Path,
    synthetic_stint_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Generate a baseline telemetry run within ``tmp_path`` and return its path."""

    destination = tmp_path / "baseline.jsonl"
    run_cli_in_tmp(
        [
            "baseline",
            str(destination),
            "--simulate",
            str(synthetic_stint_path),
        ],
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    return destination


@pytest.fixture(scope="session")
def synthetic_records(synthetic_stint_path: Path) -> List[TelemetryRecord]:
    """Load telemetry records used across segmentation/EPI tests."""

    records: List[TelemetryRecord] = []
    with synthetic_stint_path.open(encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                TelemetryRecord(
                    timestamp=float(row["timestamp"]),
                    vertical_load=float(row["vertical_load"]),
                    slip_ratio=float(row["slip_ratio"]),
                    lateral_accel=float(row["lateral_accel"]),
                    longitudinal_accel=float(row["longitudinal_accel"]),
                    yaw=float(row["yaw"]),
                    pitch=float(row["pitch"]),
                    roll=float(row["roll"]),
                    brake_pressure=float(row["brake_pressure"]),
                    locking=float(row["locking"]),
                    nfr=float(row["nfr"]),
                    si=float(row["si"]),
                    speed=float(row["speed"]),
                    yaw_rate=float(row["yaw_rate"]),
                    slip_angle=float(row["slip_angle"]),
                    steer=float(row["steer"]),
                    throttle=float(row["throttle"]),
                    gear=int(row["gear"]),
                    vertical_load_front=float(row["vertical_load_front"]),
                    vertical_load_rear=float(row["vertical_load_rear"]),
                    mu_eff_front=float(row["mu_eff_front"]),
                    mu_eff_rear=float(row["mu_eff_rear"]),
                    mu_eff_front_lateral=float(row["mu_eff_front_lateral"]),
                    mu_eff_front_longitudinal=float(row["mu_eff_front_longitudinal"]),
                    mu_eff_rear_lateral=float(row["mu_eff_rear_lateral"]),
                    mu_eff_rear_longitudinal=float(row["mu_eff_rear_longitudinal"]),
                    suspension_travel_front=float(row["suspension_travel_front"]),
                    suspension_travel_rear=float(row["suspension_travel_rear"]),
                    suspension_velocity_front=float(row["suspension_velocity_front"]),
                    suspension_velocity_rear=float(row["suspension_velocity_rear"]),
                )
            )
    return records


@pytest.fixture(scope="session")
def synthetic_bundles(synthetic_records: Iterable[TelemetryRecord]):
    """Run the EPI extractor to produce ΔNFR bundles for the synthetic stint."""

    extractor = EPIExtractor()
    return extractor.extract(list(synthetic_records))


@pytest.fixture(scope="session")
def synthetic_microsectors(
    synthetic_records: List[TelemetryRecord], synthetic_bundles
) -> List[Microsector]:
    """Microsectors obtained from the bundled synthetic stint."""

    return segment_microsectors(synthetic_records, synthetic_bundles)


@pytest.fixture(scope="session")
def car_track_thresholds() -> Dict[str, Dict[str, ThresholdProfile]]:
    """Car/track threshold profiles resolved from the JSON fixture."""

    path = Path(__file__).with_name("data") / "car_track_profiles.json"
    with path.open(encoding="utf8") as handle:
        payload = json.load(handle)
    library: Dict[str, Dict[str, ThresholdProfile]] = {}
    for car_model, tracks in payload.items():
        library[car_model] = {
            track: ThresholdProfile(**values) for track, values in tracks.items()
        }
    return library


@pytest.fixture()
def acceptance_bundle_series() -> List[EPIBundle]:
    """Synthetic bundles exercising nodal, coupling and resonance metrics."""

    timestamps = [0.0, 0.1, 0.2, 0.3]
    global_delta = [1.2, 0.8, 0.4, 0.1]
    global_si = [0.45, 0.55, 0.65, 0.75]
    tyres_delta = [0.6, 0.5, 0.4, 0.3]
    suspension_delta = [0.3, 0.2, 0.1, 0.0]
    chassis_delta = [0.3, 0.1, -0.1, -0.2]
    tyres_si = [0.5, 0.58, 0.66, 0.74]
    suspension_si = [0.48, 0.52, 0.56, 0.6]
    chassis_si = [0.46, 0.5, 0.54, 0.58]
    dt = 0.1
    bundles: List[EPIBundle] = []
    cumulative_epi = 0.0
    for index, timestamp in enumerate(timestamps):
        derivatives = {
            "tyres": 0.2 * tyres_delta[index],
            "suspension": 0.15 * suspension_delta[index],
            "chassis": 0.12 * chassis_delta[index],
        }
        node_evolution = {
            node: (derivative * dt, derivative) for node, derivative in derivatives.items()
        }
        d_epi_dt = sum(derivatives.values())
        cumulative_epi += d_epi_dt * dt
        delta_breakdown = {
            "tyres": {"total": tyres_delta[index]},
            "suspension": {"total": suspension_delta[index]},
            "chassis": {"total": chassis_delta[index]},
        }
        bundles.append(
            EPIBundle(
                timestamp=timestamp,
                epi=0.0,
                delta_nfr=global_delta[index],
                sense_index=global_si[index],
                tyres=TyresNode(
                    delta_nfr=tyres_delta[index],
                    sense_index=tyres_si[index],
                    nu_f=0.2,
                    dEPI_dt=node_evolution["tyres"][1],
                    integrated_epi=node_evolution["tyres"][0],
                    load=5200.0,
                    slip_ratio=0.04,
                    mu_eff_front=0.85,
                    mu_eff_rear=0.82,
                    mu_eff_front_lateral=0.88,
                    mu_eff_front_longitudinal=0.78,
                    mu_eff_rear_lateral=0.82,
                    mu_eff_rear_longitudinal=0.74,
                ),
                suspension=SuspensionNode(
                    delta_nfr=suspension_delta[index],
                    sense_index=suspension_si[index],
                    nu_f=0.15,
                    dEPI_dt=node_evolution["suspension"][1],
                    integrated_epi=node_evolution["suspension"][0],
                    travel_front=0.02,
                    travel_rear=0.018,
                    velocity_front=0.1,
                    velocity_rear=0.095,
                ),
                chassis=ChassisNode(
                    delta_nfr=chassis_delta[index],
                    sense_index=chassis_si[index],
                    nu_f=0.12,
                    dEPI_dt=node_evolution["chassis"][1],
                    integrated_epi=node_evolution["chassis"][0],
                    yaw=0.02,
                    pitch=0.01,
                    roll=0.015,
                    yaw_rate=0.18,
                    lateral_accel=1.2,
                    longitudinal_accel=0.4,
                ),
                brakes=BrakesNode(
                    delta_nfr=0.0,
                    sense_index=0.68,
                    nu_f=0.16,
                    dEPI_dt=0.0,
                    integrated_epi=0.0,
                    brake_pressure=0.6,
                    locking=0.02,
                ),
                transmission=TransmissionNode(
                    delta_nfr=0.0,
                    sense_index=0.7,
                    nu_f=0.11,
                    dEPI_dt=0.0,
                    integrated_epi=0.0,
                    throttle=0.7,
                    gear=4,
                    speed=32.0,
                    longitudinal_accel=0.35,
                ),
                track=TrackNode(
                    delta_nfr=0.0,
                    sense_index=0.72,
                    nu_f=0.08,
                    dEPI_dt=0.0,
                    integrated_epi=0.0,
                    axle_load_balance=0.05,
                    axle_velocity_balance=0.01,
                    yaw=0.01,
                    lateral_accel=1.1,
                ),
                driver=DriverNode(
                    delta_nfr=0.0,
                    sense_index=0.73,
                    nu_f=0.05,
                    dEPI_dt=0.0,
                    integrated_epi=0.0,
                    steer=0.12,
                    throttle=0.68,
                    style_index=0.62,
                ),
                delta_breakdown=delta_breakdown,
                dEPI_dt=d_epi_dt,
                integrated_epi=cumulative_epi,
                node_evolution=node_evolution,
            )
        )
    return bundles


@pytest.fixture()
def acceptance_records() -> List[TelemetryRecord]:
    """Telemetry streak aligned with ``acceptance_bundle_series``."""

    payload: List[TelemetryRecord] = []
    for index, (timestamp, si_value, delta_value) in enumerate(
        zip([0.0, 0.1, 0.2, 0.3], [0.45, 0.55, 0.65, 0.75], [1.2, 0.8, 0.4, 0.1])
    ):
        payload.append(
            TelemetryRecord(
                timestamp=timestamp,
                vertical_load=5050.0 + (index * 80.0),
                slip_ratio=0.03 + index * 0.005,
                lateral_accel=1.0 + index * 0.2,
                longitudinal_accel=0.4 - index * 0.05,
                yaw=0.02 * index,
                pitch=0.01 * index,
                roll=0.015 * index,
                brake_pressure=0.4 + index * 0.05,
                locking=0.01 + index * 0.005,
                nfr=500.0 + delta_value,
                si=si_value,
                speed=30.0 + index * 1.5,
                yaw_rate=0.18 + index * 0.02,
                slip_angle=0.03 + index * 0.01,
                steer=0.1 + index * 0.02,
                throttle=0.65 + index * 0.01,
                gear=4,
                vertical_load_front=2600.0 + index * 35.0,
                vertical_load_rear=2450.0 + index * 45.0,
                mu_eff_front=0.86 - index * 0.005,
                mu_eff_rear=0.83 - index * 0.004,
                mu_eff_front_lateral=0.9 - index * 0.004,
                mu_eff_front_longitudinal=0.78 - index * 0.003,
                mu_eff_rear_lateral=0.84 - index * 0.004,
                mu_eff_rear_longitudinal=0.72 - index * 0.003,
                suspension_travel_front=0.02 + index * 0.001,
                suspension_travel_rear=0.018 + index * 0.001,
                suspension_velocity_front=0.12 + index * 0.005,
                suspension_velocity_rear=0.11 + index * 0.004,
                lap="Validation",
            )
        )
    return payload


@pytest.fixture()
def acceptance_microsectors() -> List[Microsector]:
    """Microsector mock emphasising occupancy, nodal and mutation fields."""

    goals = (
        Goal(
            phase="entry",
            archetype="hairpin",
            description="Extend braking to nail the hairpin with control.",
            target_delta_nfr=0.3,
            target_sense_index=0.7,
            nu_f_target=0.18,
            nu_exc_target=0.18,
        rho_target=1.0,
        target_phase_lag=-0.05,
        target_phase_alignment=0.88,
        measured_phase_lag=-0.02,
        measured_phase_alignment=0.9,
            slip_lat_window=(-0.4, 0.4),
            slip_long_window=(-0.4, 0.4),
            yaw_rate_window=(-0.4, 0.4),
            dominant_nodes=("tyres", "brakes"),
        ),
        Goal(
            phase="apex",
            archetype="chicane",
            description="Maintain lightness to avoid saturating the second support.",
            target_delta_nfr=0.6,
            target_sense_index=0.8,
            nu_f_target=0.2,
            nu_exc_target=0.2,
        rho_target=1.0,
        target_phase_lag=0.0,
        target_phase_alignment=0.92,
        measured_phase_lag=0.01,
        measured_phase_alignment=0.91,
            slip_lat_window=(-0.35, 0.35),
            slip_long_window=(-0.35, 0.35),
            yaw_rate_window=(-0.3, 0.3),
            dominant_nodes=("suspension", "chassis"),
        ),
        Goal(
            phase="exit",
            archetype="fast",
            description="Project the exit while maintaining high pace.",
            target_delta_nfr=0.2,
            target_sense_index=0.75,
            nu_f_target=0.16,
            nu_exc_target=0.16,
        rho_target=1.0,
        target_phase_lag=0.04,
        target_phase_alignment=0.87,
        measured_phase_lag=0.05,
        measured_phase_alignment=0.86,
            slip_lat_window=(-0.3, 0.3),
            slip_long_window=(-0.3, 0.3),
            yaw_rate_window=(-0.25, 0.25),
            dominant_nodes=("driver", "transmission"),
        ),
    )
    window_occupancy = {
        "entry": {"slip_lat": 82.5, "slip_long": 79.4, "yaw_rate": 75.0},
        "apex": {"slip_lat": 72.1, "slip_long": 64.3, "yaw_rate": 58.2},
        "exit": {"slip_lat": 88.7, "slip_long": 83.9, "yaw_rate": 77.6},
    }
    phase_weights = {
        "entry": {"tyres": 1.3, "brakes": 1.1, "__default__": 1.0},
        "apex": {"suspension": 1.5, "chassis": 1.2, "__default__": 0.9},
        "exit": 0.85,
        "__default__": 1.0,
    }
    phase_lag = {"entry": -0.02, "apex": 0.01, "exit": 0.05}
    phase_alignment = {"entry": 0.9, "apex": 0.91, "exit": 0.86}
    phase_synchrony = {"entry": 0.91, "apex": 0.93, "exit": 0.88}
    microsector = Microsector(
        index=0,
        start_time=0.0,
        end_time=0.3,
        curvature=1.15,
        brake_event=True,
        support_event=True,
        delta_nfr_signature=0.62,
        goals=goals,
        phase_boundaries={"entry": (0, 1), "apex": (1, 3), "exit": (3, 4)},
        phase_samples={"entry": (0,), "apex": (1, 2), "exit": (3,)},
        active_phase="apex",
        dominant_nodes={
            "entry": ("tyres", "brakes"),
            "apex": ("suspension", "chassis"),
            "exit": ("driver", "transmission"),
        },
        phase_weights=phase_weights,
        grip_rel=1.08,
        phase_lag=phase_lag,
        phase_alignment=phase_alignment,
        phase_synchrony=phase_synchrony,
        filtered_measures={"thermal_load": 5185.0, "style_index": 0.63},
        recursivity_trace=(
            {"phase": "entry", "thermal_load": 5050.0, "style_index": 0.6},
            {"phase": "apex", "thermal_load": 5125.0, "style_index": 0.62},
        ),
        last_mutation={
            "archetype": "ataque",
            "mutated": True,
            "entropy": 0.71,
            "entropy_delta": 0.05,
            "style_delta": 0.14,
            "phase": "apex",
        },
        window_occupancy=window_occupancy,
        operator_events={},
    )
    return [microsector]

@dataclass(frozen=True)
class MiniTrackPack:
    """Container describing the synthetic track pack used in tests."""

    root: Path
    track_slug: str
    layout_code: str
    track_profile: str
    car_model: str
    car_profile: str
    tracks_dir: Path
    track_profiles_dir: Path
    modifiers_dir: Path


@pytest.fixture()
def mini_track_pack(tmp_path: Path) -> MiniTrackPack:
    """Create a minimal pack with track manifests, profiles and modifiers."""

    pack_root = tmp_path / "mini_pack"
    tracks_dir = pack_root / "data" / "tracks"
    track_profiles_dir = pack_root / "data" / "track_profiles"
    modifiers_dir = pack_root / "modifiers" / "combos"
    cars_dir = pack_root / "data" / "cars"

    for directory in (tracks_dir, track_profiles_dir, modifiers_dir, cars_dir):
        directory.mkdir(parents=True, exist_ok=True)

    track_slug = "AS"
    layout_code = "AS3"
    track_profile = "p_test_combo"
    car_model = "DEMO"
    car_profile = "demo_profile"

    tracks_dir.joinpath(f"{track_slug}.toml").write_text(
        dedent(
            """
            [config.AS3]
            name = "Mini Aston Historic"
            length_km = 5.2
            surface = "asphalt"
            track_profile = "p_test_combo"
            pit_boxes = 32
            notes = ["tight chicane", "long straight"]

            [config.AS3R]
            alias_of = "AS3"
            name = "Mini Aston Historic Reverse"
            """
        ),
        encoding="utf8",
    )

    track_profiles_dir.joinpath(f"{track_profile}.toml").write_text(
        dedent(
            """
            [meta]
            id = "p_test_combo"
            archetype = "balanced"
            length_category = "medium"

            [weights.entry]
            __default__ = 1.0
            brakes = 1.05
            tyres = 1.0

            [weights.apex]
            __default__ = 0.95
            anti_roll = 1.1

            [weights.exit]
            __default__ = 0.9
            differential = 1.2

            [hints]
            microsector_span = "compact"
            notes = ["aggressive hairpins", "short braking"]

            [hints.surface_bias]
            entry = 0.2
            exit = -0.1
            """
        ),
        encoding="utf8",
    )

    modifiers_dir.joinpath(f"{car_profile}__{track_profile}.toml").write_text(
        dedent(
            """
            [meta]
            id = "demo_profile__p_test_combo"
            car_group = "demo_profile"
            base_profile = "p_test_combo"

            [scale.weights.__default__]
            __default__ = 1.1

            [scale.weights.entry]
            __default__ = 1.25
            brakes = 1.4

            [scale.weights.exit]
            __default__ = 0.95
            differential = 1.3

            [hints]
            slip_ratio_bias = "aggressive"
            surface = "asphalt"
            """
        ),
        encoding="utf8",
    )

    cars_dir.joinpath(f"{car_model}.toml").write_text(
        dedent(
            """
            abbrev = "DEMO"
            name = "Demonstrator"
            license = "s3"
            engine_layout = "mid"
            drive = "RWD"
            weight_kg = 950
            wheel_rotation_group_deg = 30
            profile = "demo_profile"
            """
        ),
        encoding="utf8",
    )

    return MiniTrackPack(
        root=pack_root,
        track_slug=track_slug,
        layout_code=layout_code,
        track_profile=track_profile,
        car_model=car_model,
        car_profile=car_profile,
        tracks_dir=tracks_dir,
        track_profiles_dir=track_profiles_dir,
        modifiers_dir=modifiers_dir,
    )

@dataclass(frozen=True)
class CliConfigCase:
    """Represent a reusable CLI configuration scenario."""

    toml_text: str
    expected_sections: Dict[str, object]
    setup: Callable[[Path], None] | None = None


def _prefers_top_level_setup(root: Path) -> None:
    secondary = root / "secondary"
    secondary.mkdir()
    write_pyproject(secondary, "")


_CLI_CONFIG_CASES: Dict[str, CliConfigCase] = {
    "cache-section": CliConfigCase(
        toml_text="""
        [tool.tnfr_lfs.cache]
        cache_enabled = "yes"
        nu_f_cache_size = "48"
        """,
        expected_sections={
            "performance": CacheOptions(
                enable_delta_cache=True,
                nu_f_cache_size=48,
                telemetry_cache_size=48,
                recommender_cache_size=48,
            ).to_performance_config(),
        },
    ),
    "normalised-performance": CliConfigCase(
        toml_text="""
        [tool.tnfr_lfs.performance]
        cache_enabled = "no"
        max_cache_size = "12"
        telemetry_buffer_size = 42
        """,
        expected_sections={
            "performance": {
                **CacheOptions(
                    enable_delta_cache=False,
                    nu_f_cache_size=0,
                    telemetry_cache_size=0,
                    recommender_cache_size=0,
                ).to_performance_config(),
                "telemetry_buffer_size": 42,
            }
        },
    ),
    "explicit-sections": CliConfigCase(
        toml_text="""
        [tool.tnfr_lfs.core]
        host = "192.0.2.1"

        [tool.tnfr_lfs.performance]
        cache_enabled = true
        max_cache_size = 48
        """,
        expected_sections={
            "core": {"host": "192.0.2.1"},
            "performance": CacheOptions(
                enable_delta_cache=True,
                nu_f_cache_size=48,
                telemetry_cache_size=48,
                recommender_cache_size=48,
            ).to_performance_config(),
        },
    ),
    "prefers-top-level": CliConfigCase(
        toml_text="""
        [tool.tnfr_lfs.logging]
        level = "warning"
        format = "text"
        output = "stdout"
        """,
        expected_sections={
            "logging": {
                "level": "warning",
                "format": "text",
                "output": "stdout",
            }
        },
        setup=_prefers_top_level_setup,
    ),
    "logging-disabled-cache": CliConfigCase(
        toml_text="""
        [tool.tnfr_lfs.logging]
        level = "warning"
        format = "text"
        output = "stdout"

        [tool.tnfr_lfs.performance]
        cache_enabled = false
        max_cache_size = 12
        """,
        expected_sections={
            "logging": {
                "level": "warning",
                "format": "text",
                "output": "stdout",
            },
            "performance": CacheOptions(
                enable_delta_cache=False,
                nu_f_cache_size=0,
                telemetry_cache_size=0,
                recommender_cache_size=0,
            ).to_performance_config(),
        },
    ),
}


@pytest.fixture
def cli_config_case(
    request: pytest.FixtureRequest,
) -> tuple[str, Dict[str, object], Callable[[Path], None] | None]:
    """Provide reusable CLI configuration TOML snippets and expectations."""

    if not hasattr(request, "param"):
        raise pytest.UsageError("cli_config_case fixture requires a parameter")

    case_id = cast(str, request.param)
    try:
        case = _CLI_CONFIG_CASES[case_id]
    except KeyError:
        available = ", ".join(sorted(_CLI_CONFIG_CASES))
        pytest.fail(
            f"Unknown cli_config_case {case_id!r}. Available cases: {available}",
            pytrace=False,
        )

    expected_sections = dict(case.expected_sections)
    return case.toml_text, expected_sections, case.setup
