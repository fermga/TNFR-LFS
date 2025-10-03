from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tnfr_lfs.core.epi import EPIExtractor, TelemetryRecord
from tnfr_lfs.core.segmentation import Microsector, segment_microsectors
from tnfr_lfs.recommender.rules import ThresholdProfile


@pytest.fixture(scope="session")
def synthetic_stint_path() -> Path:
    """Location of the bundled synthetic telemetry stint."""

    return Path(__file__).with_name("data") / "synthetic_stint.csv"


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
                    nfr=float(row["nfr"]),
                    si=float(row["si"]),
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

