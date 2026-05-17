"""Phase 1 smoke test: every module is importable and TNFR engine is reachable."""
from __future__ import annotations

import importlib

import pytest

MODULES = (
    "lfs_telemetry.tnfr_racing",
    "lfs_telemetry.tnfr_racing.config",
    "lfs_telemetry.tnfr_racing.mapping",
    "lfs_telemetry.tnfr_racing.network_track",
    "lfs_telemetry.tnfr_racing.network_car",
    "lfs_telemetry.tnfr_racing.coupling",
    "lfs_telemetry.tnfr_racing.operators",
    "lfs_telemetry.tnfr_racing.grammar",
    "lfs_telemetry.tnfr_racing.fields",
    "lfs_telemetry.tnfr_racing.coherence",
    "lfs_telemetry.tnfr_racing.advisor",
    "lfs_telemetry.tnfr_racing.rationale",
    "lfs_telemetry.tnfr_racing.lap_filters",
)


@pytest.mark.parametrize("modname", MODULES)
def test_module_importable(modname: str) -> None:
    importlib.import_module(modname)


def test_public_exports() -> None:
    from lfs_telemetry.tnfr_racing import (
        AdvisorConfig,
        ProposedSetup,
        SetupAdvisor,
    )

    cfg = AdvisorConfig()
    advisor = SetupAdvisor(config=cfg)
    assert advisor.config is cfg
    assert hasattr(advisor, "advise")
    # ProposedSetup is now a structured dataclass — exposed but not instantiable empty.
    assert ProposedSetup.__dataclass_fields__  # type: ignore[attr-defined]


def test_tnfr_engine_reachable() -> None:
    """The canonical TNFR engine must be importable from this venv."""
    tnfr = importlib.import_module("tnfr")
    assert hasattr(tnfr, "create_nfr")
    assert hasattr(tnfr, "run_sequence")
    defs = importlib.import_module("tnfr.operators.definitions")
    assert hasattr(defs, "Emission")
    assert hasattr(defs, "Coherence")


def test_no_tnfr_leakage_outside_tnfr_racing() -> None:
    """Architectural invariant: no module under lfs_telemetry imports `tnfr`
    except inside the tnfr_racing subpackage. Enforced statically via grep."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "lfs_telemetry"
    offenders: list[str] = []
    for p in root.rglob("*.py"):
        if "tnfr_racing" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if (
                stripped.startswith("import tnfr")
                or stripped.startswith("from tnfr ")
                or stripped.startswith("from tnfr.")
            ):
                offenders.append(f"{p}: {stripped}")
    assert not offenders, "tnfr leakage detected:\n" + "\n".join(offenders)
