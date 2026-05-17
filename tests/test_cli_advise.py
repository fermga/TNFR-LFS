"""Phase 8: CLI `advise` subcommand + jargon-free shared serializers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Reuse the synthetic CarInfoBin fixture from the advisor test module.
sys.path.insert(0, str(Path(__file__).resolve().parent / "tnfr_racing"))
from test_advisor import _synthetic_baseline  # noqa: E402

from lfs_telemetry import cli  # noqa: E402
from lfs_telemetry.telemetry.lap import LapTelemetry  # noqa: E402
from lfs_telemetry.tnfr_racing.advisor import SetupAdvisor  # noqa: E402
from lfs_telemetry.tnfr_racing.serialize import (  # noqa: E402
    format_refusal,
    humanize_action,
    result_to_json,
    result_to_markdown,
)

ASSETS = Path(__file__).resolve().parents[1] / "assets"

# Same forbidden vocabulary the advisor unit tests enforce (docs §10.3).
FORBIDDEN_TERMS = (
    "EPI", "νf", "ΔNFR", "Φ_s", "Si", "operador",
    "AL", "EN", "IL", "OZ", "UM", "RA", "SHA", "VAL", "NUL",
    "THOL", "ZHIR", "NAV", "REMESH",
    "U1", "U2", "U3", "U4", "U5", "U6", "tétrada",
)


def _assert_jargon_free(text: str) -> None:
    for term in FORBIDDEN_TERMS:
        assert term not in text, (
            f"forbidden TNFR token {term!r} leaked into report:\n{text}"
        )


# ---------------------------------------------------------------------------
# Pure-Python serializer tests (no CLI invocation)
# ---------------------------------------------------------------------------


def _full_stint_result(seed: int = 20260516):
    paths = sorted(ASSETS.glob("synthetic_BL1_FBM_v2_lap*.csv"))
    assert len(paths) >= 5, "synthetic v2 stint missing from assets/"
    laps = [LapTelemetry.from_csv(p, car="FBM") for p in paths]
    advisor = SetupAdvisor(seed=seed)
    return advisor.advise(laps, _synthetic_baseline(), laps[0].car, "BL1"), laps


def _short_stint_result():
    paths = sorted(ASSETS.glob("synthetic_BL1_FBM_v2_lap*.csv"))[:2]
    laps = [LapTelemetry.from_csv(p, car="FBM") for p in paths]
    return SetupAdvisor(seed=20260516).advise(
        laps, _synthetic_baseline(), laps[0].car, "BL1",
    )


def test_format_refusal_translates_all_known_tags() -> None:
    samples = (
        "insufficient_stint: only 2 laps",
        "no_rule_fired",
        "grammar_U_violation",
        "no_coherence_positive_rule",
        "sector_decomposition_failed",
    )
    for tag in samples:
        txt = format_refusal(tag)
        assert txt and txt != tag, f"refusal tag {tag!r} not translated"
        _assert_jargon_free(txt)


def test_short_stint_serializers_are_jargon_free() -> None:
    res = _short_stint_result()
    assert res.proposed is None and res.refusal_reason is not None
    _assert_jargon_free(result_to_json(res))
    _assert_jargon_free(result_to_markdown(
        res, car_key="FBM", track_code="BL1", n_laps=2))


def test_full_stint_serializers_are_jargon_free_and_roundtrip_json() -> None:
    res, laps = _full_stint_result()
    js = result_to_json(res)
    md = result_to_markdown(
        res, car_key="FBM", track_code="BL1", n_laps=len(laps))
    _assert_jargon_free(js)
    _assert_jargon_free(md)
    payload = json.loads(js)
    assert payload["status"] in ("proposal", "refusal")
    if payload["status"] == "proposal":
        assert "actions" in payload
        assert "seed" in payload
        for act in payload["actions"]:
            assert {"kind", "target", "delta", "units",
                    "rationale_id"}.issubset(act)


def test_humanize_action_uses_physical_labels() -> None:
    res, _ = _full_stint_result()
    if res.proposed is None or not res.proposed.actions:
        pytest.skip("synthetic stint produced no actions")
    for act in res.proposed.actions:
        sub, change, rationale = humanize_action(act)
        # Subsystem must not be a raw operator name.
        assert sub and "(" in sub and ")" in sub
        _assert_jargon_free(sub)
        _assert_jargon_free(change)
        _assert_jargon_free(rationale)


# ---------------------------------------------------------------------------
# CLI dispatcher tests
# ---------------------------------------------------------------------------


def test_advise_cli_writes_both_reports(tmp_path: Path, monkeypatch,
                                        capsys) -> None:
    monkeypatch.setattr(
        "lfs_telemetry.telemetry.observables.load_car_info_bin_for",
        lambda key: _synthetic_baseline(),
    )
    paths = sorted(ASSETS.glob("synthetic_BL1_FBM_v2_lap*.csv"))
    assert len(paths) >= 5
    out_base = tmp_path / "FBM_BL1_<timestamp>"
    argv = [
        "advise",
        "--car", "FBM",
        "--track", "BL1",
        "--laps", *[str(p) for p in paths],
        "--seed", "20260516",
        "--output", str(out_base),
    ]
    rc = cli.main(argv)
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    written = sorted(tmp_path.glob("FBM_BL1_*.json")) \
        + sorted(tmp_path.glob("FBM_BL1_*.md"))
    assert len(written) == 2
    for p in written:
        text = p.read_text(encoding="utf-8")
        assert text.strip()
        _assert_jargon_free(text)
    # JSON file is valid JSON.
    js_path = next(p for p in written if p.suffix == ".json")
    json.loads(js_path.read_text(encoding="utf-8"))


def test_advise_cli_short_stint_returns_refusal(tmp_path: Path, monkeypatch,
                                                capsys) -> None:
    monkeypatch.setattr(
        "lfs_telemetry.telemetry.observables.load_car_info_bin_for",
        lambda key: _synthetic_baseline(),
    )
    paths = sorted(ASSETS.glob("synthetic_BL1_FBM_v2_lap*.csv"))[:2]
    argv = [
        "advise",
        "--car", "FBM", "--track", "BL1",
        "--laps", *[str(p) for p in paths],
        "--format", "json",
    ]
    rc = cli.main(argv)
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["status"] == "refusal"
    assert payload["refusal_reason"].startswith("insufficient_stint")
    _assert_jargon_free(captured.out)


def test_advise_cli_missing_lap_files_returns_nonzero(tmp_path: Path) -> None:
    argv = [
        "advise",
        "--car", "FBM", "--track", "BL1",
        "--laps", str(tmp_path / "does_not_exist.csv"),
    ]
    rc = cli.main(argv)
    assert rc == 2


def test_advise_cli_missing_baseline_returns_nonzero(monkeypatch) -> None:
    # Force the asset lookup to fail so the CLI surfaces the error.
    monkeypatch.setattr(
        "lfs_telemetry.telemetry.observables.load_car_info_bin_for",
        lambda key: None,
    )
    paths = sorted(ASSETS.glob("synthetic_BL1_FBM_v2_lap*.csv"))[:5]
    argv = [
        "advise",
        "--car", "ZZZ", "--track", "BL1",
        "--laps", *[str(p) for p in paths],
    ]
    rc = cli.main(argv)
    assert rc == 2


def test_python_dash_m_entrypoint_exists() -> None:
    # Just verify the module is importable as a script entry; we don't
    # actually fork a subprocess here to keep the test fast.
    import lfs_telemetry.__main__ as entry
    assert hasattr(entry, "main")
