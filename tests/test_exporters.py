import importlib
import json
import struct
import subprocess
from pathlib import Path

import pytest

from tnfr_lfs.analysis import ABResult
from tnfr_lfs.core.epi_models import EPIBundle
from tnfr_lfs.exporters import (
    CAR_MODEL_PREFIXES,
    coherence_map_exporter,
    csv_exporter,
    delta_bifurcation_exporter,
    html_exporter,
    json_exporter,
    lfs_notes_exporter,
    lfs_set_exporter,
    markdown_exporter,
    normalise_set_output_name,
    operator_trajectory_exporter,
)
from tnfr_lfs.core.operator_detection import canonical_operator_label
from tnfr_lfs.exporters.setup_plan import serialise_setup_plan
from tests.helpers import (
    BASE_NU_F,
    SUPPORTED_CAR_MODELS,
    build_epi_nodes,
    build_native_export_plan,
    build_setup_plan,
)


def build_payload():
    def build_breakdown(delta_nfr: float):
        share = delta_nfr / 7
        return {
            "tyres": {"slip_ratio": share},
            "suspension": {"travel_front": share},
            "chassis": {"yaw_rate": share},
            "brakes": {"pressure": share},
            "transmission": {"throttle": share},
            "track": {"vertical_load": share},
            "driver": {"style_index": share},
        }

    results = [
        EPIBundle(
            0.0,
            0.45,
            -5.0,
            0.82,
            delta_breakdown=build_breakdown(-5.0),
            **build_epi_nodes(-5.0, 0.82),
        ),
        EPIBundle(
            0.1,
            0.50,
            10.0,
            0.74,
            delta_breakdown=build_breakdown(10.0),
            **build_epi_nodes(10.0, 0.74),
        ),
    ]
    return {"series": results, "meta": {"session": "FP1"}}


def test_json_exporter_serialises_payload():
    payload = build_payload()
    output = json_exporter(payload)
    assert "\"session\": \"FP1\"" in output
    assert "delta_breakdown" in output


def test_csv_exporter_renders_rows():
    payload = build_payload()
    output = csv_exporter(payload)
    lines = output.strip().splitlines()
    assert (
        lines[0]
        == "timestamp,epi,delta_nfr,delta_nfr_proj_longitudinal,delta_nfr_proj_lateral,sense_index"
    )
    assert len(lines) == 3
def test_serialise_setup_plan_collects_unique_fields():
    plan = build_setup_plan()
    payload = serialise_setup_plan(plan)
    assert payload["car_model"] == "XFG"
    assert payload["sci"] == pytest.approx(0.94)
    assert len(payload["changes"]) == 2
    assert any("Telemetry indicates" in item for item in payload["rationales"])
    assert any("Optimised braking" in item for item in payload["expected_effects"])
    assert pytest.approx(payload["dsi_dparam"]["front_arb_steps"], rel=1e-6) == 0.0451
    assert pytest.approx(payload["dnfr_integral_dparam"]["front_arb_steps"], rel=1e-6) == 0.084
    assert payload["phase_sensitivities"]["apex"]["delta_nfr_integral"]["front_arb_steps"] == pytest.approx(-0.084)
    assert (
        payload["phase_dnfr_integral_dparam"]["entry"]["front_arb_steps"]
        == pytest.approx(-0.062)
    )
    assert "tyres" in payload["tnfr_rationale_by_node"]
    assert "entry" in payload["tnfr_rationale_by_phase"]
    assert "tyres" in payload["expected_effects_by_node"]
    assert "entry" in payload["expected_effects_by_phase"]
    assert "front_arb_steps" in payload["clamped_parameters"]
    assert payload["phase_axis_targets"]["entry"]["longitudinal"] == pytest.approx(0.4)
    assert payload["phase_axis_weights"]["apex"]["lateral"] == pytest.approx(0.8)
    assert (
        payload["phase_axis_summary"]["longitudinal"]["entry"]
        == "⇈+0.40"
    )
    assert any("Entry ∥" in hint for hint in payload["phase_axis_suggestions"])


def test_markdown_exporter_renders_abtest_section() -> None:
    plan = build_setup_plan()
    abtest = ABResult(
        metric="sense_index",
        baseline_laps=(0.61, 0.62),
        variant_laps=(0.66, 0.67),
        baseline_mean=0.615,
        variant_mean=0.665,
        mean_difference=0.05,
        bootstrap_low=0.045,
        bootstrap_high=0.055,
        permutation_p_value=0.0125,
        estimated_power=0.84,
        alpha=0.05,
    )
    payload = {
        "setup_plan": plan,
        "session": {"abtest": abtest},
    }
    rendered = markdown_exporter(payload)
    assert "A/B comparison" in rendered
    assert "sense_index" in rendered
    assert "Power" in rendered


def test_markdown_exporter_renders_table_and_lists():
    plan = build_setup_plan()
    output = markdown_exporter({"setup_plan": plan})
    assert "| Change | Adjustment | Rationale |" in output
    assert "brake_bias_pct" in output
    assert "Telemetry indicates oscillations" in output
    assert "**dSi/dparam**" in output
    assert "**d∫|ΔNFR|/dparam**" in output
    assert "**∫|ΔNFR| gradients per phase**" in output
    assert "**TNFR rationales per node**" in output
    assert "**Expected effects per phase**" in output
    assert "**∇NFR∥/∇NFR⊥ projection targets per phase**" in output
    assert "**∇NFR∥/∇NFR⊥ projection map per phase**" in output
    assert "Entry ∥" in output
    assert "**Priority phase suggestions**" in output
    assert "C(c/d/a)" in output
    assert "**SCI contribution**" in output


def test_lfs_notes_exporter_renders_key_instructions():
    plan = build_setup_plan()
    output = lfs_notes_exporter({"setup_plan": plan})
    assert "Quick TNFR notes" in output


def test_html_exporter_renders_extended_sections() -> None:
    abtest = ABResult(
        metric="sense_index",
        baseline_laps=(0.61, 0.62),
        variant_laps=(0.66, 0.67),
        baseline_mean=0.615,
        variant_mean=0.665,
        mean_difference=0.05,
        bootstrap_low=0.045,
        bootstrap_high=0.055,
        permutation_p_value=0.0125,
        estimated_power=0.84,
        alpha=0.05,
    )
    variability_entry = {
        "microsector": 0,
        "label": "Curva 1",
        "overall": {
            "samples": 5,
            "delta_nfr": {"stdev": 0.12},
            "sense_index": {"stdev": 0.08, "stability_score": 0.75},
        },
    }
    payload = {
        "delta_nfr": -12.3,
        "sense_index": 0.78,
        "dissonance": 0.12,
        "coupling": 0.43,
        "resonance": 0.31,
        "objectives": {
            "target_delta_nfr": -11.0,
            "target_sense_index": 0.82,
        },
        "microsector_variability": [variability_entry],
        "recursive_trace": [0.2, 0.3, 0.4],
        "pairwise_coupling": {"delta_nfr": {"tyres↔suspension": 0.12}},
        "pareto_points": [
            {"score": 0.91, "breakdown": {"delta_nfr": 0.32, "sense_index": 0.18}}
        ],
        "session": {
            "car_model": "XFG",
            "track_profile": "BL1",
            "session": "FP1",
            "abtest": abtest,
            "pareto": [
                {"score": 0.77, "breakdown": {"delta_nfr": 0.28, "sense_index": 0.22}}
            ],
            "playbook_suggestions": ["Prioritise mid-apex"],
        },
        "session_messages": ("Session affected by crosswinds",),
    }
    html = html_exporter(payload)
    expected_fragments = [
        "<h2>Global metrics</h2>",
        "<h2>Robustness</h2>",
        "<h3>Microsector variability</h3>",
        "<h2>Pareto Front</h2>",
        "<h2>A/B comparison</h2>",
        "<h2>Playbook suggestions</h2>",
        "<h2>Session profile</h2>",
        "Prioritise mid-apex",
        "Session affected by crosswinds",
        "<h2>Data sources and estimates</h2>",
    ]
    for fragment in expected_fragments:
        assert fragment in html


def test_html_exporter_handles_missing_optional_sections() -> None:
    html = html_exporter({"delta_nfr": 0.0})
    assert "Global metrics" in html
    assert "Pareto Front" not in html
    assert "A/B comparison" not in html


def test_native_encoder_writes_gearing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TNFR_LFS_NATIVE_EXPORT", "1")
    module = importlib.import_module("tnfr_lfs.exporters.lfs_native")
    importlib.reload(module)
    plan = build_native_export_plan(
        session="FP2",
        change_overrides={
            "final_drive_ratio": {"delta": 3.85},
            "gear_1_ratio": {"delta": 3.2},
            "gear_2_ratio": {"delta": 2.05},
            "gear_6_ratio": {"delta": 0.92},
        },
    )
    payload = module.encode_native_setup(plan)
    assert struct.unpack_from("<f", payload, 28)[0] == pytest.approx(3.85)
    assert struct.unpack_from("<f", payload, 32)[0] == pytest.approx(3.2)
    assert struct.unpack_from("<f", payload, 36)[0] == pytest.approx(2.05)
    assert struct.unpack_from("<f", payload, 72)[0] == pytest.approx(0.92)


def test_coherence_map_exporter_produces_track_summary(
    synthetic_microsectors, synthetic_bundles
) -> None:
    payload = {"microsectors": synthetic_microsectors, "series": synthetic_bundles}
    output = coherence_map_exporter(payload)
    data = json.loads(output)
    assert "microsectors" in data
    assert data["microsectors"]
    first = data["microsectors"][0]
    assert "coherence" in first
    assert first["coherence"]["series"]
    track = first["track"]
    assert track["end_distance"] >= track["start_distance"]
    assert data["global"]["mean_coherence"] >= 0.0


def test_operator_trajectory_exporter_serialises_events(
    synthetic_microsectors, synthetic_bundles
) -> None:
    payload = {"microsectors": synthetic_microsectors, "series": synthetic_bundles}
    output = operator_trajectory_exporter(payload)
    data = json.loads(output)
    events = data.get("events", [])
    assert events
    first = events[0]
    assert first["type"] in {
        canonical_operator_label("AL"),
        canonical_operator_label("OZ"),
        canonical_operator_label("IL"),
    }
    assert first["delta_metrics"]["peak"] == pytest.approx(
        first["delta_metrics"]["peak"], rel=1e-6
    )


def test_delta_bifurcation_exporter_detects_transitions(
    synthetic_microsectors, synthetic_bundles
) -> None:
    payload = {"microsectors": synthetic_microsectors, "series": synthetic_bundles}
    output = delta_bifurcation_exporter(payload)
    data = json.loads(output)
    assert len(data["series"]) == len(synthetic_bundles)
    stats = data["derivative_stats"]
    assert stats["count"] == len(synthetic_bundles) - 1
    assert "transitions" in data


@pytest.mark.parametrize("car_model", SUPPORTED_CAR_MODELS)
def test_lfs_set_exporter_writes_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, car_model: str
) -> None:
    monkeypatch.chdir(tmp_path)
    prefix = CAR_MODEL_PREFIXES[car_model]
    plan = build_setup_plan(car_model)
    message = lfs_set_exporter({"setup_plan": plan, "set_output": f"{prefix}_custom"})
    destination = tmp_path / f"LFS/data/setups/{prefix}_custom.set"
    assert destination.exists()
    contents = destination.read_text(encoding="utf8")
    assert "brake_bias_pct" in contents
    assert "=+2.000" in contents
    assert "TNFR-LFS setup export" in contents
    assert str(destination) in message


@pytest.mark.parametrize("car_model", SUPPORTED_CAR_MODELS)
def test_lfs_set_exporter_validates_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, car_model: str
) -> None:
    monkeypatch.chdir(tmp_path)
    plan = build_setup_plan(car_model)
    prefix = CAR_MODEL_PREFIXES[car_model]
    wrong_prefix = "XFG" if prefix != "XFG" else "FXR"
    with pytest.raises(ValueError):
        lfs_set_exporter({"setup_plan": plan, "set_output": f"{wrong_prefix}_invalid"})


def test_normalise_set_output_name_requires_prefix() -> None:
    name = normalise_set_output_name("FZR_race", "FZR")
    assert name == "FZR_race.set"
    with pytest.raises(ValueError):
        normalise_set_output_name("XFG_setup", "FZR")


def test_quickstart_reports_include_new_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    subprocess.run(["make", "quickstart"], check=True)
    baseline_dir = root / "out" / "baseline"
    occupancy_path = baseline_dir / "window_occupancy.json"
    pairwise_path = baseline_dir / "pairwise_coupling.json"
    memory_path = baseline_dir / "sense_memory.json"
    summary_path = baseline_dir / "metrics_summary.md"
    dissonance_path = baseline_dir / "dissonance_breakdown.json"

    assert occupancy_path.exists()
    occupancy = json.loads(occupancy_path.read_text(encoding="utf8"))
    assert occupancy and "window_occupancy" in occupancy[0]

    pairwise = json.loads(pairwise_path.read_text(encoding="utf8"))
    assert "global" in pairwise and "pairwise" in pairwise
    assert "delta_nfr_vs_sense_index" in pairwise["global"]

    memory = json.loads(memory_path.read_text(encoding="utf8"))
    assert "memory" in memory and isinstance(memory["memory"], list)

    dissonance = json.loads(dissonance_path.read_text(encoding="utf8"))
    assert "useful_magnitude" in dissonance

    summary = summary_path.read_text(encoding="utf8")
    assert "Useful dissonance" in summary
    assert "Coupling metrics" in summary
