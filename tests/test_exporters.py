import importlib
import json
import struct
import subprocess
from pathlib import Path

import pytest

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
from tnfr_lfs.exporters import (
    CAR_MODEL_PREFIXES,
    coherence_map_exporter,
    csv_exporter,
    delta_bifurcation_exporter,
    json_exporter,
    lfs_notes_exporter,
    lfs_set_exporter,
    markdown_exporter,
    normalise_set_output_name,
    operator_trajectory_exporter,
)
from tnfr_lfs.exporters.setup_plan import SetupChange, SetupPlan, serialise_setup_plan


BASE_NU_F = {
    "tyres": 0.18,
    "suspension": 0.14,
    "chassis": 0.12,
    "brakes": 0.16,
    "transmission": 0.11,
    "track": 0.08,
    "driver": 0.05,
}


SUPPORTED_CAR_MODELS = [
    "XFG",
    "XRG",
    "RB4",
    "FXO",
    "FXR",
    "XRR",
    "FZR",
    "FO8",
    "BF1",
]


def build_payload():
    def build_nodes(delta_nfr: float, sense_index: float):
        return dict(
            tyres=TyresNode(delta_nfr=delta_nfr / 7, sense_index=sense_index, nu_f=BASE_NU_F["tyres"]),
            suspension=SuspensionNode(delta_nfr=delta_nfr / 7, sense_index=sense_index, nu_f=BASE_NU_F["suspension"]),
            chassis=ChassisNode(delta_nfr=delta_nfr / 7, sense_index=sense_index, nu_f=BASE_NU_F["chassis"]),
            brakes=BrakesNode(delta_nfr=delta_nfr / 7, sense_index=sense_index, nu_f=BASE_NU_F["brakes"]),
            transmission=TransmissionNode(delta_nfr=delta_nfr / 7, sense_index=sense_index, nu_f=BASE_NU_F["transmission"]),
            track=TrackNode(delta_nfr=delta_nfr / 7, sense_index=sense_index, nu_f=BASE_NU_F["track"]),
            driver=DriverNode(delta_nfr=delta_nfr / 7, sense_index=sense_index, nu_f=BASE_NU_F["driver"]),
        )

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
        EPIBundle(0.0, 0.45, -5.0, 0.82, delta_breakdown=build_breakdown(-5.0), **build_nodes(-5.0, 0.82)),
        EPIBundle(0.1, 0.50, 10.0, 0.74, delta_breakdown=build_breakdown(10.0), **build_nodes(10.0, 0.74)),
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
        == "timestamp,epi,delta_nfr,delta_nfr_longitudinal,delta_nfr_lateral,sense_index"
    )
    assert len(lines) == 3


def build_setup_plan(car_model: str = "XFG") -> SetupPlan:
    return SetupPlan(
        car_model=car_model,
        session="FP1",
        sci=0.94,
        changes=(
            SetupChange(
                parameter="brake_bias_pct",
                delta=2.0,
                rationale="Rebalance braking support on entry",
                expected_effect="+2.0% bias delante",
            ),
            SetupChange(
                parameter="front_arb_steps",
                delta=-1.0,
                rationale="Tighten rotation through apex",
                expected_effect="-1 pasos barra delantera",
            ),
        ),
        rationales=("Telemetry indicates oscillations during entry phases",),
        expected_effects=("Optimised braking and roll balance",),
        sensitivities={
            "sense_index": {"brake_bias_pct": -0.0125, "front_arb_steps": 0.0451},
            "sci": {"brake_bias_pct": 0.0312, "front_arb_steps": -0.0148},
            "delta_nfr_integral": {
                "brake_bias_pct": -0.021,
                "front_arb_steps": 0.084,
            },
        },
        phase_sensitivities={
            "entry": {"delta_nfr_integral": {"front_arb_steps": -0.062}},
            "apex": {"delta_nfr_integral": {"front_arb_steps": -0.084}},
        },
        clamped_parameters=("front_arb_steps",),
        tnfr_rationale_by_node={
            "tyres": ("Ajustar presiones para estabilizar la ventana de agarre",),
            "suspension": ("Elevar soporte lateral en fases medias",),
        },
        tnfr_rationale_by_phase={
            "entry": ("Reducir transferencia hacia el eje delantero",),
            "exit": ("Aumentar tracción en salida prolongada",),
        },
        expected_effects_by_node={
            "tyres": ("Mayor estabilidad térmica",),
        },
        expected_effects_by_phase={
            "entry": ("Frenadas más consistentes",),
        },
        phase_axis_targets={
            "entry": {"longitudinal": 0.4, "lateral": 0.1},
            "apex": {"longitudinal": 0.05, "lateral": 0.3},
        },
        phase_axis_weights={
            "entry": {"longitudinal": 0.7, "lateral": 0.3},
            "apex": {"longitudinal": 0.2, "lateral": 0.8},
        },
        aero_mechanical_coherence=0.72,
        sci_breakdown={
            "sense": 0.32,
            "delta": 0.18,
            "udr": 0.12,
            "bottoming": 0.1,
            "aero": 0.09,
        },
    )


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
    assert any("Entrada ∥" in hint for hint in payload["phase_axis_suggestions"])
    assert payload["aero_mechanical_coherence"] == pytest.approx(0.72)
    assert payload["sci_breakdown"]["sense"] == pytest.approx(0.32)
    assert payload["ics_breakdown"]["sense"] == pytest.approx(0.32)


def test_markdown_exporter_renders_table_and_lists():
    plan = build_setup_plan()
    output = markdown_exporter({"setup_plan": plan})
    assert "| Cambio | Ajuste | Racional |" in output
    assert "brake_bias_pct" in output
    assert "Telemetry indicates oscillations" in output
    assert "**dSi/dparam**" in output
    assert "**d∫|ΔNFR|/dparam**" in output
    assert "**Gradientes de ∫|ΔNFR| por fase**" in output
    assert "**Racionales TNFR por nodo**" in output
    assert "**Efectos esperados por fase**" in output
    assert "**Objetivos ΔNFR∥/ΔNFR⊥ por fase**" in output
    assert "**Mapa ΔNFR∥/ΔNFR⊥ por fase**" in output
    assert "Entrada ∥" in output
    assert "**Sugerencias de fases prioritarias**" in output
    assert "C(c/d/a)" in output
    assert "**Contribución SCI**" in output


def test_lfs_notes_exporter_renders_key_instructions():
    plan = build_setup_plan()
    output = lfs_notes_exporter({"setup_plan": plan})
    assert "Instrucciones rápidas TNFR" in output


def test_native_encoder_writes_gearing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TNFR_LFS_NATIVE_EXPORT", "1")
    module = importlib.import_module("tnfr_lfs.exporters.lfs_native")
    importlib.reload(module)
    plan = SetupPlan(
        car_model="XFG",
        session="FP2",
        changes=(
            SetupChange("final_drive_ratio", 3.85, "", ""),
            SetupChange("gear_1_ratio", 3.2, "", ""),
            SetupChange("gear_2_ratio", 2.05, "", ""),
            SetupChange("gear_6_ratio", 0.92, "", ""),
        ),
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
    assert first["type"] in {"AL", "OZ", "IL"}
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
    name = normalise_set_output_name("GEN_race", "generic_gt")
    assert name == "GEN_race.set"
    with pytest.raises(ValueError):
        normalise_set_output_name("XFG_setup", "generic_gt")


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
    assert "Disonancia útil" in summary
    assert "Acoplamientos" in summary
