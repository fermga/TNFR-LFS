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
from tnfr_lfs.exporters import csv_exporter, json_exporter, markdown_exporter
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

    results = [
        EPIBundle(0.0, 0.45, -5.0, 0.82, **build_nodes(-5.0, 0.82)),
        EPIBundle(0.1, 0.50, 10.0, 0.74, **build_nodes(10.0, 0.74)),
    ]
    return {"series": results, "meta": {"session": "FP1"}}


def test_json_exporter_serialises_payload():
    payload = build_payload()
    output = json_exporter(payload)
    assert "\"session\": \"FP1\"" in output


def test_csv_exporter_renders_rows():
    payload = build_payload()
    output = csv_exporter(payload)
    lines = output.strip().splitlines()
    assert lines[0] == "timestamp,epi,delta_nfr,sense_index"
    assert len(lines) == 3


def build_setup_plan() -> SetupPlan:
    return SetupPlan(
        car_model="generic_gt",
        session="FP1",
        changes=(
            SetupChange(
                parameter="rear_ride_height",
                delta=-1.5,
                rationale="Reduce load transfer towards the front axle",
                expected_effect="Improved rotation mid-corner",
            ),
            SetupChange(
                parameter="rear_wing_angle",
                delta=2.0,
                rationale="Stabilise the car during high speed sections",
                expected_effect="Higher rear downforce",
            ),
        ),
        rationales=("Telemetry indicates oscillations during entry phases",),
        expected_effects=("Balanced aero load front to rear",),
    )


def test_serialise_setup_plan_collects_unique_fields():
    plan = build_setup_plan()
    payload = serialise_setup_plan(plan)
    assert payload["car_model"] == "generic_gt"
    assert len(payload["changes"]) == 2
    assert any("Telemetry indicates" in item for item in payload["rationales"])
    assert any("Improved rotation" in item for item in payload["expected_effects"])


def test_markdown_exporter_renders_table_and_lists():
    plan = build_setup_plan()
    output = markdown_exporter({"setup_plan": plan})
    assert "| Cambio | Ajuste | Racional |" in output
    assert "rear_ride_height" in output
    assert "Telemetry indicates oscillations" in output
