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
from tnfr_lfs.exporters import csv_exporter, json_exporter


def build_payload():
    def build_nodes(delta_nfr: float, sense_index: float):
        return dict(
            tyres=TyresNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
            suspension=SuspensionNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
            chassis=ChassisNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
            brakes=BrakesNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
            transmission=TransmissionNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
            track=TrackNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
            driver=DriverNode(delta_nfr=delta_nfr / 7, sense_index=sense_index),
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
