from tnfr_lfs.core.epi import EPIResult
from tnfr_lfs.exporters import csv_exporter, json_exporter


def build_payload():
    results = [
        EPIResult(0.0, 0.45, -5.0, 0.01),
        EPIResult(0.1, 0.50, 10.0, -0.02),
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
    assert lines[0] == "timestamp,epi,delta_nfr,delta_si"
    assert len(lines) == 3
