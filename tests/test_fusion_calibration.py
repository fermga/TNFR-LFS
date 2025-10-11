import math
from dataclasses import replace

import pytest

from tnfr_lfs.core.epi import delta_nfr_by_node, resolve_nu_f_by_node
from tnfr_lfs.ingestion.live import TelemetryFusion

from tests.helpers import build_outgauge_sample, build_outsim_sample


@pytest.mark.parametrize("slip_sequence", [[0.02, 0.06, 0.1]])
def test_fusion_calibration_mu_eff_trend_for_xfg(slip_sequence):
    fusion = TelemetryFusion()
    records = []
    baseline = None
    for index, slip in enumerate(slip_sequence):
        outsim = build_outsim_sample(index * 0.1, 20.0, slip, lateral=5.0 + index)
        outgauge = build_outgauge_sample("XFG", "BL1", 20.0, rpm=5000.0 + index * 200.0)
        record = fusion.fuse(outsim, outgauge)
        if baseline is None:
            baseline = record
        else:
            record = replace(record, reference=baseline)
            fusion._records[-1] = record
        records.append(record)

    mu_front_lateral = [record.mu_eff_front_lateral for record in records]
    assert mu_front_lateral == sorted(mu_front_lateral)

    early_delta = delta_nfr_by_node(records[1])
    late_delta = delta_nfr_by_node(records[-1])
    assert late_delta["tyres"] >= early_delta["tyres"]


def test_fusion_calibration_nu_f_converges_for_fxr():
    fusion = TelemetryFusion()
    slip_values = [0.18, 0.1, 0.05, 0.01]
    nu_f_history = []
    for index, slip in enumerate(slip_values):
        outsim = build_outsim_sample(index * 0.2, 28.0, slip, lateral=6.5 - index * 0.8)
        outgauge = build_outgauge_sample("FXR", "ASO4", 28.0, rpm=6000.0 - index * 150.0)
        record = fusion.fuse(outsim, outgauge)
        nu_map = resolve_nu_f_by_node(record).by_node
        nu_f_history.append(nu_map["tyres"])

    assert nu_f_history[0] > nu_f_history[-1]
    expected_final = 0.18 * (1.0 + min(abs(slip_values[-1]), 1.0) * 0.2)
    assert math.isclose(nu_f_history[-1], expected_final, rel_tol=0.15)
