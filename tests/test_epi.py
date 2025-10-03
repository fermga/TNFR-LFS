import pytest

from tnfr_lfs.core.epi import DeltaCalculator, EPIExtractor
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


def test_delta_calculation_against_baseline(synthetic_records):
    baseline = DeltaCalculator.derive_baseline(synthetic_records)
    sample = synthetic_records[0]
    bundle = DeltaCalculator.compute_bundle(sample, baseline, epi_value=0.0)

    assert isinstance(bundle, EPIBundle)
    assert bundle.delta_nfr == pytest.approx(sample.nfr - baseline.nfr)
    assert 0.0 <= bundle.sense_index <= 1.0
    assert bundle.track.delta_nfr + bundle.suspension.delta_nfr != 0.0
    assert bundle.dEPI_dt == pytest.approx(
        sum(
            getattr(bundle, node).nu_f * getattr(bundle, node).delta_nfr
            for node in ("tyres", "suspension", "chassis", "brakes", "transmission", "track", "driver")
        ),
        rel=1e-6,
    )
    assert bundle.integrated_epi == pytest.approx(bundle.epi)


def test_epi_extractor_creates_structured_nodes(synthetic_bundles):
    assert len(synthetic_bundles) == 17
    pivot = synthetic_bundles[5]

    assert isinstance(pivot.tyres, TyresNode)
    assert isinstance(pivot.suspension, SuspensionNode)
    assert isinstance(pivot.chassis, ChassisNode)
    assert isinstance(pivot.brakes, BrakesNode)
    assert isinstance(pivot.transmission, TransmissionNode)
    assert isinstance(pivot.track, TrackNode)
    assert isinstance(pivot.driver, DriverNode)
    assert 0.0 <= pivot.tyres.sense_index <= 1.0
    assert pivot.tyres.nu_f > 0
    assert pivot.suspension.nu_f > 0
    assert 0.0 <= pivot.track.sense_index <= 1.0
    assert sum(
        node.delta_nfr for node in (
            pivot.tyres,
            pivot.suspension,
            pivot.chassis,
            pivot.brakes,
            pivot.transmission,
            pivot.track,
            pivot.driver,
        )
    ) == pytest.approx(pivot.delta_nfr, rel=1e-6)
    assert pivot.dEPI_dt != pytest.approx(0.0)
    previous = synthetic_bundles[4]
    dt = pivot.timestamp - previous.timestamp
    expected_integral = previous.integrated_epi + pivot.dEPI_dt * dt
    assert pivot.integrated_epi == pytest.approx(expected_integral, rel=1e-3)


def test_epi_weights_shift_balance_between_load_and_slip(synthetic_records):
    default_results = EPIExtractor().extract(synthetic_records)
    slip_focused = EPIExtractor(load_weight=0.2, slip_weight=0.8).extract(synthetic_records)

    assert default_results[0].epi != pytest.approx(slip_focused[0].epi)
    assert default_results[-1].epi != pytest.approx(slip_focused[-1].epi)
    assert all(0.0 <= bundle.sense_index <= 1.0 for bundle in slip_focused)
