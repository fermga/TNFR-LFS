from tnfr_lfs.core.epi import DeltaCalculator, EPIExtractor, TelemetryRecord


def build_records():
    return [
        TelemetryRecord(0.0, 6000, 0.05, 1.2, 0.4, 520, 0.82),
        TelemetryRecord(0.1, 6500, 0.03, 1.1, 0.5, 540, 0.81),
        TelemetryRecord(0.2, 6800, 0.07, 1.0, 0.6, 560, 0.78),
    ]


def test_delta_calculation_against_baseline():
    records = build_records()
    baseline = DeltaCalculator.derive_baseline(records)
    bundle = DeltaCalculator.compute_bundle(records[0], baseline, epi_value=0.0)
    assert round(bundle.delta_nfr, 3) == -20.0
    assert 0.0 <= bundle.sense_index <= 1.0


def test_epi_and_coherence_are_computed():
    records = build_records()
    extractor = EPIExtractor()
    results = extractor.extract(records)
    assert len(results) == 3
    assert results[0].epi < results[2].epi
    assert all(0.0 <= bundle.sense_index <= 1.0 for bundle in results)
    assert results[1].sense_index >= results[0].sense_index
    assert results[1].sense_index >= results[2].sense_index
