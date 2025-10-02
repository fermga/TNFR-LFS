from tnfr_lfs.core.epi import DeltaCalculator, EPIExtractor, TelemetryRecord, compute_coherence


def build_records():
    return [
        TelemetryRecord(0.0, 6000, 0.05, 1.2, 0.4, 520, 0.82),
        TelemetryRecord(0.1, 6500, 0.03, 1.1, 0.5, 540, 0.81),
        TelemetryRecord(0.2, 6800, 0.07, 1.0, 0.6, 560, 0.78),
    ]


def test_delta_calculation_against_baseline():
    records = build_records()
    baseline = DeltaCalculator.derive_baseline(records)
    delta = DeltaCalculator.compute(records[0], baseline)
    assert round(delta.delta_nfr, 3) == -20.0
    assert round(delta.delta_si, 3) == 0.017


def test_epi_and_coherence_are_computed():
    records = build_records()
    extractor = EPIExtractor()
    results = extractor.extract(records)
    assert len(results) == 3
    coherence = compute_coherence(results)
    assert 0 <= coherence <= 1
    assert results[0].epi < results[2].epi
