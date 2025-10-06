from __future__ import annotations

from tnfr_lfs.analysis.abtest import ABResult
from tnfr_lfs.exporters.report_extended import html_exporter


def test_html_exporter_renders_key_sections() -> None:
    abtest = ABResult(
        metric="sense_index",
        baseline_laps=(0.80, 0.81),
        variant_laps=(0.90, 0.91),
        baseline_mean=0.805,
        variant_mean=0.905,
        mean_difference=0.100,
        bootstrap_low=0.080,
        bootstrap_high=0.120,
        permutation_p_value=0.0125,
        estimated_power=0.85,
        alpha=0.05,
    )
    results = {
        "session": {
            "car_model": "XFG",
            "track_profile": "BL1",
            "session": "Stint 1",
            "abtest": abtest,
            "playbook_suggestions": ["Ajusta presiones delanteras"],
        },
        "session_messages": ["Sesión evaluada correctamente."],
        "delta_nfr": -1.2,
        "sense_index": 0.88,
        "microsector_variability": [
            {
                "label": "Curva 1",
                "overall": {
                    "samples": 5,
                    "delta_nfr": {"stdev": 0.12},
                    "sense_index": {"stdev": 0.02, "stability_score": 0.95},
                },
            }
        ],
        "recursive_trace": [0.10, 0.20, 0.25],
        "pairwise_coupling": {"delta_nfr": {"tyres↔suspension": 0.18}},
        "pareto_points": [
            {
                "decision_vector": {"label": "Setup B"},
                "score": 0.95,
                "breakdown": {"lap_time": 99.0, "tyre_wear": 7.0},
                "objective_breakdown": {},
            }
        ],
        "abtest": abtest,
    }

    html = html_exporter(results)

    assert "<h2>Métricas globales</h2>" in html
    assert "<h2>Robustez</h2>" in html
    assert "Variabilidad por microsector" in html
    assert "Frente Pareto" in html
    assert "Comparación A/B" in html
    assert "Sugerencias de playbook" in html
    assert "Perfil de sesión" in html
