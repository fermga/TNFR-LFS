"""Shared helpers for AB testing fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DummyBundle:
    sense_index: float
    delta_nfr: float
    coherence_index: float
    delta_nfr_proj_longitudinal: float
    delta_nfr_proj_lateral: float


def build_metrics(lap_samples: Sequence[Sequence[float]]) -> dict[str, object]:
    bundles: list[DummyBundle] = []
    lap_indices: list[int] = []
    for lap_index, samples in enumerate(lap_samples):
        for value in samples:
            bundles.append(
                DummyBundle(
                    sense_index=value,
                    delta_nfr=value,
                    coherence_index=value,
                    delta_nfr_proj_longitudinal=value,
                    delta_nfr_proj_lateral=value,
                )
            )
            lap_indices.append(lap_index)
    return {
        "stages": {
            "coherence": {"bundles": tuple(bundles)},
            "reception": {"lap_indices": tuple(lap_indices)},
        }
    }


def scale_samples(samples: Iterable[Sequence[float]], factor: float) -> list[list[float]]:
    return [[value * factor for value in lap] for lap in samples]


__all__ = ["DummyBundle", "build_metrics", "scale_samples"]
