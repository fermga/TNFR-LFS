"""Thresholds, defaults and seed for the TNFR Setup Advisor."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AdvisorConfig:
    """Runtime configuration. All values are placeholders for Phase 1."""

    seed: int = 17
    # Phase 1 stub — populated in Phase 6 (operators) and Phase 7 (grammar).
    operator_thresholds: dict[str, float] = field(default_factory=dict)
    coherence_window_s: float = 1.0
