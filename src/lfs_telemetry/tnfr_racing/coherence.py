"""Coherence aggregates and deterministic ΔC surrogate for v1.

:func:`compute_global_coherence` wraps :func:`tnfr.metrics.coherence.compute_coherence`
with a NaN-safe fallback. Until per-domain ΔNFR hooks are wired across
all node kinds (deferred to a future phase), the advisor uses
:func:`estimate_rule_delta_coherence` — a deterministic surrogate keyed
by operator class. Each *physical* operator in the restricted advisor
palette contributes a fixed structural-gain quantum (dimensionless C
units); ``Emission`` is included with gain 0.0 because the advisor's
grammar-padding step (see :func:`advisor._pad_for_grammar`) may insert
it to satisfy U1a and we must guarantee it never inflates ΔC.

The gain magnitudes (0.02–0.05) are **v1 placeholders**: ordered by
relative TNFR operator semantics (Coherence > Coupling > Dissonance >
Reception ≈ Silence) but not field-calibrated against lap-time deltas.
The surrogate is strictly additive and deterministic; callers can rely
on it for ordering, ranking and refusal decisions without any
randomness or hidden state. The empirical validation that would turn
these into measured gains is the next-priority experiment
(see docs/TNFR_SETUP_ADVISOR.md §10.7).
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

import networkx as nx

from .operators import TriggerRule

_logger = logging.getLogger(__name__)

# Deterministic per-operator ΔC contribution (dimensionless C units).
# Only operators in the advisor's physical palette + the canonical
# grammar-padding generator (Emission, gain 0.0) are listed. Other
# canonical TNFR operators are intentionally absent: if a future rule
# emits one, the lookup falls back to 0.01 and a regression in
# operators.py's ``_KNOWN_ACTION_KINDS`` assertion would catch the
# silent broadening of the palette.
_OPERATOR_GAIN: dict[str, float] = {
    "Coherence": 0.05,
    "Coupling": 0.04,
    "Dissonance": 0.03,
    "Reception": 0.02,
    "Silence": 0.02,
    "Contraction": 0.02,
    "Emission": 0.00,  # grammar-padding (U1a); must never inflate ΔC
}


def compute_global_coherence(graph: nx.Graph) -> float:
    """Return canonical C(t) for ``graph`` (1.0 on degenerate input).

    Wraps :func:`tnfr.metrics.coherence.compute_coherence`. Returns 1.0
    when the tnfr metric raises or returns a non-finite value —
    matching the documented "empty signal → perfect coherence" boundary
    convention. The fallback path is logged at DEBUG level so a
    persistent engine fault does not stay invisible in production.
    """
    try:
        from tnfr.metrics.coherence import compute_coherence
        c = compute_coherence(graph)
        c_val = float(c if not isinstance(c, tuple) else c[0])
        if c_val != c_val or c_val < 0.0 or c_val > 1.0:  # NaN / out of range
            _logger.debug(
                "compute_global_coherence: non-finite/out-of-range value %r;"
                " returning 1.0 (boundary convention)",
                c_val,
            )
            return 1.0
        return c_val
    except Exception as exc:
        _logger.debug(
            "compute_global_coherence: tnfr engine raised %r;"
            " returning 1.0 (boundary convention)",
            exc,
        )
        return 1.0


def estimate_rule_delta_coherence(rule: TriggerRule) -> float:
    """Surrogate ΔC contribution of ``rule`` (always > 0 in v1).

    Looks up the operator class name produced by
    ``rule.operator_factory`` in the deterministic gain table.
    """
    op = rule.operator_factory()
    return float(_OPERATOR_GAIN.get(type(op).__name__, 0.01))


def estimate_sequence_delta_coherence(rules: Iterable[TriggerRule]) -> float:
    """Sum of per-rule ΔC contributions for ``rules``."""
    return float(sum(estimate_rule_delta_coherence(r) for r in rules))


__all__ = (
    "compute_global_coherence",
    "estimate_rule_delta_coherence",
    "estimate_sequence_delta_coherence",
)
