"""Wrapper over :func:`tnfr.operators.grammar_validate.validate_grammar`.

Enforces the canonical TNFR U1–U6 invariants on a proposed operator
sequence (typically the operators yielded by the triggers that fired in
this stint). On failure returns a human-readable refusal message that
the advisor surfaces verbatim to the user — we never silently drop or
reorder operators.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tnfr.operators.definitions import Operator
from tnfr.operators.grammar_validate import GrammarValidator, validate_grammar


@dataclass(frozen=True)
class GrammarResult:
    """Outcome of validating an operator sequence against U1–U6."""

    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.ok


def validate_sequence(
    operators: Sequence[Operator],
    *,
    epi_initial: float = 0.0,
) -> GrammarResult:
    """Validate ``operators`` against the canonical TNFR grammar.

    Parameters
    ----------
    operators
        Sequence of :class:`tnfr.operators.definitions.Operator` instances
        (typically obtained from ``rule.operator_factory()`` for each
        fired :class:`~lfs_telemetry.tnfr_racing.operators.TriggerRule`).
    epi_initial
        Starting EPI of the network. Use the current mean network EPI
        before applying the proposed setup change.

    Returns
    -------
    GrammarResult
        ``ok=True`` if the sequence satisfies all U1–U6 constraints.
        On failure, ``reason`` contains the validator's diagnostic
        message (never empty when ``ok=False``).
    """
    ops = list(operators)
    if not ops:
        return GrammarResult(ok=False, reason="empty operator sequence")

    # Quick path: boolean answer.
    if validate_grammar(ops, epi_initial=float(epi_initial)):
        return GrammarResult(ok=True)

    # Detailed path: ask the validator for a diagnostic message.
    validator = GrammarValidator()
    try:
        report = validator.validate(ops, epi_initial=float(epi_initial))
    except Exception as exc:  # defensive: validator API surface may evolve
        return GrammarResult(ok=False, reason=f"grammar validator error: {exc!r}")

    # ``report`` is the canonical (ok, messages) shape exposed by tnfr.
    if isinstance(report, tuple) and len(report) == 2:
        ok, msgs = report
        if ok:
            return GrammarResult(ok=True)
        text = "; ".join(str(m) for m in (msgs or [])) or "grammar U1-U6 violated"
        return GrammarResult(ok=False, reason=text)

    # Unknown shape — surface as a string but flag failure.
    return GrammarResult(ok=False, reason=str(report))


__all__ = ("GrammarResult", "validate_sequence")
