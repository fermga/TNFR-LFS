"""Tests covering the deprecation warnings for legacy phase aliases."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tnfr_lfs.core.metrics import CPHIReport, CPHIWheel, CPHIWheelComponents
from tnfr_lfs.core.phases import (
    LEGACY_PHASE_MAP,
    expand_phase_alias,
    phase_family,
    replicate_phase_aliases,
)


def _pick_legacy_alias() -> tuple[str, tuple[str, ...]]:
    alias, phases = next(iter(LEGACY_PHASE_MAP.items()))
    return alias, phases


@pytest.mark.parametrize("phase_resolver", (expand_phase_alias, phase_family))
def test_phase_resolvers_warn_for_legacy_alias(
    phase_resolver: Callable[[str], object]
) -> None:
    alias, _ = _pick_legacy_alias()
    with pytest.warns(DeprecationWarning):
        phase_resolver(alias)


def test_replicate_phase_aliases_warns_when_alias_generated() -> None:
    alias, phases = _pick_legacy_alias()
    payload = {phases[0]: 1}
    with pytest.warns(DeprecationWarning):
        result = replicate_phase_aliases(payload)
    assert alias in result


def test_cphi_report_as_legacy_mapping_warns() -> None:
    components = CPHIWheelComponents(temperature=0.2, gradient=0.3, mu=0.4)
    wheel = CPHIWheel(value=0.75, components=components, temperature_delta=0.1, gradient_rate=0.05)
    report = CPHIReport(wheels={"fl": wheel})

    with pytest.warns(DeprecationWarning):
        legacy = report.as_legacy_mapping()

    assert legacy["cphi_fl"] == pytest.approx(0.75)
