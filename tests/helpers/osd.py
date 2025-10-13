"""OSD-specific test helpers."""

from __future__ import annotations

from collections.abc import Iterable

from tnfr_lfs.cli.osd import MacroStatus, TelemetryHUD
from tnfr_core.epi import TelemetryRecord
from tnfr_core.metrics import WindowMetrics
from tnfr_lfs.exporters.setup_plan import SetupPlan

from tests.helpers.steering import build_parallel_window_metrics


def _populate_hud(records: Iterable[TelemetryRecord]) -> TelemetryHUD:
    """Build a :class:`TelemetryHUD` populated with ``records``."""

    hud = TelemetryHUD(car_model="FZR", track_name="AS5", plan_interval=0.0)
    for record in records:
        hud.append(record)
    return hud


class DummyHUD:
    """Minimal HUD stub that exposes pages and plan information."""

    def __init__(self, plan: SetupPlan) -> None:
        self._plan = plan
        self._status = MacroStatus()
        self._pages: tuple[str, str, str, str] = (
            "Page A",
            "Page B",
            "Page C",
            "Apply recommendations",
        )

    def pages(self) -> tuple[str, str, str, str]:
        return self._pages

    def plan(self) -> SetupPlan:
        return self._plan

    def update_macro_status(self, status: MacroStatus) -> None:
        self._status = status
        warning_prefix = "⚠️ " if getattr(status, "warnings", None) else ""
        self._pages = (
            self._pages[0],
            self._pages[1],
            self._pages[2],
            f"{warning_prefix}Apply recommendations",
        )


def _window_metrics_from_parallel_turn(
    slip_angles: tuple[tuple[float, float], ...]
) -> WindowMetrics:
    """Construct window metrics for a synthetic parallel turn."""

    base_overrides = {
        "si": 0.78,
        "speed": 50.0,
        "throttle": 0.45,
        "vertical_load": 4800.0,
        "vertical_load_front": 2400.0,
        "vertical_load_rear": 2400.0,
        "mu_eff_front_longitudinal": 0.96,
        "mu_eff_rear_longitudinal": 0.94,
    }
    record_overrides = [{**base_overrides} for _ in slip_angles]
    return build_parallel_window_metrics(
        slip_angles,
        yaw_rates=[0.52 + 0.04 * index for index in range(len(slip_angles))],
        steer_series=[0.2 + 0.05 * index for index in range(len(slip_angles))],
        nfr_series=[105.0 + index for index in range(len(slip_angles))],
        timestamp_step=0.5,
        record_overrides=record_overrides,
    )
