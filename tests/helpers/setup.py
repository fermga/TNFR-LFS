"""Reusable factories for constructing :class:`SetupPlan` fixtures in tests."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from tnfr_lfs.exporters.setup_plan import SetupChange, SetupPlan


_DEFAULT_CHANGE: dict[str, Any] = {
    "parameter": "",
    "delta": 0.0,
    "rationale": "",
    "expected_effect": "",
}


def _normalise_changes(
    changes: Iterable[SetupChange | Mapping[str, Any]],
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[SetupChange, ...]:
    """Convert change definitions into :class:`SetupChange` instances."""

    overrides = overrides or {}
    normalised: list[SetupChange] = []
    seen: set[str] = set()
    for change in changes:
        if isinstance(change, SetupChange):
            data: dict[str, Any] = {
                "parameter": change.parameter,
                "delta": change.delta,
                "rationale": change.rationale,
                "expected_effect": change.expected_effect,
            }
        else:
            data = dict(_DEFAULT_CHANGE)
            data.update(change)
        parameter = data.get("parameter")
        if not parameter:
            raise ValueError("Change definition must include a 'parameter'")
        if parameter in overrides:
            data.update(overrides[parameter])
        normalised.append(SetupChange(**data))
        seen.add(parameter)
    for parameter, data in overrides.items():
        if parameter in seen:
            continue
        merged = dict(_DEFAULT_CHANGE)
        merged.update({"parameter": parameter})
        merged.update(data)
        if not merged.get("parameter"):
            merged["parameter"] = parameter
        normalised.append(SetupChange(**merged))
    return tuple(normalised)


def build_setup_plan(
    car_model: str = "XFG",
    *,
    session: str | None = "FP1",
    changes: Iterable[SetupChange | Mapping[str, Any]] | None = None,
    change_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    **overrides: Any,
) -> SetupPlan:
    """Return a populated :class:`SetupPlan` with sensible defaults."""

    base_changes = (
        SetupChange(
            parameter="brake_bias_pct",
            delta=2.0,
            rationale="Rebalance braking support on entry",
            expected_effect="Shift brake bias forward by 2.0%",
        ),
        SetupChange(
            parameter="front_arb_steps",
            delta=-1.0,
            rationale="Tighten rotation through apex",
            expected_effect="Reduce front anti-roll bar by 1 step",
        ),
    )
    defaults: dict[str, Any] = {
        "car_model": car_model,
        "session": session,
        "sci": 0.94,
        "changes": base_changes,
        "rationales": ("Telemetry indicates oscillations during entry phases",),
        "expected_effects": ("Optimised braking and roll balance",),
        "sensitivities": {
            "sense_index": {"brake_bias_pct": -0.0125, "front_arb_steps": 0.0451},
            "sci": {"brake_bias_pct": 0.0312, "front_arb_steps": -0.0148},
            "delta_nfr_integral": {
                "brake_bias_pct": -0.021,
                "front_arb_steps": 0.084,
            },
        },
        "phase_sensitivities": {
            "entry": {"delta_nfr_integral": {"front_arb_steps": -0.062}},
            "apex": {"delta_nfr_integral": {"front_arb_steps": -0.084}},
        },
        "clamped_parameters": ("front_arb_steps",),
        "tnfr_rationale_by_node": {
            "tyres": ("Adjust pressures to stabilise the grip window",),
            "suspension": ("Increase lateral support through mid phases",),
        },
        "tnfr_rationale_by_phase": {
            "entry": ("Reduce load transfer towards the front axle",),
            "exit": ("Increase traction during extended exits",),
        },
        "expected_effects_by_node": {
            "tyres": ("Improved thermal stability",),
        },
        "expected_effects_by_phase": {
            "entry": ("More consistent braking",),
        },
        "phase_axis_targets": {
            "entry": {"longitudinal": 0.4, "lateral": 0.1},
            "apex": {"longitudinal": 0.05, "lateral": 0.3},
        },
        "phase_axis_weights": {
            "entry": {"longitudinal": 0.7, "lateral": 0.3},
            "apex": {"longitudinal": 0.2, "lateral": 0.8},
        },
        "aero_mechanical_coherence": 0.72,
        "sci_breakdown": {
            "sense": 0.32,
            "delta": 0.18,
            "udr": 0.12,
            "bottoming": 0.1,
            "aero": 0.09,
        },
    }
    defaults.update(overrides)
    selected_changes = changes if changes is not None else defaults.get("changes", ())
    defaults["changes"] = _normalise_changes(selected_changes, change_overrides)
    return SetupPlan(**defaults)


def build_native_export_plan(
    car_model: str = "XFG",
    *,
    session: str | None = "practice",
    changes: Iterable[SetupChange | Mapping[str, Any]] | None = None,
    change_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    **overrides: Any,
) -> SetupPlan:
    """Return a plan covering the native exporter regression payload."""

    base_changes = (
        SetupChange("front_camber_deg", -2.5, "", ""),
        SetupChange("rear_camber_deg", -1.8, "", ""),
        SetupChange("front_toe_deg", 0.1, "", ""),
        SetupChange("rear_toe_deg", -0.2, "", ""),
        SetupChange("caster_deg", 3.4, "", ""),
        SetupChange("parallel_steer", 0.25, "", ""),
        SetupChange("steering_lock_deg", 32.0, "", ""),
        SetupChange("front_ride_height", 55.0, "", ""),
        SetupChange("rear_ride_height", 60.0, "", ""),
        SetupChange("front_spring_stiffness", 75.0, "", ""),
        SetupChange("rear_spring_stiffness", 80.0, "", ""),
        SetupChange("front_rebound_clicks", 12.0, "", ""),
        SetupChange("rear_rebound_clicks", 10.0, "", ""),
        SetupChange("front_compression_clicks", 8.0, "", ""),
        SetupChange("rear_compression_clicks", 9.0, "", ""),
        SetupChange("front_arb_steps", 3200.0, "", ""),
        SetupChange("rear_arb_steps", 2800.0, "", ""),
        SetupChange("front_tyre_pressure", 190.0, "", ""),
        SetupChange("rear_tyre_pressure", 195.0, "", ""),
        SetupChange("brake_bias_pct", 68.5, "", ""),
        SetupChange("brake_max_per_wheel", 0.12, "", ""),
        SetupChange("diff_power_lock", 45.0, "", ""),
        SetupChange("diff_coast_lock", 35.0, "", ""),
        SetupChange("diff_preload_nm", 80.0, "", ""),
        SetupChange("final_drive_ratio", 3.72, "", ""),
        SetupChange("gear_1_ratio", 3.10, "", ""),
        SetupChange("gear_2_ratio", 2.12, "", ""),
        SetupChange("gear_3_ratio", 1.55, "", ""),
        SetupChange("gear_4_ratio", 1.22, "", ""),
        SetupChange("gear_5_ratio", 1.00, "", ""),
        SetupChange("gear_6_ratio", 0.88, "", ""),
        SetupChange("rear_wing_angle", 12.0, "", ""),
        SetupChange("front_wing_angle", 8.0, "", ""),
    )
    defaults: dict[str, Any] = {
        "car_model": car_model,
        "session": session,
        "changes": base_changes,
    }
    defaults.update(overrides)
    selected_changes = changes if changes is not None else defaults.get("changes", ())
    defaults["changes"] = _normalise_changes(selected_changes, change_overrides)
    return SetupPlan(**defaults)


def build_minimal_setup_plan(
    car_model: str = "FZR",
    *,
    session: str | None = None,
    changes: Iterable[SetupChange | Mapping[str, Any]] | None = None,
    change_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    **overrides: Any,
) -> SetupPlan:
    """Return a minimal plan suitable for UI driven tests."""

    defaults: dict[str, Any] = {
        "car_model": car_model,
        "session": session,
        "sci": 0.0,
        "changes": (),
        "rationales": (),
        "expected_effects": (),
        "sensitivities": {},
        "clamped_parameters": (),
    }
    defaults.update(overrides)
    selected_changes = changes if changes is not None else defaults.get("changes", ())
    defaults["changes"] = _normalise_changes(selected_changes, change_overrides)
    return SetupPlan(**defaults)


__all__ = [
    "build_setup_plan",
    "build_native_export_plan",
    "build_minimal_setup_plan",
]
