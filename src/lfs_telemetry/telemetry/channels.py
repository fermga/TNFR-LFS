"""Channel metadata registry for MoTeC-style consumers.

Maps DataFrame column names → display label, units and group, so a
visualization app can build its "channel browser" tree without
hard-coding everything.

The registry is generated once from a small declaration table plus the
per-wheel naming convention used in :mod:`.replay` and :mod:`.derived`.

Usage::

    from lfs_telemetry.telemetry.channels import CHANNELS, channel_info, ChannelInfo

    info = channel_info("speed_ms")
    info.label, info.units, info.group   # 'Speed', 'm/s', 'Vehicle'
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .channel_interpretations import (
    _INTERP_BY_GROUP,
    _focus_notes_for,
    _interpretation_for,
    _interpretation_for_lang,
)
from .i18n_es import (
    _DESCRIPTION_ES_FALLBACK,
    _GROUP_ES_FALLBACK,
    _LABEL_ES_FALLBACK,
)
from .protocol.packets import WHEEL_ORDER


@dataclass(frozen=True, slots=True)
class ChannelInfo:
    """Metadata for one telemetry channel."""

    column: str
    label: str
    units: str
    group: str
    description: str = ""
    interpretation: str = ""

    def tooltip_html(
        self,
        *,
        translate: Callable[[str], str] | None = None,
        language: str | None = None,
    ) -> str:
        """Rich HTML tooltip combining label, units, description and
        interpretation guidance. Safe to feed directly into Qt's
        ``ToolTipRole`` (Qt auto-detects HTML by the leading ``<``).
        """
        tr = translate or (lambda s: s)
        lang = (language or "en").lower()

        label = tr(self.label)
        if lang == "es" and label == self.label:
            label = _LABEL_ES_FALLBACK.get(self.label, self.label)
            # Handle wheel-suffixed labels like "Susp. travel FL" by
            # translating the base label and re-appending the corner code.
            if label == self.label:
                parts_lbl = self.label.rsplit(" ", 1)
                if (
                    len(parts_lbl) == 2
                    and parts_lbl[1] in ("FL", "FR", "RL", "RR")
                ):
                    base = _LABEL_ES_FALLBACK.get(parts_lbl[0], parts_lbl[0])
                    label = f"{base} {parts_lbl[1]}"
        units_value = tr(self.units)
        group = tr(self.group)
        if lang == "es" and group == self.group:
            group = _GROUP_ES_FALLBACK.get(self.group, self.group)

        description_text = self.description
        if lang == "es" and description_text:
            description_text = _DESCRIPTION_ES_FALLBACK.get(
                description_text,
                description_text,
            )
        description = tr(description_text) if description_text else ""

        interpretation_text = self.interpretation
        if lang == "es":
            interpretation_text = _interpretation_for_lang(
                self.column,
                self.group,
                language="es",
            )
        interpretation = tr(interpretation_text) if interpretation_text else ""

        engineer_focus, driver_focus = _focus_notes_for(
            self.column,
            self.group,
            language=lang,
        )
        engineer_header = tr("Engineer focus")
        driver_header = tr("Driver focus")
        how_to_read = tr("How to read it")
        if lang == "es":
            if engineer_header == "Engineer focus":
                engineer_header = "Foco ingeniero"
            if driver_header == "Driver focus":
                driver_header = "Foco piloto"
            if how_to_read == "How to read it":
                how_to_read = "Cómo leerlo"

        units = f" [{units_value}]" if units_value else ""
        parts = [
            f"<b>{label}</b>{units}",
            f"<i>{group} &middot; <code>{self.column}</code></i>",
        ]
        if description:
            parts.append(description)
        if interpretation:
            parts.append(
                "<div style='margin-top:4px;'>"
                f"<u>{how_to_read}</u>: {interpretation}"
                "</div>"
            )
        if engineer_focus:
            parts.append(
                "<div style='margin-top:4px;'>"
                f"<u>{engineer_header}</u>: {engineer_focus}"
                "</div>"
            )
        if driver_focus:
            parts.append(
                "<div style='margin-top:4px;'>"
                f"<u>{driver_header}</u>: {driver_focus}"
                "</div>"
            )
        return "<div style='max-width:380px;'>" + "<br>".join(parts) + "</div>"

































# (column, label, units, group, description)
_BASE: tuple[tuple[str, str, str, str, str], ...] = (
    # --- Vehicle (chassis) ---
    ("speed_ms", "Speed", "m/s", "Vehicle", "Wheel-derived speed"),
    ("rpm", "RPM", "rpm", "Engine", "Engine speed"),
    ("gear_lfs", "Gear", "", "Engine",
     "Canonical LFS gear (-1=R, 0=N, 1..N=forward)"),
    ("fuel", "Fuel", "frac", "Engine", "Fuel level (0..1)"),
    ("turbo_bar", "Turbo", "bar", "Engine", "Turbo boost"),
    # --- Driver inputs ---
    ("throttle", "Throttle", "frac", "Driver", "Throttle pedal (0..1)"),
    ("brake", "Brake", "frac", "Driver", "Brake pedal (0..1)"),
    ("clutch", "Clutch", "frac", "Driver", "Clutch pedal (0..1)"),
    ("input_steer", "Steer (raw)", "rad", "Driver", "Steering wheel angle"),
    ("input_handbrake", "Handbrake", "frac", "Driver", ""),
    ("steer_torque_nm", "Steer Torque", "Nm", "Driver", "FFB steering torque"),
    # --- Chassis dynamics ---
    ("ang_vel_x", "Roll rate", "rad/s", "Chassis", ""),
    ("ang_vel_y", "Pitch rate", "rad/s", "Chassis", ""),
    ("ang_vel_z", "Yaw rate", "rad/s", "Chassis", ""),
    ("heading", "Heading", "rad", "Chassis", ""),
    ("pitch", "Pitch", "rad", "Chassis", ""),
    ("roll", "Roll", "rad", "Chassis", ""),
    ("accel_x", "Long. accel", "m/s²", "Chassis", "+forward, -brake"),
    ("accel_y", "Lat. accel", "m/s²", "Chassis", "+right"),
    ("accel_z", "Vert. accel", "m/s²", "Chassis", "+up"),
    ("vel_x", "Vel. X", "m/s", "Chassis", "World X velocity"),
    ("vel_y", "Vel. Y", "m/s", "Chassis", "World Y velocity"),
    ("vel_z", "Vel. Z", "m/s", "Chassis", "World Z velocity"),
    ("pos_x", "Pos. X", "m", "Chassis", "World X position"),
    ("pos_y", "Pos. Y", "m", "Chassis", "World Y position"),
    ("pos_z", "Pos. Z", "m", "Chassis", "World Z position"),
    # --- Derived (chassis dynamics) ---
    ("yaw_rate_rads", "Yaw rate", "rad/s", "Derived", "Body yaw rate (z)"),
    ("yaw_rate_theoretical_rads", "Yaw rate (th)", "rad/s", "Derived",
     "Neutral-steer yaw rate from steered angle"),
    ("understeer_index", "Understeer idx", "", "Derived", "+= understeer"),
    ("transfer_long_n_real", "Long. transfer", "N", "Derived", "Real long. load Δ"),
    ("transfer_lat_n_real", "Lat. transfer", "N", "Derived", "Real lat. load Δ"),
    ("transfer_long_n_theoretical", "Long. transfer (th)", "N", "Derived", ""),
    ("transfer_lat_n_theoretical", "Lat. transfer (th)", "N", "Derived", ""),
    ("load_total_n", "Total load", "N", "Derived", "Sum of vertical loads"),
    ("load_front_frac", "Front load frac", "", "Derived", ""),
    ("load_left_frac", "Left load frac", "", "Derived", ""),
    ("load_diag_fl_rr_frac", "Diag FL/RR frac", "", "Derived", ""),
    ("brake_bias_front_real", "Brake bias front", "frac", "Derived", "Real, when braking"),
    ("ffb_load_pct", "FFB load", "frac", "Derived", "|steer_torque|/max"),
    ("steer_rate_rads", "Steer rate", "rad/s", "Derived", "d(input_steer)/dt"),
    ("steer_reversal_rate_hz", "Steer reversals", "Hz", "Derived", ""),
    # --- Derived (combined / synergy channels) ---
    ("g_total_g", "g-force total", "g", "Derived",
     "sqrt(ax²+ay²)/g; friction-circle headline magnitude"),
    ("susp_compression_front_avg_m", "Front compression", "m", "Derived",
     "Mean front-axle suspension compression"),
    ("susp_compression_rear_avg_m", "Rear compression", "m", "Derived",
     "Mean rear-axle suspension compression"),
    ("rake_compression_m", "Rake (Δ compression)", "m", "Derived",
     "Front − rear axle compression; +ve = nose-down"),
    ("slip_angle_balance_rad", "Slip balance (F−R)", "rad", "Derived",
     "mean|α_front|−mean|α_rear|; +ve=understeer"),
    ("brake_power_w", "Brake power", "W", "Derived",
     "|F_long_total|·v while braking; brake-heat proxy"),
    ("throttle_reversal_rate_hz", "Throttle reversals", "Hz", "Derived",
     "Throttle direction changes per second"),
    ("coasting", "Coasting", "bool", "Derived",
     "throttle<0.05 ∧ brake<0.05 ∧ v>3 m/s"),
    ("trail_brake_intensity", "Trail-brake intensity", "", "Derived",
     "brake × |input_steer|; combined corner-entry load"),
    ("chassis_roll_per_lat_g_rad_per_g", "Roll compliance",
     "rad/g", "Derived",
     "Instantaneous roll/ay; ARB/spring tuning probe"),
    ("chassis_pitch_per_long_g_rad_per_g", "Pitch compliance",
     "rad/g", "Derived",
     "Instantaneous pitch/ax; brake-dive / squat probe"),
    # --- Derived (track geometry; require a racing_lines/<TRACK>_racing.csv) ---
    ("track_node", "Track node", "", "Track", "Nearest centerline node index"),
    ("track_s_m", "Track s", "m", "Track", "Arc length along centerline"),
    ("track_z_m", "Track elevation", "m", "Track", "Centerline Z at this s"),
    ("track_heading_rad", "Track heading", "rad", "Track", "Centerline tangent heading"),
    ("track_curvature_1_per_m", "Track curvature", "1/m", "Track", "+left, -right"),
    ("track_radius_m", "Track radius", "m", "Track", "1/|curvature|, capped"),
    ("track_slope_pct", "Track slope", "%", "Track", "100·dz/ds along centerline"),
    ("track_width_m", "Track width", "m", "Track", "drive_right - drive_left at node"),
    ("track_offset_m", "Lateral offset", "m", "Track", "Perp. distance to centerline"),
    ("segment_id", "Segment id", "", "Track", "Geometry segment index"),
    ("accel_x_road_mps2", "Long. accel (road)", "m/s²", "Derived",
     "accel_x with road-grade gravity removed (+engine, -brake)"),
    ("yaw_misalign_rad", "Yaw misalignment", "rad", "Derived",
     "vel-heading − track-heading; trajectory slip proxy"),
    # --- Dash lights (booleans 0/1) ---
    ("dl_tc_active", "TC active", "bool", "Aids", ""),
    ("dl_abs_active", "ABS active", "bool", "Aids", ""),
    ("dl_handbrake", "Handbrake on", "bool", "Aids", ""),
    ("dl_pit_limiter", "Pit limiter", "bool", "Aids", ""),
    ("dl_oil_warn", "Oil warning", "bool", "Aids", ""),
    ("dl_battery_warn", "Battery warn", "bool", "Aids", ""),
    ("dl_signal_l", "Indicator L", "bool", "Aids", ""),
    ("dl_signal_r", "Indicator R", "bool", "Aids", ""),
    ("dl_fullbeam", "Full beam", "bool", "Aids", ""),
    ("dl_shift_light", "Shift light", "bool", "Aids", ""),
    # --- Race context ---
    ("ctx_wind", "Wind", "m/s", "Context", "Wind speed magnitude"),
)


# Per-wheel template: (suffix, label, units, group, description).
_WHEEL_TEMPLATE: tuple[tuple[str, str, str, str, str], ...] = (
    ("susp_deflect_m", "Susp. travel", "m", "Suspension", ""),
    ("vertical_load_n", "Vert. load", "N", "Suspension", ""),
    ("slip_ratio", "Slip ratio", "", "Tyre", ""),
    ("tan_slip_angle", "tan(α)", "", "Tyre", ""),
    ("x_force_n", "Lat. force", "N", "Tyre", "Tyre lateral force"),
    ("y_force_n", "Long. force", "N", "Tyre", "Tyre longitudinal force"),
    ("ang_vel_rads", "Wheel ω", "rad/s", "Tyre", ""),
    ("lean_rel_road_rad", "Camber rel.", "rad", "Suspension", ""),
    ("air_temp_c", "Tyre air temp", "°C", "Tyre", ""),
    ("slip_fraction", "Slip frac", "", "Tyre", "Sliding contact fraction"),
    ("touching", "Touching", "bool", "Tyre", ""),
    ("steer_rad", "Wheel steer", "rad", "Suspension", ""),
)


# Per-wheel template for *derived* columns.
_WHEEL_DERIVED: tuple[tuple[str, str, str, str, str], ...] = (
    ("friction_use_{c}", "Friction use", "", "Derived", "Used / available μ"),
    ("tyre_work_w_{c}", "Tyre work", "W", "Derived", "Mech. power into tyre"),
    ("wheel_{c}_susp_speed_mps", "Susp. speed", "m/s", "Suspension",
     "Damper velocity (>0 bump, <0 rebound)"),
    ("wheel_{c}_lockup", "Lockup", "bool", "Derived",
     "Braking ∧ slip_ratio<−0.3; ABS event detector"),
)


# ---------------------------------------------------------------------------
# Interpretation guide (English, plain language with telemetry context).
#
# These strings answer "what is this plot telling me?" for someone who
# knows cars but not telemetry. Where relevant they cite the LFS source
# (OutSim/OutGauge/InSim packet fields) and standard vehicle-dynamics
# concepts (slip angle/ratio, friction ellipse, weight transfer, etc.).
#
# Resolution order:
#   1. exact column-name match,
#   2. suffix match (per-wheel channels),
#   3. prefix match (derived patterns),
#   4. group fallback,
#   5. empty string.
# ---------------------------------------------------------------------------










def _build_registry() -> dict[str, ChannelInfo]:
    out: dict[str, ChannelInfo] = {}
    for col, lbl, units, group, desc in _BASE:
        out[col] = ChannelInfo(
            col, lbl, units, group, desc, _interpretation_for(col, group)
        )
    for c in WHEEL_ORDER:
        for suffix, lbl, units, group, desc in _WHEEL_TEMPLATE:
            col = f"wheel_{c}_{suffix}"
            out[col] = ChannelInfo(
                col, f"{lbl} {c}", units, group, desc,
                _interpretation_for(col, group),
            )
        for tmpl, lbl, units, group, desc in _WHEEL_DERIVED:
            col = tmpl.format(c=c)
            out[col] = ChannelInfo(
                col, f"{lbl} {c}", units, group, desc,
                _interpretation_for(col, group),
            )
    return out


CHANNELS: dict[str, ChannelInfo] = _build_registry()


def channel_info(column: str) -> ChannelInfo:
    """Return :class:`ChannelInfo` for ``column``; falls back to bare metadata."""
    info = CHANNELS.get(column)
    if info is not None:
        return info
    # Unknown column → best-effort fallback so the UI never blows up.
    return ChannelInfo(
        column=column, label=column, units="", group="Other",
        interpretation=_INTERP_BY_GROUP.get("Other", ""),
    )


def channels_by_group(columns: list[str] | None = None) -> dict[str, list[ChannelInfo]]:
    """Group known channels by ``group`` for tree-style channel browsers.

    If ``columns`` is given, restrict the listing to those (handy after
    inspecting a real DataFrame's ``df.columns``).
    """
    cols = columns if columns is not None else list(CHANNELS)
    groups: dict[str, list[ChannelInfo]] = {}
    for col in cols:
        info = channel_info(col)
        groups.setdefault(info.group, []).append(info)
    for items in groups.values():
        items.sort(key=lambda i: i.label)
    return groups


__all__ = ["CHANNELS", "ChannelInfo", "channel_info", "channels_by_group"]
