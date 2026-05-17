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

from dataclasses import dataclass
from typing import Callable

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


_GROUP_ES_FALLBACK: dict[str, str] = {
    "Driver": "Piloto",
    "Vehicle": "Vehículo",
    "Engine": "Motor",
    "Chassis": "Chasis",
    "Suspension": "Suspensión",
    "Tyre": "Neumáticos",
    "Derived": "Derivados",
    "Track": "Circuito",
    "Aids": "Ayudas",
    "Context": "Contexto",
}


_LABEL_ES_FALLBACK: dict[str, str] = {
    "Speed": "Velocidad",
    "Throttle": "Acelerador",
    "Brake": "Freno",
    "Steer (raw)": "Dirección (bruta)",
    "Steer Torque": "Par de dirección",
    "Long. accel": "Acel. long.",
    "Lat. accel": "Acel. lat.",
    "Yaw rate": "Tasa de guiñada",
    "Understeer idx": "Índice de subviraje",
    "Brake bias front": "Reparto de frenada delante",
    "Fuel": "Combustible",
}


_DESCRIPTION_ES_FALLBACK: dict[str, str] = {
    "Wheel-derived speed": "Velocidad derivada de ruedas",
    "Engine speed": "Régimen del motor",
    "Throttle pedal (0..1)": "Pedal de acelerador (0..1)",
    "Brake pedal (0..1)": "Pedal de freno (0..1)",
    "Steering wheel angle": "Ángulo del volante",
    "FFB steering torque": "Par de dirección FFB",
    "+forward, -brake": "+adelante, -frenada",
    "+right": "+derecha",
    "Fuel level (0..1)": "Nivel de combustible (0..1)",
}


_INTERP_BY_COLUMN_ES: dict[str, str] = {
    "speed_ms": (
        "Velocidad longitudinal del coche. Compárala en entrada,"
        " ápice y salida para detectar dónde ganas o pierdes tiempo."
    ),
    "throttle": (
        "Posición del acelerador (0..1). Una rampa suave al salir"
        " de curva suele indicar buena tracción y control."
    ),
    "brake": (
        "Presión de freno (0..1). Busca un pico inicial fuerte y"
        " una liberación progresiva hacia el giro (trail braking)."
    ),
    "input_steer": (
        "Ángulo de volante. Trazas limpias y sin serrucho suelen"
        " indicar un coche estable y una conducción precisa."
    ),
    "steer_torque_nm": (
        "Par de dirección (FFB). Si se aplana o satura, puedes estar"
        " perdiendo información del tren delantero."
    ),
    "accel_x": (
        "Aceleración longitudinal. Negativa al frenar, positiva al"
        " acelerar. Útil para comparar eficacia de frenada y salida."
    ),
    "accel_y": (
        "Aceleración lateral. El pico absoluto en curva refleja el"
        " agarre lateral realmente usado."
    ),
    "ang_vel_z": (
        "Tasa de guiñada. Relaciona cuánto rota el coche con lo que"
        " pides con el volante en cada fase de la curva."
    ),
    "yaw_rate_rads": (
        "Guiñada medida del chasis. Compárala con la teórica para"
        " diagnosticar subviraje/sobreviraje."
    ),
    "understeer_index": (
        "Índice de balance: positivo=subviraje, negativo=sobreviraje."
        " Mira en qué fase de curva aparece el pico."
    ),
    "brake_bias_front_real": (
        "Reparto real de frenada delante. Si se va demasiado delante,"
        " tenderás a bloquear delante; demasiado atrás, inestabilidad."
    ),
    "fuel": (
        "Nivel de combustible (0..1). La pendiente de la curva sirve"
        " para estimar consumo por vuelta y ventana de parada."
    ),
}


_FOCUS_BY_COLUMN_EN: dict[str, tuple[str, str]] = {
    "speed_ms": (
        "Engineer: compare minimum-speed and corner-exit speed by lap"
        " and sector to separate line losses from power losses.",
        "Driver: focus on carrying entry speed without delaying"
        " throttle pickup at the apex.",
    ),
    "throttle": (
        "Engineer: quantify throttle smoothness and time-at-full-"
        "throttle; spikes often correlate with traction instability.",
        "Driver: build one clean throttle ramp at exit instead of"
        " repeated stabs.",
    ),
    "brake": (
        "Engineer: check initial bite, release rate, and consistency"
        " across laps to tune bias and pedal map.",
        "Driver: brake hard early, then bleed pressure smoothly into"
        " turn-in to keep front grip.",
    ),
    "input_steer": (
        "Engineer: steering oscillations indicate instability or"
        " over-sensitive front axle setup.",
        "Driver: reduce micro-corrections; one decisive arc is usually"
        " faster than multiple small fixes.",
    ),
    "steer_torque_nm": (
        "Engineer: monitor clipping/saturation to preserve FFB"
        " information quality.",
        "Driver: if steering goes numb at apex, reduce steering angle"
        " demand and re-balance entry.",
    ),
    "accel_x": (
        "Engineer: compare peak decel and brake-release timing for"
        " braking performance benchmarking.",
        "Driver: maximize straight-line decel, then release progressively"
        " as steering is added.",
    ),
    "accel_y": (
        "Engineer: use peak lateral-g by corner to identify where grip"
        " is under-used.",
        "Driver: aim to reach consistent lateral-g peaks without"
        " scrubbing mid-corner speed.",
    ),
    "understeer_index": (
        "Engineer: map peaks by corner phase to decide whether changes"
        " belong to front geometry, rear support, or diff.",
        "Driver: positive peaks suggest waiting for rotation; negative"
        " peaks ask for calmer throttle/steer timing.",
    ),
    "brake_bias_front_real": (
        "Engineer: track bias drift under load to validate setup and"
        " ABS behavior.",
        "Driver: if rear gets nervous on entry, move a touch forward;"
        " if front locks first, move rearward.",
    ),
}


_FOCUS_BY_COLUMN_ES: dict[str, tuple[str, str]] = {
    "speed_ms": (
        "Ingeniero: compara velocidad mínima y de salida por vuelta y"
        " sector para separar pérdidas por trazada y por potencia.",
        "Piloto: prioriza conservar velocidad de entrada sin retrasar"
        " la apertura de gas en el ápice.",
    ),
    "throttle": (
        "Ingeniero: mide suavidad de gas y tiempo a fondo; picos suelen"
        " correlacionar con falta de tracción.",
        "Piloto: construye una rampa limpia de gas en salida, evita"
        " golpecitos repetidos.",
    ),
    "brake": (
        "Ingeniero: revisa mordida inicial, velocidad de liberación y"
        " consistencia para ajustar reparto y mapa de freno.",
        "Piloto: frena fuerte al inicio y suelta de forma progresiva"
        " al empezar el giro.",
    ),
    "input_steer": (
        "Ingeniero: oscilaciones de dirección apuntan a inestabilidad"
        " o exceso de sensibilidad del tren delantero.",
        "Piloto: reduce microcorrecciones; un arco claro suele ser más"
        " rápido que varias correcciones pequeñas.",
    ),
    "steer_torque_nm": (
        "Ingeniero: vigila saturación/clipping para mantener calidad de"
        " información en el FFB.",
        "Piloto: si el volante se queda 'muerto' en ápice, pide menos"
        " ángulo y reequilibra la entrada.",
    ),
    "accel_x": (
        "Ingeniero: compara pico de deceleración y timing de liberación"
        " de freno para evaluar rendimiento de frenada.",
        "Piloto: maximiza frenada en recta y libera progresivamente al"
        " añadir dirección.",
    ),
    "accel_y": (
        "Ingeniero: usa pico de g lateral por curva para detectar dónde"
        " no se está explotando el agarre.",
        "Piloto: busca picos de g lateral repetibles sin arrastrar"
        " velocidad en mitad de curva.",
    ),
    "understeer_index": (
        "Ingeniero: localiza picos por fase de curva para decidir si"
        " tocar geometría delantera, apoyo trasero o diferencial.",
        "Piloto: picos positivos piden esperar rotación; negativos"
        " piden suavizar gas/dirección.",
    ),
    "brake_bias_front_real": (
        "Ingeniero: sigue la deriva del reparto bajo carga para validar"
        " setup y actuación del ABS.",
        "Piloto: si el eje trasero se mueve en entrada, adelanta un"
        " poco; si bloquea delante, atrasa.",
    ),
}


def _interpretation_for_lang(
    column: str,
    group: str,
    *,
    language: str,
) -> str:
    if language != "es":
        return _interpretation_for(column, group)
    if column in _INTERP_BY_COLUMN_ES:
        return _INTERP_BY_COLUMN_ES[column]
    return _GROUP_ES_FALLBACK.get(group, _interpretation_for(column, group))


def _focus_notes_for(
    column: str,
    group: str,
    *,
    language: str,
) -> tuple[str, str]:
    if language == "es":
        if column in _FOCUS_BY_COLUMN_ES:
            return _FOCUS_BY_COLUMN_ES[column]
        return (
            "Ingeniero: usa este canal para comparar consistencia por"
            " vuelta/sector y validar cambios de setup.",
            "Piloto: observa tendencia y repetibilidad; una señal más"
            " estable suele traducirse en más ritmo.",
        )
    if column in _FOCUS_BY_COLUMN_EN:
        return _FOCUS_BY_COLUMN_EN[column]
    return (
        "Engineer: use this channel to compare consistency by lap and"
        " sector, and to validate setup changes.",
        "Driver: track trend and repeatability; a cleaner, steadier"
        " trace usually means more pace.",
    )


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

_INTERP_BY_COLUMN: dict[str, str] = {
    # -------- Vehicle / Engine (OutGauge + OutSim) --------
    "speed_ms": (
        "Forward ground speed (m/s) from OutGauge. Overlay two laps on "
        "the same corner: the one carrying more speed at the apex is "
        "usually the better line. Sudden drops mean braking, wheel "
        "lock, or a traction loss; long flat plateaus mark sections "
        "where you are speed-limited by aero/drag rather than grip."
    ),
    "rpm": (
        "Engine speed (revs/min) from OutGauge. Flat-top traces against "
        "the rev limit = upshifts left too late (you are hitting the "
        "limiter); large RPM drops on upshift = gear ratio too long or "
        "shift performed off the power band. For naturally-aspirated "
        "engines the sweet spot is staying inside the upper third of "
        "the torque curve."
    ),
    "gear_lfs": (
        "Currently engaged gear (R, N, 1..N) from OutGauge. Use this "
        "to audit shift points and downshifts: dropping a gear before "
        "the brakes are released loads the driven wheels mid-corner "
        "and can pitch the car. Compare with throttle/brake traces."
    ),
    "fuel": (
        "Fuel level as a fraction of tank capacity (0..1) from "
        "OutGauge. The slope is the instantaneous consumption rate; "
        "the FuelTracker module turns it into laps-remaining and pit "
        "windows. A sudden drop is usually a refuel event."
    ),
    "eng_temp_c": (
        "Coolant temperature (°C) from OutGauge. Steady creep above "
        "the normal band signals blocked radiator vents, body damage, "
        "or persistent over-revving. Most LFS engines tolerate up to "
        "~110 °C before power loss; sustained higher figures shorten "
        "engine life."
    ),
    "oil_temp_c": (
        "Oil temperature (°C). It reacts slower than coolant and is "
        "the better proxy for long-term thermal stress. Healthy race "
        "range is roughly 90–125 °C depending on the car."
    ),
    "oil_pressure_bar": (
        "Engine oil pressure (bar). Drops under high lateral g "
        "indicate oil starvation (oil sloshing away from the pickup); "
        "persistently low pressure suggests a hot, thin oil or wear."
    ),
    "turbo_bar": (
        "Turbo boost pressure (bar) above atmospheric. A flat trace "
        "at peak boost = wastegate regulating correctly. Slow ramps "
        "at low RPM are turbo lag; sudden dips mid-throttle are "
        "blow-off events or wastegate flutter."
    ),

    # -------- Driver inputs (OutGauge) --------
    "throttle": (
        "Throttle pedal position (0..1) from OutGauge. The textbook "
        "shape exits a corner as a smooth, monotonic ramp from 0 to "
        "1; step inputs that bounce off 1.0 indicate aggressive "
        "throttle that the rear axle may not handle. Long 1.0 "
        "plateaus mark full-throttle sections; you can measure "
        "‘time at full throttle’ per lap to compare driving styles."
    ),
    "brake": (
        "Brake pedal position (0..1) from OutGauge. A good braking "
        "trace ramps quickly to a high initial peak (just below "
        "lock-up), then bleeds smoothly toward zero as you turn in "
        "(trail-braking). Stair-stepped releases reveal pedal "
        "modulation or ABS cycling; very flat plateaus near 1.0 mean "
        "you are pushing the brake into ABS or wheel lock."
    ),
    "clutch": (
        "Clutch pedal (0..1). Only meaningful around standing starts "
        "and gear shifts. A long partial-depress trace on launch is "
        "clutch slipping (used deliberately to manage wheelspin)."
    ),
    "input_steer": (
        "Steering wheel angle (rad), positive = left. Smooth, "
        "single-arc traces indicate well-judged corner entry; a "
        "‘sawtooth’ pattern of small reversals shows over-correction "
        "or a nervous chassis. The peak amplitude tied to corner "
        "radius hints at understeer/oversteer balance."
    ),
    "steer_torque_nm": (
        "Steering rack torque (N·m) — the physical FFB reaction that "
        "would be felt at the wheel. Sudden sign changes denote rear "
        "snap; large flat plateaus across mid-corner mean the front "
        "tyres are saturated (understeer)."
    ),
    "input_handbrake": (
        "Handbrake input (0..1). Only used for handbrake turns, drift, "
        "or rallycross. A spike in normal racing is almost always a "
        "mistake."
    ),

    # -------- Chassis (OutSim — vehicle-axis IMU) --------
    "accel_x": (
        "Longitudinal acceleration in the vehicle frame (m/s²). "
        "Positive = accelerating, negative = braking. A clean braking "
        "trace shows a sharp negative step to the friction limit and "
        "stays flat (constant deceleration) before bleeding to 0 at "
        "turn-in. Reference figures: ~1 g braking = 9.81 m/s²; race "
        "slicks can pull 1.5–2 g."
    ),
    "accel_y": (
        "Lateral acceleration (m/s², +left). Its absolute peak in a "
        "corner equals the lateral grip you actually used. Cross-plot "
        "against accel_x (a ‘g–g diagram’) to see how well you fill "
        "the friction ellipse — missing quadrants reveal areas where "
        "you are leaving grip on the table."
    ),
    "accel_z": (
        "Vertical acceleration (m/s²). Spikes are bumps, curb hits, "
        "or landings after a crest. Sustained values above 1 g "
        "compression indicate a heavy aero load or a steep "
        "compression."
    ),
    "ang_vel_z": (
        "Yaw rate (rad/s) about the vertical axis — how fast the car "
        "rotates in plan view. The pure-geometry expectation is "
        "v / R (speed ÷ corner radius); see understeer_index for the "
        "deviation."
    ),
    "ang_vel_x": (
        "Roll rate (rad/s) about the longitudinal axis. Big spikes "
        "in fast direction changes (chicanes) indicate soft anti-roll "
        "bars or springs and slow weight transfer."
    ),
    "ang_vel_y": (
        "Pitch rate (rad/s) about the lateral axis. Positive pulses "
        "under braking (nose-dive), negative under throttle "
        "(squat). Magnitude scales with brake/throttle force divided "
        "by front/rear suspension stiffness."
    ),
    "pitch": (
        "Chassis pitch angle (rad). Positive in braking (nose down), "
        "negative under acceleration. Excessive pitch hurts "
        "aerodynamic platform and changes mechanical grip balance."
    ),
    "roll": (
        "Chassis roll angle (rad). Large roll in steady-state corners "
        "points to soft springs or insufficient ARB; very stiff cars "
        "will show almost flat traces but can skip on curbs."
    ),
    "heading": (
        "Chassis yaw (heading) angle in world frame (rad). Mostly "
        "useful when overlaid on the track map; not very informative "
        "as a time-series trace by itself."
    ),

    # -------- Lap / track distance (InSim + derived) --------
    "current_lap_dist_m": (
        "Distance travelled since the start/finish line (m). Resets "
        "to 0 each lap. Use it as an x-axis to align corners between "
        "laps for direct comparison."
    ),
    "indexed_distance_m": (
        "Distance along the racing-line index (m); does not reset at "
        "the line. Preferred for season-long datasets that span "
        "multiple stints."
    ),

    # -------- Derived chassis dynamics --------
    "yaw_rate_rads": (
        "Measured yaw rate (rad/s). Compared against the theoretical "
        "neutral-car yaw rate it tells you the balance: see "
        "understeer_index."
    ),
    "yaw_rate_theoretical_rads": (
        "Yaw rate a perfectly neutral car would have at this speed "
        "and steering angle (v · tan(δ) / wheelbase). The gap to the "
        "measured yaw rate is what the front/rear axles are actually "
        "delivering."
    ),
    "beta_deg": (
        "Vehicle side-slip angle (°) = atan(v_lateral / v_long). On "
        "a road car a healthy maximum is 3–6°; a sustained value "
        "above 8° means the car is sliding noticeably and is likely "
        "losing time. Useful to spot oversteer or aggressive trail-"
        "braking."
    ),
    "understeer_index": (
        "(yaw_theoretical − yaw_measured) / yaw_theoretical. "
        "Positive = understeer (front pushing wide), negative = "
        "oversteer (rear yawing more than the steering asks for). "
        "Note where the spikes happen: entry, mid-corner or exit "
        "diagnose different setup problems (front geometry, weight "
        "transfer, diff)."
    ),
    "transfer_long_n_real": (
        "Longitudinal weight transfer (N). The textbook value is "
        "m · a_x · h_cg / wheelbase: positive under acceleration "
        "(load shifts rearward), negative under braking (load shifts "
        "forward). Compare to the static axle weights to see relative "
        "load swing."
    ),
    "transfer_lat_n_real": (
        "Lateral weight transfer (N). Driven by m · a_y · h_cg / "
        "track. Its split between front and rear axles depends on "
        "roll stiffness; biasing roll stiffness forward adds front "
        "transfer (more understeer) and vice-versa."
    ),
    "load_total_n": (
        "Sum of all four wheel vertical loads (N). Should hover "
        "around the car’s weight (m · g) plus aero downforce. Brief "
        "dips below static weight indicate a wheel airborne or all-"
        "round unloading over a crest."
    ),
    "load_front_frac": (
        "Fraction of total vertical load carried by the front axle "
        "(0..1). Static values for road cars sit around 0.55–0.60 "
        "(front-engined) or 0.40–0.45 (rear-engined); braking "
        "transients can push it past 0.75."
    ),
    "load_left_frac": (
        "Fraction of total load on the left side of the car (0..1). "
        "Right-hand corners load the left side; magnitude depends on "
        "lateral g and CG height."
    ),
    "load_diag_fl_rr_frac": (
        "Diagonal load: (FL + RR) / total. Differences from 0.5 in "
        "steady-state cornering indicate chassis cross-weight; in "
        "transients it reveals chassis twist (especially on curbs "
        "and crests)."
    ),
    "brake_bias_front_real": (
        "Actual front brake bias under braking (front_brake_force / "
        "total_brake_force). If it diverges from the dialled-in "
        "value, either ABS is intervening on one axle or one set of "
        "tyres has locked up. Front-bias too high = front lock-up; "
        "too low = rear lock-up and snap oversteer."
    ),
    "ffb_load_pct": (
        "Force-feedback torque utilisation (0..1). 1.0 sustained = "
        "the wheelbase is clipping; raise the in-game FFB setting "
        "down or the wheelbase ‘strength’ up so the peaks fit. "
        "Clipped FFB loses front-end information."
    ),
    "steer_rate_rads": (
        "Steering angular velocity (rad/s). High peaks are abrupt "
        "corrections; pros usually keep peak rates below ~4 rad/s in "
        "fast corners."
    ),
    "steer_reversal_rate_hz": (
        "Rate of steering-input sign changes (Hz). High values "
        "(>2 Hz sustained) reveal hand-wrestling and corner exit "
        "instability; low and steady values characterise a clean, "
        "‘planted’ lap."
    ),
    "accel_x_road_mps2": (
        "Longitudinal acceleration corrected for road slope. "
        "Isolates what brakes/engine actually do from the help or "
        "hindrance of a slope. Use this instead of accel_x for power-"
        "and-brake analysis on undulating tracks."
    ),
    "velocity_heading_rad": (
        "Direction of the velocity vector in the world frame (rad). "
        "Combined with the chassis heading it produces beta and "
        "yaw_misalign."
    ),
    "yaw_misalign_rad": (
        "Chassis-heading minus velocity-heading (rad). A practical "
        "proxy for the vehicle slip angle on track; positive while "
        "the rear is stepping out in a right-hander, negative in a "
        "left-hander."
    ),

    # -------- Track geometry (racing_lines/<TRACK>_racing.csv) --------
    "track_node": (
        "Nearest racing-line node index. Discrete integer; useful as "
        "a join key but not as a y-axis."
    ),
    "track_s_m": (
        "Arc-length distance along the racing line (m). Monotonic, "
        "ideal for x-axis when comparing laps that re-cross start/"
        "finish at different positions."
    ),
    "track_z_m": "Track elevation at the current racing-line node (m).",
    "track_heading_rad": (
        "Tangent heading of the racing line at this node (rad). The "
        "difference between chassis heading and this is essentially "
        "the car’s slip angle relative to the ideal line."
    ),
    "track_curvature_1_per_m": (
        "Signed racing-line curvature κ (1/m); + = left-hander, − = "
        "right-hander, ~0 = straight. The reciprocal gives the local "
        "corner radius."
    ),
    "track_radius_m": (
        "Local corner radius R = 1/|κ| (m). Smaller radius = tighter "
        "corner. Sets the upper-bound cornering speed via "
        "v = sqrt(μ · g · R) for given grip."
    ),
    "track_slope_pct": (
        "Track gradient (%) along the racing line. Positive = uphill. "
        "Used together with accel_x to remove slope effects."
    ),
    "track_width_m": (
        "Width of the AI drive corridor (m) at this node — the usable "
        "racing surface."
    ),
    "drive_left_local": (
        "Distance from racing line to the left edge of the AI drive "
        "corridor (≤0). Use it together with track_offset_m to see "
        "how close to the edge you are."
    ),
    "drive_right_local": (
        "Distance from racing line to the right edge of the AI drive "
        "corridor (≥0)."
    ),
    "limit_left_local": (
        "Distance to the hard left limit (asphalt edge) (≤0). "
        "Crossing this is officially ‘off-track’."
    ),
    "limit_right_local": (
        "Distance to the hard right limit (asphalt edge) (≥0)."
    ),
    "track_offset_m": (
        "Lateral offset (m) of the car from the racing line "
        "(centre-line). Compare with the drive/limit channels to see "
        "if you are clipping the apex, running wide or going off."
    ),
    "segment_kind": (
        "Categorical tag for the current track segment: straight | "
        "left | right. Useful to group statistics by segment type."
    ),
    "segment_id": (
        "Integer ID of the current segment (straight or corner). "
        "Lets you aggregate metrics per corner across laps and "
        "stints."
    ),

    # -------- Aids / dashboard lights (OutGauge ShowLights bits) --------
    "dl_tc_active": (
        "Traction-control intervention flag. Frequent activations "
        "indicate you are demanding more longitudinal grip than the "
        "driven tyres can deliver — back off throttle on exits or "
        "soften the diff."
    ),
    "dl_abs_active": (
        "ABS intervention flag. Frequent activations mean brake "
        "pressure exceeds the locking threshold for those tyres — "
        "ease off, or shift bias toward the saturating axle."
    ),
    "dl_pit_limiter": "Pit-lane speed limiter engaged.",
    "dl_handbrake": "Handbrake engaged warning.",
    "dl_shift_light": (
        "Shift-light trigger (engine near its optimum upshift RPM). "
        "Use as a coaching aid against the rpm trace."
    ),
    "dl_oil_warn": "Oil warning (low pressure or high temperature).",
    "dl_battery_warn": "Battery / alternator warning.",
    "dl_fullbeam": "High beam on.",
    "dl_signal_l": "Left turn indicator.",
    "dl_signal_r": "Right turn indicator.",
}


_INTERP_BY_SUFFIX: tuple[tuple[str, str], ...] = (
    ("_susp_deflect_m", (
        "Suspension travel at this corner (m). Positive = compressed "
        "relative to ride height. Short spikes are bumps; sustained "
        "compression reflects steady weight transfer. Watching all "
        "four corners side-by-side reveals body roll, pitch and "
        "diagonal load."
    )),
    ("_vertical_load_n", (
        "Vertical load (N) on this tyre. If it touches 0 the tyre is "
        "airborne (no grip). A left/right imbalance in steady state "
        "indicates body roll; a front/rear imbalance indicates pitch."
    )),
    ("_slip_ratio", (
        "Longitudinal slip ratio (dimensionless). 0 = pure rolling; "
        "positive = driving (tyre spinning faster than ground), "
        "negative = braking (tyre rotating slower). Peak tyre "
        "longitudinal grip occurs around |0.10–0.15|; sustained "
        "values past that mean the tyre is sliding."
    )),
    ("_tan_slip_angle", (
        "tan(α), where α is the tyre slip angle — the angle between "
        "the tyre’s heading and its velocity vector. Peak lateral "
        "grip lives around α ≈ 6–8° (tan ≈ 0.10–0.14). Bigger means "
        "the tyre is sliding sideways rather than gripping."
    )),
    ("_x_force_n", (
        "Lateral force generated by the tyre contact patch (N). Sign "
        "tracks the cornering direction. Saturation at a flat plateau "
        "is the maximum lateral grip the tyre can provide."
    )),
    ("_y_force_n", (
        "Longitudinal force at the contact patch (N): positive when "
        "driving forward, negative when braking. Per the friction "
        "ellipse, generating large lateral force reduces the maximum "
        "longitudinal force you can add at the same time."
    )),
    ("_ang_vel_rads", (
        "Wheel angular velocity (rad/s). Multiply by tyre radius to "
        "get the rolling speed; comparing it with ground speed gives "
        "the slip ratio."
    )),
    ("_lean_rel_road_rad", (
        "Wheel camber angle relative to the road surface (rad). The "
        "dynamic camber — what the tyre actually sees — and the key "
        "input for lateral grip from a cambered tyre."
    )),
    ("_air_temp_c", (
        "Inflation-air temperature inside the tyre (°C) as modelled "
        "by LFS. It rises with energy input and is the most stable "
        "indicator of tyre work; a sudden rise signals abuse "
        "(lock-ups, slides). Carcass and surface temperatures are not "
        "available separately in OutSim/OutGauge."
    )),
    ("_slip_fraction", (
        "Fraction of the contact patch that is sliding (0..1). "
        "Values near 1 mean the tyre is past the linear region; "
        "very useful to spot ‘scrubbing’ on the limit."
    )),
    ("_touching", (
        "1 when the tyre is in contact with a surface, 0 when it is "
        "airborne. Use it to mask other tyre channels that become "
        "meaningless mid-air."
    )),
    ("_steer_rad", (
        "Steered angle of this road wheel (rad). Differs from the "
        "steering-wheel angle by the steering ratio and the "
        "Ackermann geometry (inner wheel steers more than outer in "
        "a tight corner)."
    )),
    ("_susp_speed_mps", (
        "Damper velocity (m/s). Positive = compressing (bump); "
        "negative = extending (rebound). The HS/LS histogram in the "
        "Damper tab splits this into high-speed (bump strikes) and "
        "low-speed (chassis motion) regimes; a balanced car has "
        "well-filled low-speed lobes and brief, capped high-speed "
        "tails."
    )),
)


_INTERP_BY_PATTERN: tuple[tuple[str, str], ...] = (
    ("friction_use_", (
        "Tyre friction-ellipse utilisation (0..1) = sqrt(Fx² + Fy²) / "
        "(μ · Fz). 1.0 means the contact patch is fully used and you "
        "cannot add more lateral or longitudinal force without losing "
        "grip elsewhere. If just one wheel reaches saturation while "
        "the others lag, the setup is unbalanced (anti-roll, ride "
        "height, pressures, alignment)."
    )),
    ("tyre_work_w_", (
        "Mechanical power dissipated by the tyre (W) — the energy "
        "going into heat and wear. Integrate over a lap and divide "
        "across the four tyres to see which corner is the hottest "
        "worked and likely degrading first."
    )),
)


_INTERP_BY_GROUP: dict[str, str] = {
    "Vehicle": (
        "Basic car state — speed, position, lap times. Sourced from "
        "the OutGauge UDP stream that LFS emits."
    ),
    "Engine": (
        "Powertrain channels: RPM, gear, fuel and engine/oil "
        "temperatures from OutGauge."
    ),
    "Driver": (
        "Driver inputs: throttle, brake, clutch, handbrake, steering "
        "and FFB torque. The cleanest indicator of driving style."
    ),
    "Chassis": (
        "OutSim chassis dynamics: 3-axis acceleration (m/s²), "
        "3-axis angular velocity (rad/s) and Euler attitude angles."
    ),
    "Suspension": (
        "Per-corner suspension state: travel, vertical load, damper "
        "velocity and steered wheel angle."
    ),
    "Tyre": (
        "Per-corner tyre behaviour: slip ratio, slip angle, "
        "longitudinal/lateral forces, inflation temperature and "
        "contact-patch flags."
    ),
    "Aids": (
        "OutGauge ShowLights bitfield: state of TC, ABS, pit "
        "limiter, indicators and warning lamps."
    ),
    "Derived": (
        "Quantities computed by Studio from the raw OutSim/OutGauge "
        "channels — understeer index, weight transfers, friction "
        "utilisation, etc."
    ),
    "Lap": (
        "Lap-relative distances and indices used to align traces "
        "between laps."
    ),
    "Track": (
        "Static track geometry sampled at the current racing-line "
        "node: curvature, radius, slope, width and the lateral "
        "offset of the car within the corridor."
    ),
    "Context": (
        "Session context: car, track, weather, wind. Used to filter "
        "and group captures."
    ),
}


def _interpretation_for(column: str, group: str) -> str:
    if column in _INTERP_BY_COLUMN:
        return _INTERP_BY_COLUMN[column]
    for suffix, text in _INTERP_BY_SUFFIX:
        if column.endswith(suffix):
            return text
    for prefix, text in _INTERP_BY_PATTERN:
        if column.startswith(prefix):
            return text
    return _INTERP_BY_GROUP.get(group, "")


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


__all__ = ["ChannelInfo", "CHANNELS", "channel_info", "channels_by_group"]
