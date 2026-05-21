"""EN interpretation tables, bilingual focus tables and the
helpers that resolve ``ChannelInfo.interpretation`` and the
engineer/driver focus pair for the tooltip renderer.

Extracted from :mod:`lfs_telemetry.telemetry.channels`. The
Spanish interpretation tables live in :mod:`.i18n_es` and are
re-imported here so the helpers can resolve either language.
"""
from __future__ import annotations

from .i18n_es import (
    _FOCUS_BY_COLUMN_ES,
    _INTERP_BY_COLUMN_ES,
    _INTERP_BY_GROUP_ES,
    _INTERP_BY_PATTERN_ES,
    _INTERP_BY_SUFFIX_ES,
)

__all__ = [
    "_focus_notes_for",
    "_interpretation_for",
    "_interpretation_for_lang",
]


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
    for suffix, text in _INTERP_BY_SUFFIX_ES:
        if column.endswith(suffix):
            return text
    for prefix, text in _INTERP_BY_PATTERN_ES:
        if column.startswith(prefix):
            return text
    if group in _INTERP_BY_GROUP_ES:
        return _INTERP_BY_GROUP_ES[group]
    # Nunca caer en texto vacío/genérico si no hay traducción puntual:
    # prioriza la explicación inglesa completa por canal.
    return _interpretation_for(column, group)

_FOCUS_BY_SUFFIX_EN: tuple[tuple[str, tuple[str, str]], ...] = (
    ("_slip_ratio", (
        "Engineer: find where each axle exceeds optimal slip under brake/"
        "traction to tune bias, diff and pedal maps.",
        "Driver: target one smooth brake release and one smooth throttle"
        " pickup to keep slip in the productive range.",
    )),
    ("_tan_slip_angle", (
        "Engineer: compare front vs rear slip-angle demand to localize"
        " understeer/oversteer by corner phase.",
        "Driver: avoid adding steering once the tyre is already sliding;"
        " prioritize rotation timing.",
    )),
    ("_vertical_load_n", (
        "Engineer: monitor load distribution across the axle to validate"
        " ARB/spring and ride-height changes.",
        "Driver: smoother pedal and steering transitions reduce abrupt"
        " load spikes and improve consistency.",
    )),
    ("_susp_speed_mps", (
        "Engineer: use HS/LS split to separate platform-control issues"
        " from curb/bump compliance issues.",
        "Driver: if the car feels bouncy after curbs, calm steering and"
        " throttle timing over rough sections.",
    )),
)
_FOCUS_BY_SUFFIX_ES: tuple[tuple[str, tuple[str, str]], ...] = (
    ("_slip_ratio", (
        "Ingeniero: identifica dónde cada eje supera el slip óptimo en"
        " frenada/tracción para ajustar reparto, diferencial y mapas.",
        "Piloto: busca una suelta de freno y una apertura de gas limpias"
        " para mantener el slip en zona productiva.",
    )),
    ("_tan_slip_angle", (
        "Ingeniero: compara demanda de deriva delante/detrás para localizar"
        " subviraje/sobreviraje por fase de curva.",
        "Piloto: evita pedir más volante cuando el neumático ya desliza;"
        " prioriza el timing de rotación.",
    )),
    ("_vertical_load_n", (
        "Ingeniero: monitoriza reparto de carga por eje para validar"
        " cambios de barras, muelles y alturas.",
        "Piloto: transiciones más suaves de pedales y volante reducen"
        " picos de carga y mejoran consistencia.",
    )),
    ("_susp_speed_mps", (
        "Ingeniero: usa el reparto HS/LS para separar problemas de"
        " plataforma de problemas de absorción de baches/pianos.",
        "Piloto: si el coche rebota tras piano, suaviza volante y gas"
        " al pasar por zonas rotas.",
    )),
)
_FOCUS_BY_PATTERN_EN: tuple[tuple[str, tuple[str, str]], ...] = (
    ("friction_use_", (
        "Engineer: identify which corner saturates first and redistribute"
        " load/grip with setup before chasing driving fixes.",
        "Driver: if one wheel saturates early, delay peak input and"
        " straighten exits before full throttle.",
    )),
    ("tyre_work_w_", (
        "Engineer: track thermal workload per corner to forecast tyre"
        " degradation and pressure evolution.",
        "Driver: reduce prolonged slides and wheelspin to keep tyre work"
        " and temperatures under control.",
    )),
)
_FOCUS_BY_PATTERN_ES: tuple[tuple[str, tuple[str, str]], ...] = (
    ("friction_use_", (
        "Ingeniero: localiza qué rueda satura primero y redistribuye"
        " carga/agarre con setup antes de corregir al piloto.",
        "Piloto: si una rueda satura pronto, retrasa el pico de mando y"
        " endereza más antes de gas a fondo.",
    )),
    ("tyre_work_w_", (
        "Ingeniero: sigue trabajo térmico por rueda para anticipar"
        " degradación y evolución de presiones.",
        "Piloto: reduce deslizamientos largos y patinaje para mantener"
        " temperatura y desgaste bajo control.",
    )),
)
_FOCUS_BY_GROUP_EN: dict[str, tuple[str, str]] = {
    "Driver": (
        "Engineer: benchmark input timing and smoothness corner-by-corner"
        " to explain lap-time deltas.",
        "Driver: prioritize repeatable traces over one-off peaks; pace"
        " comes from consistency.",
    ),
    "Engine": (
        "Engineer: optimize shift points, thermal margins and fuel usage"
        " over stint length.",
        "Driver: keep the engine in its useful band and avoid limiter"
        " time that gives no extra acceleration.",
    ),
    "Chassis": (
        "Engineer: correlate inertia channels with setup changes to isolate"
        " balance and platform issues.",
        "Driver: read where the car rotates vs where it pushes, then adapt"
        " entry speed and release timing.",
    ),
    "Suspension": (
        "Engineer: verify platform control and curb compliance per corner"
        " to tune dampers/springs/ARBs.",
        "Driver: smoother curb usage and steering timing reduces upset and"
        " preserves grip on exits.",
    ),
    "Tyre": (
        "Engineer: monitor saturation, temperature trend and force balance"
        " to protect peak grip over the stint.",
        "Driver: avoid over-driving the tyre beyond peak slip windows;"
        " smoothness preserves speed.",
    ),
    "Track": (
        "Engineer: segment performance by curvature/slope/radius to target"
        " setup and gearing where it matters.",
        "Driver: use geometry channels to commit to consistent references"
        " for turn-in, apex and exit.",
    ),
}
_FOCUS_BY_GROUP_ES: dict[str, tuple[str, str]] = {
    "Driver": (
        "Ingeniero: compara timing y suavidad de mandos curva a curva"
        " para explicar deltas de vuelta.",
        "Piloto: prioriza trazas repetibles frente a picos aislados;"
        " el ritmo llega por consistencia.",
    ),
    "Engine": (
        "Ingeniero: optimiza puntos de cambio, margen térmico y consumo"
        " durante el stint.",
        "Piloto: mantén el motor en su banda útil y evita tiempo en"
        " limitador que no acelera más.",
    ),
    "Chassis": (
        "Ingeniero: correlaciona canales inerciales con cambios de setup"
        " para aislar balance y plataforma.",
        "Piloto: identifica dónde rota y dónde empuja para ajustar"
        " velocidad de entrada y liberación.",
    ),
    "Suspension": (
        "Ingeniero: valida control de plataforma y absorción de piano por"
        " rueda para afinar amortiguación/muelles/barras.",
        "Piloto: un uso más limpio de piano y volante preserva apoyo y"
        " tracción de salida.",
    ),
    "Tyre": (
        "Ingeniero: vigila saturación, deriva térmica y equilibrio de"
        " fuerzas para sostener agarre pico en el stint.",
        "Piloto: evita sobreconducir fuera de la ventana de slip pico;"
        " la suavidad conserva neumático y ritmo.",
    ),
    "Track": (
        "Ingeniero: segmenta rendimiento por curvatura/pendiente/radio"
        " para enfocar setup y desarrollo.",
        "Piloto: usa la geometría para fijar referencias consistentes de"
        " entrada, ápice y salida.",
    ),
}
def _focus_notes_for(
    column: str,
    group: str,
    *,
    language: str,
) -> tuple[str, str]:
    if language == "es":
        if column in _FOCUS_BY_COLUMN_ES:
            return _FOCUS_BY_COLUMN_ES[column]
        for suffix, notes in _FOCUS_BY_SUFFIX_ES:
            if column.endswith(suffix):
                return notes
        for prefix, notes in _FOCUS_BY_PATTERN_ES:
            if column.startswith(prefix):
                return notes
        if group in _FOCUS_BY_GROUP_ES:
            return _FOCUS_BY_GROUP_ES[group]
        return (
            "Ingeniero: compara este canal por vuelta y por sector para"
            " separar problema de setup de problema de ejecución.",
            "Piloto: busca trazas limpias y repetibles; menor ruido suele"
            " traducirse en más confianza y ritmo.",
        )
    if column in _FOCUS_BY_COLUMN_EN:
        return _FOCUS_BY_COLUMN_EN[column]
    for suffix, notes in _FOCUS_BY_SUFFIX_EN:
        if column.endswith(suffix):
            return notes
    for prefix, notes in _FOCUS_BY_PATTERN_EN:
        if column.startswith(prefix):
            return notes
    if group in _FOCUS_BY_GROUP_EN:
        return _FOCUS_BY_GROUP_EN[group]
    return (
        "Engineer: use this channel to compare consistency by lap and"
        " sector, and to validate setup changes.",
        "Driver: track trend and repeatability; a cleaner, steadier"
        " trace usually means more pace.",
    )


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
    "g_total_g": (
        "Total acceleration magnitude in g, sqrt(ax²+ay²)/g. The "
        "headline number on a friction-circle plot: how close the "
        "car is to the combined-grip envelope at any instant. "
        "Sustained values close to the tyre/aero limit mean little "
        "margin is left for steering or throttle."
    ),
    "susp_compression_front_avg_m": (
        "Mean of front-axle suspension compression (m). A direct "
        "ride-height proxy on the front axle; rises under braking "
        "and aero load, falls over crests. Trend across a lap "
        "exposes bottoming risk and aero-platform stability."
    ),
    "susp_compression_rear_avg_m": (
        "Mean of rear-axle suspension compression (m). Same idea as "
        "the front channel but for rear ride height; rises on power "
        "squat and high downforce, drops over crests and on lift-"
        "off."
    ),
    "rake_compression_m": (
        "Front-minus-rear axle compression (m). Positive ⇒ nose-down "
        "attitude (typical on entry / under heavy braking); negative "
        "⇒ tail-down (power-on, big aero on rear). Aero-sensitive "
        "cars are very fussy about this trace staying inside a "
        "narrow band through the corner."
    ),
    "slip_angle_balance_rad": (
        "Kinematic balance: mean(|α_front|) − mean(|α_rear|) in rad. "
        "Positive ⇒ front tyres are sliding more than the rear "
        "(understeer); negative ⇒ rear is sliding more (oversteer). "
        "Complements understeer_index — this one is a direct slip-"
        "angle reading, the other one is yaw-rate based."
    ),
    "brake_power_w": (
        "Instantaneous mechanical power going into the brakes (W) "
        "while the pedal is down: |sum of longitudinal tyre force| × "
        "speed. Integrate per lap or per braking zone to get a "
        "reliable brake-heat / pad-wear proxy and to spot zones "
        "where bias or cooling needs work."
    ),
    "throttle_reversal_rate_hz": (
        "Throttle direction-change rate (Hz). Mirror of "
        "steer_reversal_rate_hz for the right pedal: high sustained "
        "values mean pedal-tap / lift-stab driving which upsets "
        "longitudinal balance and burns rear tyres. Smooth drivers "
        "keep this low except in clear traction-management bursts."
    ),
    "coasting": (
        "Boolean flag: throttle < 0.05 ∧ brake < 0.05 ∧ speed > "
        "3 m/s. Marks mid-corner coasting / lift-off phases — useful "
        "to study lift-off oversteer technique and to quantify how "
        "much of the lap is spent neither braking nor accelerating."
    ),
    "trail_brake_intensity": (
        "brake × |input_steer|. Non-zero only when both pedal and "
        "hand are engaged, i.e. on trail-braking corner entry. "
        "Higher peaks ⇒ more combined load on the front tyres; "
        "compare per-corner to balance entry rotation against "
        "front-end overload."
    ),
    "chassis_roll_per_lat_g_rad_per_g": (
        "Instantaneous roll compliance: roll / (accel_y/g) in rad/g, "
        "NaN below ~0.2 g lateral load. Direct probe for ARB and "
        "spring choice — a sudden jump in this ratio at a given "
        "corner phase usually means the suspension just hit a "
        "bump-stop or the inside wheel went light."
    ),
    "chassis_pitch_per_long_g_rad_per_g": (
        "Instantaneous pitch compliance: pitch / (accel_x/g) in "
        "rad/g, NaN below ~0.2 g longitudinal load. Quantifies "
        "brake-dive (negative ax) and power-squat (positive ax) "
        "behaviour; the right channel for judging bump/rebound "
        "damping on the front and rear axles."
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
