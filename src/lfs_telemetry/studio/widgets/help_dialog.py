"""Help dialog: general telemetry interpretation guide.

Renders an HTML overview of every channel group plus the per-channel
interpretation table from :mod:`lfs_telemetry.telemetry.channels`. The
goal is that someone who knows nothing about telemetry can read each
plot at a glance.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ...telemetry import channels_by_group


_GROUP_ORDER: tuple[str, ...] = (
    "Driver", "Vehicle", "Engine", "Chassis",
    "Suspension", "Tyre", "Derived", "Track", "Aids", "Lap", "Context",
)

_GROUP_INTRO: dict[str, str] = {
    "Driver": (
        "What the driver is doing with the controls. These traces "
        "(throttle, brake, clutch, handbrake, steering, FFB torque) "
        "are sourced directly from the LFS OutGauge UDP stream and "
        "give the most direct read on driving style. Comparing two "
        "laps on the driver channels alone usually explains most of "
        "the lap-time delta."
    ),
    "Vehicle": (
        "Basic car state: speed, position and lap timing. Overlaying "
        "speed for two laps shows exactly where you gain or lose "
        "time; integrating the speed difference gives the cumulative "
        "Δt that the chart already plots when ‘Δt vs ref’ is enabled."
    ),
    "Engine": (
        "Powertrain channels — RPM, gear, fuel, engine/oil "
        "temperatures, turbo boost. Use them to audit shift points, "
        "monitor thermal headroom and budget fuel."
    ),
    "Chassis": (
        "OutSim chassis dynamics: 3-axis acceleration (m/s²), "
        "3-axis angular velocity (rad/s) and Euler attitude. This is "
        "where understeer/oversteer, weight transfer and platform "
        "stability are read from."
    ),
    "Suspension": (
        "Per-corner suspension state: travel, vertical load, damper "
        "velocity, dynamic camber and the steered wheel angle. The "
        "Dampers tab summarises HS/LS damper histograms (the "
        "industry-standard ~50 mm/s threshold separating chassis "
        "motions from bump strikes)."
    ),
    "Tyre": (
        "Per-corner tyre channels — slip ratio, slip angle, "
        "longitudinal/lateral forces, inflation temperature, contact "
        "patch. The friction ellipse |F|/(μ·Fz) ties them all "
        "together: a tyre at saturation cannot give more lateral "
        "without losing longitudinal grip, and vice-versa."
    ),
    "Derived": (
        "Quantities Studio computes from the raw OutSim/OutGauge "
        "channels: understeer index, longitudinal / lateral load "
        "transfers, real brake bias, friction utilisation, steering "
        "smoothness… These are usually the most diagnostic plots."
    ),
    "Track": (
        "Static racing-line geometry sampled at the current node — "
        "curvature, radius, slope, width and the lateral offset of "
        "the car within the corridor. Lets you interpret the rest "
        "of the channels in their physical context (e.g. high "
        "lateral g where the radius is tight is expected)."
    ),
    "Aids": (
        "OutGauge ShowLights bitfield — state of TC, ABS, pit "
        "limiter, shift light, turn indicators and warning lamps. "
        "Frequent TC/ABS activations mean you are demanding more "
        "longitudinal grip than the tyres can supply."
    ),
    "Lap": (
        "Lap-relative distances and indices used to align traces "
        "across multiple laps."
    ),
    "Context": (
        "Session context: car, track, weather, wind. Used as "
        "metadata to filter and compare captures."
    ),
}

_HOW_TO_READ = """
<h2>How to read Studio</h2>
<ul>
  <li><b>Hover any channel</b> in the right-hand tree, or any plot,
      to see a tooltip with what it measures and how to interpret it.</li>
  <li><b>X axis</b>: distance (recommended for comparing laps) or
      time (better for inspecting individual transients).</li>
  <li><b>Δt vs ref</b>: with two or more laps on a distance x-axis,
      the first row shows the cumulative time delta against the
      reference lap. Rising = losing time; falling = gaining.</li>
  <li><b>Friction circle / Load transfer presets</b>: one-click
      channel selections for the most common diagnostic views.</li>
  <li><b>Sectors tab</b>: splits the lap into sectors, marking your
      personal best sector and the theoretical best.</li>
  <li><b>Dampers tab</b>: high-speed / low-speed damper velocity
      histogram (cross-over near ~50 mm/s). A balanced car has
      well-filled low-speed lobes and brief, capped high-speed
      tails; if one wheel never fills the HS bin it is over-damped,
      if it only fills HS it is under-damped.</li>
  <li><b>Race dashboard</b>: in live sessions you see real-time
      splits, projected lap, nearby traffic and fuel autonomy.</li>
</ul>

<h2>Reading shortcuts (driving and physics)</h2>
<ul>
  <li><b>Throttle &amp; brake</b> should ‘shake hands’ at corner
      entry/exit — released brake as throttle starts to climb. Long
      simultaneous applications are deliberate trail-braking or a
      mistake. Step inputs that bounce off 1.0 suggest aggressive
      pedal work that the tyres may not absorb.</li>
  <li><b>Steering</b>: smooth single-arc traces = well-judged turn-in.
      A sawtooth pattern of small reversals is over-correction or a
      nervous chassis (steer_reversal_rate_hz quantifies it).</li>
  <li><b>g–g diagram</b>: plot accel_x vs accel_y. A car using the
      whole friction ellipse fills the circle; empty quadrants show
      where grip is being left unused (typically braking-into-turn
      transitions).</li>
  <li><b>understeer_index</b>: +ve = front pushing wide (front tyres
      saturated), −ve = rear yawing more than the steering input asks
      (oversteer). Where the spikes appear matters: <i>entry</i> spikes
      tend to come from soft front geometry or cold front tyres;
      <i>exit</i> spikes from too-much-power-too-early or a loose
      differential setting.</li>
  <li><b>friction_use_*</b> close to 1.0 on a single wheel = the
      contact patch is saturated. If only one wheel does this while
      the others lag, the load distribution is off (anti-roll bars,
      pressures, alignment, ride height).</li>
  <li><b>Tyre slip targets</b>: peak longitudinal grip lives near
      slip_ratio |0.10–0.15|; peak lateral grip near slip angle
      6–8° (tan ≈ 0.10–0.14). Past those, the tyre is sliding more
      than gripping.</li>
  <li><b>Weight transfer</b>: transfer_long ≈ m·a_x·h_cg/wheelbase,
      transfer_lat ≈ m·a_y·h_cg/track. Both scale with CG height —
      lowering ride height is the cheapest way to reduce transfer.</li>
</ul>

<h2>Where the data comes from</h2>
<p>LFS exposes three telemetry streams that Studio listens to:</p>
<ul>
  <li><b>OutSim</b> — chassis IMU at high rate (position, velocity,
      3-axis acceleration in m/s², 3-axis angular velocity in rad/s,
      Euler angles, per-wheel info for slip ratio, slip angle,
      forces, contact and suspension travel).</li>
  <li><b>OutGauge</b> — dashboard state (RPM, speed, gear, throttle,
      brake, clutch, handbrake, steering, fuel, engine/oil
      temperatures, turbo, show-lights bitfield).</li>
  <li><b>InSim</b> — session events (lap times, splits, race
      control, chat).</li>
</ul>
<p>Use <i>Tools → Configure LFS…</i> to write the matching entries
into <code>cfg.txt</code> automatically.</p>
"""


# ----------------------------------------------------------------------
# Setup tab ↔ telemetry channels mapping
# ----------------------------------------------------------------------
# Each entry describes one slider/parameter from the LFS garage Setup
# screen (sourced from the official LFS wiki — Basic Setup Guide,
# Advanced Setup Guide and Technical Reference) and lists the
# telemetry channels in Studio that reveal whether that setting is
# right or wrong, with a short note on *what* to look for.

_SetupParam = tuple[
    str,             # parameter name (matches LFS garage label)
    str,             # what it does (plain language)
    tuple[tuple[str, str], ...],  # (channel_id, why it matters)
    str,             # tuning shortcut / how to read the trace
]

_SETUP_GROUPS: tuple[tuple[str, str, tuple[_SetupParam, ...]], ...] = (
    (
        "Brakes",
        "LFS exposes two adjustments: <b>Maximum brake torque per "
        "wheel</b> (Nm) and <b>Brake balance</b> (front/rear bias, "
        "5–95%). Goal per the Advanced Setup Guide: the wheel at the "
        "grippiest braking point of the lap should be just shy of "
        "locking; balance compensates for engine braking on the "
        "driven axle (RWD = slightly forward, FWD = slightly "
        "rearward).",
        (
            (
                "Max brake torque per wheel (Nm)",
                "Peak clamping torque when the pedal is at 1.0. Too "
                "high and any wheel will lock; too low and you cannot "
                "reach the friction limit at the grippiest braking "
                "zone.",
                (
                    ("brake",
                     "The pedal trace shows what you commanded; "
                     "verify you really are reaching ~1.0 at the "
                     "hardest braking zone."),
                    ("fl_slip_fraction / fr_slip_fraction / "
                     "rl_slip_fraction / rr_slip_fraction",
                     "Slip fraction near 1.0 (or pegged) with brake "
                     "applied = that wheel is locking. Reduce Max "
                     "torque or back off the pedal."),
                    ("fl_slip_ratio … rr_slip_ratio",
                     "Under braking they go negative; |slip_ratio| "
                     "much beyond 0.15 means past the longitudinal "
                     "peak — tyre sliding more than gripping."),
                    ("fl_long_force_n … rr_long_force_n",
                     "Per-wheel braking force. If one wheel's |Fx| "
                     "saturates well before the others, that wheel is "
                     "locking first."),
                    ("aid_abs_active",
                     "OutGauge ShowLights ABS bit. Constant flicker "
                     "= you are commanding more brake than the tyre "
                     "can hold; the ABS is doing the work for you."),
                    ("accel_x",
                     "Peak deceleration plateau. A clean braking "
                     "event should sit near the tyre limit "
                     "(1.0–1.5 g road, 2.0+ g slicks) flat-topped, "
                     "not spiky."),
                ),
                "Look at the four <code>*_slip_fraction</code> "
                "traces during the hardest stop of the lap: the "
                "best Max-torque is the one where they all peak "
                "together just below 1.0.",
            ),
            (
                "Brake balance (% front)",
                "Splits total clamping between front and rear. "
                "Wiki: front-locking → reduce % front; rear-locking "
                "→ increase % front. Account for engine braking on "
                "the driven axle.",
                (
                    ("brake_bias_real_pct",
                     "Derived effective front bias including engine "
                     "braking. Compare against the slider you set; "
                     "the gap is the engine-braking contribution."),
                    ("fl_slip_fraction & fr_slip_fraction vs "
                     "rl_slip_fraction & rr_slip_fraction",
                     "Whichever axle saturates first is the one with "
                     "too much bias. Wiki Example 1 = front locks "
                     "first; Example 2 = rear locks first."),
                    ("fl_long_force_n + fr_long_force_n vs "
                     "rl_long_force_n + rr_long_force_n",
                     "Axle Fx sum ratio at peak braking should match "
                     "your intended bias once weight transfer is in."),
                    ("transfer_long_n_real",
                     "Longitudinal weight transfer "
                     "(≈ m·a_x·h_cg/wheelbase). Tells you how much "
                     "load has already moved forward and how much "
                     "the rear axle can still hold."),
                    ("understeer_index",
                     "Big +ve spike when you stand on the brakes = "
                     "front locked / pushing; big −ve = rear stepping "
                     "out. Trail-braking should show only a small "
                     "−ve excursion."),
                ),
                "Trail-brake into a medium corner: if "
                "<code>understeer_index</code> goes deeply negative "
                "and the rears reach <code>slip_fraction</code> ≈ 1 "
                "first → move bias forward.",
            ),
        ),
    ),
    (
        "Suspension — Springs (Stiffness)",
        "Wiki: tune by <i>spring frequency</i>, not absolute N/m. "
        "Sensible window ≈ 1.7–3.0 Hz for road/GT cars, 4–8 Hz for "
        "downforce cars. Higher rear frequency (RWD) or higher front "
        "(FWD) by 0.15–0.25 Hz biases the car towards understeer/"
        "oversteer respectively.",
        (
            (
                "Spring stiffness (front / rear, N/mm)",
                "Stiffer spring = less travel under the same load, "
                "less body roll, less mechanical grip over bumps; "
                "softer = more grip on smooth tracks but more "
                "platform movement.",
                (
                    ("fl_susp_deflect_m … rr_susp_deflect_m",
                     "Static and peak compression. If a corner uses "
                     "<25% of the available travel the spring may be "
                     "too stiff; if it bottoms out frequently it is "
                     "too soft or ride height too low."),
                    ("fl_susp_speed_mps … rr_susp_speed_mps",
                     "Spring rate sets the natural frequency together "
                     "with damping. Persistent oscillation after a "
                     "kerb = underdamped or wrong spring."),
                    ("fl_vertical_load_n … rr_vertical_load_n",
                     "Load reaching the contact patch. With a stiffer "
                     "spring, transients are sharper (faster rise/"
                     "fall) — useful to confirm the change took "
                     "effect."),
                    ("roll_rad / pitch_rad",
                     "Chassis attitude. Stiffer springs reduce both "
                     "the steady-state roll in long corners and the "
                     "pitch dive under braking."),
                    ("understeer_index",
                     "Persistent bias confirms the front/rear "
                     "frequency split chose the intended direction."),
                ),
                "Open the <b>Dampers tab</b> alongside the deflection "
                "traces: if the bell curve of suspension velocity is "
                "centred far from 0 mm/s the springs are not in their "
                "linear range — usually too stiff for the surface.",
            ),
        ),
    ),
    (
        "Suspension — Bump & Rebound Damping",
        "Wiki: bump damping resists compression, rebound resists "
        "extension. Bump controls the <i>unsprung</i> mass (wheel), "
        "rebound controls the <i>sprung</i> mass (chassis). Aim ≈ "
        "0.8× critical damping on rebound, 0.5–0.75× rebound on "
        "bump. Use the HS/LS split: low-speed → driver inputs and "
        "weight transfer; high-speed → kerbs and bumps. The "
        "industry cross-over sits near ~50 mm/s.",
        (
            (
                "Bump (compression) damping",
                "Resists the wheel moving up into the body. Too "
                "high = wheel skips over bumps, lost grip; too low "
                "= wheel slams into the bump stops.",
                (
                    ("fl_susp_speed_mps … rr_susp_speed_mps",
                     "Positive velocity histogram is the bump side. "
                     "Healthy LS lobe up to ~50 mm/s; HS tail short "
                     "and capped. A massive HS tail = under-damped "
                     "bump; no HS tail at all = over-damped."),
                    ("fl_susp_deflect_m … rr_susp_deflect_m",
                     "Watch for double bumps / chatter after a kerb. "
                     "Long oscillation = too soft; instant jolt that "
                     "doesn't move = too hard."),
                    ("fl_vertical_load_n … rr_vertical_load_n",
                     "Vertical load that survives the bump tells you "
                     "if the contact patch is being preserved."),
                    ("fl_touching … rr_touching",
                     "If a wheel briefly leaves the road over kerbs, "
                     "bump damping is likely too stiff."),
                ),
                "Race a lap then open the Dampers tab. The bump "
                "histogram you want looks like a peaked Gaussian "
                "centred near 0 with a small skew toward positive HS.",
            ),
            (
                "Rebound (extension) damping",
                "Resists the wheel moving down away from the body. "
                "Too high = wheel can't return to road after a "
                "compression (jacks down, loses grip on the inside "
                "wheel); too low = floaty chassis, slow transient "
                "response.",
                (
                    ("fl_susp_speed_mps … rr_susp_speed_mps",
                     "Negative velocity side (extension). Should "
                     "settle quickly after a corner exit; long "
                     "negative tail = under-damped rebound."),
                    ("pitch_rad",
                     "Dive on entry, squat on exit. Excess rebound "
                     "on the lifting axle keeps the platform skewed "
                     "for too long."),
                    ("roll_rad",
                     "After an apex, roll should release smoothly; "
                     "ringing = rebound too soft."),
                    ("fl_vertical_load_n … rr_vertical_load_n",
                     "Inside wheel on a long corner. If load decays "
                     "to near 0 and stays there, the inside wheel is "
                     "being jacked down by excessive rebound."),
                    ("understeer_index",
                     "Transient damping primarily tunes corner "
                     "entry/exit balance — read US/OS spikes at "
                     "those instants."),
                ),
                "Basic Setup Guide quick-reference: <i>entry "
                "understeer → soften front compression + soften rear "
                "rebound; exit oversteer → soften rear compression + "
                "soften front rebound.</i>",
            ),
        ),
    ),
    (
        "Suspension — Anti-Roll Bars",
        "Wiki: ARB reduces body roll without making the spring "
        "stiffer; downside is that one-wheel bumps transfer to the "
        "other side, hurting mechanical grip. Use ARB <i>balance</i> "
        "to fine-tune US/OS once springs are set neutral. Don't let "
        "the ARB:spring roll-stiffness ratio exceed ≈ 1.0.",
        (
            (
                "Front / Rear ARB stiffness",
                "Stiffer front ARB → more understeer in steady-state "
                "corners; stiffer rear ARB → more oversteer.",
                (
                    ("roll_rad",
                     "Peak steady roll in a long corner — main "
                     "indicator of overall ARB+spring stiffness."),
                    ("roll_rate_rads",
                     "Roll velocity at turn-in. Stiffer ARBs raise "
                     "roll rate, snapping the chassis to attitude "
                     "faster — too snappy and the car becomes "
                     "nervous."),
                    ("fl_vertical_load_n vs fr_vertical_load_n / "
                     "rl_vertical_load_n vs rr_vertical_load_n",
                     "Lateral load split between left/right on the "
                     "same axle. A stiffer ARB on an axle steals "
                     "load from the inside wheel and transfers it "
                     "to the outside wheel, reducing that axle's "
                     "grip first."),
                    ("transfer_lat_n_real",
                     "Lateral load transfer ≈ m·a_y·h_cg/track. "
                     "ARB redistributes the per-axle share of that "
                     "total."),
                    ("understeer_index",
                     "Mid-corner sign tells you whether the current "
                     "ARB balance leans US (+) or OS (−)."),
                    ("friction_use_fl … friction_use_rr",
                     "|F|/(μ·Fz) per corner. The wheel that "
                     "saturates first in a sustained corner is on "
                     "the axle that needs the ARB softened."),
                ),
                "Mid-corner US → soften front ARB or stiffen rear; "
                "mid-corner OS → the opposite. Confirm with the "
                "load-difference traces.",
            ),
        ),
    ),
    (
        "Suspension — Ride Height",
        "Wiki: lowest practical ride height = lowest CG = less load "
        "transfer = more grip, but you mustn't run out of travel "
        "or bottom out. Tune <i>last</i>, once springs and dampers "
        "are set.",
        (
            (
                "Front / Rear ride height reduction (mm)",
                "Distance from chassis reference to ground at rest. "
                "Affects CG height, mechanical/aero balance and "
                "available bump travel.",
                (
                    ("fl_susp_deflect_m … rr_susp_deflect_m",
                     "Look for instances at 0 travel left = bottoming "
                     "out. The Advanced Guide accepts the suspension "
                     "<i>just</i> bottoming once or twice per lap if "
                     "handling isn't upset."),
                    ("transfer_long_n_real / transfer_lat_n_real",
                     "Both scale with h_cg. Lower the car and these "
                     "magnitudes drop directly."),
                    ("pitch_rad",
                     "Static pitch (squat/rake). The Wiki recommends "
                     "a small static dive to compensate acceleration "
                     "squat (note: LFS does not yet model the aero "
                     "effect of pitch)."),
                    ("accel_z",
                     "Big negative spikes = bumpstop hits / kerb "
                     "strikes. Raise ride height or stiffen bump "
                     "damping if these are causing handling issues."),
                ),
                "Run the lap, view <code>*_susp_deflect_m</code> on "
                "distance axis: lower the car in 2–3 mm steps until "
                "you see the deflection traces using the upper "
                "~85–95% of available travel without persistent "
                "bottoming.",
            ),
        ),
    ),
    (
        "Steering — Maximum Lock",
        "Wiki: range 9°–36°. Lower lock = less sensitive, more "
        "precise, harder to catch oversteer. Higher lock = easier "
        "to catch slides but twitchy on centre.",
        (
            (
                "Maximum steering lock (°)",
                "How far the front wheels move at full controller "
                "input.",
                (
                    ("steering_input",
                     "Normalised −1..+1 input from OutGauge. With "
                     "high lock you'll see small numeric values doing "
                     "lots of work."),
                    ("steer_rate_rads",
                     "Hand-speed at the wheel. Higher lock = lower "
                     "rate needed for the same angle on track."),
                    ("steer_reversal_rate_hz",
                     "Reversals/s. If you constantly saw on centre, "
                     "lock is probably too high for your input "
                     "device."),
                    ("fl_steer_rad / fr_steer_rad",
                     "Actual front-wheel angles. With Ackermann "
                     "(<100% Parallel Steer) inner and outer wheels "
                     "differ."),
                ),
                "Compare <code>steer_rate_rads</code> across laps: "
                "lower lock should produce smoother peak rates and "
                "fewer reversals.",
            ),
        ),
    ),
    (
        "Steering — Caster (race cars)",
        "Wiki: caster is shown for all cars but only adjustable on "
        "race cars (inclination is fixed by geometry). Caster adds "
        "negative camber to the steered wheel <i>linearly</i> with "
        "lock; helps the outside front stay flat in a turn but "
        "needs heavier steering.",
        (
            (
                "Caster angle (°)",
                "Tilt of the kingpin axis viewed from the side. More "
                "caster → more self-centring, more dynamic camber, "
                "more straight-line stability.",
                (
                    ("fl_lean_rel_road_rad / fr_lean_rel_road_rad",
                     "Live camber relative to road during cornering. "
                     "More caster = more negative camber on the "
                     "outside front in long corners."),
                    ("fl_air_temp_c / fr_air_temp_c (inner/middle/"
                     "outer in F9 view)",
                     "Camber correctness shows up as even left-to-"
                     "right tyre-band temperatures. Studio surfaces "
                     "the inflation-air temp; correlate with the F9 "
                     "tyre-bar wear screen recommended in the Wiki."),
                    ("fl_y_force_n / fr_y_force_n",
                     "Lateral force per front wheel: extra caster "
                     "helps the outside wheel produce more Fy in a "
                     "sustained corner."),
                    ("ffb_torque",
                     "Steering-rack torque feedback. More caster "
                     "literally makes the wheel heavier — verify by "
                     "the bigger plateau on the trace."),
                    ("understeer_index",
                     "Quick-reference (Wiki): entry US → more caster; "
                     "entry OS → less caster."),
                ),
                "Push entry US: try +1 caster. Watch "
                "<code>fl/fr_lean_rel_road_rad</code> for the right "
                "−2° to −4° band at peak lateral g.",
            ),
        ),
    ),
    (
        "Steering — Parallel Steer (Ackermann)",
        "Wiki: 100% = wheels stay parallel; 0% = full Ackermann (inner "
        "wheel takes more lock than outer). True Ackermann reduces "
        "tyre scrub at low-speed corners; race cars usually run high "
        "% with static front toe-out instead.",
        (
            (
                "Parallel Steer (%)",
                "How much extra angle the inside wheel takes versus "
                "the outside as lock is applied.",
                (
                    ("fl_steer_rad vs fr_steer_rad",
                     "Direct read of the dynamic toe applied. The "
                     "difference grows as 100% → 0%."),
                    ("fl_tan_slip_angle / fr_tan_slip_angle",
                     "Inner vs outer front slip angles. With correct "
                     "Ackermann they should converge on a similar "
                     "magnitude around the apex."),
                    ("fl_air_temp_c / fr_air_temp_c",
                     "Wrong Ackermann shows as uneven front-tyre "
                     "temperatures, especially in slow corners."),
                ),
                "Tight hairpins exaggerate Ackermann effects: compare "
                "the two front <code>tan_slip_angle</code> traces "
                "through the slowest corner of the lap.",
            ),
        ),
    ),
    (
        "Wheels — Toe",
        "Wiki: front toe-out (negative toe-in) helps turn-in but "
        "makes straights nervous; front toe-in stabilises. Rear toe-"
        "in stabilises RWD against power-on oversteer; rear toe-out "
        "is used on FWD to provoke rotation. All toe increases tyre "
        "drag and heat.",
        (
            (
                "Front toe-in (°)",
                "Wheel pre-angle (closer at front = toe-in, closer "
                "at back = toe-out).",
                (
                    ("fl_tan_slip_angle / fr_tan_slip_angle",
                     "Static toe shows up as a non-zero baseline "
                     "even in a straight line."),
                    ("fl_air_temp_c / fr_air_temp_c",
                     "Excess toe heats and wears the front tyres."),
                    ("steer_reversal_rate_hz",
                     "Toe-out on the front raises straight-line "
                     "nervousness → more micro-corrections."),
                    ("understeer_index",
                     "Less front toe-out (or some toe-in) damps "
                     "entry oversteer; more toe-out attacks entry "
                     "understeer."),
                ),
                "Tail-wagging on the straight → add toe-in 0.1° at a "
                "time; lazy turn-in → toe-out 0.1°.",
            ),
            (
                "Rear toe-in (°)",
                "Same definition at the rear. Even small changes "
                "(0.3°) move the car a lot.",
                (
                    ("rl_tan_slip_angle / rr_tan_slip_angle",
                     "Rear slip-angle baseline and how it evolves "
                     "on throttle exit."),
                    ("understeer_index",
                     "Wiki: more rear toe-in → more resistance to "
                     "(power) oversteer. Less or toe-out → more "
                     "rotation, useful on FWD."),
                    ("rl_air_temp_c / rr_air_temp_c",
                     "Excess rear toe heats and wears the rears, "
                     "especially over a stint."),
                    ("yaw_rate_rads",
                     "Rear toe-in tames yaw transients; toe-out "
                     "amplifies them at corner exit."),
                ),
                "RWD with power-on snap → +0.1° rear toe-in. FWD "
                "stubborn understeer mid-corner → −0.1°.",
            ),
        ),
    ),
    (
        "Wheels — Camber",
        "Wiki: aim for a flat contact patch under peak lateral g — "
        "use the F9 in-game tyre-bar view as the primary truth, but "
        "the telemetry channels give you a real-time live read.",
        (
            (
                "Static camber (°)",
                "Wheel tilt at rest. Negative camber compensates "
                "body-roll-induced positive camber on the outside "
                "wheel during cornering.",
                (
                    ("fl_lean_rel_road_rad … rr_lean_rel_road_rad",
                     "Live camber vs road. Target ≈ −0.5° to −1.5° "
                     "on the loaded outside wheel at peak lateral g."),
                    ("fl_air_temp_c … rr_air_temp_c",
                     "Across the band: <i>inner hotter than outer</i>"
                     " = too much negative camber; <i>outer hotter "
                     "than inner</i> = not enough."),
                    ("fl_vertical_load_n … rr_vertical_load_n",
                     "Loaded wheel is where camber matters most; "
                     "correlate live camber with peak load."),
                    ("fl_x_force_n … rr_x_force_n",
                     "Too much camber reduces braking grip "
                     "noticeably — watch peak |Fx| at the heaviest "
                     "stop."),
                ),
                "Open the relevant <code>*_lean_rel_road_rad</code> "
                "and <code>*_air_temp_c</code> traces at the apex "
                "of a long corner — same logic as the Basic Setup "
                "Guide tyre-temperature table.",
            ),
        ),
    ),
    (
        "Tyres — Type / Compound",
        "Wiki: road cars choose between Normal and Super; race cars "
        "between R1 (softest) and R4 (hardest). Optimum is the "
        "compound that holds <i>working temperature</i> through the "
        "stint without overheating.",
        (
            (
                "Tyre compound",
                "Trade-off between peak grip and durability / "
                "temperature window.",
                (
                    ("fl_air_temp_c … rr_air_temp_c",
                     "Inflation-air temperature. Sustained reds = too "
                     "soft for the workload; permanent blues = too "
                     "hard or pressures too high."),
                    ("tyre_work_w_fl … tyre_work_w_rr",
                     "Power being dissipated as heat in the contact "
                     "patch. Big disparity between a corner and the "
                     "average = that tyre is on the edge for this "
                     "compound."),
                    ("friction_use_fl … friction_use_rr",
                     "If a softer compound saturates everywhere = "
                     "you're at the grip limit; if a harder one "
                     "never saturates = unused grip = potential lap "
                     "time."),
                    ("fl_slip_fraction … rr_slip_fraction",
                     "Persistent values above ≈ 0.5 say the tyre is "
                     "sliding more than gripping → wrong compound or "
                     "wrong pressures."),
                ),
                "Run a representative stint, then check that all "
                "four <code>*_air_temp_c</code> traces sit in the "
                "green band for most of each lap.",
            ),
        ),
    ),
    (
        "Tyres — Pressure",
        "Wiki: lower pressure = bigger contact patch and more "
        "absolute grip, but more rolling resistance, more heat and "
        "more wear. Higher pressure = sharper response, cooler "
        "running, less ultimate grip. Slight over-inflation is safer "
        "than slight under-inflation.",
        (
            (
                "Tyre pressure (bar / psi)",
                "Cold inflation pressure.",
                (
                    ("fl_air_temp_c … rr_air_temp_c",
                     "Pressure is the main lever for the steady-state "
                     "temperature; lower it 0.05 bar to add a few °C."),
                    ("fl_lean_rel_road_rad … rr_lean_rel_road_rad",
                     "Together with the F9 tyre-bar wear pattern: "
                     "centre hotter = over-inflated; edges hotter = "
                     "under-inflated."),
                    ("fl_vertical_load_n … rr_vertical_load_n",
                     "Combined with pressure, sets the contact-patch "
                     "size. A sharp load transient with high pressure "
                     "shows up as a quicker rise."),
                    ("steer_rate_rads",
                     "Higher pressure = sharper response → smaller "
                     "rates needed for the same line change."),
                ),
                "Adjust pressure one step at a time; verify with the "
                "<code>*_air_temp_c</code> stint trend before changing "
                "anything else.",
            ),
        ),
    ),
    (
        "Final drive — Gear ratios",
        "Wiki: top gear set so you hit max-power RPM at end of the "
        "longest straight; first gear set as tall as possible "
        "without bogging; ratios spaced so RPM drop reduces with "
        "each upshift.",
        (
            (
                "Final drive ratio + Individual gear ratios",
                "Translates engine RPM into wheel torque and top "
                "speed.",
                (
                    ("rpm",
                     "Trace shape across a straight tells you whether "
                     "top gear is right. Hitting limiter mid-straight "
                     "= too short; never reaching peak power = too "
                     "tall."),
                    ("gear",
                     "Detect short-shifts, mis-shifts and how often "
                     "you sit on a given gear."),
                    ("speed_ms / speed_kmh",
                     "Combined with RPM gives the actual gear curve. "
                     "Plot speed vs RPM coloured by gear."),
                    ("throttle",
                     "If throttle saturates at 1.0 between two gears "
                     "but speed barely climbs, that gear is too tall."),
                    ("accel_x",
                     "Acceleration steps at each upshift should "
                     "decrease smoothly. A flat-spot after a shift = "
                     "RPM falling below the torque peak."),
                ),
                "On a long straight: <code>rpm</code> should reach "
                "max-power RPM at the braking point in top gear; "
                "<code>accel_x</code> after each shift should never "
                "drop below the previous gear's exit value.",
            ),
        ),
    ),
    (
        "Final drive — Differential type",
        "Wiki: open diff and locked diff are rarely used. Clutch-pack "
        "LSD (Salisbury) and viscous LSD are the racing choices. "
        "Clutch-pack adds <i>power</i> + <i>coast</i> + <i>preload</i>"
        " sliders; viscous adds a single viscosity slider.",
        (
            (
                "Diff type (open / locked / viscous / clutch-pack)",
                "How torque is allocated between the two wheels on the "
                "driven axle when their speeds differ.",
                (
                    ("rl_ang_vel_rads vs rr_ang_vel_rads "
                     "(or fl/fr on FWD)",
                     "Wheel-speed difference is the direct read on the "
                     "diff: open = big gap on a corner, locked = ~0, "
                     "LSD = small controlled gap."),
                    ("rl_slip_ratio vs rr_slip_ratio",
                     "Driven wheel slip ratios. Open diff = inside "
                     "wheel will pull massive slip on throttle; LSD "
                     "splits it."),
                    ("rl_long_force_n vs rr_long_force_n",
                     "Torque actually transmitted. Confirms whether "
                     "the slipping wheel is also the one with low Fx."),
                    ("understeer_index",
                     "On RWD a tight diff = more power oversteer; on "
                     "FWD a tight power side = more turn-in."),
                ),
                "Plot left-vs-right wheel speed on the driven axle "
                "exiting a hairpin: zero gap = locked behaviour, huge "
                "gap = open / loose LSD.",
            ),
            (
                "Diff slip limits — Power lock / Coast lock / Preload",
                "Clutch-pack: power lock = how much it locks under "
                "throttle, coast lock = under engine braking / lift-"
                "off, preload = baseline lock at zero torque.",
                (
                    ("rl_ang_vel_rads vs rr_ang_vel_rads",
                     "Power lock too low → outside wheel laggy on "
                     "exit. Coast lock too high → rear hops or locks "
                     "on entry."),
                    ("rl_slip_ratio vs rr_slip_ratio",
                     "Inside-wheel slip on throttle = power lock "
                     "insufficient."),
                    ("yaw_rate_rads / understeer_index",
                     "On RWD: high power lock = more exit OS, high "
                     "coast lock = stabilises entry but kills "
                     "rotation. Wiki: oversteer on the brakes → add "
                     "preload; understeer on initial squeeze → remove "
                     "preload."),
                    ("brake",
                     "Use it together with coast lock: study what "
                     "happens between brake release and throttle-on."),
                ),
                "On a long throttle-on exit, compare driven-axle "
                "slip ratios: if the inside is far higher → increase "
                "<i>power lock</i>.",
            ),
        ),
    ),
    (
        "Final drive — Front Torque Bias (AWD only)",
        "Wiki: % of engine torque sent to the front wheels on AWD. "
        "Lower % = more rear-biased / oversteer-prone; higher % = "
        "more front-biased / understeer-prone.",
        (
            (
                "Front torque bias (%)",
                "Static torque split front:rear.",
                (
                    ("fl_long_force_n + fr_long_force_n vs "
                     "rl_long_force_n + rr_long_force_n",
                     "Axle Fx sum on throttle reveals the realised "
                     "split; compare with the slider."),
                    ("fl_slip_ratio + fr_slip_ratio vs rl_slip_ratio "
                     "+ rr_slip_ratio",
                     "If only one axle saturates on throttle the bias "
                     "is wrong for the available grip."),
                    ("understeer_index",
                     "On throttle: more front bias = more power "
                     "understeer; less = more power oversteer."),
                    ("fl_air_temp_c + fr_air_temp_c vs rl_air_temp_c "
                     "+ rr_air_temp_c",
                     "Wiki: a balanced AWD setup tends to equalise "
                     "tyre temperatures between axles."),
                ),
                "Long uphill exit: if the rears overheat and the "
                "fronts stay cool → move 5% more torque forward.",
            ),
        ),
    ),
    (
        "Downforce — Wing angles",
        "Wiki: higher wing = more grip and more drag. Aero balance "
        "should roughly match weight distribution to stay neutral "
        "with speed. Front-bias understeers at high speed; rear-bias "
        "oversteers.",
        (
            (
                "Front / Rear wing angle (°)",
                "Aerodynamic downforce per axle.",
                (
                    ("speed_ms / speed_kmh",
                     "Top speed gain/loss is the most visible "
                     "consequence of changing wing."),
                    ("fl_vertical_load_n + fr_vertical_load_n vs "
                     "rl_vertical_load_n + rr_vertical_load_n",
                     "At high speed, increased load above the static "
                     "weight is the actual downforce being generated."),
                    ("fl_susp_deflect_m … rr_susp_deflect_m",
                     "Rise of static compression with speed = "
                     "downforce settling the platform. Make sure you "
                     "don't run out of travel on the fastest part of "
                     "the lap."),
                    ("understeer_index",
                     "Compute it at high speed only — if it shifts +ve "
                     "vs low-speed corners → too much rear wing; if "
                     "−ve → too much front."),
                    ("accel_x",
                     "Peak braking deceleration grows with speed in "
                     "downforce cars — that's your aero working."),
                ),
                "Take a high-speed corner and a slow corner. If the "
                "fast corner is much more understeery than the slow "
                "one → reduce rear wing (or add front).",
            ),
        ),
    ),
)


class HelpDialog(QDialog):
    """Modal dialog with a channel-and-interpretation reference."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Channel & telemetry guide")
        self.resize(960, 720)

        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(False)
        browser.setHtml(self._build_html())
        browser.setStyleSheet(
            "QTextBrowser { font-size: 11pt; }"
            "h1 { margin-top: 14px; }"
            "h2 { margin-top: 18px; border-bottom: 1px solid #555; }"
            "h3 { margin-top: 12px; color: #d8d8d8; }"
            "table { border-collapse: collapse; margin-top: 6px; }"
            "td, th { padding: 4px 8px; border-bottom: 1px solid #333; "
            "vertical-align: top; }"
            "th { text-align: left; }"
            "code { color: #c0c0c0; }"
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(browser, 1)
        layout.addWidget(buttons)

        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_html(self) -> str:
        groups = channels_by_group()
        parts: list[str] = [
            "<h1>Channel &amp; telemetry guide</h1>",
            "<p>This panel explains, in plain language, what every "
            "channel measures and how to read it. No telemetry "
            "background required: general driving and car knowledge "
            "is enough.</p>",
            _HOW_TO_READ,
            self._build_setup_html(),
            "<h2>Channels by group</h2>",
        ]
        ordered = [g for g in _GROUP_ORDER if g in groups] + [
            g for g in sorted(groups) if g not in _GROUP_ORDER
        ]
        for group in ordered:
            intro = _GROUP_INTRO.get(group, "")
            parts.append(f"<h3>{group}</h3>")
            if intro:
                parts.append(f"<p>{intro}</p>")
            parts.append(
                "<table width='100%'>"
                "<tr><th width='22%'>Channel</th><th width='10%'>Unit</th>"
                "<th>What it is &amp; how to read it</th></tr>"
            )
            for info in groups[group]:
                desc = info.description or ""
                interp = info.interpretation or ""
                cell: list[str] = []
                if desc:
                    cell.append(desc)
                if interp:
                    cell.append(f"<i>{interp}</i>")
                cell_html = "<br>".join(cell) if cell else "&mdash;"
                units = info.units or ""
                parts.append(
                    f"<tr><td><b>{info.label}</b><br>"
                    f"<code>{info.column}</code></td>"
                    f"<td>{units}</td>"
                    f"<td>{cell_html}</td></tr>"
                )
            parts.append("</table>")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Setup tab mapping
    # ------------------------------------------------------------------

    def _build_setup_html(self) -> str:
        parts: list[str] = [
            "<h2>Setup tab \u2194 telemetry channels</h2>",
            "<p>One row per slider in the LFS garage Setup screen, "
            "with the channels you should look at to know whether "
            "the change worked. Background condensed from the "
            "official LFS wiki "
            "(<i>Basic Setup Guide</i>, <i>Advanced Setup Guide</i>"
            ", <i>Technical Reference</i>).</p>",
        ]
        for group_name, intro, params in _SETUP_GROUPS:
            parts.append(f"<h3>{group_name}</h3>")
            if intro:
                parts.append(f"<p>{intro}</p>")
            parts.append(
                "<table width='100%'>"
                "<tr><th width='22%'>Setup parameter</th>"
                "<th width='32%'>Channels that show the effect</th>"
                "<th>What to read &amp; tuning shortcut</th></tr>"
            )
            for name, what, channels, shortcut in params:
                ch_rows: list[str] = []
                for ch_id, why in channels:
                    ch_rows.append(
                        f"<b><code>{ch_id}</code></b> &mdash; {why}"
                    )
                ch_html = "<br>".join(ch_rows)
                parts.append(
                    f"<tr><td><b>{name}</b><br>"
                    f"<i>{what}</i></td>"
                    f"<td>{ch_html}</td>"
                    f"<td>{shortcut}</td></tr>"
                )
            parts.append("</table>")
        return "\n".join(parts)


__all__ = ["HelpDialog"]
