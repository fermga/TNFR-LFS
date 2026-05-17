"""Deterministic operator → physical-delta rule table (``TRIGGER_RULES``).

Each :class:`TriggerRule` couples a *physical predicate* over the
coupled track↔car network metrics with a canonical TNFR operator and a
:class:`PhysicalAction` (the actual setup delta the advisor will
propose). Predicates are pure functions of ``metrics`` — a dict keyed by
node name (the names produced by :mod:`network_track` / :mod:`network_car`)
that carries ``epi``, ``vf``, ``theta`` and the seed ``meta`` per node.

The 8 v1 rules are physically grounded in suspension / brake / tyre
first-principles. They are deliberately conservative: each rule fires
only on sustained signatures (mean over a stint, not single samples)
and maps to a *single* tunable delta with conservative magnitude.

See ``docs/TNFR_SETUP_ADVISOR.md`` §7 for the full physical rationale.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from tnfr.operators.definitions import (
    Coherence,
    Contraction,
    Coupling,
    Dissonance,
    Reception,
    Silence,
)
from tnfr.operators.definitions import Operator

from .mapping import FRONT_WHEELS, REAR_WHEELS, WHEEL_ORDER

# -- public dataclasses -------------------------------------------------


@dataclass(frozen=True)
class PhysicalAction:
    """A single, atomic, reversible setup delta in LFS units.

    Attributes
    ----------
    kind
        One of: ``damper_rebound``, ``damper_bump``, ``spring``,
        ``arb``, ``tyre_pressure``, ``brake_bias``, ``camber``, ``toe``,
        ``ride_height``.
    target
        Wheel (``FL``/``FR``/``RL``/``RR``), axle (``front``/``rear``)
        or ``global``.
    delta
        Signed magnitude in :attr:`units`.
    units
        SI-ish LFS units: ``clicks``, ``N/mm``, ``N·mm/deg``, ``kPa``,
        ``%``, ``deg``.
    rationale_id
        ID of the firing rule (``TriggerRule.name``). Carried through
        the advisor pipeline so the UI can show *why* without
        re-evaluating predicates.
    """

    kind: str
    target: str
    delta: float
    units: str
    rationale_id: str


NodeMetrics = Mapping[str, Mapping[str, Any]]
Predicate = Callable[[NodeMetrics], bool]
OperatorFactory = Callable[[], Operator]


@dataclass(frozen=True)
class TriggerRule:
    """Maps a physical predicate to a TNFR operator + setup action."""

    name: str
    description: str
    operator_factory: OperatorFactory
    predicate: Predicate
    action: PhysicalAction
    tags: tuple[str, ...] = field(default_factory=tuple)


# -- metric helpers -----------------------------------------------------


def _get(metrics: NodeMetrics, node: str, key: str, default: float = float("nan")) -> float:
    """Return ``metrics[node][key]`` as float (NaN if missing)."""
    if node not in metrics:
        return default
    val = metrics[node].get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _meta(metrics: NodeMetrics, node: str, key: str, default: float = float("nan")) -> float:
    """Return ``metrics[node]['meta'][key]`` as float."""
    if node not in metrics:
        return default
    meta = metrics[node].get("meta") or {}
    val = meta.get(key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def agg_corner_phase(metrics: NodeMetrics, phase: str, key: str = "epi") -> float:
    """Mean ``key`` over all ``corner.*.<phase>`` nodes (NaN if none)."""
    vals: list[float] = []
    for name, attrs in metrics.items():
        if not name.startswith("corner."):
            continue
        if attrs.get("phase") != phase:
            # fall back to suffix parsing if 'phase' attr missing
            if not name.endswith("." + phase):
                continue
        try:
            vals.append(float(attrs[key]))
        except (KeyError, TypeError, ValueError):
            continue
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def extract_node_metrics(graph) -> dict[str, dict[str, Any]]:
    """Project a coupled graph to the metrics dict the predicates expect.

    Reads canonical TNFR attributes (``EPI``, ``νf``, ``theta``) plus the
    seed ``meta`` and any ``kind`` / ``phase`` / ``sector_id`` /
    ``wheel`` / ``axle`` tags previously set by the network builders.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, data in graph.nodes(data=True):
        entry: dict[str, Any] = {
            "epi": float(data.get("EPI", float("nan"))),
            "vf": float(data.get("νf", float("nan"))),
            "theta": float(data.get("theta", float("nan"))),
            "meta": dict(data.get("meta", {}) or {}),
        }
        for tag in ("kind", "phase", "sector_id", "wheel", "axle"):
            if tag in data:
                entry[tag] = data[tag]
        out[name] = entry
    return out


# -- predicate constructors --------------------------------------------


def _osc_predicate(wheel: str, phase: str, vf_thr: float = 8.0, epi_thr: float = 0.55) -> Predicate:
    node = f"wheel.{wheel}"

    def _p(m: NodeMetrics) -> bool:
        vf = _get(m, node, "vf")
        epi_w = _get(m, node, "epi")
        epi_phase = agg_corner_phase(m, phase)
        return vf > vf_thr and epi_w > 0.6 and epi_phase > epi_thr

    return _p


def _balance_predicate(
    high: str, low: str, phase: str, gap: float = 0.15, phase_thr: float = 0.55,
) -> Predicate:
    """Fires when ``epi(high) - epi(low) >= gap`` during ``phase``."""

    def _p(m: NodeMetrics) -> bool:
        eh = _get(m, high, "epi")
        el = _get(m, low, "epi")
        ep = agg_corner_phase(m, phase)
        return (eh - el) >= gap and ep > phase_thr

    return _p


def _friction_predicate(wheel: str, thr: float = 0.85) -> Predicate:
    node = f"wheel.{wheel}"

    def _p(m: NodeMetrics) -> bool:
        return _meta(m, node, "friction_use_mean") >= thr

    return _p


def _lateral_imbalance_predicate(axle: str, gap: float = 0.20) -> Predicate:
    wheels = FRONT_WHEELS if axle == "front" else REAR_WHEELS
    l, r = (f"wheel.{w}" for w in wheels)

    def _p(m: NodeMetrics) -> bool:
        a = _get(m, l, "epi")
        b = _get(m, r, "epi")
        return abs(a - b) >= gap

    return _p


def _brake_bias_predicate(target_front: float = 0.60, tol: float = 0.04) -> Predicate:
    def _p(m: NodeMetrics) -> bool:
        bf = _meta(m, "brake.front", "brake_bias_front_real_mean")
        return bf == bf and abs(bf - target_front) >= tol  # NaN-safe

    return _p


def _thermal_predicate(axle: str, t_max_c: float = 130.0) -> Predicate:
    wheels = FRONT_WHEELS if axle == "front" else REAR_WHEELS

    def _p(m: NodeMetrics) -> bool:
        temps = [_meta(m, f"wheel.{w}", "temp_mean_c") for w in wheels]
        temps = [t for t in temps if t == t]  # drop NaN
        if not temps:
            return False
        return (sum(temps) / len(temps)) >= t_max_c

    return _p


# --- additional predicate constructors (Phase 9 full expansion) -------
# Every predicate below maps to one structural signature derived from
# the canonical TNFR nodal dynamics on the coupled track↔car network.
# The operator choices intentionally avoid bifurcation-class primitives
# (Mutation / SelfOrganization / Resonance / Transition) so that the
# rule sequence remains grammar-stable end-to-end; instead, each new
# rule is modelled as a Coherence (rest-state retune), Coupling
# (bind two sub-networks), Contraction (relieve a saturated channel)
# or Silence (let the channel rest) operation, which is the most
# physically faithful reading for setup-tuning deltas anyway.


def _chassis_load_predicate(
    axle: str, epi_thr: float = 0.65, vf_max: float = 3.0,
) -> Predicate:
    """Sustained structural load on the axle with a *slow* internal
    rate → suspension is swallowing load without oscillating; the
    spring rate set-point itself must be re-tuned (Coherence).
    """
    node = f"axle.{axle}"

    def _p(m: NodeMetrics) -> bool:
        epi = _get(m, node, "epi")
        vf = _get(m, node, "vf")
        return epi >= epi_thr and vf == vf and vf <= vf_max

    return _p


def _resonance_predicate(
    axle: str, vf_min: float = 4.0, vf_gap: float = 1.5,
) -> Predicate:
    """Axle and chassis share a dominant internal rate → coupled
    modes that bleed energy from the chassis into the suspension.
    Stiffen bump on that axle to detune (Coupling).
    """
    node = f"axle.{axle}"

    def _p(m: NodeMetrics) -> bool:
        vf_a = _get(m, node, "vf")
        vf_c = _get(m, "chassis", "vf")
        if not (vf_a == vf_a and vf_c == vf_c):
            return False
        return vf_a >= vf_min and abs(vf_a - vf_c) <= vf_gap

    return _p


def _thermal_asymmetry_predicate(axle: str, gap_c: float = 12.0) -> Predicate:
    """Left-vs-right wheel temperature gap on an axle ≥ ``gap_c`` °C
    → camber off target; the geometric rest-state needs a retune
    (Coherence).
    """
    wheels = FRONT_WHEELS if axle == "front" else REAR_WHEELS
    l, r = (f"wheel.{w}" for w in wheels)

    def _p(m: NodeMetrics) -> bool:
        tl = _meta(m, l, "temp_mean_c")
        tr = _meta(m, r, "temp_mean_c")
        if not (tl == tl and tr == tr):
            return False
        return abs(tl - tr) >= gap_c

    return _p


def _slip_diff_predicate(axle: str, thr: float = 0.15) -> Predicate:
    """Left-right slip-fraction RMS on an axle ≥ ``thr`` → cross-wheel
    kinematic drift consistent with mis-toed geometry; the toe set-
    point needs a retune (Coherence).
    """
    node = f"axle.{axle}"

    def _p(m: NodeMetrics) -> bool:
        s = _meta(m, node, "slip_diff_rms")
        return s == s and s >= thr

    return _p


def _brake_force_concentration_predicate(
    axle: str, factor: float = 1.30,
) -> Predicate:
    """One axle absorbs ``factor`` × more brake-force RMS than the
    other → structural channel saturated; shift bias away to relieve
    the saturated channel (Contraction).
    """
    other = "rear" if axle == "front" else "front"

    def _p(m: NodeMetrics) -> bool:
        a = _meta(m, f"brake.{axle}", "brake_force_rms_n")
        b = _meta(m, f"brake.{other}", "brake_force_rms_n")
        if not (a == a and b == b) or b <= 1.0:
            return False
        return a / b >= factor

    return _p


def _corner_drift_predicate(
    from_phase: str, to_phase: str, gap: float = 0.20,
) -> Predicate:
    """Mean structural load across corners drops by ≥ ``gap`` between
    two phases → the structural state is being lost across the phase
    boundary. Compensate via ride-height adjustment (Coherence).
    """

    def _p(m: NodeMetrics) -> bool:
        a = agg_corner_phase(m, from_phase)
        b = agg_corner_phase(m, to_phase)
        if not (a == a and b == b):
            return False
        return (a - b) >= gap

    return _p


# -- canonical rule table ----------------------------------------------
#
# Every rule below is *derivable* from the canonical TNFR nodal
# dynamics evaluated on the coupled track↔car network produced by
# Phase 3. The advisor uses a deliberately restricted *physical*
# operator palette — only operators that map onto a concrete setup
# adjustment a driver can apply at the garage:
#
#   * Coherence       — re-tune the rest set-point of a sub-system
#                       (dampers, springs, brake bias, camber, toe,
#                        ride height).
#   * Dissonance      — front↔rear structural imbalance at a phase
#                       (resolve via ARB delta).
#   * Reception       — friction overload at a wheel (absorb via
#                       pressure).
#   * Coupling        — left↔right mismatch on an axle, or coupled
#                       axle↔chassis modes (bind via ARB or bump).
#   * Silence         — sustained thermal overload on an axle (let
#                       the channel rest by lowering pressure).
#   * Contraction     — brake-force concentration on one axle
#                       (relieve the saturated channel via bias).
#
# This palette is NOT closed under the canonical U1–U6 grammar on its
# own: a stint where only ARB rules fire would emit a Dissonance-only
# sequence, violating U2 (destabilizer without stabilizer); a stint
# where ``epi_mean`` ≈ 0 would violate U1a (no canonical generator).
# Closure is enforced at advisor level by :func:`advisor._pad_for_grammar`
# which auto-prepends ``Emission`` (U1a) / ``Coherence`` (U2) and
# auto-appends ``Silence`` (U1b) as needed. Padding operators carry no
# PhysicalAction and contribute 0 to the surrogate ΔC.
#
# Magnitudes are deliberately conservative single-step deltas; the
# advisor's ``synthesize_setup`` aggregates them into a final
# coherent setup proposal.

def _osc_rule(wheel: str, phase: str) -> TriggerRule:
    """Build a Coherence damper rule for ``wheel``·``phase``.

    Convention: entry/braking → rebound (extension control while loading
    the outside); apex → rebound (sustain plateau); exit → bump
    (compression control under traction transfer).
    """
    kind = "damper_bump" if phase == "exit" else "damper_rebound"
    delta = +1.0 if phase == "exit" else +2.0
    name = f"{wheel}_oscillation_{phase}"
    pretty_phase = {"entry": "entry", "apex": "apex", "exit": "exit"}[phase]
    desc = (
        f"Wheel {wheel} shows sustained high-frequency vertical-load "
        f"oscillation at corner {pretty_phase} → stiffen "
        f"{'bump' if kind == 'damper_bump' else 'rebound'} to "
        f"re-stabilize the local mode."
    )
    return TriggerRule(
        name=name,
        description=desc,
        operator_factory=Coherence,
        predicate=_osc_predicate(wheel, phase),
        action=PhysicalAction(
            kind=kind, target=wheel, delta=delta,
            units="clicks", rationale_id=name,
        ),
        tags=("damper", phase),
    )


TRIGGER_RULES: tuple[TriggerRule, ...] = (
    # === DAMPER: 4 wheels × 3 phases = 12 Coherence rules ============
    _osc_rule("FL", "entry"),
    _osc_rule("FL", "apex"),
    _osc_rule("FL", "exit"),
    _osc_rule("FR", "entry"),
    _osc_rule("FR", "apex"),
    _osc_rule("FR", "exit"),
    _osc_rule("RL", "entry"),
    _osc_rule("RL", "apex"),
    _osc_rule("RL", "exit"),
    _osc_rule("RR", "entry"),
    _osc_rule("RR", "apex"),
    _osc_rule("RR", "exit"),
    # === ARB BALANCE: 4 Dissonance rules =============================
    TriggerRule(
        name="apex_understeer",
        description=(
            "Front axle saturated relative to rear at apex → reduce "
            "rear ARB to restore yaw response (re-balance the front↔"
            "rear structural mismatch)."
        ),
        operator_factory=Dissonance,
        predicate=_balance_predicate("axle.front", "axle.rear", "apex"),
        action=PhysicalAction(
            kind="arb", target="rear", delta=-2.0,
            units="clicks", rationale_id="apex_understeer",
        ),
        tags=("arb", "balance"),
    ),
    TriggerRule(
        name="apex_oversteer",
        description=(
            "Rear axle saturated relative to front at apex → reduce "
            "front ARB to load the front and re-balance the structural "
            "mismatch."
        ),
        operator_factory=Dissonance,
        predicate=_balance_predicate("axle.rear", "axle.front", "apex"),
        action=PhysicalAction(
            kind="arb", target="front", delta=-2.0,
            units="clicks", rationale_id="apex_oversteer",
        ),
        tags=("arb", "balance"),
    ),
    TriggerRule(
        name="exit_understeer",
        description=(
            "Front axle saturated relative to rear on exit → stiffen "
            "front ARB to free the rear under traction."
        ),
        operator_factory=Dissonance,
        predicate=_balance_predicate("axle.front", "axle.rear", "exit"),
        action=PhysicalAction(
            kind="arb", target="front", delta=+2.0,
            units="clicks", rationale_id="exit_understeer",
        ),
        tags=("arb", "balance"),
    ),
    TriggerRule(
        name="exit_oversteer",
        description=(
            "Rear axle saturated relative to front on exit → stiffen "
            "rear ARB to share lateral load with the inside wheel."
        ),
        operator_factory=Dissonance,
        predicate=_balance_predicate("axle.rear", "axle.front", "exit"),
        action=PhysicalAction(
            kind="arb", target="rear", delta=+2.0,
            units="clicks", rationale_id="exit_oversteer",
        ),
        tags=("arb", "balance"),
    ),
    # === FRICTION (Reception): per-wheel μ-saturation, 4 rules =======
    TriggerRule(
        name="friction_saturation_FL",
        description=(
            "Sustained μ-use ≥ 0.85 on front-left → raise FL pressure "
            "to shrink contact patch and absorb the friction overload."
        ),
        operator_factory=Reception,
        predicate=_friction_predicate("FL"),
        action=PhysicalAction(
            kind="tyre_pressure", target="FL", delta=+1.0,
            units="kPa", rationale_id="friction_saturation_FL",
        ),
        tags=("tyre", "friction"),
    ),
    TriggerRule(
        name="friction_saturation_FR",
        description=(
            "Sustained μ-use ≥ 0.85 on front-right → raise FR pressure "
            "to shrink contact patch and absorb the friction overload."
        ),
        operator_factory=Reception,
        predicate=_friction_predicate("FR"),
        action=PhysicalAction(
            kind="tyre_pressure", target="FR", delta=+1.0,
            units="kPa", rationale_id="friction_saturation_FR",
        ),
        tags=("tyre", "friction"),
    ),
    TriggerRule(
        name="friction_saturation_RL",
        description=(
            "Sustained μ-use ≥ 0.85 on rear-left → raise RL pressure "
            "to shrink contact patch and absorb the friction overload."
        ),
        operator_factory=Reception,
        predicate=_friction_predicate("RL"),
        action=PhysicalAction(
            kind="tyre_pressure", target="RL", delta=+1.0,
            units="kPa", rationale_id="friction_saturation_RL",
        ),
        tags=("tyre", "friction"),
    ),
    TriggerRule(
        name="friction_saturation_RR",
        description=(
            "Sustained μ-use ≥ 0.85 on rear-right → raise RR pressure "
            "to shrink contact patch and absorb the friction overload."
        ),
        operator_factory=Reception,
        predicate=_friction_predicate("RR"),
        action=PhysicalAction(
            kind="tyre_pressure", target="RR", delta=+1.0,
            units="kPa", rationale_id="friction_saturation_RR",
        ),
        tags=("tyre", "friction"),
    ),
    # === LATERAL COUPLING: 2 Coupling rules ==========================
    TriggerRule(
        name="lateral_imbalance_front",
        description=(
            "Cross-axle structural mismatch on front → couple via ARB "
            "to redistribute load and converge structural state."
        ),
        operator_factory=Coupling,
        predicate=_lateral_imbalance_predicate("front"),
        action=PhysicalAction(
            kind="arb", target="front", delta=+1.0,
            units="clicks", rationale_id="lateral_imbalance_front",
        ),
        tags=("arb", "coupling"),
    ),
    TriggerRule(
        name="lateral_imbalance_rear",
        description=(
            "Cross-axle structural mismatch on rear → couple via ARB "
            "to redistribute load and converge structural state."
        ),
        operator_factory=Coupling,
        predicate=_lateral_imbalance_predicate("rear"),
        action=PhysicalAction(
            kind="arb", target="rear", delta=+1.0,
            units="clicks", rationale_id="lateral_imbalance_rear",
        ),
        tags=("arb", "coupling"),
    ),
    # === THERMAL: per-axle, 2 rules (let the channel rest) ===========
    TriggerRule(
        name="thermal_axle_overload_front",
        description=(
            "Mean front-axle tyre temperature ≥ 130 °C → drop pressure "
            "to enlarge contact and let the axle rest."
        ),
        operator_factory=Silence,
        predicate=_thermal_predicate("front"),
        action=PhysicalAction(
            kind="tyre_pressure", target="front", delta=-2.0,
            units="kPa", rationale_id="thermal_axle_overload_front",
        ),
        tags=("tyre", "thermal"),
    ),
    TriggerRule(
        name="thermal_axle_overload_rear",
        description=(
            "Mean rear-axle tyre temperature ≥ 130 °C → drop pressure "
            "to enlarge contact and let the axle rest."
        ),
        operator_factory=Silence,
        predicate=_thermal_predicate("rear"),
        action=PhysicalAction(
            kind="tyre_pressure", target="rear", delta=-2.0,
            units="kPa", rationale_id="thermal_axle_overload_rear",
        ),
        tags=("tyre", "thermal"),
    ),
    # === BRAKES: 1 bias retune + 2 force-relief rules ================
    TriggerRule(
        name="brake_bias_drift",
        description=(
            "Realized brake bias diverges from setup target by ≥ 4 % "
            "→ nudge bias back toward target."
        ),
        operator_factory=Coherence,
        predicate=_brake_bias_predicate(),
        action=PhysicalAction(
            kind="brake_bias", target="global", delta=+0.5,
            units="%", rationale_id="brake_bias_drift",
        ),
        tags=("brake",),
    ),
    TriggerRule(
        name="brake_force_concentration_front",
        description=(
            "Front brake-force RMS ≥ 1.3× rear → reduce front bias to "
            "shed the concentrated load."
        ),
        operator_factory=Contraction,
        predicate=_brake_force_concentration_predicate("front"),
        action=PhysicalAction(
            kind="brake_bias", target="global", delta=-1.0,
            units="%", rationale_id="brake_force_concentration_front",
        ),
        tags=("brake", "contraction"),
    ),
    TriggerRule(
        name="brake_force_concentration_rear",
        description=(
            "Rear brake-force RMS ≥ 1.3× front → increase front bias "
            "to relieve the rear."
        ),
        operator_factory=Contraction,
        predicate=_brake_force_concentration_predicate("rear"),
        action=PhysicalAction(
            kind="brake_bias", target="global", delta=+1.0,
            units="%", rationale_id="brake_force_concentration_rear",
        ),
        tags=("brake", "contraction"),
    ),
    # === SPRINGS: 2 rest-state retunes ===============================
    TriggerRule(
        name="chassis_load_saturation_front",
        description=(
            "Front axle holds high structural load with a slow "
            "internal rate → suspension is swallowing load without "
            "oscillating; stiffen front springs to retune the rest "
            "set-point."
        ),
        operator_factory=Coherence,
        predicate=_chassis_load_predicate("front"),
        action=PhysicalAction(
            kind="spring", target="front", delta=+5.0,
            units="N/mm", rationale_id="chassis_load_saturation_front",
        ),
        tags=("spring",),
    ),
    TriggerRule(
        name="chassis_load_saturation_rear",
        description=(
            "Rear axle holds high structural load with a slow "
            "internal rate → suspension is swallowing load without "
            "oscillating; stiffen rear springs to retune the rest "
            "set-point."
        ),
        operator_factory=Coherence,
        predicate=_chassis_load_predicate("rear"),
        action=PhysicalAction(
            kind="spring", target="rear", delta=+5.0,
            units="N/mm", rationale_id="chassis_load_saturation_rear",
        ),
        tags=("spring",),
    ),
    # === COUPLED AXLE↔CHASSIS MODES: 2 detune rules ==================
    TriggerRule(
        name="axle_chassis_resonance_front",
        description=(
            "Front axle and chassis share a dominant internal rate → "
            "energy is coupling between them; stiffen front bump "
            "damping to detune."
        ),
        operator_factory=Coupling,
        predicate=_resonance_predicate("front"),
        action=PhysicalAction(
            kind="damper_bump", target="front", delta=+1.0,
            units="clicks", rationale_id="axle_chassis_resonance_front",
        ),
        tags=("damper", "coupling"),
    ),
    TriggerRule(
        name="axle_chassis_resonance_rear",
        description=(
            "Rear axle and chassis share a dominant internal rate → "
            "energy is coupling between them; stiffen rear bump "
            "damping to detune."
        ),
        operator_factory=Coupling,
        predicate=_resonance_predicate("rear"),
        action=PhysicalAction(
            kind="damper_bump", target="rear", delta=+1.0,
            units="clicks", rationale_id="axle_chassis_resonance_rear",
        ),
        tags=("damper", "coupling"),
    ),
    # === GEOMETRY: camber + toe rest-state retunes, 4 rules ==========
    TriggerRule(
        name="camber_misalignment_front",
        description=(
            "Left↔right front tyre temperature gap ≥ 12 °C → camber "
            "off target; retune front camber."
        ),
        operator_factory=Coherence,
        predicate=_thermal_asymmetry_predicate("front"),
        action=PhysicalAction(
            kind="camber", target="front", delta=-0.2,
            units="deg", rationale_id="camber_misalignment_front",
        ),
        tags=("camber", "geometry"),
    ),
    TriggerRule(
        name="camber_misalignment_rear",
        description=(
            "Left↔right rear tyre temperature gap ≥ 12 °C → camber "
            "off target; retune rear camber."
        ),
        operator_factory=Coherence,
        predicate=_thermal_asymmetry_predicate("rear"),
        action=PhysicalAction(
            kind="camber", target="rear", delta=-0.2,
            units="deg", rationale_id="camber_misalignment_rear",
        ),
        tags=("camber", "geometry"),
    ),
    TriggerRule(
        name="toe_misalignment_front",
        description=(
            "Front-axle left↔right slip-fraction RMS ≥ 0.15 → toe "
            "drift consistent with scrub; trim front toe-out."
        ),
        operator_factory=Coherence,
        predicate=_slip_diff_predicate("front"),
        action=PhysicalAction(
            kind="toe", target="front", delta=-0.05,
            units="deg", rationale_id="toe_misalignment_front",
        ),
        tags=("toe", "geometry"),
    ),
    TriggerRule(
        name="toe_misalignment_rear",
        description=(
            "Rear-axle left↔right slip-fraction RMS ≥ 0.15 → toe "
            "drift consistent with scrub; add a touch of rear toe-in."
        ),
        operator_factory=Coherence,
        predicate=_slip_diff_predicate("rear"),
        action=PhysicalAction(
            kind="toe", target="rear", delta=+0.05,
            units="deg", rationale_id="toe_misalignment_rear",
        ),
        tags=("toe", "geometry"),
    ),
    # === RIDE HEIGHT: phase-boundary bridges, 2 rules ================
    TriggerRule(
        name="ride_height_drift_braking_to_apex",
        description=(
            "Mean corner load drops sharply from entry to apex → the "
            "structural state collapses across the braking-to-apex "
            "transition; lower front ride height to bridge the phase "
            "boundary."
        ),
        operator_factory=Coherence,
        predicate=_corner_drift_predicate("entry", "apex"),
        action=PhysicalAction(
            kind="ride_height", target="front", delta=-2.0,
            units="mm", rationale_id="ride_height_drift_braking_to_apex",
        ),
        tags=("ride_height",),
    ),
    TriggerRule(
        name="ride_height_drift_apex_to_exit",
        description=(
            "Mean corner load drops sharply from apex to exit → the "
            "structural state is lost under traction; lower rear ride "
            "height to bridge the apex-to-exit transition."
        ),
        operator_factory=Coherence,
        predicate=_corner_drift_predicate("apex", "exit"),
        action=PhysicalAction(
            kind="ride_height", target="rear", delta=-2.0,
            units="mm", rationale_id="ride_height_drift_apex_to_exit",
        ),
        tags=("ride_height",),
    ),
)

# Sanity: rule names must be unique (the advisor uses them as IDs).
assert len({r.name for r in TRIGGER_RULES}) == len(TRIGGER_RULES), (
    "TRIGGER_RULES contains duplicate rule names"
)
# Sanity: rationale_id must also be unique because rule_learning.py keys
# its empirical outcome tables on it, not on the rule name. Silent
# duplicates here would merge unrelated outcomes during re-ranking.
assert len({r.action.rationale_id for r in TRIGGER_RULES}) == len(TRIGGER_RULES), (
    "TRIGGER_RULES contains duplicate action.rationale_id values"
)
# Sanity: every PhysicalAction must target a kind the sensitivities
# narrator knows about, otherwise the advisor will silently emit
# low-confidence generic sentences for that action.
_KNOWN_ACTION_KINDS = frozenset({
    "damper_rebound", "damper_bump", "spring", "arb",
    "tyre_pressure", "brake_bias", "camber", "toe", "ride_height",
})
assert all(r.action.kind in _KNOWN_ACTION_KINDS for r in TRIGGER_RULES), (
    "TRIGGER_RULES contains an unknown PhysicalAction.kind; update "
    "sensitivities._KIND_DISPATCH first"
)


# -- evaluation ---------------------------------------------------------


def evaluate_triggers(
    metrics: NodeMetrics, rules: tuple[TriggerRule, ...] = TRIGGER_RULES,
) -> tuple[TriggerRule, ...]:
    """Return rules whose predicate fires under ``metrics``.

    Order is preserved — callers can rely on it for determinism.
    """
    fired: list[TriggerRule] = []
    for rule in rules:
        try:
            if rule.predicate(metrics):
                fired.append(rule)
        except Exception:
            # Predicates must be NaN-safe; a raising predicate is a
            # programming error and is treated as "did not fire".
            continue
    return tuple(fired)


__all__ = (
    "PhysicalAction",
    "TriggerRule",
    "TRIGGER_RULES",
    "ConsolidatedAdjustment",
    "SetupSynthesis",
    "agg_corner_phase",
    "evaluate_triggers",
    "extract_node_metrics",
    "synthesize_setup",
    "synthesize_actions",
)


# -- setup synthesis ---------------------------------------------------
#
# The advisor fires N rules → N atomic PhysicalActions. The driver
# wants ONE coherent setup, not N independent suggestions. Synthesis
# aggregates every action that targets the *same* (kind, target) into
# a single net delta and tracks which rules contributed.
#
# This is the structural-closure step of the nodal cycle: the advisor
# has read the network's ΔNFR field through TRIGGER_RULES; synthesis
# *re-emits* a single, coherent ΔNFR back to the setup (Coherence
# closure on the operator sequence).


@dataclass(frozen=True)
class ConsolidatedAdjustment:
    """One aggregated setup change.

    Attributes
    ----------
    kind, target, units
        Same semantics as :class:`PhysicalAction`.
    net_delta
        Signed sum of every contributing rule's delta.
    contributing_rules
        Tuple of ``TriggerRule.name`` whose actions were merged here.
    confidence
        ``len(contributing_rules)`` (more independent rules pointing
        the same way → higher confidence). Capped at 5 for display.
    """

    kind: str
    target: str
    net_delta: float
    units: str
    contributing_rules: tuple[str, ...]
    confidence: int


@dataclass(frozen=True)
class SetupSynthesis:
    """Final consolidated setup proposal.

    Attributes
    ----------
    adjustments
        Tuple of :class:`ConsolidatedAdjustment`, sorted by absolute
        ``net_delta`` (largest first), then by ``kind``/``target``.
    fired_rules
        Names of every rule that fired (in order).
    conflict_groups
        Tuples of ``(kind, target, rules_pushing_up, rules_pushing_down)``
        for any (kind, target) pair where deltas have opposite signs.
        These are surfaced explicitly: the structural field is
        ambiguous and the driver should treat the net delta as a
        weak signal.
    """

    adjustments: tuple[ConsolidatedAdjustment, ...]
    fired_rules: tuple[str, ...]
    conflict_groups: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...]


def synthesize_setup(
    fired: tuple[TriggerRule, ...],
) -> SetupSynthesis:
    """Collapse N fired rules → 1 coherent setup proposal.

    See :func:`synthesize_actions` for the aggregation rule. This is a
    thin wrapper that extracts each rule's :class:`PhysicalAction`
    while preserving ``rationale_id`` as the contributing-rule key.
    """
    return synthesize_actions(
        tuple(r.action for r in fired),
        rule_names=tuple(r.name for r in fired),
    )


def synthesize_actions(
    actions: tuple[PhysicalAction, ...],
    rule_names: tuple[str, ...] | None = None,
) -> SetupSynthesis:
    """Aggregate atomic :class:`PhysicalAction` deltas into one setup.

    Actions sharing ``(kind, target)`` are merged into a single
    :class:`ConsolidatedAdjustment`; deltas are summed algebraically
    so opposing actions cancel and reinforcing ones stack. Conflicts
    (mixed-sign deltas on the same channel) are surfaced separately.

    Closure rationale: this is the operator-sequence projection back
    to the physical state space — the discrete analog of integrating
    ∂EPI/∂t over the stint window. Each action is one Δ-impulse on
    its channel; the net delta is the structural response the setup
    must absorb to re-stabilize the network.
    """
    if rule_names is None:
        rule_names = tuple(a.rationale_id for a in actions)
    if len(rule_names) != len(actions):
        raise ValueError("rule_names length must match actions length")

    groups: dict[tuple[str, str, str], list[tuple[float, str]]] = {}
    for act, name in zip(actions, rule_names):
        key = (act.kind, act.target, act.units)
        groups.setdefault(key, []).append((act.delta, name))

    adjustments: list[ConsolidatedAdjustment] = []
    conflicts: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = []
    for (kind, target, units), entries in groups.items():
        net = sum(d for d, _ in entries)
        names = tuple(n for _, n in entries)
        conf = min(5, len(entries))
        adjustments.append(
            ConsolidatedAdjustment(
                kind=kind, target=target, net_delta=float(net),
                units=units, contributing_rules=names, confidence=conf,
            )
        )
        ups = tuple(n for d, n in entries if d > 0)
        downs = tuple(n for d, n in entries if d < 0)
        if ups and downs:
            conflicts.append((kind, target, ups, downs))

    adjustments.sort(key=lambda c: (-abs(c.net_delta), c.kind, c.target))
    return SetupSynthesis(
        adjustments=tuple(adjustments),
        fired_rules=tuple(rule_names),
        conflict_groups=tuple(conflicts),
    )
