# TNFR Setup Equivalence Guide

This guide maps traditional setup adjustments to the key metrics used by
the TNFR × LFS framework. Each table is organised by subsystem and
explains how to interpret the `∇NFR⊥`, `ν_f`, and `C(t)` indicators within the
recommendation flow and the live HUD. All TNFR readings come from the
Live for Speed OutSim/OutGauge telemetry, so both broadcasters must be
enabled and `OutSim Opts ff` configured to transmit the extended wheel packet
(Fz loads, forces, and deflections) required by the telemetry fusion
module.【F:tnfr_lfs/ingestion/fusion.py†L200-L284】 Enable the extended
OutGauge payload (`OutGauge Opts …`) as well so the pipeline includes the
layer temperatures and actual pressures: enable at least the
`OG_EXT_TYRE_TEMP`, `OG_EXT_TYRE_PRESS`, and `OG_EXT_BRAKE_TEMP` flags to have
the broadcaster send the 20 floats in the extended block (inner/middle/outer
profiles, pressures, and caliper temperatures). Otherwise TNFR × LFS keeps the
last sample or reports “no data”.【F:tnfr_lfs/ingestion/fusion.py†L594-L657】 The
global `slip_ratio` and `slip_angle` metrics are derived directly from the
load-weighted average of the per-wheel channels within the OutSim packet and
fall back to the kinematic estimate only when the signal arrives incomplete.【F:tnfr_lfs/ingestion/fusion.py†L111-L175】【F:tnfr_lfs/ingestion/fusion.py†L242-L291】
When the capture comes from CSV, the reader preserves missing columns as
`math.nan` so the indicators render as “no data” instead of fabricating
values.【F:tnfr_lfs/ingestion/outsim_client.py†L87-L155】

> **Note on ∇NFR:** the `∇NFR∥` and `∇NFR⊥` metrics represent the projection of
> the ΔNFR nodal gradient onto the longitudinal and lateral axes. They
> complement balance analysis but never replace the actual load channels.
> Whenever an adjustment depends on absolute vertical force, compare the nodal
> reading with the extended OutSim `Fz`/`ΔFz` records.

## Required signals per metric

- **ΔNFR (nodal gradient) / ∇NFR∥/∇NFR⊥ (projections)** – integrate the Fz loads
  (and their ΔFz variations), wheel forces, and deflections reported by OutSim
  together with the engine regime, pedal inputs, and ABS/TC lights provided by
  OutGauge to evaluate the longitudinal/lateral gradient distribution. Use the
  `Fz`/`ΔFz` channels directly whenever you need absolute forces before adjusting
  the setup.【F:tnfr_lfs/ingestion/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】
- **ν_f (natural frequency)** – leverages the Fz load distribution,
  `slip_ratio`/`slip_angle`, speed, and `yaw_rate` emitted by OutSim combined with
  the driving-style signals (`throttle`, `gear`) from OutGauge to classify nodes
  and target bands.【F:tnfr_lfs/ingestion/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L648-L710】 The `slip_ratio` and `slip_angle`
  used here correspond to the four-wheel average transmitted by OutSim, ensuring
  the frequency labels react to the actual axle grip and only falling back to the
  kinematic estimate when no valid readings are available.【F:tnfr_lfs/ingestion/fusion.py†L111-L175】
- **C(t) (structural coherence)** – built from the nodal ΔNFR distribution, the
  `mu_eff_*` coefficients derived from OutSim accelerations, and the ABS/TC
  activity reported by OutGauge.【F:tnfr_lfs/ingestion/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】【F:tnfr_lfs/core/coherence.py†L65-L125】
- **Ackermann budgets / Slide Catch** – depend exclusively on the `slip_angle_*`
  channels and the `yaw_rate` that OutSim transmits for each wheel; if the source
  omits those fields TNFR × LFS will display `"no data"` for the related
  readings.
- **Aero balance drift (`aero_balance_drift`)** – summarises rake from the OutSim
  `pitch` signal and axle suspension travel together with the `μ_front - μ_rear`
  contrast without blending synthetic inputs.
- **Tyre temperatures/pressures** – reuse the OutGauge readings whenever the
  samples are positive and finite; if the extended block is disabled the pipeline
  keeps the previous reading or `"no data"`, avoiding artificial extrapolations.
- **CPHI (Contact Patch Health Index)** – requires `slip_ratio`, `slip_angle`,
  lateral/longitudinal forces, and wheel loads from the extended OutSim packet.
  Without those signals TNFR × LFS labels the patch health as `"no data"`, so the
  planner will not apply tyre-balance adjustments. The reports and the HUD share
  red/amber/green thresholds (0.62/0.78/0.90) to flag when intervention is needed
  or when the patch is ready for attack stints.

## Aerodynamic subsystem

| Adjustment | TNFR metric | Reading | Application |
| --- | --- | --- | --- |
| Front wing | `∇NFR⊥` | Positive lateral deltas point to an overload on the front axle during the middle phase. | Reduce angle or front load when `C(t)` remains high (>0.75) to restore balance without sacrificing coherence and validate via the `Fz_front`/`Fz_rear` trends that the distribution follows the adjustment. |
| Rear wing | `∇NFR⊥` | Persistent negative lateral deltas in traction microsectors indicate insufficient rear support. | Add one or two clicks of wing when `ν_f` drops below the optimal band and the HUD reports "Δaero ↘", confirming that `Fz_rear` regains the target load after the change. |
| Gurney flap | `C(t)` | Declining structural coherence together with oversteer recommendations suggests induced turbulence. | Swap the flap for a lower-load variant until `C(t)` returns to the 0.70–0.85 band while `ν_f` settles back on the "optimal" label. |

**CLI example**

```bash
$ tnfr_lfs suggest stint.jsonl --car-model FZR --track AS5 \
    | jq '.recommendations[] | select(.parameter=="rear_wing_angle")'
{
  "parameter": "rear_wing_angle",
  "metric": "∇NFR⊥",
  "value": -0.18,
  "rationale": "Rear ∇NFR⊥ beyond threshold in traction",
  "expected_effect": "Increase rear load"
}
```

Cross the output with the table above: a negative traction-side `∇NFR⊥`
matches the **Rear wing** row, so increase rear load until the HUD shows
`ν_f` in green and `C(t)` > 0.75.

## Suspension subsystem

| Adjustment | TNFR metric | Reading | Application |
| --- | --- | --- | --- |
| Spring stiffness | `ν_f` | A natural frequency outside the target band triggers "very low"/"very high" alerts. | Increase or decrease `spring_rate` in small steps until `ν_f` returns to the "optimal" label and `C(t)` stays stable. |
| Damping (rebound) | `C(t)` | Decoherence spikes after quick transitions indicate the damper is not following the TNFR profile. | Adjust rebound to smooth the response until `C(t)` starts rising again after each transition. |
| Anti-roll bars | `∇NFR⊥` | Lateral imbalance between axles grows when the bars act asymmetrically in medium corners. | Rebalance the front/rear bar based on the sign of `∇NFR⊥`; matching `ν_f` between axles prevents dissonance. |

**HUD/OSD example**

```text
Page A  ΔNFR [--^--]  ν_f: very low  C(t): 0.62
Recommendation: "Rear stiffness +1 → recover ν_f band"
```

When the HUD shows a "very low" `ν_f` label and `C(t)` drops below 0.65,
check the **Spring stiffness** row: increase stiffness until the badge flips
to "optimal" and the coherence bar (`C(t) ███░`) grows in parallel.

## Tyre subsystem

| Adjustment | TNFR metric | Reading | Application |
| --- | --- | --- | --- |
| Hot pressure | `∇NFR⊥` | Irregular lateral deltas between laps reveal pressure differences. | Adjust pressures to smooth `∇NFR⊥` and stabilise `ν_f` around the profile’s nominal frequency, confirming with per-wheel `Fz` traces that loads converge after the change. |
| Camber | `C(t)` | Declining coherence during load phases indicates the contact patch is not following the load. | Increase negative camber until `C(t)` aligns with the profile table reference (>0.78). |
| Toe | `ν_f`, `Φsync` | Rapid jumps in the `ν_f~` indicator after direction changes point to excessive toe; `Φsync` gaps on entry/exit trigger corrections (front toe-out when it drops, rear toe-in when it spikes). | Adjust toe to remove artefacts in the `ν_f~` signal and recover the "optimal" band; use the `Φsync` guidance to open front toe when entry sync is missing and close rear toe when exit sync increases. |

**CLI example**

```bash
$ tnfr_lfs analyze stint.jsonl --export json \
    | jq '.phase_messages[] | select(.metric=="∇NFR⊥" and .phase=="apex")'
  {
  "phase": "apex",
  "metric": "∇NFR⊥",
  "status": "high",
  "message": "Front pressures imbalanced"
}
```

The ``apex`` phase with a high `∇NFR⊥` maps to the **Hot pressure** row,
so equalise pressures until the metric returns to its nominal zone and the
alert disappears in the next ``tnfr_lfs analyze`` run.

## Brake and balance subsystem

| Adjustment | TNFR metric | Reading | Application |
| --- | --- | --- | --- |
| Brake bias | `∇NFR⊥` | Positive spikes under heavy braking indicate front overload. | Shift the bias rearward until the HUD or CLI reduces the reported peaks and confirm via the `Fz_front`/`ΔFz_front` traces that load redistributes as expected. |
| ABS/TC power | `C(t)` | An oscillating coherence index after ABS entries reveals aggressive intervention. | Soften ABS/TC to keep `C(t)` steady while `ν_f` stays within band. |
| Longitudinal heave | `ν_f` | A "very high" `ν_f` badge on straights suggests excessive stiffness under braking. | Soften front packers or bump stops to align the frequency with the target. |

**Combined example**

```bash
$ tnfr_lfs osd --host 127.0.0.1 --outsim-port 4123 \
    --outgauge-port 3000 --insim-port 29999
# During braking, page A shows:
∇NFR⊥: +0.24  ν_f: optimal  C(t): 0.81  → "High front bias"
```

The HUD links straight to the **Brake bias** row, signalling that the balance
must move rearward until `∇NFR⊥` stays below the contextual threshold and the
recommendation clears.

---

For deeper insight into how the `∇NFR⊥`, `ν_f`, and `C(t)` metrics are computed,
review the `tnfr_lfs.core` modules described in the [API reference](api_reference.md).
