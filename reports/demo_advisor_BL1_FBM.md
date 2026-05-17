# TNFR Setup Advisor — proposal — FBM @ BL1 (5-lap stint)

- Projected coherence: **+0.370** (1.000 -> 1.000)
- Stability constraints: pass
- Seed: `17` - baseline `69b5b6c690a77f5a` - stint `3c3918bece6dd445`

## Proposed changes

| # | Subsystem | Change | Why |
|---|-----------|--------|-----|
| 1 | Brake bias (% front) (whole car) | -1 % | brake force concentration front - Brake bias moves from 60.0% front to 59.0% front; under threshold braking the rear axle will start to limit deceleration sooner. |
| 2 | Spring rate (front axle) | +5 N/mm | chassis load saturation front - Ride frequency at the corner moves from 2.85 Hz to 3.01 Hz (asphalt target 1.5-2.5 Hz on a road car, 2.5-3.5 Hz on a stiff race car). |
| 3 | Spring rate (rear axle) | +5 N/mm | chassis load saturation rear - Ride frequency at the corner moves from 2.85 Hz to 3.01 Hz (asphalt target 1.5-2.5 Hz on a road car, 2.5-3.5 Hz on a stiff race car). |
| 4 | Camber (front axle) | -0.2 deg | camber misalignment front - Static camber at the corner moves from -2.01° to -2.21°; expect tyre temperatures to redistribute across the tread and peak lateral grip to shift inboard. |
| 5 | Camber (rear axle) | -0.2 deg | camber misalignment rear - Static camber at the corner moves from -2.01° to -2.21°; expect tyre temperatures to redistribute across the tread and peak lateral grip to shift inboard. |
| 6 | Toe-in (front axle) | -0.05 deg | toe misalignment front - Static toe-in at the corner moves from +0.00° to -0.05°; turn-in response sharpens with more toe-out and straight-line stability improves with more toe-in. |
| 7 | Toe-in (rear axle) | +0.05 deg | toe misalignment rear - Static toe-in at the corner moves from +0.00° to +0.05°; turn-in response sharpens with more toe-out and straight-line stability improves with more toe-in. |
| 8 | Ride height (rear axle) | -2 mm | ride height drift apex to exit - Ride height at the rear moves by -2.0 mm (lower the rear); static rake shifts 0.8 mrad nose-up, which moves aerodynamic balance and lowers the mechanical roll-centre at that axle. |

## Consolidated optimal setup

Aggregated net delta per channel — every individual recommendation above contributes to this single coherent setup proposal:

| Subsystem | Net change | Confidence | Driven by |
|-----------|-----------:|-----------:|-----------|
| Spring rate (front axle) | **+5 N/mm** | *---- | 1 signal |
| Spring rate (rear axle) | **+5 N/mm** | *---- | 1 signal |
| Ride height (rear axle) | **-2 mm** | *---- | 1 signal |
| Brake bias (% front) (whole car) | **-1 %** | *---- | 1 signal |
| Camber (front axle) | **-0.2 deg** | *---- | 1 signal |
| Camber (rear axle) | **-0.2 deg** | *---- | 1 signal |
| Toe-in (front axle) | **-0.05 deg** | *---- | 1 signal |
| Toe-in (rear axle) | **+0.05 deg** | *---- | 1 signal |

## Diagnostics

- Global structural coherence before changes. = 1.000
- Projected structural coherence after applying the proposed setup deltas (deterministic surrogate). = 1.000
- 8 of 8 fired rules retained after coherence-positive filter. = 8.000
