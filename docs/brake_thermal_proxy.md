# Brake thermal proxy

The brake thermal proxy keeps fade and ventilation analytics alive when Live for Speed does not expose real brake temperatures. Native readings only appear when the simulator broadcasts the extended OutGauge payload (`OG_EXT_BRAKE_TEMP`); otherwise the stream publishes `0 °C` markers that the fusion layer treats as missing data.【F:tnfr_lfs/acquisition/fusion.py†L248-L321】【F:tnfr_lfs/acquisition/fusion.py†L1064-L1126】 The proxy integrates brake work on every wheel and applies configurable cooling so downstream modules continue to operate with finite series.

## Operating model

1. **Energy injection** – Each tick integrates the mechanical work `m·a·v` attributed to the active wheel and scales it by the instantaneous normal load, adding the result to the disc/caliper temperature estimate.【F:tnfr_lfs/analysis/brake_thermal.py†L9-L122】
2. **Convective dissipation** – Cooling scales with the square root of vehicle speed and two coefficients (still air and convective) that approximate forced ventilation; the estimator clamps the output between the configured ambient temperature and 1200 °C to avoid runaway values.【F:tnfr_lfs/analysis/brake_thermal.py†L27-L122】
3. **Seeding with telemetry** – Whenever OutGauge delivers plausible brake temperatures the estimator observes them, anchoring the model to the real signal before continuing its integration.【F:tnfr_lfs/acquisition/fusion.py†L1078-L1107】

The estimator maintains an independent state per wheel so asymmetries in load or pedal application propagate naturally to the four temperatures.【F:tnfr_lfs/analysis/brake_thermal.py†L65-L122】

## Modes and overrides

The proxy honours the `mode` configured in the active pack (`[thermal.brakes]`) and can be overridden per session with the `TNFR_LFS_BRAKE_THERMAL` environment variable.【F:tnfr_lfs/acquisition/fusion.py†L616-L733】 The following modes are available:

- **`auto`** (default) – Consume OutGauge temperatures when available and fall back to the estimator when the broadcaster sends `0 °C` markers or omits the block.【F:tnfr_lfs/acquisition/fusion.py†L1078-L1107】
- **`off`** – Skip estimation and forward the raw OutGauge samples (or their last valid values) so engineers can validate hardware sensors without synthetic blending.【F:tnfr_lfs/acquisition/fusion.py†L1107-L1117】
- **`force`** – Ignore OutGauge altogether and rely exclusively on the estimator, useful for legacy cars without brake telemetry or for simulated runs that only provide OutSim loads.【F:tnfr_lfs/acquisition/fusion.py†L1117-L1126】

## Assumptions and limitations

- Brake power is derived from OutSim longitudinal acceleration and wheel loads; low OutSim sampling rates (< 10 Hz) may under-represent short spikes compared to high-frequency loggers.【F:tnfr_lfs/analysis/brake_thermal.py†L27-L122】
- The estimator assumes OutSim and OutGauge streams are time-aligned within the usual Live for Speed tolerance (one or two packets); it does not attempt to compensate for network-induced lag.【F:tnfr_lfs/acquisition/fusion.py†L248-L321】
- When OutGauge publishes finite but noisy temperatures the proxy preserves them instead of smoothing, preventing divergence from the simulator’s ground truth.【F:tnfr_lfs/acquisition/fusion.py†L1078-L1107】

## Configurable parameters

Global defaults live under `[thermal.brakes]` in `config/global.toml` and can be overridden per car via `data/cars/*.toml`.【F:config/global.toml†L13-L20】

- `ambient` – Lower clamp for the temperature integration (°C).
- `heat_capacity` – Effective thermal capacity of the caliper/disc assembly (J/°C).
- `heating_efficiency` – Fraction of mechanical work that becomes heat.
- `convective_coefficient` – Scaling for cooling at speed.
- `still_air_coefficient` – Baseline cooling when the car is stationary.
- `minimum_brake_input` – Pedal threshold below which the estimator stops injecting energy.
- `mode` – Default operating mode (`auto`, `off`, `force`).

Adjust these parameters to match each car’s hardware or test rigs; the CLI and HUD will automatically apply the active profile when processing telemetry.
