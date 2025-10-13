# `tnfr_core.equations.constants` module
Centralised constant definitions shared across TNFR × LFS modules.

## Attributes
- `WHEEL_SUFFIXES = ('fl', 'fr', 'rl', 'rr')`
- `WHEEL_LABELS = MappingProxyType({'fl': 'FL', 'fr': 'FR', 'rl': 'RL', 'rr': 'RR'})`
- `TEMPERATURE_MEAN_KEYS = MappingProxyType({suffix: f'tyre_temp_{suffix}' for suffix in WHEEL_SUFFIXES})`
- `TEMPERATURE_STD_KEYS = MappingProxyType({suffix: f'{TEMPERATURE_MEAN_KEYS[suffix]}_std' for suffix in WHEEL_SUFFIXES})`
- `BRAKE_TEMPERATURE_MEAN_KEYS = MappingProxyType({suffix: f'brake_temp_{suffix}' for suffix in WHEEL_SUFFIXES})`
- `BRAKE_TEMPERATURE_STD_KEYS = MappingProxyType({suffix: f'{BRAKE_TEMPERATURE_MEAN_KEYS[suffix]}_std' for suffix in WHEEL_SUFFIXES})`
- `PRESSURE_MEAN_KEYS = MappingProxyType({suffix: f'tyre_pressure_{suffix}' for suffix in WHEEL_SUFFIXES})`
- `PRESSURE_STD_KEYS = MappingProxyType({suffix: f'{PRESSURE_MEAN_KEYS[suffix]}_std' for suffix in WHEEL_SUFFIXES})`

