# `tnfr_lfs.exporters.lfs_native` module
Future Live for Speed native exporter utilities.

## Classes
### `NativeSetupVector`
Lightweight container for serialised setup changes.

This mirrors the information required to convert a :class:`SetupPlan`
into the binary/text representation consumed directly by Live for Speed.

## Functions
- `build_native_vector(plan: SetupPlan) -> NativeSetupVector`
  - Extract the decision vector from a setup plan.
- `encode_native_setup(plan: SetupPlan) -> bytes`
  - Encode a setup plan into the native LFS payload.

The implementation is guarded behind ``TNFR_LFS_NATIVE_EXPORT`` to avoid
surprising side effects while the mapping logic is still a work in
progress.

## Attributes
- `FEATURE_FLAG_LFS_NATIVE_EXPORT = bool(os.getenv('TNFR_LFS_NATIVE_EXPORT'))`

