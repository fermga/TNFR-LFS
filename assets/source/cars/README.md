# Drop LFS `<CAR>_CAR_info.bin` exports here

This directory is the default search path that
`telemetry.observables.load_car_info_bin_for` uses when resolving a
baseline car export (e.g. for `discover_car_specs` /
`CarSpecStore.merge_from_car_info`). The directory is honoured by the
core telemetry layer and by any downstream tooling that loads car
baselines — the shipped Studio bundle no longer ships a Setup tab UI
(it remains in `src/lfs_telemetry/studio/widgets/setup_tab.py` for
developer use but is excluded from the frozen build).

## How to generate one

1. Launch LFS in Programmer Mode:

       LFS.exe /prog

2. Drive the car you want to baseline (e.g. **FBM**) into Single Player.
3. Choose **Save CAR_info.bin** from the programmer menu. LFS writes the
   file to `LFS/data/`.
4. Copy the file here with the right name, e.g. `FBM_CAR_info.bin`
   (uppercase short name + `_CAR_info.bin`).

You can override this directory entirely by setting the
`LFS_TELEMETRY_CAR_INFO_DIR` environment variable.

## Mod cars

LFS does not currently export `CAR_info.bin` for mod cars. For mod-car
detection / radar dimensions see `../mods/mod_sizes.json` instead.
