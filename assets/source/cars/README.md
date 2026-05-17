# Drop LFS `<CAR>_CAR_info.bin` exports here

This directory is the default search path the Studio Setup tabs and the
TNFR Setup Advisor use when resolving a baseline car export.

## How to generate one

1. Launch LFS in Programmer Mode:

       LFS.exe /prog

2. Drive the car you want to baseline (e.g. **FBM**) into Single Player.
3. Choose **Save CAR_info.bin** from the programmer menu. LFS writes the
   file to `LFS/data/`.
4. Either:
   * Click **Import CAR_info.bin…** in the Studio Setup tab and select
     the exported file (it is copied here automatically), or
   * Copy the file here manually with the right name, e.g.
     `FBM_CAR_info.bin`.

You can override this directory entirely by setting the
`LFS_TELEMETRY_CAR_INFO_DIR` environment variable.

## Mod cars

LFS does not currently export `CAR_info.bin` for mod cars. For mod-car
detection / radar dimensions see `../mods/mod_sizes.json` instead.
