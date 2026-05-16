# Reconnaissance: `C:\LFS\data\wld\*` (SRWORL / SRLGHT)

Snapshot date: end of session after `.knw` integration.

## Inventory

| file                | magic   | size    | notes                                |
|---------------------|---------|---------|--------------------------------------|
| `BLACKWOOD.wld`     | SRWORL  |   8 MB  | static world geometry (only one)     |
| `BLACKWOOD.lgh`     | SRLGHT  |  78 MB  | lightmap bake                        |
| `ASTON.lgh`         | SRLGHT  | 147 MB  | lightmap bake                        |
| `AUTOCROSS.lgh`     | SRLGHT  |  43 MB  | lightmap bake                        |
| `FERN BAY.lgh`      | SRLGHT  |  31 MB  | lightmap bake                        |
| `KYOTO RING.lgh`    | SRLGHT  | 354 MB  | lightmap bake                        |
| `Rockingham.lgh`    | SRLGHT  |  43 MB  | lightmap bake                        |
| `SOUTH CITY.lgh`    | SRLGHT  | 356 MB  | lightmap bake                        |
| `WESTHILL.lgh`      | SRLGHT  | 117 MB  | lightmap bake                        |
| `*.lok` (8 files)   | —       | varies  | **encrypted, discard**               |

## SRWORL header (`BLACKWOOD.wld`)

```
00  53 52 57 4f 52 4c 00 fc 17 04 01 01 00 00 00 00   SRWORL...........
10  f1 40 2e c0 32 9b 4d 42 68 b3 0a 3e 47 50 20 54   .@..2.MBh..>GP T
20  52 41 43 4b 00 00 00 00 ...                       RACK....
3c  0d 00 00 00 09 00 00 00 92 01 92 00 00 00 00 00   <-- TOC begins
4c  6c 01 00 00 5e 00 00 00 fe 00 00 00 ff ff ff ff   <-- TOC ends with -1
```

* **Magic**: `SRWORL\0\xfc` (8 B) + build/version `17 04 01 01` (4 B) + 4 B pad.
* **Header floats** (offset 0x10..0x1b): `-2.7227, 51.40, 0.1354` —
  meaning unknown (origin?  scale?  bbox center?).
* **Track name** (offset 0x1c..0x3b, 32 B C-string): `"GP TRACK"`.
* **TOC** (offset 0x3c..0x57): 8 little-endian u32/i32 values:
  `13, 9, 0x00920192, 0, 364, 94, 254, -1`.  The value `0x0192 = 402`
  matches the count of the first XYZ chunk discovered below.
* **Body** starts around offset 0x5c.

## SRWORL body — chunked, NOT a uniform record array

Heuristic sweep with the recon tool found 5 runs of consecutive f32 XYZ
triplets that fit inside a generous BLACKWOOD bbox:

| chunk offset | XYZ triplets | last sample XYZ          |
|--------------|--------------|--------------------------|
| 0x0000005c   | 401          | (0, 0, NaN)               |
| 0x0000560c   | 182          | (-5, +5, ...)             |
| 0x0000b834   | 698          | (0, 0, NaN)               |
| 0x00043d40   | 794          | (+3172, ..., +13.95)      |
| 0x00048638   | 376          | (0, ...)                  |

* The very first chunk (offset 0x5c, 401 triplets) is exactly the count
  hinted at by the TOC entry `0x0192 = 402` (likely 1-header + 401-data).
* Long runs are surrounded by tag/length fields (chunk headers).
* Pretending the file is one big 36-byte-record array gives ~199k of
  238k records with “plausible” XYZ purely by accident — so the format
  is **chunked** with multiple record types, not a flat vertex pool.

XYZ value bbox across all 36-B aligned candidates (overestimate; many
are random alignments):

```
X ∈ [ -600, +2564]   (BLACKWOOD GP loop is ≈ ±600 m)
Y ∈ [ -600,  +977]
Z ∈ [ -172,  +463]
```

The X/Y range is consistent with the real BLACKWOOD GP coordinate
frame; Z reaches +463 m so this includes scenery (hills, sky meshes),
not just the track surface.

## SRLGHT files (all eight)

```
SRLGHT\0\xfc <4 B build> 00 00 72 fe 00 00 b9 00 a8 aa aa 06 ...
```

All eight share the same header signature.  The size scales roughly with
track area × textured-surface complexity (FERN BAY = 31 MB, SOUTH CITY
= 356 MB).  Combined with the magic (`SR LIGHT`), this is the
**pre-baked lightmap dataset** used by the renderer.  It is essentially
texels + UV indices — irrelevant to telemetry / surface geometry.

## Verdict

* **SRWORL** is the only target with potential telemetry value, BUT:
  * Only one file exists (BLACKWOOD GP) – not generalisable.
  * Format is chunked with no public spec; reverse-engineering the TOC
    entries would take several focused sessions.
  * `.knw` already gives us the canonical racing line for ~49 layouts,
    including BLACKWOOD.
  * Banking / surface elevation could be reconstructed approximately
    from the PTH centre-line gradient + the `.knw` lateral offsets.
* **SRLGHT** is a renderer asset, not a telemetry source.  Skip.

## Recommended next moves

In decreasing value/effort order:

1. **Stop here on `wld/`** unless we specifically need BLACKWOOD surface
   reconstruction.  Document and move on.
2. Consider new telemetry surfaces from sources we have NOT yet
   exhausted:
   * `setup/*.set` – car setups (springs / dampers / gear ratios).
   * `replays/*.spr|.mpr` – full timeline replays with inputs.
   * `script/*.lfs` – LFS scripted layouts.
   * `cars/*.car` – car geometry / mass / aero.
3. Only revisit SRWORL if a specific BLACKWOOD-only validation case
   demands it (e.g. validating slope-corrected accel against the real
   surface normal).
