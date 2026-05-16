"""Parser for LFS PIN files in C:/LFS/data/smx/*.pin.

PIN ("Path INfo" / per-environment header) is a tiny 32-byte file shipped
once per environment (AS, AU, BL, FE, KY, LA, RO, SO, WE).  It declares
the number of base layout configurations the environment ships with and
the world bounding-box of that environment in metres.

Format (reverse-engineered, validated against the 9 LFS-shipped files):

  Bytes 0..5   : ASCII magic 'LFSPIN'
  Bytes 6..11  : reserved (always 0x00)
  Bytes 12..15 : uint32 little-endian — base layout count
  Bytes 16..19 : int32 Q16.16 — world X min (metres)
  Bytes 20..23 : int32 Q16.16 — world X max (metres)
  Bytes 24..27 : int32 Q16.16 — world Y min (metres)
  Bytes 28..31 : int32 Q16.16 — world Y max (metres)

Per-env layout counts confirmed against shipped *.pth inventory:
  AS=9, AU=4, BL=4, FE=6, KY=8, LA=2, RO=11, SO=7, WE=7   (sum = 58 base)
  Reverse layouts ("R" suffix) are derived in-engine and are not counted.

Coordinate system matches OutSim and PTH: X east, Y north, Z up.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Iterable

from .pth import DEFAULT_SMX_DIR, FIXED_POINT_DIVISOR

PIN_MAGIC = b"LFSPIN"
PIN_BYTES = 32


@dataclass(slots=True, frozen=True)
class PinInfo:
    """Per-environment header parsed from a PIN file."""
    env: str                # e.g. "BL"
    layout_count: int       # number of base layouts (reverses excluded)
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float

    @property
    def width_m(self) -> float:
        return self.x_max_m - self.x_min_m

    @property
    def height_m(self) -> float:
        return self.y_max_m - self.y_min_m

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max)."""
        return (self.x_min_m, self.y_min_m, self.x_max_m, self.y_max_m)

    def contains_xy(self, x: float, y: float, *, margin_m: float = 0.0) -> bool:
        """True if (x, y) lies within the world bbox extended by `margin_m`."""
        return (
            self.x_min_m - margin_m <= x <= self.x_max_m + margin_m
            and self.y_min_m - margin_m <= y <= self.y_max_m + margin_m
        )


def parse_pin(path: str | _Path) -> PinInfo:
    """Parse a PIN file from disk."""
    p = _Path(path)
    return parse_pin_bytes(p.read_bytes(), env=p.stem.upper())


def parse_pin_bytes(data: bytes, env: str = "") -> PinInfo:
    if len(data) != PIN_BYTES:
        raise ValueError(f"PIN must be exactly {PIN_BYTES} bytes (got {len(data)})")
    if data[:6] != PIN_MAGIC:
        raise ValueError(f"PIN bad magic: {data[:6]!r}")
    if any(b != 0 for b in data[6:12]):
        raise ValueError(f"PIN reserved bytes 6..11 not zero: {data[6:12]!r}")
    layout_count = struct.unpack_from("<I", data, 12)[0]
    xmin_i, xmax_i, ymin_i, ymax_i = struct.unpack_from("<4i", data, 16)
    return PinInfo(
        env=env,
        layout_count=int(layout_count),
        x_min_m=xmin_i / FIXED_POINT_DIVISOR,
        x_max_m=xmax_i / FIXED_POINT_DIVISOR,
        y_min_m=ymin_i / FIXED_POINT_DIVISOR,
        y_max_m=ymax_i / FIXED_POINT_DIVISOR,
    )


# ----------------------------------------------------------------------------
# Discovery helpers
# ----------------------------------------------------------------------------

def list_pin_files(smx_dir: _Path | str = DEFAULT_SMX_DIR) -> list[_Path]:
    """Return all *.pin files in the LFS smx directory, sorted."""
    return sorted(_Path(smx_dir).glob("*.pin"))


def load_all(smx_dir: _Path | str = DEFAULT_SMX_DIR) -> dict[str, PinInfo]:
    """Load every PIN and return {env: PinInfo}."""
    out: dict[str, PinInfo] = {}
    for f in list_pin_files(smx_dir):
        try:
            info = parse_pin(f)
            out[info.env] = info
        except Exception:  # noqa: BLE001
            continue
    return out


def find_env_for_xy(
    x: float,
    y: float,
    pins: Iterable[PinInfo] | None = None,
    *,
    smx_dir: _Path | str = DEFAULT_SMX_DIR,
    margin_m: float = 50.0,
) -> list[PinInfo]:
    """Return every PinInfo whose bbox contains (x, y) within `margin_m`.

    Multiple environments may share overlapping bboxes (e.g. LA and AU both
    cover origin-centred areas), so this returns all matches and lets the
    caller disambiguate using PTH spatial matching.
    """
    if pins is None:
        pins = load_all(smx_dir).values()
    return [p for p in pins if p.contains_xy(x, y, margin_m=margin_m)]
