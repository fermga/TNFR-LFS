"""Parser for LFS ``.knw`` (AI knowledge) files.

These small (≈ 0.4–1.0 KB) binary files live under ``C:\\LFS\\data\\knw`` —
one per ``<LAYOUT>_<CAR>`` combination — and describe how the canonical
LFS AI drives a given track with a given car: a coarse segmentation of
the PTH centerline plus a lateral offset per segment.

Reverse-engineered schema (validated against 1718 install files, 89
layouts × 20 cars):

::

    File header (12 B):
        char[6] magic   = "LFSKNW"
        u16     version = 7
        u32     build   (date stamp; not interpreted)

    Header record (24 B) — always the first record:
        u32 ai_seed              # per-car run seed
        f32 lap_factor           # ~0.27..1.5
        f32 speed_metric_a_ms    # car-dependent target speed (m/s)
        f32 speed_metric_b_ms    # car-dependent target speed (m/s)
        f32 car_constant         # per-car constant (FBM=42, BF1=95, FXR=100, …)
        u32 packed_counts        # low u16 = pth_node_count
                                 # high u16 = segment_count == len(segments)

    Segment record (24 B) — repeated ``segment_count`` times:
        u32 reserved             # always 0 in the wild
        u32 flags                # bit-packed (low byte varies in a regular pattern)
        u32 node_start           # start index on the PTH centerline
        u32 node_end             # end index on the PTH centerline
                                 # invariant: node_end[i] == node_start[i+1]
        f32 offset_delta         # small correction (~±0.1)
        f32 lateral_offset_m     # target lateral offset from centerline (m)

The chain of (``node_start``, ``node_end``) intervals wraps around the
closed PTH loop, so the file is one full lap.
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path

from ..constants import SPEED_MS_TO_KMH

_LOG = logging.getLogger(__name__)

KNW_MAGIC = b"LFSKNW"
# Two header u16 values are observed across the canonical install:
#   0x0700 (1792) — AS/BL/SO/WE layouts
#   0x0600 (1536) — FE/KY layouts
# Both parse identically; treat both as canonical.
KNW_VERSIONS_KNOWN: frozenset[int] = frozenset({0x0700, 0x0600})
KNW_VERSION = 0x0700   # the most common; kept for backward compatibility
HEADER_SIZE = 12
RECORD_SIZE = 24

DEFAULT_KNW_DIR = Path(r"C:\LFS\data\knw")


@dataclass(slots=True)
class KnwSegment:
    """One coarse segment of the AI's racing line."""
    index: int
    flags: int
    node_start: int
    node_end: int
    offset_delta: float
    lateral_offset_m: float


@dataclass(slots=True)
class KnwInfo:
    """Parsed contents of one ``.knw`` file."""
    layout: str
    car: str
    version: int
    build_stamp: bytes
    ai_seed: int
    lap_factor: float
    speed_metric_a_ms: float
    speed_metric_b_ms: float
    car_constant: float
    pth_node_count: int
    segment_count: int
    segments: list[KnwSegment] = field(default_factory=list)

    @property
    def speed_metric_a_kmh(self) -> float:
        return self.speed_metric_a_ms * SPEED_MS_TO_KMH

    @property
    def speed_metric_b_kmh(self) -> float:
        return self.speed_metric_b_ms * SPEED_MS_TO_KMH


def _split_layout_car(stem: str) -> tuple[str, str]:
    """``"AS1_FBM"`` → ``("AS1", "FBM")``; falls back to ``("", "")``."""
    if "_" not in stem:
        return "", ""
    layout, _, car = stem.rpartition("_")
    return layout.upper(), car.upper()


def parse_knw_bytes(data: bytes, *, layout: str = "", car: str = "") -> KnwInfo:
    """Parse a raw ``.knw`` byte string."""
    if len(data) < HEADER_SIZE + RECORD_SIZE:
        raise ValueError(
            f"knw file too short ({len(data)} B); need at least "
            f"{HEADER_SIZE + RECORD_SIZE} B"
        )
    if data[:6] != KNW_MAGIC:
        raise ValueError(
            f"not an LFS .knw file (magic {data[:6]!r} != {KNW_MAGIC!r})"
        )
    version = struct.unpack_from("<H", data, 6)[0]
    if version not in KNW_VERSIONS_KNOWN:
        _LOG.warning("unexpected .knw version 0x%04X (known: %s)",
                     version,
                     ", ".join(f"0x{v:04X}" for v in sorted(KNW_VERSIONS_KNOWN)))
    build_stamp = bytes(data[8:HEADER_SIZE])

    body = data[HEADER_SIZE:]
    if len(body) % RECORD_SIZE != 0:
        raise ValueError(
            f"knw body length {len(body)} not a multiple of {RECORD_SIZE} B"
        )
    n_records = len(body) // RECORD_SIZE

    ai_seed, lap_factor, sm_a, sm_b, car_constant, packed = struct.unpack_from(
        "<IffffI", body, 0,
    )
    pth_node_count = packed & 0xFFFF
    segment_count = (packed >> 16) & 0xFFFF

    if segment_count != n_records - 1:
        # Non-fatal: trust the on-disk records, warn for visibility.
        _LOG.warning(
            "segment_count in header (%d) != records-1 (%d) for %s_%s",
            segment_count, n_records - 1, layout, car,
        )
        segment_count = n_records - 1

    segments: list[KnwSegment] = []
    for i in range(segment_count):
        off = (i + 1) * RECORD_SIZE
        reserved, flags, n_start, n_end, delta, lat = struct.unpack_from(
            "<IIIIff", body, off,
        )
        if reserved != 0:
            _LOG.debug("segment %d reserved=%d (non-zero)", i, reserved)
        segments.append(KnwSegment(
            index=i,
            flags=flags,
            node_start=n_start,
            node_end=n_end,
            offset_delta=delta,
            lateral_offset_m=lat,
        ))

    return KnwInfo(
        layout=layout,
        car=car,
        version=version,
        build_stamp=build_stamp,
        ai_seed=ai_seed,
        lap_factor=lap_factor,
        speed_metric_a_ms=sm_a,
        speed_metric_b_ms=sm_b,
        car_constant=car_constant,
        pth_node_count=pth_node_count,
        segment_count=segment_count,
        segments=segments,
    )


def parse_knw(path: Path | str) -> KnwInfo:
    """Parse the ``.knw`` file at ``path``; layout/car inferred from stem."""
    p = Path(path)
    layout, car = _split_layout_car(p.stem)
    return parse_knw_bytes(p.read_bytes(), layout=layout, car=car)


def list_knw_files(knw_dir: Path | str = DEFAULT_KNW_DIR) -> list[Path]:
    """Return every ``*.knw`` file under ``knw_dir`` (alphabetical)."""
    d = Path(knw_dir)
    if not d.exists():
        return []
    return sorted(d.glob("*.knw"))


def load_all_for_layout(
    layout: str,
    knw_dir: Path | str = DEFAULT_KNW_DIR,
) -> dict[str, KnwInfo]:
    """Return ``{car: KnwInfo}`` for every car that has a ``.knw`` on ``layout``."""
    d = Path(knw_dir)
    layout_u = layout.upper()
    out: dict[str, KnwInfo] = {}
    for p in d.glob(f"{layout_u}_*.knw"):
        try:
            info = parse_knw(p)
        except ValueError as exc:
            _LOG.warning("skipping %s: %s", p, exc)
            continue
        if info.car:
            out[info.car] = info
    return out


def load_for(
    layout: str,
    car: str,
    knw_dir: Path | str = DEFAULT_KNW_DIR,
) -> KnwInfo | None:
    """Return the parsed ``.knw`` for ``(layout, car)`` or ``None``."""
    p = Path(knw_dir) / f"{layout.upper()}_{car.upper()}.knw"
    if not p.exists():
        return None
    return parse_knw(p)


__all__ = [
    "DEFAULT_KNW_DIR",
    "HEADER_SIZE",
    "KNW_MAGIC",
    "KNW_VERSION",
    "KNW_VERSIONS_KNOWN",
    "RECORD_SIZE",
    "KnwInfo",
    "KnwSegment",
    "list_knw_files",
    "load_all_for_layout",
    "load_for",
    "parse_knw",
    "parse_knw_bytes",
]
