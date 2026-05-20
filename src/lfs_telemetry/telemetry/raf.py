"""LFS Replay Analyser File (RAF) v2 reader + lap splitter.

The RAF format is documented at https://www.lfs.net/programmer/raf and
is the file LFS generates when *analysing a replay* — it is the only
official path for getting per-sample telemetry out of someone else's
replay (``.mpr``/``.spr``). This module reads RAF v2 files and exports
each detected lap as a CSV in the schema used by the rest of the app
(see :mod:`lfs_telemetry.telemetry.replay`), so RAF-imported laps can
be loaded by :class:`LapTelemetry` and compared like any other capture.

Key compatibility notes
-----------------------

* RAF axes (X right, Y forward, Z up) match the OutSim convention this
  app uses internally; no axis remapping is needed.
* RAF lap boundaries are detected from the ``index distance`` channel
  (track-ruler measurement, monotonically increasing within a lap and
  resetting to ~0 at the start/finish line). This is the same signal
  LFS uses to anchor its sectors.
* RAF gives one G value per axis quantised to ``signed char * 20``
  (range ±6 g). We promote to m/s² using ``g = 9.80665`` to keep the
  enriched view consistent with native captures.
* The CSV columns we do not have a source for in RAF (per-wheel slip
  ratio/tangential slip, OBH counts, pit-stop counts, FFB torque,
  steer torque…) are written empty so downstream consumers fall back
  to their default-when-missing behaviour. This is intentional: the
  goal is to enable *comparison* of two drivers' laps, not to claim
  feature parity with a live capture.
"""

from __future__ import annotations

import csv
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol.packets import WHEEL_ORDER

_G = 9.80665  # m/s² — same constant the rest of the package uses
_MAGIC = b"LFSRAF"
_HEADER_SIZE = 1024
_BLOCK_SIZE = 192
_WHEEL_BLOCK_SIZE = 32
_WHEEL_BLOCK_OFFSET = 64
_NUM_WHEELS = 4

# RAF wheel order, as confirmed by reading the file alongside OutSim2
# captures of the same car: RL, RR, FL, FR. This matches the wire
# order used in :mod:`packets` (WHEEL_ORDER_WIRE), so we re-use that
# tuple verbatim and the resulting CSV column names line up 1:1 with
# native captures.
_RAF_WHEEL_ORDER = WHEEL_ORDER


@dataclass(slots=True, frozen=True)
class RafHeader:
    """Parsed RAF header (subset useful for downstream code)."""

    raf_version: int
    update_interval_ms: int
    header_size: int
    block_size: int
    wheel_block_size: int
    wheel_block_offset: int
    num_blocks: int
    short_track_name: str
    track_ruler_length_m: float
    player: str
    car: str
    track: str
    config: str
    weather: str
    lfs_version: str
    player_flags: int
    num_wheels: int
    hlvc_legal: int
    num_splits: int
    splits_ms: tuple[int, int, int, int]
    mass_kg: float
    sprung_mass_kg: float
    rear_antiroll_n_per_m: float
    front_antiroll_n_per_m: float
    final_drive: float
    num_gears: int
    gear_ratios: tuple[float, ...]


def _cstr(buf: bytes) -> str:
    end = buf.find(b"\x00")
    if end >= 0:
        buf = buf[:end]
    try:
        return buf.decode("latin-1").strip()
    except UnicodeDecodeError:
        return buf.decode("latin-1", errors="replace").strip()


def parse_raf_header(buf: bytes) -> RafHeader:
    """Parse the first :data:`_HEADER_SIZE` bytes of a RAF file."""
    if len(buf) < _HEADER_SIZE:
        raise ValueError(
            f"RAF header truncated: got {len(buf)} bytes, "
            f"need at least {_HEADER_SIZE}",
        )
    if buf[:6] != _MAGIC:
        raise ValueError("not a RAF file (missing LFSRAF magic)")
    raf_version = buf[8]
    if raf_version > 2:
        raise ValueError(
            f"unsupported RAF version {raf_version} (this reader handles v2)",
        )
    update_interval_ms = buf[9]
    header_size = struct.unpack_from("<H", buf, 12)[0]
    block_size = struct.unpack_from("<H", buf, 14)[0]
    wheel_block_size = struct.unpack_from("<H", buf, 16)[0]
    wheel_block_offset = struct.unpack_from("<H", buf, 18)[0]
    num_blocks = struct.unpack_from("<i", buf, 20)[0]
    short_track_name = _cstr(buf[24:28])
    track_ruler_length_m = struct.unpack_from("<f", buf, 28)[0]
    player = _cstr(buf[32:64])
    car = _cstr(buf[64:96])
    track = _cstr(buf[96:128])
    config = _cstr(buf[128:144])
    weather = _cstr(buf[144:160])
    lfs_version = _cstr(buf[160:168])
    player_flags = buf[168]
    num_wheels = buf[169]
    hlvc_legal = buf[170]
    num_splits = buf[171]
    s1, s2, s3, s4 = struct.unpack_from("<iiii", buf, 172)
    mass_kg = struct.unpack_from("<f", buf, 188)[0]
    sprung_mass_kg = struct.unpack_from("<f", buf, 192)[0]
    r_antiroll = struct.unpack_from("<f", buf, 196)[0]
    f_antiroll = struct.unpack_from("<f", buf, 200)[0]
    final_drive = struct.unpack_from("<f", buf, 204)[0]
    num_gears = buf[208]
    gear_ratios = struct.unpack_from("<7f", buf, 212)
    return RafHeader(
        raf_version=raf_version,
        update_interval_ms=update_interval_ms,
        header_size=header_size,
        block_size=block_size,
        wheel_block_size=wheel_block_size,
        wheel_block_offset=wheel_block_offset,
        num_blocks=num_blocks,
        short_track_name=short_track_name,
        track_ruler_length_m=track_ruler_length_m,
        player=player,
        car=car,
        track=track,
        config=config,
        weather=weather,
        lfs_version=lfs_version,
        player_flags=player_flags,
        num_wheels=num_wheels,
        hlvc_legal=hlvc_legal,
        num_splits=num_splits,
        splits_ms=(int(s1), int(s2), int(s3), int(s4)),
        mass_kg=float(mass_kg),
        sprung_mass_kg=float(sprung_mass_kg),
        rear_antiroll_n_per_m=float(r_antiroll),
        front_antiroll_n_per_m=float(f_antiroll),
        final_drive=float(final_drive),
        num_gears=int(num_gears),
        gear_ratios=tuple(float(x) for x in gear_ratios[:num_gears]),
    )


def _parse_data_block(
    buf: bytes, off: int, header: RafHeader,
) -> dict[str, Any]:
    """Decode one fixed-size data block at ``buf[off:off+block_size]``."""
    throttle = struct.unpack_from("<f", buf, off + 0)[0]
    brake = struct.unpack_from("<f", buf, off + 4)[0]
    input_steer = struct.unpack_from("<f", buf, off + 8)[0]
    clutch = struct.unpack_from("<f", buf, off + 12)[0]
    handbrake = struct.unpack_from("<f", buf, off + 16)[0]
    gear_raf = buf[off + 20]  # 0=R, 1=N, 2+=forward gear
    lat_g_q = struct.unpack_from("<b", buf, off + 21)[0]
    fwd_g_q = struct.unpack_from("<b", buf, off + 22)[0]
    up_g_q = struct.unpack_from("<b", buf, off + 23)[0]
    speed_ms = struct.unpack_from("<f", buf, off + 24)[0]
    car_distance_m = struct.unpack_from("<f", buf, off + 28)[0]
    pos_x_q, pos_y_q, pos_z_q = struct.unpack_from("<iii", buf, off + 32)
    engine_rads = struct.unpack_from("<f", buf, off + 44)[0]
    index_distance_m = struct.unpack_from("<f", buf, off + 48)[0]
    rx, ry, rz = struct.unpack_from("<hhh", buf, off + 52)
    fx, fy, fz = struct.unpack_from("<hhh", buf, off + 58)

    # Quantised G → m/s² (signed char * 20 means ±6 g range, see spec)
    accel_y = (lat_g_q / 20.0) * _G  # lateral (X = right)
    accel_x = (fwd_g_q / 20.0) * _G  # forward
    accel_z = (up_g_q / 20.0) * _G   # vertical

    # Position scale: 1 m = 65536 in RAF
    pos_x = pos_x_q / 65536.0
    pos_y = pos_y_q / 65536.0
    pos_z = pos_z_q / 65536.0

    # Heading from forward-vector (spec: anti-clockwise from above)
    fx_f = fx / 32767.0
    fy_f = fy / 32767.0
    heading = math.atan2(-fx_f, fy_f) if (fx or fy) else 0.0
    # Pitch from forward-Z (right-handed, up positive)
    fz_f = fz / 32767.0
    pitch = math.asin(max(-1.0, min(1.0, fz_f)))
    # Roll from right-vector Z (right is X axis)
    rz_f = rz / 32767.0
    roll = math.asin(max(-1.0, min(1.0, rz_f)))

    # OutSim convention: gear 0 = reverse, 1 = neutral, 2 = first.
    # RAF uses the same convention, so pass it through directly.
    gear = int(gear_raf)

    # rpm from engine_rads (rad/s → rpm)
    rpm = engine_rads * 60.0 / (2.0 * math.pi) if engine_rads else 0.0

    row: dict[str, Any] = {
        "throttle": float(throttle),
        "brake": float(brake),
        "clutch": float(clutch),
        "input_throttle": float(throttle),
        "input_brake": float(brake),
        "input_steer": float(input_steer),
        "input_clutch": float(clutch),
        "input_handbrake": float(handbrake),
        "gear": gear,
        "speed_ms": float(speed_ms),
        "rpm": float(rpm),
        "accel_x": float(accel_x),
        "accel_y": float(accel_y),
        "accel_z": float(accel_z),
        "pos_x": float(pos_x),
        "pos_y": float(pos_y),
        "pos_z": float(pos_z),
        "heading": float(heading),
        "pitch": float(pitch),
        "roll": float(roll),
        "indexed_distance_m": float(index_distance_m),
        "engine_ang_vel_rads": float(engine_rads),
        "_car_distance_m": float(car_distance_m),  # internal helper
    }

    # Per-wheel dynamic block
    wb_off0 = off + header.wheel_block_offset
    for i, corner in enumerate(_RAF_WHEEL_ORDER):
        wb = wb_off0 + i * header.wheel_block_size
        susp = struct.unpack_from("<f", buf, wb + 0)[0]
        steer_rad = struct.unpack_from("<f", buf, wb + 4)[0]
        vload = struct.unpack_from("<f", buf, wb + 8)[0]
        x_force = struct.unpack_from("<f", buf, wb + 12)[0]
        y_force = struct.unpack_from("<f", buf, wb + 16)[0]
        ang_vel = struct.unpack_from("<f", buf, wb + 20)[0]
        lean = struct.unpack_from("<f", buf, wb + 24)[0]
        air_temp = buf[wb + 28]
        slip_frac = buf[wb + 29]
        row[f"wheel_{corner}_susp_deflect_m"] = float(susp)
        row[f"wheel_{corner}_steer_rad"] = float(steer_rad)
        row[f"wheel_{corner}_vertical_load_n"] = float(vload)
        row[f"wheel_{corner}_x_force_n"] = float(x_force)
        row[f"wheel_{corner}_y_force_n"] = float(y_force)
        row[f"wheel_{corner}_ang_vel_rads"] = float(ang_vel)
        row[f"wheel_{corner}_lean_rel_road_rad"] = float(lean)
        row[f"wheel_{corner}_air_temp_c"] = int(air_temp)
        row[f"wheel_{corner}_slip_fraction"] = int(slip_frac)
        # Channels we have no RAF source for — left blank so the
        # downstream enrichment falls back to its missing-data path.
        row[f"wheel_{corner}_slip_ratio"] = ""
        row[f"wheel_{corner}_tan_slip_angle"] = ""
        row[f"wheel_{corner}_touching"] = ""

    return row


def parse_raf(path: str | Path) -> tuple[RafHeader, list[dict[str, Any]]]:
    """Read a RAF file and return ``(header, rows)``.

    ``rows`` is a list of per-sample dicts, one per data block, each
    carrying our CSV schema fields (plus the internal ``_car_distance_m``
    helper). No lap splitting is performed here — call
    :func:`split_into_laps` or :func:`raf_to_lap_csvs` for that.
    """
    p = Path(path)
    data = p.read_bytes()
    header = parse_raf_header(data)
    if header.block_size != _BLOCK_SIZE:
        # The spec explicitly allows block_size to grow; degrade
        # gracefully by treating any extra bytes per block as padding.
        pass
    rows: list[dict[str, Any]] = []
    cursor = header.header_size
    end = cursor + header.block_size * header.num_blocks
    if end > len(data):
        # Truncated file: read as many full blocks as we can.
        end = len(data) - ((len(data) - cursor) % header.block_size)
    dt_ms = max(int(header.update_interval_ms), 1)
    sample = 0
    while cursor + header.block_size <= end:
        row = _parse_data_block(data, cursor, header)
        row["time_ms"] = int(sample * dt_ms)
        row["car"] = header.car
        row["ctx_track"] = header.track
        row["ctx_weather"] = header.weather
        row["ctx_view_car"] = header.car
        row["ctx_lfs_version"] = header.lfs_version
        rows.append(row)
        cursor += header.block_size
        sample += 1
    return header, rows


def split_into_laps(
    header: RafHeader, rows: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group samples into laps using the ``indexed_distance_m`` channel.

    A new lap begins whenever the track-ruler index distance resets
    (drops by more than half the ruler length) compared with the
    previous sample. The first segment before any reset is treated as
    "lap 0" (out-lap / partial lap before the first start/finish
    crossing), which downstream sectors logic discards if it lacks a
    valid start anchor.
    """
    if not rows:
        return []
    ruler = max(float(header.track_ruler_length_m), 1.0)
    laps: list[list[dict[str, Any]]] = [[]]
    prev_idx = rows[0].get("indexed_distance_m", 0.0)
    lap_start_dist = rows[0].get("_car_distance_m", 0.0)
    lap_start_time = rows[0].get("time_ms", 0)
    for row in rows:
        cur_idx = float(row.get("indexed_distance_m", 0.0))
        if prev_idx - cur_idx > 0.5 * ruler:
            # Wrap → new lap
            laps.append([])
            lap_start_dist = float(row.get("_car_distance_m", 0.0))
            lap_start_time = int(row.get("time_ms", 0))
        # Anchor per-lap distance/time so each lap CSV starts at 0
        row = dict(row)
        row["current_lap_dist_m"] = (
            float(row.get("_car_distance_m", 0.0)) - lap_start_dist
        )
        row["time_ms"] = int(row.get("time_ms", 0)) - int(lap_start_time)
        laps[-1].append(row)
        prev_idx = cur_idx
    # Drop empty lists (can happen if the very first sample is a wrap)
    return [lap for lap in laps if lap]


def raf_to_lap_csvs(
    raf_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    skip_outlap: bool = True,
    min_samples_per_lap: int = 100,
) -> list[Path]:
    """Read ``raf_path``, split into laps, write one CSV per lap.

    Returns the list of CSV paths written. Each CSV uses the same
    schema as :func:`lfs_telemetry.telemetry.replay.write_csv_replay`,
    so :meth:`LapTelemetry.from_csv` can load them directly.

    Parameters
    ----------
    raf_path:
        Source ``.raf`` file.
    out_dir:
        Destination folder. Defaults to ``<raf_path>.laps/`` next to
        the input file. Created if missing.
    skip_outlap:
        Drop the leading partial segment before the first start/finish
        crossing (typical "lap 0" in a replay). Default ``True``.
    min_samples_per_lap:
        Discard lap segments shorter than this many samples (replay
        cut at mid-lap, pause, etc.). Default 100 samples ≈ 1 s at
        100 Hz RAF cadence.
    """
    raf_path = Path(raf_path)
    if out_dir is None:
        out_dir = raf_path.with_suffix("")
        out_dir = out_dir.parent / f"{out_dir.name}_raf_laps"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header, rows = parse_raf(raf_path)
    laps = split_into_laps(header, rows)

    # Drop the lead-in segment if requested
    if skip_outlap and len(laps) > 1:
        laps = laps[1:]
    laps = [lap for lap in laps if len(lap) >= min_samples_per_lap]

    # Import locally to avoid a hard cyclic import at module load
    from .replay import _FIELDS, _SCHEMA_HEADER

    # Sanitize names for file system use
    def _clean(s: str) -> str:
        keep = "-_."
        out = []
        for ch in s:
            if ch.isalnum() or ch in keep:
                out.append(ch)
            else:
                out.append("_")
        return "".join(out).strip("_") or "x"

    track = _clean(header.short_track_name or header.track or "TRACK")
    car = _clean(header.car or "CAR")
    player = _clean(header.player or "RAF")
    stem = f"raf_{player}_{track}_{car}"

    written: list[Path] = []
    for i, lap in enumerate(laps, start=1):
        csv_path = out_dir / f"{stem}_lap{i:02d}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            fp.write(_SCHEMA_HEADER + "\n")
            writer = csv.DictWriter(
                fp, fieldnames=_FIELDS, extrasaction="ignore",
            )
            writer.writeheader()
            for row in lap:
                writer.writerow(row)
        written.append(csv_path)
    return written


__all__ = [
    "RafHeader",
    "parse_raf_header",
    "parse_raf",
    "split_into_laps",
    "raf_to_lap_csvs",
]
