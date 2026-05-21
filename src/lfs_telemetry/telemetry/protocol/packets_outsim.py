"""OutSim packet structures (extracted from packets.py for MH4).

Re-exported by :mod:`lfs_telemetry.telemetry.protocol.packets` for
backward compatibility. Importers should keep using ``from
lfs_telemetry.telemetry.protocol.packets import OutSimPacket`` etc.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import ClassVar

# ---------------------------------------------------------------------------
# OutSim
# ---------------------------------------------------------------------------

# Layout (LFS default OutSim Opts = 0):
#   uint32 Time
#   float  AngVel[3]   (rad/s, world)
#   float  Heading, Pitch, Roll  (rad)
#   float  Accel[3]    (m/s^2, local car frame)
#   float  Vel[3]      (m/s,  world)
#   int32  Pos[3]      (1/65536 m, world)
#   int32  ID          (only if "OutSim ID" != 0)
_OUTSIM_FMT = "<I" + "f" * 3 + "f" * 3 + "f" * 3 + "f" * 3 + "i" * 3
OUTSIM_SIZE = struct.calcsize(_OUTSIM_FMT)              # 64
OUTSIM_SIZE_WITH_ID = OUTSIM_SIZE + 4                   # 68


@dataclass(slots=True)
class OutSimPacket:
    """Decoded OutSim physics packet (SI units, world frame unless noted)."""

    time_ms: int
    ang_vel: tuple[float, float, float]      # rad/s
    heading: float                            # rad
    pitch: float                              # rad
    roll: float                               # rad
    accel: tuple[float, float, float]         # m/s^2 (local car frame)
    vel: tuple[float, float, float]           # m/s   (world)
    pos: tuple[float, float, float]           # meters (world)
    packet_id: int | None = None

    _STRUCT: ClassVar[struct.Struct] = struct.Struct(_OUTSIM_FMT)

    @classmethod
    def parse(cls, data: bytes) -> OutSimPacket:
        if len(data) not in (OUTSIM_SIZE, OUTSIM_SIZE_WITH_ID):
            raise ValueError(f"unexpected OutSim packet size: {len(data)}")
        unpacked = cls._STRUCT.unpack_from(data, 0)
        (
            t,
            ax, ay, az,
            hd, pt, rl,
            acx, acy, acz,
            vx, vy, vz,
            px, py, pz,
        ) = unpacked
        pid = None
        if len(data) == OUTSIM_SIZE_WITH_ID:
            pid = struct.unpack_from("<i", data, OUTSIM_SIZE)[0]
        return cls(
            time_ms=t,
            ang_vel=(ax, ay, az),
            heading=hd,
            pitch=pt,
            roll=rl,
            accel=(acx, acy, acz),
            vel=(vx, vy, vz),
            pos=(px / 65536.0, py / 65536.0, pz / 65536.0),
            packet_id=pid,
        )




# ---------------------------------------------------------------------------
# OutSim extended (OutSimPack2) — driven by OSOpts hex flags
# ---------------------------------------------------------------------------

# OSOpts flags (cfg.txt "OutSim Opts" hex value).
OSO_HEADER = 0x001
OSO_ID = 0x002
OSO_TIME = 0x004
OSO_MAIN = 0x008
OSO_INPUTS = 0x010
OSO_DRIVE = 0x020
OSO_DISTANCE = 0x040
OSO_WHEELS = 0x080
OSO_EXTRA_1 = 0x100

OSO_ALL = (
    OSO_HEADER | OSO_ID | OSO_TIME | OSO_MAIN | OSO_INPUTS
    | OSO_DRIVE | OSO_DISTANCE | OSO_WHEELS | OSO_EXTRA_1
)  # 0x1ff — recommended config

OUTSIM_WHEEL_FMT = "<7f4B2f"
OUTSIM_WHEEL_SIZE = struct.calcsize(OUTSIM_WHEEL_FMT)        # 40
assert OUTSIM_WHEEL_SIZE == 40, OUTSIM_WHEEL_SIZE
OUTSIMPACK2_FULL_SIZE = 280


@dataclass(slots=True)
class OutSimWheel:
    """Per-wheel telemetry from extended OutSim (OSO_WHEELS).

    Each :class:`OutSimPack2` carries 4 of these in :data:`WHEEL_ORDER`
    (RL, RR, FL, FR).
    """

    susp_deflect_m: float       # compression from unloaded
    steer_rad: float            # incl. Ackermann + toe
    x_force_n: float            # right (lateral, car frame)
    y_force_n: float            # forward (longitudinal, car frame)
    vertical_load_n: float      # perpendicular to surface
    ang_vel_rads: float
    lean_rel_road_rad: float    # anti-clockwise viewed from rear
    air_temp_c: int             # tyre air temperature (degrees C)
    slip_fraction_byte: int     # 0..255, fraction of contact patch sliding
    touching: int               # 1 if touching ground else 0
    slip_ratio: float
    tan_slip_angle: float       # tan(slip angle)

    @property
    def slip_fraction(self) -> float:
        """Slip fraction as a 0..1 float."""
        return self.slip_fraction_byte / 255.0


@dataclass(slots=True)
class OutSimPack2:
    """Decoded extended OutSim packet (driven by OSOpts).

    Fields are populated only if the corresponding OSOpts flag is set in
    ``opts``. Use :func:`outsim2_size` to predict the wire size for a given
    OSOpts mask.
    """

    opts: int
    header: str | None = None
    packet_id: int | None = None
    time_ms: int | None = None
    # OSO_MAIN
    ang_vel: tuple[float, float, float] | None = None
    heading: float | None = None
    pitch: float | None = None
    roll: float | None = None
    accel: tuple[float, float, float] | None = None
    vel: tuple[float, float, float] | None = None
    pos: tuple[float, float, float] | None = None
    # OSO_INPUTS
    throttle: float | None = None
    brake: float | None = None
    input_steer: float | None = None
    clutch: float | None = None
    handbrake: float | None = None
    # OSO_DRIVE
    gear: int | None = None
    engine_ang_vel_rads: float | None = None
    max_torque_at_vel_nm: float | None = None
    # OSO_DISTANCE
    current_lap_dist_m: float | None = None
    indexed_distance_m: float | None = None
    # OSO_WHEELS
    wheels: list[OutSimWheel] | None = None
    # OSO_EXTRA_1
    steer_torque_nm: float | None = None

    @classmethod
    def parse(cls, data: bytes, opts: int) -> OutSimPack2:
        expected = outsim2_size(opts)
        if len(data) != expected:
            raise ValueError(
                f"OutSimPack2 size {len(data)} != expected {expected} for opts=0x{opts:x}"
            )
        off = 0
        out = cls(opts=opts)
        if opts & OSO_HEADER:
            (raw,) = struct.unpack_from("<4s", data, off)
            off += 4
            out.header = raw.split(b"\x00", 1)[0].decode("latin-1", "replace")
        if opts & OSO_ID:
            (out.packet_id,) = struct.unpack_from("<i", data, off)
            off += 4
        if opts & OSO_TIME:
            (out.time_ms,) = struct.unpack_from("<I", data, off)
            off += 4
        if opts & OSO_MAIN:
            (avx, avy, avz, hd, pt, rl, acx, acy, acz,
             vx, vy, vz, px, py, pz) = struct.unpack_from("<12f3i", data, off)
            off += struct.calcsize("<12f3i")
            out.ang_vel = (avx, avy, avz)
            out.heading = hd
            out.pitch = pt
            out.roll = rl
            out.accel = (acx, acy, acz)
            out.vel = (vx, vy, vz)
            out.pos = (px / 65536.0, py / 65536.0, pz / 65536.0)
        if opts & OSO_INPUTS:
            (thr, brk, ist, clu, hnd) = struct.unpack_from(
                "<5f", data, off)
            off += 20
            out.throttle = thr
            out.brake = brk
            out.input_steer = ist
            out.clutch = clu
            out.handbrake = hnd
        if opts & OSO_DRIVE:
            (gear, _sp1, _sp2, _sp3, eng, mxt) = struct.unpack_from(
                "<4B2f", data, off)
            off += struct.calcsize("<4B2f")
            out.gear = gear
            out.engine_ang_vel_rads = eng
            out.max_torque_at_vel_nm = mxt
        if opts & OSO_DISTANCE:
            (cld, ixd) = struct.unpack_from("<2f", data, off)
            off += 8
            out.current_lap_dist_m = cld
            out.indexed_distance_m = ixd
        if opts & OSO_WHEELS:
            wheels: list[OutSimWheel] = []
            for _ in range(4):
                (sd, st, xf, yf, vl, av, lr, at, sf, tc, _sp,
                 sr, ta) = struct.unpack_from(OUTSIM_WHEEL_FMT, data, off)
                off += OUTSIM_WHEEL_SIZE
                wheels.append(OutSimWheel(
                    susp_deflect_m=sd, steer_rad=st,
                    x_force_n=xf, y_force_n=yf, vertical_load_n=vl,
                    ang_vel_rads=av, lean_rel_road_rad=lr,
                    air_temp_c=at, slip_fraction_byte=sf, touching=tc,
                    slip_ratio=sr, tan_slip_angle=ta,
                ))
            out.wheels = wheels
        if opts & OSO_EXTRA_1:
            (stq, _spare) = struct.unpack_from("<2f", data, off)
            off += 8
            out.steer_torque_nm = stq
        return out


def outsim2_size(opts: int) -> int:
    """Return wire size (bytes) of an OutSimPack2 with the given OSOpts."""
    size = 0
    if opts & OSO_HEADER:
        size += 4
    if opts & OSO_ID:
        size += 4
    if opts & OSO_TIME:
        size += 4
    if opts & OSO_MAIN:
        size += 60
    if opts & OSO_INPUTS:
        size += 20
    if opts & OSO_DRIVE:
        size += 12
    if opts & OSO_DISTANCE:
        size += 8
    if opts & OSO_WHEELS:
        size += 4 * OUTSIM_WHEEL_SIZE  # 160
    if opts & OSO_EXTRA_1:
        size += 8
    return size


# Sanity check at import time — fail loud if a constant drifts.
assert outsim2_size(OSO_ALL) == OUTSIMPACK2_FULL_SIZE, (
    outsim2_size(OSO_ALL), OUTSIMPACK2_FULL_SIZE
)


