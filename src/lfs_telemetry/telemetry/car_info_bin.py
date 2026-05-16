"""Parser for the official LFS ``CAR_info.bin`` export.

LFS 0.7+ ships a small binary export per car that documents the canonical
chassis / engine / drivetrain / wheels in a stable on-disk layout. The file
is the ground-truth source for mass, weight distribution, wheelbase, track,
CG height, drivetrain layout, gear ratios, and per-wheel suspension specs
— values that we cannot infer from OutSim/OutGauge telemetry alone.

Layout reference: https://www.lfs.net/programmer/carinfo (section
"CAR_info.bin"). Offsets and units verified against LFS 0.7G files.

This module is read-only and stdlib-only: it intentionally does not depend
on any other telemetry module so it can be exercised from `__main__` or
tests without a running InSim session.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

__all__ = [
    "CarInfoWheel",
    "CarInfoBin",
    "parse_car_info_bin",
    "DRIVE_NAMES",
]

DRIVE_NAMES: dict[int, str] = {
    0: "Unknown",
    1: "RWD",
    2: "FWD",
    3: "AWD",
}

# ``CoG`` fixed-point uses 65536 = 1 metre per the spec.
_FX_SCALE = 65536.0

# Per-wheel block size and layout-derived field offsets (within each block).
# Each wheel block is 128 bytes. Only the fields we actually surface are
# unpacked here; the rest of the block (geometry helpers, debug flags) is
# left untouched and can be added in a follow-up if needed.
_WHEEL_BLOCK_SIZE = 128
# Offset 0   : tyre type (byte)
# Offset 4   : pressure(float, kPa), air temp, toe-in (rad)
# Offset 16  : contact patch x,y,z (float, metres in body frame), unsprung mass
# Offset 32  : tyre width, sidewall height proportion, rim radius, rim width
# Offset 48  : spring const, damping (compression / rebound), anti-roll
# Offset 64  : camber, inclination, caster, scrub radius
# Offset 80  : moment of inertia, current susp deflection, max susp deflection
# Offset 96  : tyre vertical spring rate, current vertical deflection
_WHEEL_NAMES = ("RL", "RR", "FL", "FR")


@dataclass(slots=True, frozen=True)
class CarInfoWheel:
    """One wheel block from CAR_info.bin (subset)."""

    name: str                  # "RL" / "RR" / "FL" / "FR"
    tyre_type: int             # byte, LFS-internal tyre compound id
    contact_x_m: float         # body-frame coordinates of the contact patch
    contact_y_m: float
    contact_z_m: float
    unsprung_mass_kg: float
    tyre_width_m: float
    sidewall_height_prop: float
    rim_radius_m: float
    rim_width_m: float
    spring_const: float        # N/m
    damping_comp: float        # N·s/m
    damping_rebound: float
    anti_roll: float           # N/m (per official RAF spec)
    camber_rad: float
    inclination_rad: float
    caster_rad: float
    scrub_radius_m: float
    moment_inertia: float
    susp_deflection_m: float
    max_susp_deflection_m: float
    tyre_vert_spring: float
    tyre_vert_deflection: float
    # Setup-screen fields (officially documented at offset 4 of each wheel
    # block in CAR_info.bin; LFS in-game garage exposes them as kPa / °C /
    # toe-in degrees).
    tyre_pressure_kpa: float
    air_temp_c: float
    toe_in_rad: float


@dataclass(slots=True)
class CarInfoBin:
    """Parsed view over a CAR_info.bin file (subset surfaced for telemetry).

    The fields we expose are precisely those that ``CarSpec`` consumes plus a
    few extras (gears, brake balance) that are useful for diagnostics.
    """

    file_version: int
    short_name: str            # 4-letter LFS short name ("XFG", "FBM", ...)
    passengers: int
    # Centre of gravity in body frame, metres. The X axis points right, Y
    # forward, Z up (LFS convention).
    cg_x_m: float
    cg_y_m: float
    cg_z_m: float
    # CG expressed in chassis-relative fractions (0..1). Some cars only set
    # the absolute CoG block, others only the relative one — we expose both.
    cg_x_rel: float
    cg_y_rel: float
    cg_z_rel: float
    # Fuel tank position (body-frame metres).
    fuel_tank_x_m: float
    fuel_tank_y_m: float
    fuel_tank_z_m: float
    # Engine.
    max_torque_nm: float
    max_torque_rpm: float
    max_power_kw: float
    max_power_rpm: float
    # Chassis bulk numbers (the spec calls these "approx" because LFS
    # actually derives them from per-wheel geometry; they are still the
    # authoritative figures for setup / strategy code).
    fuel_capacity_l: float
    mass_kg: float
    wheelbase_m: float
    weight_dist_front: float
    # Drivetrain.
    forward_gears: int
    drive: str                 # one of DRIVE_NAMES values
    torque_split: float        # AWD only (fraction to front, 0..1)
    drivetrain_efficiency: float
    gear_ratios: tuple[float, ...]   # [reverse, 1st, 2nd, ...]
    final_drive: float
    # Steering / brakes.
    parallel_steer: float
    brake_strength_nm: float
    brake_balance_front: float
    # Wheels (always 4, in RL, RR, FL, FR order per spec).
    wheels: tuple[CarInfoWheel, CarInfoWheel, CarInfoWheel, CarInfoWheel]

    # Derived helpers ------------------------------------------------------
    _MAGIC: ClassVar[bytes] = b"LFS_CI"

    @property
    def track_front_m(self) -> float:
        """Front track width from wheel contact patches."""
        return abs(self.wheels[2].contact_x_m - self.wheels[3].contact_x_m)

    @property
    def track_rear_m(self) -> float:
        return abs(self.wheels[0].contact_x_m - self.wheels[1].contact_x_m)

    @property
    def cg_height_m(self) -> float:
        """CG height above ground.

        Prefer the absolute CoG (offset 64 fixed-point). If the file leaves
        it zero we fall back to the relative-Z field, which is roughly the
        fraction of body height above the chassis reference plane.
        """
        if self.cg_z_m:
            return abs(self.cg_z_m)
        return abs(self.cg_z_rel)

    def to_car_spec_kwargs(self) -> dict:
        """Return kwargs ready for :class:`CarSpec(**kwargs)`."""
        return dict(
            mass_kg=self.mass_kg,
            wheelbase_m=self.wheelbase_m,
            track_front_m=self.track_front_m,
            track_rear_m=self.track_rear_m,
            cg_height_m=self.cg_height_m,
            weight_dist_front=self.weight_dist_front,
            driven=self.drive if self.drive in ("RWD", "FWD", "AWD") else "RWD",
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_wheel(buf: bytes, off: int, name: str) -> CarInfoWheel:
    tyre_type = buf[off]
    (pressure, air_temp, toe_in) = struct.unpack_from("<fff", buf, off + 4)
    (cx, cy, cz, unsprung) = struct.unpack_from("<ffff", buf, off + 16)
    (tw, sw_h, rim_r, rim_w) = struct.unpack_from("<ffff", buf, off + 32)
    (spring, damp_c, damp_r, anti_roll) = struct.unpack_from(
        "<ffff", buf, off + 48)
    (camber, incl, caster, scrub) = struct.unpack_from(
        "<ffff", buf, off + 64)
    (moi, susp_def, max_susp_def, _) = struct.unpack_from(
        "<ffff", buf, off + 80)
    (tyre_vert_spr, tyre_vert_def, _, _) = struct.unpack_from(
        "<ffff", buf, off + 96)
    return CarInfoWheel(
        name=name, tyre_type=tyre_type,
        contact_x_m=cx, contact_y_m=cy, contact_z_m=cz,
        unsprung_mass_kg=unsprung,
        tyre_width_m=tw, sidewall_height_prop=sw_h,
        rim_radius_m=rim_r, rim_width_m=rim_w,
        spring_const=spring, damping_comp=damp_c, damping_rebound=damp_r,
        anti_roll=anti_roll,
        camber_rad=camber, inclination_rad=incl, caster_rad=caster,
        scrub_radius_m=scrub,
        moment_inertia=moi, susp_deflection_m=susp_def,
        max_susp_deflection_m=max_susp_def,
        tyre_vert_spring=tyre_vert_spr,
        tyre_vert_deflection=tyre_vert_def,
        tyre_pressure_kpa=pressure,
        air_temp_c=air_temp,
        toe_in_rad=toe_in,
    )


def parse_car_info_bin(path: str | Path) -> CarInfoBin:
    """Parse one CAR_info.bin file.

    Raises ``ValueError`` if the magic header is missing or the file is too
    small to be a valid export.
    """
    buf = Path(path).read_bytes()
    if len(buf) < 384 + 4 * _WHEEL_BLOCK_SIZE:
        raise ValueError(
            f"{path}: file too small ({len(buf)} bytes) for CAR_info.bin")
    if not buf.startswith(CarInfoBin._MAGIC):
        raise ValueError(f"{path}: missing 'LFS_CI' magic header")

    file_version = buf[7]
    short_name = buf[8:12].split(b"\x00", 1)[0].decode("latin-1", "replace")
    passengers = buf[12]

    # CoG absolute (fixed-point, 65536 = 1 m).
    (cg_x_fx, cg_y_fx, cg_z_fx) = struct.unpack_from("<iii", buf, 64)
    cg_x_m = cg_x_fx / _FX_SCALE
    cg_y_m = cg_y_fx / _FX_SCALE
    cg_z_m = cg_z_fx / _FX_SCALE

    # CoG relative (float fractions).
    (cg_x_rel, cg_y_rel, cg_z_rel) = struct.unpack_from("<fff", buf, 76)

    # Fuel tank position (body-frame floats).
    (ft_x, ft_y, ft_z) = struct.unpack_from("<fff", buf, 88)

    # Engine.
    (max_torque, t_rpm, max_power, p_rpm) = struct.unpack_from(
        "<ffff", buf, 256)

    # Chassis.
    (fuel_cap, mass, wb, wdf) = struct.unpack_from("<ffff", buf, 272)

    # Drivetrain header.
    forward_gears = buf[288]
    drive_byte = buf[289]
    drive = DRIVE_NAMES.get(drive_byte, "Unknown")
    (torque_split, drive_eff, _) = struct.unpack_from("<fff", buf, 292)

    # 8 gear ratios at offset 304 (reverse + 7 forward slots).
    gear_floats = struct.unpack_from("<8f", buf, 304)
    final_drive = struct.unpack_from("<f", buf, 336)[0]

    # Steering / brakes.
    (parallel_steer, brake_str, brake_bal) = struct.unpack_from(
        "<fff", buf, 352)

    # Wheels — 4 blocks of 128 bytes starting at 384, order RL, RR, FL, FR.
    wheels = tuple(
        _parse_wheel(buf, 384 + i * _WHEEL_BLOCK_SIZE, _WHEEL_NAMES[i])
        for i in range(4)
    )

    return CarInfoBin(
        file_version=file_version,
        short_name=short_name,
        passengers=passengers,
        cg_x_m=cg_x_m, cg_y_m=cg_y_m, cg_z_m=cg_z_m,
        cg_x_rel=cg_x_rel, cg_y_rel=cg_y_rel, cg_z_rel=cg_z_rel,
        fuel_tank_x_m=ft_x, fuel_tank_y_m=ft_y, fuel_tank_z_m=ft_z,
        max_torque_nm=max_torque, max_torque_rpm=t_rpm,
        max_power_kw=max_power, max_power_rpm=p_rpm,
        fuel_capacity_l=fuel_cap, mass_kg=mass,
        wheelbase_m=wb, weight_dist_front=wdf,
        forward_gears=forward_gears, drive=drive,
        torque_split=torque_split,
        drivetrain_efficiency=drive_eff,
        gear_ratios=tuple(gear_floats[: 1 + max(forward_gears, 0)]),
        final_drive=final_drive,
        parallel_steer=parallel_steer,
        brake_strength_nm=brake_str,
        brake_balance_front=brake_bal,
        wheels=wheels,  # type: ignore[arg-type]
    )
