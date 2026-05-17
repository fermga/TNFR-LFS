"""Live for Speed binary packet structures.

References: ``docs/InSim.txt`` and ``docs/OutSimPack.txt`` shipped with LFS.
All packets are little-endian.

We model the full set of LFS data sources useful for structural analysis:

* **OutSim** – physics packet (UDP).

  - Basic layout (``OutSim Opts 0``): :class:`OutSimPacket`, 64/68 B.
  - Extended layout (``OutSim Opts > 0``): :class:`OutSimPack2`, up to 280 B
    when ``OutSim Opts 1ff`` (all flags). Provides per-wheel forces, slip,
    suspension deflection, lap distance, steering torque, and inputs.

  Recommended ``cfg.txt`` block for full coverage::

      OutSim Mode 2
      OutSim Delay 1
      OutSim IP 127.0.0.1
      OutSim Port 30000
      OutSim ID 0
      OutSim Opts 1ff

  (``Mode 2`` selects the extended OutSimPack2 layout. ``InSim`` has no
  cfg.txt entry — enable it at runtime with ``/insim 29999`` in the LFS
  console or by launching ``LFS.exe /insim=29999``.)

* **OutGauge** – dashboard packet (UDP). Configured via ``cfg.txt``::

      OutGauge Mode 1
      OutGauge Delay 1
      OutGauge IP 127.0.0.1
      OutGauge Port 30001
      OutGauge ID 0

* **InSim** – control/event channel (TCP). Subset implemented:
  IS_ISI, IS_VER, IS_TINY, IS_SMALL, IS_STA, IS_RST, IS_NCN, IS_CNL,
  IS_NPL, IS_PLP, IS_PLL, IS_LAP, IS_SPX, IS_PIT, IS_PSF, IS_PLA,
  IS_CCH, IS_PEN, IS_TOC, IS_FLG, IS_PFL, IS_FIN, IS_RES, IS_NLP,
  IS_MCI, IS_CON, IS_OBH, IS_HLV, IS_SLC, IS_CSC, IS_CIM, IS_MAL.

We target InSim version 10 (LFS 0.7G+) but the wire format for the packets
we parse is identical to v9 except for IS_VER reporting ``insim_ver=10``.
"""

from __future__ import annotations

import math
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
    def parse(cls, data: bytes) -> "OutSimPacket":
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
# OutGauge
# ---------------------------------------------------------------------------

# Layout (LFS default OutGauge Opts = 0):
#   uint32 Time
#   char   Car[4]
#   uint16 Flags
#   byte   Gear           (0 = R, 1 = N, 2 = 1st, ...)
#   byte   PLID
#   float  Speed          m/s
#   float  RPM
#   float  Turbo          bar
#   float  EngTemp        deg C
#   float  Fuel           0..1
#   float  OilPressure    bar
#   float  OilTemp        deg C
#   uint32 DashLights
#   uint32 ShowLights
#   float  Throttle       0..1
#   float  Brake          0..1
#   float  Clutch         0..1
#   char   Display1[16]
#   char   Display2[16]
#   int32  ID             (only if "OutGauge ID" != 0)
_OUTGAUGE_FMT = "<I4sHBB" + "f" * 7 + "II" + "fff" + "16s16s"
OUTGAUGE_SIZE = struct.calcsize(_OUTGAUGE_FMT)          # 92
OUTGAUGE_SIZE_WITH_ID = OUTGAUGE_SIZE + 4               # 96


@dataclass(slots=True)
class OutGaugePacket:
    time_ms: int
    car: str
    flags: int
    gear: int
    player_id: int
    speed_ms: float
    rpm: float
    turbo_bar: float
    eng_temp_c: float
    fuel: float
    oil_pressure_bar: float
    oil_temp_c: float
    dash_lights: int
    show_lights: int
    throttle: float
    brake: float
    clutch: float
    display1: str
    display2: str
    packet_id: int | None = None

    _STRUCT: ClassVar[struct.Struct] = struct.Struct(_OUTGAUGE_FMT)

    @classmethod
    def parse(cls, data: bytes) -> "OutGaugePacket":
        if len(data) not in (OUTGAUGE_SIZE, OUTGAUGE_SIZE_WITH_ID):
            raise ValueError(f"unexpected OutGauge packet size: {len(data)}")
        unpacked = cls._STRUCT.unpack_from(data, 0)
        (
            t, car, flags, gear, plid,
            speed, rpm, turbo, eng_t, fuel, oilp, oilt,
            dash, show,
            thr, brk, clu,
            d1, d2,
        ) = unpacked
        pid = None
        if len(data) == OUTGAUGE_SIZE_WITH_ID:
            pid = struct.unpack_from("<i", data, OUTGAUGE_SIZE)[0]
        return cls(
            time_ms=t,
            car=decode_car_id(car),
            flags=flags,
            gear=gear,
            player_id=plid,
            speed_ms=speed,
            rpm=rpm,
            turbo_bar=turbo,
            eng_temp_c=eng_t,
            fuel=fuel,
            oil_pressure_bar=oilp,
            oil_temp_c=oilt,
            dash_lights=dash,
            show_lights=show,
            throttle=thr,
            brake=brk,
            clutch=clu,
            display1=d1.split(b"\x00", 1)[0].decode("latin-1", "replace"),
            display2=d2.split(b"\x00", 1)[0].decode("latin-1", "replace"),
            packet_id=pid,
        )


# ---------------------------------------------------------------------------
# InSim (minimal subset)
# ---------------------------------------------------------------------------

ISP_NONE = 0
ISP_ISI = 1     # InSim init (client → host)
ISP_VER = 2     # version info
ISP_TINY = 3    # generic 4-byte request/response
ISP_SMALL = 4
ISP_STA = 5     # state info (race/replay/spr_x)
ISP_RST = 17    # race start
ISP_NCN = 18    # new connection (host)
ISP_CNL = 19    # connection left
ISP_NPL = 21    # new player joining
ISP_PLP = 22    # player pits (telepit, slot kept)
ISP_PLL = 23    # player leaves (spectates, slot freed)
ISP_LAP = 24    # lap time
ISP_SPX = 25    # split time
ISP_PIT = 26    # pit stop start
ISP_PSF = 27    # pit stop finish
ISP_PLA = 28    # pit lane (enter/exit)
ISP_CCH = 29    # camera changed
ISP_PEN = 30    # penalty given / cleared
ISP_TOC = 31    # take over car (driver swap)
ISP_FLG = 32    # flag shown (yellow/blue)
ISP_PFL = 33    # player flags changed (TC/ABS/help toggles)
ISP_FIN = 34    # finished race (provisional)
ISP_RES = 35    # race result confirmed
ISP_NLP = 37    # node and lap info (compact, all players)
ISP_MCI = 38    # multi-car info (CompCar array)
ISP_MSL = 40    # message send local (display message in this LFS instance)
ISP_CON = 50    # contact between cars
ISP_OBH = 51    # object hit
ISP_HLV = 52    # hot lap validity (off-track / wall etc.)
ISP_SLC = 62    # connection selected a car
ISP_CSC = 63    # car state changed (start/stop engine)
ISP_CIM = 64    # connection interface mode (which screen the driver is on)
ISP_MAL = 65    # mods allowed on host

# Tiny SubTypes (values verified against LFS docs/InSim.txt)
TINY_NONE = 0
TINY_VER = 1
TINY_CLOSE = 2
TINY_PING = 3
TINY_REPLY = 4
TINY_VTC = 5
TINY_SCP = 6
TINY_SST = 7    # request: send IS_STA (state info)
TINY_GTM = 8
TINY_MPE = 9
TINY_ISM = 10
TINY_REN = 11
TINY_CLR = 12
TINY_NCN = 13
TINY_NPL = 14   # request all current players (gets a flood of IS_NPL)
TINY_RES = 15
TINY_NLP = 16
TINY_MCI = 17
TINY_REO = 18
TINY_RST = 19
TINY_AXI = 20
TINY_AXC = 21
TINY_RIP = 22
TINY_NCI = 23   # request IS_NCI for all guests (admin host only)
TINY_ALC = 24
TINY_AXM = 25
TINY_SLC = 26   # request IS_SLC for all connections
TINY_MAL = 27   # request IS_MAL listing the allowed mods

# Small SubTypes (4th byte of IS_SMALL packets)
SMALL_NONE = 0
SMALL_SSP = 1
SMALL_SSG = 2
SMALL_VTA = 3   # vote action (race end/restart/qualify)
SMALL_TMS = 4
SMALL_STP = 5
SMALL_RTP = 6
SMALL_NLI = 7
SMALL_ALC = 8
SMALL_LCS = 9
SMALL_LCL = 10
SMALL_AII = 11

# Vote actions (carried in SMALL_VTA UVal)
VOTE_NONE = 0
VOTE_END = 1
VOTE_RESTART = 2
VOTE_QUALIFY = 3

# Pit lane facts (IS_PLA.fact)
PITLANE_EXIT = 0
PITLANE_ENTER = 1
PITLANE_NO_PURPOSE = 2
PITLANE_DT = 3      # drive-through
PITLANE_SG = 4      # stop-go

# ISI flags (subset)
ISF_LOCAL = 0x0004
ISF_CON = 0x0040    # Send IS_CON (car-to-car contact)
ISF_OBH = 0x0080    # Send IS_OBH (object hit)
ISF_NLP = 0x0010    # Send IS_NLP packets
ISF_MCI = 0x0020    # Send IS_MCI packets
ISF_HLV = 0x0100    # Send IS_HLV packets
ISF_AXM_LOAD = 0x0200
ISF_AXM_EDIT = 0x0400
ISF_REQ_JOIN = 0x0800

# IS_NPL SetF flags
SETF_SYMM_WHEELS = 1
SETF_TC_ENABLE = 2
SETF_ABS_ENABLE = 4

# IS_HLV HLVC reasons
HLVC_GROUND = 0
HLVC_WALL = 1
HLVC_SPEEDING = 4
HLVC_OUT_OF_BOUNDS = 5

_HLVC_NAMES = {
    HLVC_GROUND: "ground",
    HLVC_WALL: "wall",
    HLVC_SPEEDING: "speeding",
    HLVC_OUT_OF_BOUNDS: "out_of_bounds",
}


def hlvc_name(code: int) -> str:
    return _HLVC_NAMES.get(code, f"hlvc_{code}")


# OutGauge dash_lights / show_lights bits (DL_*).
#
# Source: InSim.txt v10 (LFS 0.7G) ``Dashboard Lights`` table. Bit 11 used
# to be reserved (we previously called it ``DL_SPARE``) but LFS now uses
# it for ENGINE damage — ``DLF_ENGINE_SEVERE`` (0x1000_0000) is the high
# bit that escalates the warning.
DL_SHIFT = 0x00000001
DL_FULLBEAM = 0x00000002
DL_HANDBRAKE = 0x00000004
DL_PITSPEED = 0x00000008
DL_TC = 0x00000010
DL_SIGNAL_L = 0x00000020
DL_SIGNAL_R = 0x00000040
DL_SIGNAL_ANY = 0x00000080
DL_OILWARN = 0x00000100
DL_BATTERY = 0x00000200
DL_ABS = 0x00000400
DL_ENGINE = 0x00000800   # engine damage warning
DL_FOG_REAR = 0x00001000
DL_FOG_FRONT = 0x00002000
DL_DIPPED = 0x00004000
DL_FUELWARN = 0x00008000
DL_SIDELIGHTS = 0x00010000
DL_NEUTRAL = 0x00020000
DLF_ENGINE_SEVERE = 0x10000000  # set together with DL_ENGINE for severe damage

_DL_NAMES: tuple[tuple[int, str], ...] = (
    (DL_SHIFT,        "shift"),
    (DL_FULLBEAM,     "fullbeam"),
    (DL_HANDBRAKE,    "handbrake"),
    (DL_PITSPEED,     "pit_limiter"),
    (DL_TC,           "tc"),
    (DL_SIGNAL_L,     "signal_l"),
    (DL_SIGNAL_R,     "signal_r"),
    (DL_SIGNAL_ANY,   "signal_any"),
    (DL_OILWARN,      "oil_warn"),
    (DL_BATTERY,      "battery"),
    (DL_ABS,          "abs"),
    (DL_ENGINE,       "engine"),
    (DL_FOG_REAR,     "fog_rear"),
    (DL_FOG_FRONT,    "fog_front"),
    (DL_DIPPED,       "dipped"),
    (DL_FUELWARN,     "fuel_warn"),
    (DL_SIDELIGHTS,   "sidelights"),
    (DL_NEUTRAL,      "neutral"),
    (DLF_ENGINE_SEVERE, "engine_severe"),
)


def decode_dash_lights(bits: int) -> list[str]:
    """Return active OutGauge dash light names for ``bits`` (DL_*)."""
    return [name for mask, name in _DL_NAMES if bits & mask]


# IS_OBH OBHFlags bits.
OBH_LAYOUT_OBJECT = 0x01    # an autocross layout object
OBH_CAN_MOVE = 0x02    # the object can be moved
OBH_WAS_MOVING = 0x04    # was moving when hit
OBH_ON_SPOT = 0x08    # was at its original spot


# IS_MCI CCI_* (CompCar Info bits).
CCI_BLUE = 0x01
CCI_YELLOW = 0x02
CCI_OOB = 0x04   # out-of-bounds (off track / spinning)
CCI_RETIRED = 0x08   # retired from the race
CCI_LAG = 0x20
CCI_FIRST = 0x40
CCI_LAST = 0x80


# ---------------------------------------------------------------------------
# IS_PIT.Work bits (PSE_*) — what was changed during the pit stop.
# ---------------------------------------------------------------------------
PSE_NOTHING = 0x0000
PSE_STOP = 0x0001
PSE_FR_DAM = 0x0002
PSE_FR_WHL = 0x0004
PSE_LE_FR_DAM = 0x0008
PSE_LE_FR_WHL = 0x0010
PSE_RI_FR_DAM = 0x0020
PSE_RI_FR_WHL = 0x0040
PSE_LE_RE_DAM = 0x0080
PSE_LE_RE_WHL = 0x0100
PSE_RI_RE_DAM = 0x0200
PSE_RI_RE_WHL = 0x0400
PSE_BODY_MINOR = 0x0800
PSE_BODY_MAJOR = 0x1000
PSE_SETUP = 0x2000
PSE_REFUEL = 0x4000

_PSE_NAMES: tuple[tuple[int, str], ...] = (
    (PSE_STOP,       "stop"),
    (PSE_FR_DAM,     "front_damage"),
    (PSE_FR_WHL,     "front_wheels"),
    (PSE_LE_FR_DAM,  "front_left_damage"),
    (PSE_LE_FR_WHL,  "front_left_wheel"),
    (PSE_RI_FR_DAM,  "front_right_damage"),
    (PSE_RI_FR_WHL,  "front_right_wheel"),
    (PSE_LE_RE_DAM,  "rear_left_damage"),
    (PSE_LE_RE_WHL,  "rear_left_wheel"),
    (PSE_RI_RE_DAM,  "rear_right_damage"),
    (PSE_RI_RE_WHL,  "rear_right_wheel"),
    (PSE_BODY_MINOR, "body_minor"),
    (PSE_BODY_MAJOR, "body_major"),
    (PSE_SETUP,      "setup"),
    (PSE_REFUEL,     "refuel"),
)


def decode_pit_work(bits: int) -> list[str]:
    """Return active pit-work names for an ``IS_PIT.work`` bitfield."""
    return [name for mask, name in _PSE_NAMES if bits & mask]


# ---------------------------------------------------------------------------
# IS_RST.Flags / host options (HOSTF_*) — race rules in force.
# ---------------------------------------------------------------------------
HOSTF_CAN_VOTE = 0x01
HOSTF_CAN_SELECT = 0x02
HOSTF_MID_RACE_JOIN = 0x04
HOSTF_MUST_PIT = 0x08
HOSTF_CAN_RESET = 0x10
HOSTF_FCV = 0x20   # force cockpit view
HOSTF_CRUISE = 0x40

_HOSTF_NAMES: tuple[tuple[int, str], ...] = (
    (HOSTF_CAN_VOTE,      "can_vote"),
    (HOSTF_CAN_SELECT,    "can_select"),
    (HOSTF_MID_RACE_JOIN, "mid_race_join"),
    (HOSTF_MUST_PIT,      "must_pit"),
    (HOSTF_CAN_RESET,     "can_reset"),
    (HOSTF_FCV,           "force_cockpit_view"),
    (HOSTF_CRUISE,        "cruise"),
)


def decode_host_flags(bits: int) -> list[str]:
    """Return active host-rule names for an ``IS_RST.flags`` bitfield."""
    return [name for mask, name in _HOSTF_NAMES if bits & mask]


# ---------------------------------------------------------------------------
# IS_RST.RaceLaps encoding (InSim.txt ``RaceLaps`` paragraph).
# ---------------------------------------------------------------------------
def race_laps_kind(race_laps: int) -> tuple[str, int | None]:
    """Decode the ``IS_STA.race_laps`` / ``IS_RST.race_laps`` byte.

    Returns one of:
      * ``("practice", None)``  — 0 means practice / no race scheduled.
      * ``("laps", n)``         — 1–99 = literal lap count.
      * ``("laps", n)``         — 100–190 = (rl-100)*10 + 100 laps.
      * ``("hours", h)``        — 191–238 = (rl-190) hours.
      * ``("no_timing", None)`` — 255 means timing disabled.
      * ``("unknown", None)``   — anything else.
    """
    if race_laps == 0:
        return ("practice", None)
    if 1 <= race_laps <= 99:
        return ("laps", race_laps)
    if 100 <= race_laps <= 190:
        return ("laps", (race_laps - 100) * 10 + 100)
    if 191 <= race_laps <= 238:
        return ("hours", race_laps - 190)
    if race_laps == 255:
        return ("no_timing", None)
    return ("unknown", None)


# ---------------------------------------------------------------------------
# IS_FLG / IS_PEN / IS_CIM enums.
# ---------------------------------------------------------------------------
# Flag values shown in IS_FLG.flag
FLG_BLUE = 1
FLG_YELLOW = 2

# IS_PEN penalty codes (NewPen / OldPen).
PENALTY_NONE = 0
PENALTY_DT = 1   # drive-through
PENALTY_DT_VALID = 2
PENALTY_SG = 3   # stop-go
PENALTY_SG_VALID = 4
PENALTY_30 = 5   # +30 s
PENALTY_45 = 6   # +45 s

_PENALTY_NAMES = {
    PENALTY_NONE: "none",
    PENALTY_DT: "drive_through",
    PENALTY_DT_VALID: "drive_through_valid",
    PENALTY_SG: "stop_go",
    PENALTY_SG_VALID: "stop_go_valid",
    PENALTY_30: "plus_30s",
    PENALTY_45: "plus_45s",
}


def penalty_name(code: int) -> str:
    return _PENALTY_NAMES.get(code, f"penalty_{code}")


# IS_PEN reason (PENR_*).
PENR_UNKNOWN = 0
PENR_ADMIN = 1
PENR_WRONG_WAY = 2
PENR_FALSE_START = 3
PENR_SPEEDING = 4
PENR_STOP_SHORT = 5
PENR_STOP_LATE = 6

_PENR_NAMES = {
    PENR_UNKNOWN:     "unknown",
    PENR_ADMIN:       "admin",
    PENR_WRONG_WAY:   "wrong_way",
    PENR_FALSE_START: "false_start",
    PENR_SPEEDING:    "pit_speeding",
    PENR_STOP_SHORT:  "stop_short",
    PENR_STOP_LATE:   "stop_late",
}


def penalty_reason_name(code: int) -> str:
    return _PENR_NAMES.get(code, f"reason_{code}")


# IS_PFL player flags (subset of IS_NPL PIF_*).
PIF_SWAPSIDE = 0x0001
PIF_RESERVED_2 = 0x0002
PIF_RESERVED_4 = 0x0004
PIF_AUTOGEARS = 0x0008
PIF_SHIFTER = 0x0010
PIF_RESERVED_20 = 0x0020
PIF_HELP_B = 0x0040   # brake help
PIF_AXIS_CLUTCH = 0x0080
PIF_INPITS = 0x0100
PIF_AUTOCLUTCH = 0x0200
PIF_MOUSE = 0x0400
PIF_KB_NO_HELP = 0x0800
PIF_KB_STABILISED = 0x1000
PIF_CUSTOM_VIEW = 0x2000


# IS_CIM modes.
CIM_NORMAL = 0
CIM_OPTIONS = 1
CIM_HOST_OPTS = 2
CIM_GARAGE = 3
CIM_CAR_SELECT = 4
CIM_TRACK_SELECT = 5
CIM_SHIFTU = 6   # SHIFT+U mode (free-look / autocross editor)

_CIM_NAMES = {
    CIM_NORMAL:       "normal",
    CIM_OPTIONS:      "options",
    CIM_HOST_OPTS:    "host_options",
    CIM_GARAGE:       "garage",
    CIM_CAR_SELECT:   "car_select",
    CIM_TRACK_SELECT: "track_select",
    CIM_SHIFTU:       "shift_u",
}


def cim_mode_name(code: int) -> str:
    return _CIM_NAMES.get(code, f"cim_{code}")


# IS_CIM garage submodes (when mode == CIM_GARAGE) — useful for context-aware
# overlays: NRM_WHEEL_TEMPS means the driver is staring at the F9 panel etc.
GRG_INFO = 0
GRG_COLOURS = 1
GRG_BRAKE_TC = 2
GRG_SUSPENSION = 3
GRG_STEER_SUSPENSION = 4
GRG_DRIVE = 5
GRG_GEARS = 6
GRG_TYRES = 7
GRG_PASSENGERS = 8


# IS_CSC action.
CSC_STOP = 0
CSC_START = 1


# Wheel order used by every per-wheel LFS data structure.
WHEEL_ORDER: tuple[str, ...] = ("RL", "RR", "FL", "FR")


def build_isi_packet(
    *,
    udp_port: int = 0,
    flags: int = ISF_LOCAL,
    prefix: str = "!",
    interval_ms: int = 0,
    admin_password: str = "",
    iname: str = "lfs-telemetry",
    insim_ver: int = 10,
) -> bytes:
    """Build an IS_ISI (InSim init) packet (44 bytes).

    Layout::

        byte  Size       (44)
        byte  Type       (ISP_ISI = 1)
        byte  ReqI       (0)
        byte  Zero       (0)
        word  UDPPort
        word  Flags
        byte  InSimVer   (10 for LFS 0.7G+; backward-compatible with 0.7E)
        byte  Prefix
        word  Interval
        char  Admin[16]
        char  IName[16]
    """
    # InSim v9+ encodes the Size field as bytes/4 (so 44 -> 11).
    pkt = struct.pack(
        "<BBBBHHBBH16s16s",
        44 // 4,
        ISP_ISI,
        0,
        0,
        udp_port,
        flags,
        int(insim_ver) & 0xFF,
        ord(prefix[0]) if prefix else 0,
        interval_ms,
        admin_password.encode("latin-1")[:16].ljust(16, b"\x00"),
        iname.encode("latin-1")[:16].ljust(16, b"\x00"),
    )
    return pkt


@dataclass(slots=True)
class InSimHeader:
    size: int
    type: int
    req_i: int
    data: bytes  # raw payload after the 4-byte header

    @classmethod
    def parse(cls, data: bytes) -> "InSimHeader":
        if len(data) < 4:
            raise ValueError("short InSim packet")
        size, ptype, reqi, _zero = struct.unpack_from("<BBBB", data, 0)
        return cls(size=size, type=ptype, req_i=reqi, data=data[4:size])


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
    def parse(cls, data: bytes, opts: int) -> "OutSimPack2":
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


# ---------------------------------------------------------------------------
# InSim packet payload parsers
# ---------------------------------------------------------------------------
#
# Conventions (from InSim.txt, version 9, used by LFS 0.7E+):
#
# * Every InSim packet starts with: byte Size, byte Type, byte ReqI, byte X.
# * The 4th header byte (``X``) is **packet-specific** — for many event
#   packets it carries the PLID (player id); for handshake packets it is a
#   reserved zero. We therefore keep raw header parsing in :class:`InSimHeader`
#   but each IS_xxx dataclass below is parsed from the **full packet bytes**
#   (offset 0 = Size byte) so it can recover that 4th byte itself.
# * Since v9, the ``Size`` byte is the packet size in **bytes** (previous
#   versions stored size/4). All sizes documented below are in bytes.


def _cstr(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("latin-1", "replace")


# ---------------------------------------------------------------------------
# Car-ID decoder shared by OutGauge.Car[4], IS_NPL.CName[4], IS_SLC.CName[4],
# IS_RES.CName[4] and the 32-bit SkinIDs in IS_MAL.
# ---------------------------------------------------------------------------
#
# LFS InSim spec (0.7F):
#   * The 4-byte ``CName`` / ``Car`` fields hold either:
#       - a stock-car short name (1-3 ASCII letters/digits + NUL pad), or
#       - a mod ``SkinID`` in "compressed format" (3 bytes ID + NUL pad,
#         little-endian, matching the unsigned 32-bit SkinIDs sent in
#         IS_MAL).
#   * The canonical text form for a mod ID is 6 lowercase hex chars
#     (the format used by Detect&Monitor's ``cars/mod_sizes.car`` and by
#     the URLs at ``lfs.net/files/vehmods/<6HEX>``).
#
# Decoding naively as ``raw.split(b"\x00",1)[0].decode("latin-1")`` (as the
# old ``_cstr`` did for these fields) yielded garbage like ``\xfe\xf1V`` for
# mods, which then propagated into capture filenames and broke
# :class:`CarSpecStore` lookups. ``decode_car_id`` handles both cases
# uniformly.
_STOCK_CARS: frozenset[str] = frozenset({
    "UF1", "XFG", "XRG", "LX4", "LX6", "RB4", "FXO", "XRT", "RAC",
    "FZ5", "UFR", "XFR", "FXR", "XRR", "FZR", "MRT", "FBM", "FOX",
    "FO8", "BF1",
})


def decode_car_id(raw: bytes) -> str:
    """Decode a 4-byte LFS car identifier.

    * Stock car (e.g. ``b"FBM\\x00"``) → ASCII short name in upper case.
    * Mod SkinID (e.g. ``b"\\xfe\\xf1\\x56\\x00"``) → 6-char lowercase hex
      string formed from the little-endian 32-bit value (→ ``"56f1fe"``).
    * Empty / malformed input → ``""``.
    """
    if not raw:
        return ""
    raw4 = raw[:4].ljust(4, b"\x00")
    head = raw4.rstrip(b"\x00")
    if head:
        try:
            ascii_name = head.decode("ascii")
        except UnicodeDecodeError:
            ascii_name = ""
        if ascii_name and ascii_name.upper() in _STOCK_CARS:
            return ascii_name.upper()
    u32 = int.from_bytes(raw4, "little")
    if u32 == 0:
        return ""
    return f"{u32:06x}"


@dataclass(slots=True)
class InSimVersion:
    """IS_VER — server replies with this after IS_ISI handshake. Size = 20."""

    req_i: int
    version: str        # LFS version e.g. "0.7E"
    product: str        # "DEMO" / "S1" / "S2" / "S3"
    insim_ver: int

    @classmethod
    def parse(cls, data: bytes) -> "InSimVersion":
        # offsets 0-3: Size, Type, ReqI, Zero
        # offset 4: char Version[8]
        # offset 12: char Product[6]
        # offset 18: byte InSimVer
        # offset 19: byte Spare
        reqi = data[2]
        ver, prod, insim_ver, _sp = struct.unpack_from("<8s6sBB", data, 4)
        return cls(req_i=reqi, version=_cstr(ver), product=_cstr(prod),
                   insim_ver=insim_ver)


@dataclass(slots=True)
class InSimState:
    """IS_STA — overall race/replay state. Size = 28."""

    replay_speed: float       # 1.0 = normal
    flags: int                # ISS_* bits
    in_game_cam: int
    view_plid: int
    num_players: int
    num_connections: int
    num_finished: int
    race_in_progress: int     # 0=none, 1=race, 2=qualifying
    qual_minutes: int
    race_laps: int
    track: str                # 6 chars (e.g. "BL1", "AS5R")
    weather: int
    wind: int

    @classmethod
    def parse(cls, data: bytes) -> "InSimState":
        # Payload at offset 4: float, word, 8 bytes, 6s, 2 bytes
        (rs, fl, cam, vplid, np_, nc, nf, rip, qm, rl,
         _sp2, _sp3, track, wx, wd) = struct.unpack_from(
            "<fHBBBBBBBB BB 6s BB", data, 4)
        return cls(replay_speed=rs, flags=fl, in_game_cam=cam, view_plid=vplid,
                   num_players=np_, num_connections=nc, num_finished=nf,
                   race_in_progress=rip, qual_minutes=qm, race_laps=rl,
                   track=_cstr(track), weather=wx, wind=wd)


@dataclass(slots=True)
class InSimRaceStart:
    """IS_RST — race start info. Size = 28."""

    req_i: int
    race_laps: int
    qual_minutes: int
    num_players: int
    timing: int
    track: str
    weather: int
    wind: int
    flags: int                # RACE_* bits
    num_nodes: int
    finish_node: int
    split1_node: int
    split2_node: int
    split3_node: int

    @classmethod
    def parse(cls, data: bytes) -> "InSimRaceStart":
        reqi = data[2]
        # Payload at offset 4 (total packet size = 28 bytes):
        # RaceLaps, QualMins, NumP, Timing,
        # Track[6], Weather, Wind,
        # Flags(word), NumNodes(word), Finish(word),
        # Split1(word), Split2(word), Split3(word).
        (rl, qm, np_, tm,
         track, wx, wd, fl,
         nn, fn, s1, s2, s3) = struct.unpack_from(
            "<BBBB6sBBHHHHHH", data, 4)
        return cls(req_i=reqi, race_laps=rl, qual_minutes=qm, num_players=np_,
                   timing=tm, track=_cstr(track), weather=wx, wind=wd,
                   flags=fl, num_nodes=nn, finish_node=fn,
                   split1_node=s1, split2_node=s2, split3_node=s3)


@dataclass(slots=True)
class InSimNewPlayer:
    """IS_NPL — new player joining race. Size = 76.

    Note: the trailing player-handicap fields (RWAdj, FWAdj) are exposed
    as raw bytes since their byte layout is unstable across LFS revisions.
    """

    player_id: int
    connection_id: int
    player_type: int
    flags: int                # PIF_* bits
    player_name: str
    plate: str
    car_name: str             # short id e.g. "FOX", "FO8", "BF1"
    skin_name: str
    tyres: tuple[int, int, int, int]   # compounds in WHEEL_ORDER (RL, RR, FL, FR)
    handicap_mass_kg: int
    handicap_t_res: int       # tyre intake restriction %
    model: int
    passengers: int           # bitfield
    rear_wheel_adjust: int    # RWAdj (raw byte)
    front_wheel_adjust: int   # FWAdj (raw byte)
    set_flags: int            # SETF_*
    num_in_race: int
    config: int
    fuel_pct: int             # 0..100

    @classmethod
    def parse(cls, data: bytes) -> "InSimNewPlayer":
        plid = data[3]
        # Payload at offset 4: UCID, PType, Flags(word),
        # PName[24], Plate[8], CName[4], SName[16],
        # Tyres[4], H_Mass, H_TRes, Model, Pass,
        # RWAdj, FWAdj, Sp[2], SetF, NumP, Config, Fuel.
        (ucid, ptype, flags,
         pname, plate, cname, sname,
         t1, t2, t3, t4,
         hm, htr, model, pass_,
         rwadj, fwadj, _s1, _s2,
         setf, nump, config, fuel) = struct.unpack_from(
            "<BBH 24s 8s 4s 16s 4B 4B 4B 4B", data, 4)
        return cls(player_id=plid, connection_id=ucid, player_type=ptype,
                   flags=flags,
                   player_name=_cstr(pname), plate=_cstr(plate),
                   car_name=decode_car_id(cname), skin_name=_cstr(sname),
                   tyres=(t1, t2, t3, t4),
                   handicap_mass_kg=hm, handicap_t_res=htr,
                   model=model, passengers=pass_,
                   rear_wheel_adjust=rwadj, front_wheel_adjust=fwadj,
                   set_flags=setf, num_in_race=nump,
                   config=config, fuel_pct=fuel)


# Scale factor to convert IS_LAP / IS_SPX ``fuel200`` (0..200) into a
# percentage (0..100). Use as: ``fuel_pct = fuel200 / FUEL_SCALE``.
FUEL_SCALE = 2.0


@dataclass(slots=True)
class InSimLap:
    """IS_LAP — lap time. Size = 20."""

    player_id: int
    laps_done: int
    lap_time_ms: int
    elapsed_time_ms: int
    flags: int
    penalty: int
    num_stops: int
    fuel200: int              # fuel * 2 (0..200) — divide by FUEL_SCALE for %

    @classmethod
    def parse(cls, data: bytes) -> "InSimLap":
        plid = data[3]
        # Payload at offset 4: LTime u32, ETime u32, LapsDone u16, Flags u16,
        # Sp0, Penalty, NumStops, Fuel200 (4 bytes).
        (ltime, etime, lapsdone, flags,
         _sp0, pen, stops, fuel200) = struct.unpack_from("<IIHH BBBB", data, 4)
        return cls(player_id=plid, laps_done=lapsdone,
                   lap_time_ms=ltime, elapsed_time_ms=etime,
                   flags=flags, penalty=pen, num_stops=stops, fuel200=fuel200)


@dataclass(slots=True)
class InSimSplit:
    """IS_SPX — split time. Size = 16."""

    player_id: int
    split_time_ms: int
    elapsed_time_ms: int
    split: int                # 1, 2 or 3
    penalty: int
    num_stops: int
    fuel200: int

    @classmethod
    def parse(cls, data: bytes) -> "InSimSplit":
        plid = data[3]
        (stime, etime, split, pen, stops, fuel200) = struct.unpack_from(
            "<II BBBB", data, 4)
        return cls(player_id=plid, split_time_ms=stime, elapsed_time_ms=etime,
                   split=split, penalty=pen, num_stops=stops, fuel200=fuel200)


@dataclass(slots=True)
class InSimPit:
    """IS_PIT — pit-stop start. Size = 24."""

    player_id: int
    laps_done: int
    flags: int                # PIF_* bits
    fuel_add: int             # 0..255 (0 = no fuel added)
    penalty: int
    num_stops: int
    tyres: tuple[int, int, int, int]
    work: int                 # PSE_* bitfield (what was changed)

    @classmethod
    def parse(cls, data: bytes) -> "InSimPit":
        plid = data[3]
        # Payload: LapsDone(word), Flags(word), FuelAdd(byte), Penalty(byte),
        # NumStops(byte), Sp3(byte), Tyres[4], Work(uint), Spare(uint).
        (lapsdone, flags,
         fuel_add, pen, stops, _sp3,
         t1, t2, t3, t4,
         work, _spare) = struct.unpack_from("<HH BBBB 4B II", data, 4)
        return cls(player_id=plid, laps_done=lapsdone, flags=flags,
                   fuel_add=fuel_add, penalty=pen, num_stops=stops,
                   tyres=(t1, t2, t3, t4), work=work)


@dataclass(slots=True)
class InSimPitStopFinish:
    """IS_PSF — pit-stop finish. Size = 12."""

    player_id: int
    stop_time_ms: int

    @classmethod
    def parse(cls, data: bytes) -> "InSimPitStopFinish":
        plid = data[3]
        # Payload: STime(u32), Spare(u32) = 8 bytes (12 - 4).
        (stime, _spare) = struct.unpack_from("<II", data, 4)
        return cls(player_id=plid, stop_time_ms=stime)


@dataclass(slots=True)
class InSimHotLapValid:
    """IS_HLV — hot-lap validity broken. Size = 16.

    Layout (post v9):
      Size, Type, ReqI, PLID,                         (4 bytes header)
      HLVC, Sp1, Time(word),                          (4 bytes)
      CarContOBJ C (8 bytes): Direction byte, Heading byte,
                              Speed byte, Zbyte, X short, Y short.

    ``Time`` is a looping 16-bit counter in 10-ms ticks.
    """

    player_id: int
    hlvc: int                 # HLVC_GROUND / WALL / SPEEDING / OUT_OF_BOUNDS
    time_ms: int              # looping; multiply word by 10 to get ms
    car_speed_ms: float       # Speed byte from CarContOBJ (1 unit = 1 m/s)
    car_direction_rad: float  # Direction byte (256 = 360°)
    car_heading_rad: float    # Heading byte
    car_x_m: float            # X short / 16
    car_y_m: float            # Y short / 16

    @classmethod
    def parse(cls, data: bytes) -> "InSimHotLapValid":
        plid = data[3]
        # Offset 4: HLVC, Sp1, Time(word). Then CarContOBJ at offset 8.
        (hlvc, _sp1, time_w,
         direction, heading, speed, _zbyte,
         x_s, y_s) = struct.unpack_from("<BBH BBBB hh", data, 4)
        return cls(
            player_id=plid, hlvc=hlvc,
            time_ms=time_w * 10,
            car_speed_ms=float(speed),
            car_direction_rad=direction * (math.tau / 256.0),
            car_heading_rad=heading * (math.tau / 256.0),
            car_x_m=x_s / 16.0,
            car_y_m=y_s / 16.0,
        )


@dataclass(slots=True)
class CompCar:
    """One car entry inside an IS_MCI packet (28 B)."""

    node: int
    lap: int
    player_id: int
    position: int
    info: int                 # CCI_* bits (BLUE=1, YELLOW=2, OOB=4, LAG=32)
    x_m: float
    y_m: float
    z_m: float
    speed_ms: float
    direction_rad: float
    heading_rad: float
    ang_vel_rads: float

    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<HHBBBB iii HHHh")

    @classmethod
    def parse(cls, data: bytes, off: int) -> "CompCar":
        (node, lap, plid, pos, info, _sp,
         x, y, z, speed, direction, heading, ang_vel) = cls._STRUCT.unpack_from(
            data, off)
        return cls(
            node=node, lap=lap, player_id=plid, position=pos, info=info,
            x_m=x / 65536.0, y_m=y / 65536.0, z_m=z / 65536.0,
            speed_ms=speed * 100.0 / 32768.0,
            direction_rad=direction * (math.tau / 65536.0),
            heading_rad=heading * (math.tau / 65536.0),
            ang_vel_rads=ang_vel * (math.tau / 16384.0),
        )


@dataclass(slots=True)
class InSimMCI:
    """IS_MCI — multi-car info. Variable size (4 + 4 + N*28)."""

    cars: list[CompCar]

    @classmethod
    def parse(cls, data: bytes) -> "InSimMCI":
        # InSim header (4 B): Size, Type, ReqI, NumC.
        # CompCar array (NumC × 28 B) starts immediately at offset 4.
        # Total packet size = 4 + NumC * 28.
        numc = data[3]
        cars = [CompCar.parse(data, 4 + i * 28) for i in range(numc)]
        return cls(cars=cars)


@dataclass(slots=True)
class NodeLap:
    """Per-player entry inside an IS_NLP packet (6 B)."""

    node: int
    lap: int
    player_id: int
    position: int

    @classmethod
    def parse(cls, data: bytes, off: int) -> "NodeLap":
        node, lap, plid, pos = struct.unpack_from("<HHBB", data, off)
        return cls(node=node, lap=lap, player_id=plid, position=pos)


@dataclass(slots=True)
class InSimNodeLap:
    """IS_NLP — compact node/lap snapshot for every car. Variable size."""

    entries: list[NodeLap]

    @classmethod
    def parse(cls, data: bytes) -> "InSimNodeLap":
        # Layout: Size, Type, ReqI, NumP, then NumP × NodeLap(6).
        nump = data[3]
        return cls(entries=[NodeLap.parse(data, 4 + i * 6) for i in range(nump)])


@dataclass(slots=True)
class InSimObjectHit:
    """IS_OBH — object hit (autocross object, kerb, wall mesh). Size = 24."""

    player_id: int
    closing_speed_ms: float    # SpClose decoded as m/s (low 12 bits / 10.0)
    time_ms: int               # 10 ms ticks (looping)
    contact_direction_rad: float
    contact_heading_rad: float
    contact_speed_ms: int
    contact_x_m: float
    contact_y_m: float
    map_x_m: float
    map_y_m: float
    map_z_m: float
    object_index: int
    flags: int

    @classmethod
    def parse(cls, data: bytes) -> "InSimObjectHit":
        plid = data[3]
        # Payload at offset 4: SpClose(word), Time(word),
        # CarContOBJ(8 B): Direction, Heading, Speed, Zbyte, X(short), Y(short)
        # then map X(word), Y(word), Zbyte(byte), Sp1(byte), Index(byte), OBHFlags(byte).
        (sp_close, time_w,
         cdir, chead, cspeed, _czb, cx, cy,
         mx, my, mz, _sp1, idx, flags) = struct.unpack_from(
            "<HH BBBB hh HH BBBB", data, 4)
        return cls(
            player_id=plid,
            # InSim.txt: SpClose low 12 bits, 10 units == 1 m/s.
            closing_speed_ms=(sp_close & 0x0FFF) / 10.0,
            time_ms=time_w * 10,
            contact_direction_rad=cdir * (math.tau / 256.0),
            contact_heading_rad=chead * (math.tau / 256.0),
            contact_speed_ms=cspeed,
            contact_x_m=cx / 16.0,
            contact_y_m=cy / 16.0,
            map_x_m=mx / 16.0,
            map_y_m=my / 16.0,
            map_z_m=mz / 4.0,
            object_index=idx,
            flags=flags,
        )


@dataclass(slots=True)
class InSimNewConnection:
    """IS_NCN — new connection on the host. Size = 56."""

    connection_id: int          # UCID (0 = host)
    user_name: str              # LFS account name (ASCII)
    player_name: str            # nickname (may contain LFS colour codes)
    admin: int                  # 1 if admin
    total: int                  # total connections incl. host
    flags: int                  # bit 2 = remote

    @classmethod
    def parse(cls, data: bytes) -> "InSimNewConnection":
        ucid = data[3]
        # Payload at offset 4: UName[24], PName[24], Admin, Total, Flags, Sp3.
        uname, pname, admin, total, flags, _sp3 = struct.unpack_from(
            "<24s24sBBBB", data, 4)
        return cls(connection_id=ucid,
                   user_name=_cstr(uname), player_name=_cstr(pname),
                   admin=admin, total=total, flags=flags)


@dataclass(slots=True)
class InSimConnectionLeft:
    """IS_CNL — a connection left the host. Size = 8."""

    connection_id: int
    reason: int                 # LEAVR_* (timeout, kicked, banned, ...)
    total: int                  # remaining connections incl. host

    @classmethod
    def parse(cls, data: bytes) -> "InSimConnectionLeft":
        ucid = data[3]
        reason, total, _sp2, _sp3 = struct.unpack_from("<BBBB", data, 4)
        return cls(connection_id=ucid, reason=reason, total=total)


@dataclass(slots=True)
class InSimSelectedCar:
    """IS_SLC — a connection selected a car (empty if no car). Size = 8.

    ``car_name`` is decoded with :func:`decode_car_id` so mods come through
    as 6-char lowercase hex instead of garbled latin-1 bytes.
    """

    connection_id: int          # UCID (0 = host)
    car_name: str               # stock short-name or 6-hex mod ID (empty if none)

    @classmethod
    def parse(cls, data: bytes) -> "InSimSelectedCar":
        ucid = data[3]
        cname = data[4:8]
        return cls(connection_id=ucid, car_name=decode_car_id(cname))


@dataclass(slots=True)
class InSimPitLane:
    """IS_PLA — pit-lane enter / exit notification. Size = 8."""

    player_id: int
    fact: int                   # PITLANE_EXIT / ENTER / NO_PURPOSE / DT / SG

    @classmethod
    def parse(cls, data: bytes) -> "InSimPitLane":
        plid = data[3]
        fact, _sp1, _sp2, _sp3 = struct.unpack_from("<BBBB", data, 4)
        return cls(player_id=plid, fact=fact)


@dataclass(slots=True)
class InSimModsAllowed:
    """IS_MAL — list of mods allowed on the host. Variable size.

    Per InSim.txt, ``SkinID`` is a 32-bit unsigned in "compressed format" —
    we surface it as the canonical 6-char lowercase hex string so it can
    be compared directly to the mod IDs we get from IS_SLC / IS_NPL / OutGauge.

    ``num_mods == 0`` means "all mods allowed" (the cleared / unrestricted
    state).
    """

    connection_id: int          # UCID of the connection that updated the list
    flags: int
    mod_ids: tuple[str, ...]    # 6-hex lowercase, one per mod

    @classmethod
    def parse(cls, data: bytes) -> "InSimModsAllowed":
        # InSim header: Size, Type, ReqI, NumM. Then UCID, Flags, Sp2, Sp3.
        # Total packet size = 8 + NumM * 4.
        num_mods = data[3]
        ucid, flags, _sp2, _sp3 = struct.unpack_from("<BBBB", data, 4)
        ids: list[str] = []
        for i in range(num_mods):
            raw = data[8 + i * 4:12 + i * 4]
            u32 = int.from_bytes(raw, "little")
            ids.append(f"{u32:06x}")
        return cls(connection_id=ucid, flags=flags, mod_ids=tuple(ids))


@dataclass(slots=True)
class InSimSmall:
    """Generic IS_SMALL packet (8 B): SubT + UVal."""

    sub_t: int
    u_val: int

    @classmethod
    def parse(cls, data: bytes) -> "InSimSmall":
        sub_t = data[3]
        (u_val,) = struct.unpack_from("<I", data, 4)
        # Specialise on subtype where we have a richer model.
        if sub_t == SMALL_VTA:
            return InSimVoteAction(sub_t=sub_t, u_val=u_val, action=u_val)
        return cls(sub_t=sub_t, u_val=u_val)


@dataclass(slots=True)
class InSimVoteAction(InSimSmall):
    """SMALL_VTA — vote action carried inside IS_SMALL (race end/restart/qualify)."""

    action: int = 0             # VOTE_END / VOTE_RESTART / VOTE_QUALIFY


def build_tiny_packet(sub_t: int, req_i: int = 0) -> bytes:
    """Build an IS_TINY packet (4 B). Used for keepalive (TINY_NONE) and queries.

    InSim v9+ encodes Size as bytes/4 (so 4 -> 1).
    """
    return struct.pack("<BBBB", 1, ISP_TINY, req_i, sub_t)


# ---------------------------------------------------------------------------
# Race-event packets added in this revision (CON, FIN, RES, TOC, PEN, FLG,
# PFL, PLP, PLL, CCH, CIM, CSC). All sizes verified against InSim.txt v10.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CarContact:
    """One side of an IS_CON car-to-car contact (16 B)."""

    player_id: int
    info: int                 # CCI_* bits
    steer: int                # signed byte (-127..127)
    throttle: int             # high nibble of ThrBrk (0..15)
    brake: int                # low nibble of ThrBrk (0..15)
    clutch: int               # high nibble of CluHan
    handbrake: int            # low nibble of CluHan
    gear: int                 # high nibble of GearSp (0=R, 1=N, 2..=fwd gears)
    speed_ms: int             # byte (m/s)
    direction_rad: float      # byte (256 = 360°)
    heading_rad: float        # byte (256 = 360°)
    accel_forward: int        # signed byte, m/s²
    accel_right: int          # signed byte, m/s²
    x_m: float                # short / 16
    y_m: float                # short / 16

    _STRUCT: ClassVar[struct.Struct] = struct.Struct("<BBb BBB BBB bb hh")
    # PLID, Info, Sp2, Steer, ThrBrk, CluHan, GearSp, Speed, Direction,
    # Heading, AccelF, AccelR, X, Y.  Wait: we need PLID(B), Info(B), Sp2(B),
    # Steer(b), ThrBrk(B), CluHan(B), GearSp(B), Speed(B), Direction(B),
    # Heading(B), AccelF(b), AccelR(b), X(h), Y(h)  → 1+1+1+1+1+1+1+1+1+1+1+1+2+2 = 16 ✓

    @classmethod
    def parse(cls, data: bytes, off: int) -> "CarContact":
        (plid, info, _sp2, steer,
         thrbrk, cluhan, gearsp,
         speed, direction, heading,
         accel_f, accel_r,
         x_s, y_s) = struct.unpack_from(
            "<BBBb BBB BBB bb hh", data, off)
        return cls(
            player_id=plid, info=info, steer=steer,
            throttle=(thrbrk >> 4) & 0x0F,
            brake=thrbrk & 0x0F,
            clutch=(cluhan >> 4) & 0x0F,
            handbrake=cluhan & 0x0F,
            gear=(gearsp >> 4) & 0x0F,
            speed_ms=speed,
            direction_rad=direction * (math.tau / 256.0),
            heading_rad=heading * (math.tau / 256.0),
            accel_forward=accel_f,
            accel_right=accel_r,
            x_m=x_s / 16.0,
            y_m=y_s / 16.0,
        )


@dataclass(slots=True)
class InSimCarContact:
    """IS_CON — contact between two cars. Size = 40."""

    closing_speed_ms: float   # SpClose low 12 bits / 10.0
    time_ms: int              # word * 10
    a: CarContact
    b: CarContact

    @classmethod
    def parse(cls, data: bytes) -> "InSimCarContact":
        # Offset 4: SpClose(word), Time(word) = 4 bytes.
        sp_close, time_w = struct.unpack_from("<HH", data, 4)
        return cls(
            closing_speed_ms=(sp_close & 0x0FFF) / 10.0,
            time_ms=time_w * 10,
            a=CarContact.parse(data, 8),
            b=CarContact.parse(data, 24),
        )


@dataclass(slots=True)
class InSimFinish:
    """IS_FIN — provisional race finish for one player. Size = 20."""

    player_id: int
    total_time_ms: int        # 0 if forfeit
    best_lap_ms: int
    num_stops: int
    confirm: int              # CONF_* bits (mentioned, penalty, disq, ...)
    laps_done: int
    flags: int                # PIF_* at time of finish

    @classmethod
    def parse(cls, data: bytes) -> "InSimFinish":
        plid = data[3]
        (ttime, btime,
         _sp_a, num_stops, confirm, _sp_b,
         laps_done, flags) = struct.unpack_from("<II BBBB HH", data, 4)
        return cls(
            player_id=plid, total_time_ms=ttime, best_lap_ms=btime,
            num_stops=num_stops, confirm=confirm,
            laps_done=laps_done, flags=flags,
        )


@dataclass(slots=True)
class InSimResult:
    """IS_RES — confirmed race result for one player. Size = 84."""

    player_id: int
    user_name: str
    player_name: str
    plate: str
    car_name: str
    total_time_ms: int
    best_lap_ms: int
    num_stops: int
    confirm: int
    laps_done: int
    flags: int
    result_num: int           # finishing position (0 = first)
    num_results: int          # total results in this race
    penalty_seconds: int

    @classmethod
    def parse(cls, data: bytes) -> "InSimResult":
        plid = data[3]
        (uname, pname, plate, cname,
         ttime, btime,
         _sp_a, num_stops, confirm, _sp_b,
         laps_done, flags,
         result_num, num_results, pseconds) = struct.unpack_from(
            "<24s24s8s4s II BBBB HH BBH", data, 4)
        return cls(
            player_id=plid,
            user_name=_cstr(uname), player_name=_cstr(pname),
            plate=_cstr(plate), car_name=decode_car_id(cname),
            total_time_ms=ttime, best_lap_ms=btime,
            num_stops=num_stops, confirm=confirm,
            laps_done=laps_done, flags=flags,
            result_num=result_num, num_results=num_results,
            penalty_seconds=pseconds,
        )


@dataclass(slots=True)
class InSimTakeOverCar:
    """IS_TOC — driver swap (Take Over Car). Size = 8."""

    player_id: int
    old_connection_id: int
    new_connection_id: int

    @classmethod
    def parse(cls, data: bytes) -> "InSimTakeOverCar":
        plid = data[3]
        old_ucid, new_ucid, _sp2, _sp3 = struct.unpack_from("<BBBB", data, 4)
        return cls(player_id=plid,
                   old_connection_id=old_ucid,
                   new_connection_id=new_ucid)


@dataclass(slots=True)
class InSimPenalty:
    """IS_PEN — penalty given to or cleared from a player. Size = 8."""

    player_id: int
    old_penalty: int          # PENALTY_*
    new_penalty: int
    reason: int               # PENR_*

    @classmethod
    def parse(cls, data: bytes) -> "InSimPenalty":
        plid = data[3]
        old_pen, new_pen, reason, _sp3 = struct.unpack_from("<BBBB", data, 4)
        return cls(player_id=plid,
                   old_penalty=old_pen, new_penalty=new_pen, reason=reason)


@dataclass(slots=True)
class InSimFlag:
    """IS_FLG — flag shown (yellow/blue). Size = 8."""

    player_id: int
    off_on: int               # 0 = off, 1 = on
    flag: int                 # FLG_BLUE / FLG_YELLOW
    car_behind: int           # PLID of car causing the flag (blue flag)

    @classmethod
    def parse(cls, data: bytes) -> "InSimFlag":
        plid = data[3]
        off_on, flag, car_behind, _sp3 = struct.unpack_from("<BBBB", data, 4)
        return cls(player_id=plid, off_on=off_on, flag=flag,
                   car_behind=car_behind)


@dataclass(slots=True)
class InSimPlayerFlags:
    """IS_PFL — player flags changed mid-stint. Size = 8."""

    player_id: int
    flags: int                # PIF_* bits

    @classmethod
    def parse(cls, data: bytes) -> "InSimPlayerFlags":
        plid = data[3]
        (flags, _sp) = struct.unpack_from("<HH", data, 4)
        return cls(player_id=plid, flags=flags)


@dataclass(slots=True)
class InSimPlayerTelepit:
    """IS_PLP — player has teleported to pits (slot kept). Size = 4."""

    player_id: int

    @classmethod
    def parse(cls, data: bytes) -> "InSimPlayerTelepit":
        return cls(player_id=data[3])


@dataclass(slots=True)
class InSimPlayerLeaves:
    """IS_PLL — player leaves race (spectates, slot freed). Size = 4."""

    player_id: int

    @classmethod
    def parse(cls, data: bytes) -> "InSimPlayerLeaves":
        return cls(player_id=data[3])


@dataclass(slots=True)
class InSimCameraChange:
    """IS_CCH — camera changed for a player. Size = 8."""

    player_id: int
    camera: int               # ISS_* view-camera index (in_game_cam)

    @classmethod
    def parse(cls, data: bytes) -> "InSimCameraChange":
        plid = data[3]
        camera, _sp1, _sp2, _sp3 = struct.unpack_from("<BBBB", data, 4)
        return cls(player_id=plid, camera=camera)


@dataclass(slots=True)
class InSimInterfaceMode:
    """IS_CIM — connection interface mode (screen the user is on). Size = 8."""

    connection_id: int        # UCID
    mode: int                 # CIM_*
    sub_mode: int             # GRG_* when mode == CIM_GARAGE
    sel_type: int             # context-specific (selected item index)

    @classmethod
    def parse(cls, data: bytes) -> "InSimInterfaceMode":
        ucid = data[3]
        mode, sub_mode, sel_type, _sp3 = struct.unpack_from("<BBBB", data, 4)
        return cls(connection_id=ucid, mode=mode,
                   sub_mode=sub_mode, sel_type=sel_type)


@dataclass(slots=True)
class InSimCarStateChanged:
    """IS_CSC — car engine started or stopped. Size = 12."""

    player_id: int
    action: int               # CSC_STOP / CSC_START
    time_ms: int              # u32

    @classmethod
    def parse(cls, data: bytes) -> "InSimCarStateChanged":
        plid = data[3]
        # Layout (post v9): PLID at header[3]; payload at offset 4:
        # CSCAction(byte), Sp1, Sp2, Sp3, Time(u32).
        action, _sp1, _sp2, _sp3, time_ms = struct.unpack_from(
            "<BBBB I", data, 4)
        return cls(player_id=plid, action=action, time_ms=time_ms)



# Sound codes for IS_MSL.
SND_SILENT = 0
SND_MESSAGE = 1
SND_SYSMESSAGE = 2
SND_INVALIDKEY = 3
SND_ERROR = 4


def build_msl_packet(message: str, sound: int = SND_MESSAGE) -> bytes:
    """Build an IS_MSL packet (132 B) to display ``message`` locally in LFS.

    The message is shown in this LFS instance only (does not broadcast on
    multiplayer chat). Up to 127 ASCII bytes are sent (the 128th is a
    null terminator). Non-ASCII characters are encoded with the LFS
    color-tag fallback ``^7`` and silently truncated.
    """
    raw = message.encode("latin-1", errors="replace")[:127]
    msg_field = raw + b"\x00" * (128 - len(raw))
    # Size = 132 -> 132 / 4 = 33.
    return struct.pack("<BBBB", 33, ISP_MSL, 0, int(sound) & 0xFF) + msg_field
