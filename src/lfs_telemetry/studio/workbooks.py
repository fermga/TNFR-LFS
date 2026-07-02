"""Workbook / Worksheet / Component data model for the Telemetry tab.

MoTeC i2-inspired: a *Workbook* groups several *Worksheets* (tabs), and
each Worksheet hosts an ordered list of *Components*. A Component is a
self-contained visual (line graph, bar column, histogram, XY scatter,
gauge, track map, report) with its own list of channels and rendering
options. The Studio's Telemetry tab loads one Workbook at a time and
lets the user edit/save it.

This module is intentionally **pure data + I/O**: no Qt widgets, no
pyqtgraph. The studio widgets layer consumes the model and renders it.
That keeps the schema testable without booting the GUI.

Persistence
-----------
Workbooks live as JSON files under the user's writable config dir,
typically::

    %APPDATA%/LFS-Race-Engineer/LFS Telemetry Studio/workbooks/*.json

Path is resolved via :func:`user_workbooks_dir`, which uses
``QStandardPaths.AppConfigLocation`` so it matches every other Studio
setting. A handful of read-only **built-in templates** ship inside the
package (see :func:`builtin_templates`); the user can clone any of them
into the writable dir to start tweaking.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Bump when an incompatible schema change is shipped. ``from_dict`` will
# reject anything newer than ``SCHEMA_VERSION``; older files are accepted
# (and silently up-converted in memory) so users don't lose their work.
SCHEMA_VERSION = 1

# Closed set of component types the renderer knows about. Files
# referencing an unknown type are loaded with the component dropped so
# the rest of the workbook still opens.
COMPONENT_TYPES: tuple[str, ...] = (
    "graph",      # line plot vs distance/time (overlay/normalize aware)
    "bar",        # per-wheel/per-channel column snapshot at cursor
    "gauge",      # single numeric/arc readout
    "histogram",  # 1-D distribution of a channel
    "xy",         # scatter of two channels (e.g. G-G)
    "trackmap",   # mini track map colour-coded by a channel
    "report",     # tabular summary (min/max/avg per sector)
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

def _new_id() -> str:
    """Short, URL-safe component id (12 hex chars from uuid4)."""
    return uuid.uuid4().hex[:12]


@dataclass
class Component:
    """One visual on a worksheet."""

    type: str
    title: str
    channels: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)

    def __post_init__(self) -> None:
        if self.type not in COMPONENT_TYPES:
            raise ValueError(
                f"unknown component type {self.type!r}; "
                f"expected one of {COMPONENT_TYPES}"
            )
        if not isinstance(self.channels, list):
            raise TypeError("Component.channels must be a list[str]")
        if not all(isinstance(c, str) for c in self.channels):
            raise TypeError("Component.channels must contain only str")
        if not isinstance(self.options, dict):
            raise TypeError("Component.options must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Component:
        return cls(
            type=str(raw["type"]),
            title=str(raw.get("title", "")),
            channels=list(raw.get("channels", []) or []),
            options=dict(raw.get("options", {}) or {}),
            id=str(raw.get("id") or _new_id()),
        )


@dataclass
class Worksheet:
    """A tab inside the workbook, holding an ordered list of components."""

    title: str
    components: list[Component] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "components": [c.to_dict() for c in self.components],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Worksheet:
        comps: list[Component] = []
        for entry in raw.get("components", []) or []:
            try:
                comps.append(Component.from_dict(entry))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping invalid component on worksheet %r: %s",
                    raw.get("title"), exc,
                )
        return cls(title=str(raw.get("title", "")), components=comps)


@dataclass
class Workbook:
    """Top-level container persisted to a single JSON file."""

    name: str
    worksheets: list[Worksheet] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "name": self.name,
            "worksheets": [w.to_dict() for w in self.worksheets],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Workbook:
        version = int(raw.get("schema_version", 1))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"workbook schema_version {version} is newer than "
                f"supported {SCHEMA_VERSION}; please upgrade the app"
            )
        return cls(
            name=str(raw.get("name", "")),
            worksheets=[
                Worksheet.from_dict(w) for w in raw.get("worksheets", []) or []
            ],
            schema_version=version,
        )


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._\- ]+")


def _safe_filename(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", name).strip(" .") or "workbook"
    return cleaned + ".json"


def load_workbook(path: Path) -> Workbook:
    """Parse a workbook JSON file into a :class:`Workbook` instance."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"workbook file {path} is not a JSON object")
    return Workbook.from_dict(raw)


def save_workbook(workbook: Workbook, path: Path) -> None:
    """Serialize *workbook* to *path* (parent dir is created on demand)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(workbook.to_dict(), indent=2, ensure_ascii=False)
    # Atomic-ish write: stage to .tmp then replace, so a crash mid-save
    # never leaves a half-written workbook on disk.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def user_workbooks_dir() -> Path:
    """Writable folder where user-saved workbooks live.

    Resolved lazily so importing this module never requires a running
    QApplication (the constants module is loaded by headless tests too).
    """
    try:
        from PySide6.QtCore import QStandardPaths
    except Exception:  # pragma: no cover - PySide6 always present at runtime
        from .. import lfs_paths
        # Fallback for environments without Qt: drop next to QSettings ini.
        return Path.home() / ".config" / lfs_paths.QSETTINGS_ORG / "workbooks"
    base = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)
    if not base:  # pragma: no cover - QStandardPaths always returns something
        base = str(Path.home() / ".config")
    return Path(base) / "workbooks"


def list_user_workbooks() -> list[Path]:
    """Return every ``*.json`` workbook found under the user dir, sorted."""
    folder = user_workbooks_dir()
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.json") if p.is_file())


def save_user_workbook(workbook: Workbook) -> Path:
    """Save *workbook* under the user dir using its ``name`` as filename."""
    target = user_workbooks_dir() / _safe_filename(workbook.name)
    save_workbook(workbook, target)
    return target


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

_WHEELS = ("FL", "FR", "RL", "RR")


def _graph(
    title: str,
    channels: Iterable[str],
    *,
    overlay: bool = True,
    normalize: bool = False,
) -> Component:
    return Component(
        type="graph",
        title=title,
        channels=list(channels),
        options={"overlay": overlay, "normalize": normalize},
    )


def _bar(title: str, channels: Iterable[str], units: str = "") -> Component:
    return Component(
        type="bar",
        title=title,
        channels=list(channels),
        options={"units": units, "at_cursor": True},
    )


def _xy(title: str, x: str, y: str, *, color_by: str = "lap") -> Component:
    return Component(
        type="xy",
        title=title,
        channels=[x, y],
        options={"x": x, "y": y, "color_by": color_by},
    )


def _driver_inputs() -> Worksheet:
    return Worksheet(
        title="Driver Inputs",
        components=[
            _graph("Throttle + Brake", ["throttle", "brake"]),
            _graph(
                "Steering",
                ["input_steer"],
                overlay=False,
            ),
            _graph("Speed", ["speed_ms"], overlay=False),
            _graph("Clutch", ["clutch"], overlay=False),
        ],
    )


def _tyres() -> Worksheet:
    return Worksheet(
        title="Tyres",
        components=[
            _graph(
                "Slip ratio (4 wheels)",
                [f"wheel_{c}_slip_ratio" for c in _WHEELS],
            ),
            _graph(
                "Vertical load (4 wheels)",
                [f"wheel_{c}_vertical_load_n" for c in _WHEELS],
            ),
            _graph(
                "Tyre temperature (4 wheels)",
                [f"wheel_{c}_air_temp_c" for c in _WHEELS],
            ),
            _bar(
                "Tyre temp snapshot",
                [f"wheel_{c}_air_temp_c" for c in _WHEELS],
                units="°C",
            ),
        ],
    )


def _suspension() -> Worksheet:
    return Worksheet(
        title="Suspension",
        components=[
            _graph(
                "Suspension travel (4 wheels)",
                [f"wheel_{c}_susp_deflect_m" for c in _WHEELS],
            ),
            _graph(
                "Damper velocity (4 wheels)",
                [f"wheel_{c}_susp_speed_mps" for c in _WHEELS],
            ),
            _graph(
                "Ride height",
                [f"wheel_{c}_susp_deflect_m" for c in _WHEELS],
                normalize=True,
            ),
        ],
    )


def _brakes() -> Worksheet:
    return Worksheet(
        title="Brakes",
        components=[
            _graph("Brake pedal", ["brake"], overlay=False),
            _graph(
                "Brake bias (real) + Pedal",
                ["brake_bias_front_real", "brake"],
                normalize=True,
            ),
            _graph(
                "Wheel lock indicator (4 wheels)",
                [f"wheel_{c}_slip_ratio" for c in _WHEELS],
            ),
        ],
    )


def _chassis() -> Worksheet:
    return Worksheet(
        title="Chassis / Aero",
        components=[
            _graph(
                "Long. + Lat. acceleration",
                ["accel_x", "accel_y"],
            ),
            _graph(
                "Yaw rate vs Steer",
                ["yaw_rate_rads", "input_steer"],
                normalize=True,
            ),
            _xy("G-G diagram", x="accel_y", y="accel_x"),
        ],
    )


def _engine() -> Worksheet:
    return Worksheet(
        title="Engine",
        components=[
            _graph("RPM", ["rpm"], overlay=False),
            _graph("Gear", ["gear_lfs"], overlay=False),
            _graph("Throttle", ["throttle"], overlay=False),
        ],
    )


def _default_workbook_factories() -> dict[str, callable]:
    """Map of template-name → factory producing a fresh :class:`Workbook`.

    Factories so each call returns a brand-new, mutable instance — never
    a shared object the user could accidentally mutate.
    """
    return {
        "Default": lambda: Workbook(
            name="Default",
            worksheets=[
                _driver_inputs(),
                _tyres(),
                _suspension(),
                _brakes(),
                _chassis(),
                _engine(),
            ],
        ),
        "Driver Inputs": lambda: Workbook(
            name="Driver Inputs", worksheets=[_driver_inputs()],
        ),
        "Tyres": lambda: Workbook(name="Tyres", worksheets=[_tyres()]),
        "Suspension": lambda: Workbook(
            name="Suspension", worksheets=[_suspension()],
        ),
        "Brakes": lambda: Workbook(name="Brakes", worksheets=[_brakes()]),
        "Chassis / Aero": lambda: Workbook(
            name="Chassis / Aero", worksheets=[_chassis()],
        ),
        "Engine": lambda: Workbook(name="Engine", worksheets=[_engine()]),
    }


def builtin_template_names() -> list[str]:
    """Names of the built-in templates, in menu order."""
    return list(_default_workbook_factories().keys())


def builtin_template(name: str) -> Workbook:
    """Return a fresh copy of the built-in template called *name*."""
    factories = _default_workbook_factories()
    if name not in factories:
        raise KeyError(
            f"unknown built-in workbook template {name!r}; "
            f"available: {builtin_template_names()}"
        )
    return factories[name]()


def default_workbook() -> Workbook:
    """The workbook shown the first time the Telemetry tab opens."""
    return builtin_template("Default")


__all__ = [
    "COMPONENT_TYPES",
    "SCHEMA_VERSION",
    "Component",
    "Workbook",
    "Worksheet",
    "builtin_template",
    "builtin_template_names",
    "default_workbook",
    "list_user_workbooks",
    "load_workbook",
    "save_user_workbook",
    "save_workbook",
    "user_workbooks_dir",
]
