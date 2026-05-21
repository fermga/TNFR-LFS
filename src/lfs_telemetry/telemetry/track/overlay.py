"""Top-down map overlay for the Studio track-map dock.

Each LFS *environment* (BL, AS, KY…) ships as one square ``.tif``
top-down render. Tracks within an environment (BL1, BL2R, AS3…) all
reuse the same image. The TIFs are plain rasters with no
georeferencing tags, so we anchor them to LFS world coordinates by:

1. Locating the racing-line CSVs that share the environment prefix
   (``racing_lines/<ENV>*_racing.csv``).
2. Computing the union bounding box of ``(x_line_m, y_line_m)``.
3. Centring the square TIF on that bbox and scaling it so the bbox
   fills a configurable fraction of the image (LFS official maps
   include a generous slice of surrounding scenery, so the racing
   line typically covers ~50–70 % of the image diagonal).

The defaults work out of the box for stock LFS tracks. Per-environment
fine tuning lives in :data:`OVERLAY_CONFIG_FILENAME`
(``config/track_overlays.json``) — every field is optional and falls
back to :data:`DEFAULT_CALIBRATION`.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import candidate_racing_lines_dirs

_LOG = logging.getLogger(__name__)

OVERLAY_CONFIG_FILENAME = "track_overlays.json"


# ---------------------------------------------------------------------------
# Calibration model
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class OverlayCalibration:
    """Per-environment placement tweaks applied to the auto-fit extent.

    Attributes
    ----------
    fill_fraction:
        How much of the image side the racing-line bbox should occupy
        (0..1). 0.6 means the bbox fills 60 % of the image, leaving
        20 % padding on each side for surroundings. Real-world LFS
        maps need ~0.55–0.7.
    scale:
        Multiplicative correction on top of ``fill_fraction``. Use to
        nudge the overall image size without redefining the fraction.
    dx_m, dy_m:
        Constant shift applied to the bottom-left corner of the
        overlay (positive moves the image right / up in world coords).
    flip_y:
        If true (default for raster inputs), flip the image vertically
        before placing it — raster row 0 is at the *top* whereas LFS
        world Y increases upward.
    rotate_deg:
        Reserved for future use; ignored for now (LFS maps are
        rendered with the same compass rose as the world axes).
    """
    fill_fraction: float = 0.60
    scale: float = 1.0
    dx_m: float = 0.0
    dy_m: float = 0.0
    flip_y: bool = True
    rotate_deg: float = 0.0


DEFAULT_CALIBRATION = OverlayCalibration()


# ---------------------------------------------------------------------------
# Filesystem discovery
# ---------------------------------------------------------------------------


def track_to_environment(track: str) -> str | None:
    """Return the 2-letter environment code (BL, AS, …) for a track id.

    LFS uses a fixed naming convention: the first two upper-case
    letters of a track code identify the environment. ``"BL2R"`` →
    ``"BL"``, ``"AS3"`` → ``"AS"``. Returns ``None`` when the input
    does not match the convention.
    """
    if not track:
        return None
    code = "".join(ch for ch in track[:2].upper() if ch.isalpha())
    return code if len(code) == 2 else None


def candidate_overlay_dirs() -> list[Path]:
    """Search dirs for the per-environment ``<ENV>.tif`` images."""
    cands: list[Path] = []
    cwd = Path.cwd()
    cands.append(cwd / "assets" / "tracks")
    cands.append(cwd / "tracks")
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cands.append(Path(meipass) / "assets" / "tracks")
        cands.append(Path(meipass) / "tracks")
    if sys.argv:
        exe_dir = Path(sys.argv[0]).resolve().parent
        cands.append(exe_dir / "assets" / "tracks")
        cands.append(exe_dir / "tracks")
    repo_root = Path(__file__).resolve().parents[4]
    cands.append(repo_root / "assets" / "tracks")
    cands.append(repo_root / "tracks")
    # De-duplicate while keeping order.
    seen: set[Path] = set()
    out: list[Path] = []
    for p in cands:
        rp = p.resolve() if p.exists() else p
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def find_overlay_image(env: str) -> Path | None:
    """Locate ``<ENV>.tif`` (or .png) under any candidate dir."""
    if not env:
        return None
    env_u = env.upper()
    for base in candidate_overlay_dirs():
        for ext in (".tif", ".tiff", ".png"):
            candidate = base / f"{env_u}{ext}"
            if candidate.exists():
                return candidate
    return None


def candidate_overlay_config_dirs() -> list[Path]:
    """Search dirs for the calibration JSON."""
    cands: list[Path] = []
    cwd = Path.cwd()
    cands.append(cwd / "config")
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cands.append(Path(meipass) / "config")
    if sys.argv:
        cands.append(Path(sys.argv[0]).resolve().parent / "config")
    repo_root = Path(__file__).resolve().parents[4]
    cands.append(repo_root / "config")
    seen: set[Path] = set()
    out: list[Path] = []
    for p in cands:
        rp = p.resolve() if p.exists() else p
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def load_overlay_calibrations(
    path: Path | None = None,
) -> dict[str, OverlayCalibration]:
    """Load per-environment calibrations from a JSON file.

    Missing file or malformed entries are tolerated — the caller
    falls back to :data:`DEFAULT_CALIBRATION` for unknown envs.
    """
    candidates = [path] if path is not None else [
        d / OVERLAY_CONFIG_FILENAME for d in candidate_overlay_config_dirs()
    ]
    for cand in candidates:
        if cand is None or not cand.exists():
            continue
        try:
            raw = json.loads(cand.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("could not read overlay config %s: %s", cand, exc)
            continue
        if not isinstance(raw, dict):
            continue
        out: dict[str, OverlayCalibration] = {}
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            kwargs = {
                f.name: value[f.name]
                for f in OverlayCalibration.__dataclass_fields__.values()
                if f.name in value
            }
            try:
                out[str(key).upper()] = replace(
                    DEFAULT_CALIBRATION, **kwargs
                )
            except TypeError as exc:
                _LOG.debug("ignoring overlay entry %s: %s", key, exc)
        return out
    return {}


# ---------------------------------------------------------------------------
# Auto-fit using racing-line bbox
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class OverlayExtent:
    """World-coord placement of an overlay image.

    The image, after the optional Y-flip, occupies the rectangle
    ``[x0, x0 + width] × [y0, y0 + height]`` in LFS world meters.
    """
    x0_m: float
    y0_m: float
    width_m: float
    height_m: float
    flip_y: bool


def _racing_line_bbox(env: str) -> tuple[float, float, float, float] | None:
    """Union bbox of ``(x_line_m, y_line_m)`` across all variants.

    Returns ``(x_min, y_min, x_max, y_max)`` or ``None`` when no
    matching CSV is found.
    """
    env_u = env.upper()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    seen: set[Path] = set()
    for base in candidate_racing_lines_dirs():
        if not base.exists():
            continue
        for csv in sorted(base.glob(f"{env_u}*_racing.csv")):
            rp = csv.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            try:
                df = pd.read_csv(csv, usecols=["x_line_m", "y_line_m"])
            except (OSError, ValueError, KeyError) as exc:
                _LOG.debug("skipping %s: %s", csv, exc)
                continue
            xs.append(df["x_line_m"].to_numpy(dtype=float))
            ys.append(df["y_line_m"].to_numpy(dtype=float))
    if not xs:
        return None
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return None
    x = x[finite]
    y = y[finite]
    return float(x.min()), float(y.min()), float(x.max()), float(y.max())


def compute_overlay_extent(
    env: str,
    calibration: OverlayCalibration | None = None,
) -> OverlayExtent | None:
    """Auto-fit the square TIF for ``env`` against its racing-line bbox.

    The TIF is treated as a square. We centre it on the racing-line
    bbox centre and pick a side length such that the larger bbox
    dimension covers ``calibration.fill_fraction`` of the image.
    """
    cal = calibration or DEFAULT_CALIBRATION
    bbox = _racing_line_bbox(env)
    if bbox is None:
        return None
    x_min, y_min, x_max, y_max = bbox
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    bw = max(x_max - x_min, 1.0)
    bh = max(y_max - y_min, 1.0)
    fill = max(0.05, min(1.0, float(cal.fill_fraction)))
    side = (max(bw, bh) / fill) * float(cal.scale)
    half = side / 2.0
    return OverlayExtent(
        x0_m=cx - half + float(cal.dx_m),
        y0_m=cy - half + float(cal.dy_m),
        width_m=side,
        height_m=side,
        flip_y=bool(cal.flip_y),
    )


__all__ = [
    "DEFAULT_CALIBRATION",
    "OVERLAY_CONFIG_FILENAME",
    "OverlayCalibration",
    "OverlayExtent",
    "candidate_overlay_dirs",
    "candidate_overlay_config_dirs",
    "compute_overlay_extent",
    "find_overlay_image",
    "load_overlay_calibrations",
    "track_to_environment",
]
