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
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import candidate_racing_lines_dirs

_LOG = logging.getLogger(__name__)

OVERLAY_CONFIG_FILENAME = "track_overlays.json"
_USER_DIRNAME = "lfs-telemetry-viewer"


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


def user_overlay_config_path() -> Path:
    """User-writable per-environment overlay calibration JSON.

    Lives next to the lap cache so all user-state for the viewer
    converges in one place. The directory is created on demand.
    """
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        root = Path(base) / _USER_DIRNAME
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / _USER_DIRNAME
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        root = (
            Path(xdg) if xdg else Path.home() / ".config"
        ) / _USER_DIRNAME
    return root / OVERLAY_CONFIG_FILENAME


def _read_calibrations_from(
    path: Path,
) -> dict[str, OverlayCalibration]:
    """Parse one calibration JSON file. Returns ``{}`` on any failure."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("could not read overlay config %s: %s", path, exc)
        return {}
    if not isinstance(raw, dict):
        return {}
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
            out[str(key).upper()] = replace(DEFAULT_CALIBRATION, **kwargs)
        except TypeError as exc:
            _LOG.debug("ignoring overlay entry %s: %s", key, exc)
    return out


def load_overlay_calibrations(
    path: Path | None = None,
) -> dict[str, OverlayCalibration]:
    """Load per-environment calibrations.

    Resolution order (each later step overrides earlier ones):

    1. The first bundled ``config/track_overlays.json`` we find under
       :func:`candidate_overlay_config_dirs` (or the explicit ``path``
       argument when provided).
    2. The user-scoped override at :func:`user_overlay_config_path`,
       which is where the Track-map dock's "Calibrate map" dialog
       saves interactive nudges.

    Missing files or malformed entries are tolerated — the caller
    falls back to :data:`DEFAULT_CALIBRATION` for unknown envs.
    """
    merged: dict[str, OverlayCalibration] = {}

    # Step 1: bundled defaults (or explicit path).
    candidates: list[Path]
    if path is not None:
        candidates = [path]
    else:
        candidates = [
            d / OVERLAY_CONFIG_FILENAME
            for d in candidate_overlay_config_dirs()
        ]
    for cand in candidates:
        if cand is None or not cand.exists():
            continue
        merged.update(_read_calibrations_from(cand))
        break

    # Step 2: user override (only when no explicit path was forced).
    if path is None:
        user = user_overlay_config_path()
        if user.exists():
            merged.update(_read_calibrations_from(user))

    return merged


def save_user_overlay_calibration(
    env: str, calibration: OverlayCalibration,
) -> Path:
    """Persist *calibration* for *env* to :func:`user_overlay_config_path`.

    Read-modify-write so other environments' overrides are preserved.
    Returns the path written to. Raises :class:`OSError` on I/O errors.
    """
    env_u = str(env).upper()
    path = user_overlay_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw: dict[str, dict] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                raw = {
                    str(k).upper(): v
                    for k, v in existing.items()
                    if isinstance(v, dict)
                }
        except json.JSONDecodeError:
            # Corrupted file: start over rather than refuse to save.
            raw = {}
    raw[env_u] = {
        "fill_fraction": float(calibration.fill_fraction),
        "scale": float(calibration.scale),
        "dx_m": float(calibration.dx_m),
        "dy_m": float(calibration.dy_m),
        "flip_y": bool(calibration.flip_y),
        "rotate_deg": float(calibration.rotate_deg),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


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


def compute_overlay_extent_for_image(
    image_size_px: tuple[int, int],
    calibration: OverlayCalibration | None = None,
) -> OverlayExtent:
    """Deterministic 1 m/px placement centred on the world origin.

    LFS renders every track overview at exactly 1 metre per pixel and
    centres the image on world coordinate ``(0, 0)``. So a TIF of size
    ``W × H`` pixels occupies ``[-W/2, W/2] × [-H/2, H/2]`` in metres,
    with the image's Y axis flipped relative to LFS world Y (north is
    up in world coords; row 0 is at the top of the image).

    ``calibration`` is applied as a *residual* tweak: ``scale`` enlarges
    the rectangle around its centre (rarely needed; defaults to 1.0),
    and ``dx_m, dy_m`` shift it. ``flip_y`` lets a user override the
    default Y-flip if a particular TIF is exported with row 0 at the
    bottom.
    """
    cal = calibration or DEFAULT_CALIBRATION
    width_px, height_px = image_size_px
    if width_px <= 0 or height_px <= 0:
        # Degenerate image; fall back to a unit square so the caller
        # can still render something rather than crashing.
        width_px = max(1, width_px)
        height_px = max(1, height_px)
    scale = float(cal.scale) if cal.scale > 0 else 1.0
    width_m = float(width_px) * scale
    height_m = float(height_px) * scale
    x0 = -width_m / 2.0 + float(cal.dx_m)
    y0 = -height_m / 2.0 + float(cal.dy_m)
    return OverlayExtent(
        x0_m=x0,
        y0_m=y0,
        width_m=width_m,
        height_m=height_m,
        flip_y=bool(cal.flip_y),
    )


def compute_overlay_extent(
    env: str,
    calibration: OverlayCalibration | None = None,
) -> OverlayExtent | None:
    """Legacy auto-fit against the racing-line bbox.

    Kept for callers that don't have the source image dimensions
    handy. Prefer :func:`compute_overlay_extent_for_image` whenever
    you've already loaded the TIF — it produces the canonical
    1 m/px placement centred on the world origin, matching how LFS
    renders the track overviews.
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
    "candidate_overlay_config_dirs",
    "candidate_overlay_dirs",
    "compute_overlay_extent",
    "compute_overlay_extent_for_image",
    "find_overlay_image",
    "load_overlay_calibrations",
    "save_user_overlay_calibration",
    "track_to_environment",
    "user_overlay_config_path",
]
