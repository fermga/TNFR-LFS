"""VR mirror: polls visible overlay modules and pushes them to OpenVR.

Keeps the on-screen ``_LiveModuleWindow`` instances as the single source
of truth for "what's enabled and how it looks". On every tick we:

1. Inspect the live tab's widget map.
2. For each *visible* module: ensure a SteamVR overlay exists and
   upload its current ``render_to_image()``.
3. For each module that became hidden / was destroyed: hide or destroy
   the matching SteamVR overlay.

This means the user gets full visual parity for free — there is one
content model (the Qt widget), one paint pipeline, and the VR sink is
just another delivery target.

Designed as an opt-in attachment so the studio works exactly as before
when VR is disabled or unavailable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import QObject, QTimer

from .openvr_overlay import OpenVROverlaySink, OverlayPose

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..widgets.live_modules import _LiveModuleWindow

log = logging.getLogger(__name__)

# 30 Hz default — visually smooth, low CPU. The user's HMD compositor
# reprojects to its native rate (typically 90 Hz) so we don't need to
# match that here. Heads-up data rarely benefits from > 30 Hz updates.
DEFAULT_TICK_HZ = 30


# Per-module default poses. Coordinates are HMD-relative meters
# (x=right, y=up, z=forward-negative). Values chosen so multiple
# enabled modules don't overlap by default. The user will be able to
# tweak these from the Overlay tab in a later step.
_DEFAULT_POSES: dict[str, OverlayPose] = {
    "speed": OverlayPose(x=-0.30, y=-0.20, z=-1.5, width_m=0.20),
    "gear": OverlayPose(x=0.00, y=-0.20, z=-1.5, width_m=0.18),
    "rpm": OverlayPose(x=0.30, y=-0.20, z=-1.5, width_m=0.30),
    "fuel_pct": OverlayPose(x=-0.55, y=-0.05, z=-1.5, width_m=0.18),
    "fuel_laps": OverlayPose(x=-0.55, y=-0.20, z=-1.5, width_m=0.18),
    "delta": OverlayPose(x=0.00, y=-0.05, z=-1.5, width_m=0.40),
    "gap_ahead": OverlayPose(x=0.55, y=-0.05, z=-1.5, width_m=0.18),
    "gap_behind": OverlayPose(x=0.55, y=-0.20, z=-1.5, width_m=0.18),
    "grip": OverlayPose(x=-0.55, y=0.10, z=-1.5, width_m=0.18),
    "session_info": OverlayPose(x=0.55, y=0.10, z=-1.5, width_m=0.30),
    "flags": OverlayPose(x=0.00, y=0.25, z=-1.5, width_m=0.30),
    "pit_limiter": OverlayPose(x=0.00, y=0.10, z=-1.5, width_m=0.20),
    "gmeter": OverlayPose(x=-0.40, y=-0.40, z=-1.5, width_m=0.25),
    "radar": OverlayPose(x=0.40, y=-0.40, z=-1.5, width_m=0.25),
    "minimap": OverlayPose(x=-0.55, y=0.30, z=-1.5, width_m=0.30),
    "gap_compass": OverlayPose(x=0.55, y=0.30, z=-1.5, width_m=0.25),
    "tc_abs": OverlayPose(x=0.00, y=-0.40, z=-1.5, width_m=0.25),
}


# ---------------------------------------------------------------------------
# Mirror
# ---------------------------------------------------------------------------


WidgetProvider = Callable[[], "dict[str, _LiveModuleWindow]"]


class VrMirror(QObject):
    """Polls a widget map and mirrors visible modules to a VR sink.

    Parameters
    ----------
    provider:
        Callable returning the *current* ``{module_id: widget}`` map.
        Pass ``LiveTab._widgets`` reader; we re-read every tick so the
        mirror picks up newly-toggled modules without explicit notify.
    sink:
        Optional pre-built sink. If ``None`` we lazily create an
        :class:`OpenVROverlaySink` the first time the mirror is enabled.
    tick_hz:
        Upload rate. Default 30 Hz.
    parent:
        Qt parent. The internal ``QTimer`` is owned by ``self`` so it
        gets cleaned up at parent destruction time.
    """

    def __init__(
        self,
        provider: WidgetProvider,
        *,
        sink: OpenVROverlaySink | None = None,
        tick_hz: int = DEFAULT_TICK_HZ,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._provider = provider
        self._sink: OpenVROverlaySink | None = sink
        self._tick_ms = max(10, int(round(1000 / max(1, tick_hz))))
        self._timer = QTimer(self)
        self._timer.setInterval(self._tick_ms)
        self._timer.timeout.connect(self._tick)
        # Tracks which module ids currently have an overlay handle, so
        # we can hide/destroy the ones whose widget disappeared.
        self._known: set[str] = set()
        self._enabled = False

    # ----- Lifecycle ---------------------------------------------------

    def enable(self) -> bool:
        """Start mirroring. Returns ``True`` if the VR sink is alive.

        Calling this when ``openvr`` isn't installed or SteamVR isn't
        running leaves the mirror dormant and returns ``False`` — but
        does NOT raise, so UI code can simply attempt enable() and
        report the boolean to the user.
        """
        if self._sink is None:
            self._sink = OpenVROverlaySink()
        if not self._sink.available:
            log.info(
                "VR mirror requested but unavailable: %s",
                self._sink.init_error,
            )
            self._enabled = False
            return False
        self._enabled = True
        self._timer.start()
        return True

    def disable(self) -> None:
        """Stop mirroring and tear down every active overlay handle."""
        self._timer.stop()
        self._enabled = False
        if self._sink is None:
            return
        for mid in list(self._known):
            self._sink.destroy_overlay(mid)
        self._known.clear()

    def shutdown(self) -> None:
        """Disable + close the underlying OpenVR session."""
        self.disable()
        if self._sink is not None:
            self._sink.shutdown()
            self._sink = None

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_available(self) -> bool:
        if self._sink is None:
            # Probe lazily so the UI can decide whether to even show
            # the "Enable VR" toggle.
            self._sink = OpenVROverlaySink()
        return self._sink.available

    # ----- Tick --------------------------------------------------------

    def _tick(self) -> None:  # pragma: no cover - exercised in real VR
        sink = self._sink
        if not self._enabled or sink is None or not sink.available:
            return
        try:
            widgets = self._provider()
        except Exception as exc:
            log.debug("VrMirror provider failed: %s", exc)
            return

        seen: set[str] = set()
        for mid, win in widgets.items():
            if win is None:
                continue
            try:
                visible = bool(win.isVisible())
            except RuntimeError:
                # Widget already deleted on the C++ side.
                continue
            if not visible:
                continue
            seen.add(mid)
            if mid not in self._known:
                pose = _DEFAULT_POSES.get(mid, OverlayPose())
                if not sink.ensure_overlay(mid, pose=pose):
                    continue
                self._known.add(mid)
            try:
                img = win.render_to_image()
            except Exception as exc:
                log.debug("render_to_image failed for %s: %s", mid, exc)
                continue
            sink.upload(mid, img)

        # Hide overlays for modules that disappeared from the map or
        # were toggled off in the UI.
        for mid in list(self._known):
            if mid not in seen:
                sink.set_visible(mid, False)


__all__ = ["VrMirror", "DEFAULT_TICK_HZ"]
