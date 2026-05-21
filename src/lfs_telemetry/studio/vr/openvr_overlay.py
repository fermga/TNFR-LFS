"""OpenVR / SteamVR overlay sink for live overlay modules.

This module lets every :class:`_LiveModuleWindow` from
:mod:`lfs_telemetry.studio.widgets.live_modules` push its current
``render_to_image()`` output to a SteamVR ``IVROverlay`` so the user
sees the same overlay layer in their HMD that they configured on the
desktop.

Design rules (kept deliberately small):

* **Single content model** — pixels come from
  ``_LiveModuleWindow.render_to_image()``. No re-painting, no parallel
  rendering pipeline. Visual parity with the desktop window is by
  construction.
* **One IVROverlay handle per module**, keyed by ``MODULE_ID``. SteamVR
  supports many simultaneous overlays, which mirrors our N independent
  draggable windows.
* **Graceful absence**: if ``openvr`` isn't installed, or SteamVR isn't
  running, or any OpenVR call fails at init time, the adapter logs the
  reason, sets ``available=False``, and the rest of the studio keeps
  working. We never crash the main app because the user doesn't have
  a headset on.
* **Windows-only** in practice (matches the rest of this project's
  shipping target). The ``vr`` extra in ``pyproject.toml`` already
  guards the dep with ``sys_platform == 'win32'``.

The adapter does NOT decide *when* to tick or *which* modules are
active — that is the orchestrator's job (added in a later step). It
just exposes a small imperative API: ``ensure_overlay``, ``upload``,
``set_visible``, ``destroy_overlay``, ``shutdown``.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PySide6.QtGui import QImage

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy-import wrapper so absence of the dep / SteamVR is non-fatal.
# ---------------------------------------------------------------------------


def _try_import_openvr() -> Any | None:
    try:
        import openvr  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        log.info("openvr not importable: %s", exc)
        return None
    return openvr


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OverlayPose:
    """Minimal HMD-relative pose for an overlay panel.

    Coordinates are in meters, OpenVR convention:
      * +X right, +Y up, +Z toward the user (out of the HMD's screen).
    A pose at ``(0, 0, -1.5)`` puts the panel 1.5 m straight ahead.
    """

    x: float = 0.0
    y: float = -0.2
    z: float = -1.5
    width_m: float = 0.40

    def to_matrix34(self) -> tuple[tuple[float, ...], ...]:
        """Build a 3x4 row-major identity rotation + translation matrix."""
        return (
            (1.0, 0.0, 0.0, self.x),
            (0.0, 1.0, 0.0, self.y),
            (0.0, 0.0, 1.0, self.z),
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass
class _OverlayEntry:
    handle: int
    width_m: float
    visible: bool = False
    last_size: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class VRRuntimeStatus:
    """Snapshot of the live SteamVR runtime as we see it.

    Used by the UI to confirm that (a) we are talking to a real HMD and
    (b) LFS is the currently focused VR scene application — i.e. our
    overlay layer is going to be composited on top of LFS's frames.
    """

    hmd_connected: bool = False
    hmd_model: str | None = None
    scene_app_pid: int | None = None
    scene_app_name: str | None = None  # e.g. "LFS.exe"
    scene_app_is_lfs: bool = False


def _process_name_for_pid(pid: int) -> str | None:
    """Best-effort PID → executable basename on Windows.

    Used to confirm whether the active VR scene-app is ``LFS.exe``.
    Returns ``None`` if anything fails (non-Windows host, access
    denied, process gone, etc.) — we never raise to the caller.
    """
    if pid is None or pid <= 0:
        return None
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:  # pragma: no cover - non-Windows
        return None
    try:
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        OpenProcess = kernel32.OpenProcess
        OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        OpenProcess.restype = wintypes.HANDLE
        QueryFullProcessImageNameW = kernel32.QueryFullProcessImageNameW
        QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        QueryFullProcessImageNameW.restype = wintypes.BOOL
        CloseHandle = kernel32.CloseHandle
        CloseHandle.argtypes = [wintypes.HANDLE]
        CloseHandle.restype = wintypes.BOOL

        h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return None
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(len(buf))
            if not QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return None
            from pathlib import Path as _Path
            return _Path(buf.value).name or None
        finally:
            CloseHandle(h)
    except Exception:  # pragma: no cover - defensive
        return None


class OpenVROverlaySink:
    """Holds a SteamVR overlay-app session and one IVROverlay per module.

    Thread-safety: all OpenVR calls are serialized behind ``self._lock``
    so a future render thread and the Qt main thread can both poke the
    sink without tearing handles.
    """

    APP_KEY = "lfs_race_engineer.overlay"
    APP_NAME = "LFS Race Engineer Overlay"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._openvr = _try_import_openvr()
        self._vr_overlay: Any | None = None
        self._initialized = False
        self._init_error: str | None = None
        self._entries: dict[str, _OverlayEntry] = {}

        if self._openvr is None:
            self._init_error = "openvr package not installed"
            return

        try:
            self._openvr.init(self._openvr.VRApplication_Overlay)
            self._vr_overlay = self._openvr.VROverlay()
            self._initialized = True
            log.info("OpenVR overlay sink initialized")
        except Exception as exc:  # pragma: no cover - runtime only
            self._init_error = f"{type(exc).__name__}: {exc}"
            log.info("OpenVR init failed: %s", self._init_error)
            self._openvr = None
            self._vr_overlay = None

    # ----- Properties --------------------------------------------------

    @property
    def available(self) -> bool:
        """``True`` if SteamVR is up and overlays can be created."""
        return self._initialized and self._vr_overlay is not None

    @property
    def init_error(self) -> str | None:
        return self._init_error

    # ----- Overlay lifecycle ------------------------------------------

    def ensure_overlay(
        self, module_id: str, *, pose: OverlayPose | None = None,
    ) -> bool:
        """Create the IVROverlay for ``module_id`` if missing.

        Returns ``True`` on success, ``False`` if VR isn't available.
        Idempotent — repeated calls just return ``True``.
        """
        if pose is None:
            pose = OverlayPose()
        if not self.available:
            return False
        with self._lock:
            if module_id in self._entries:
                return True
            try:
                key = f"{self.APP_KEY}.{module_id}"
                friendly = f"LFS RE — {module_id}"
                handle = self._vr_overlay.createOverlay(key, friendly)
                self._vr_overlay.setOverlayWidthInMeters(
                    handle, pose.width_m,
                )
                # Anchor to HMD (tracked device 0). LFS users overwhelmingly
                # want HUD-style overlays; world-locked variants are a later
                # configuration toggle.
                hmd_idx = 0
                self._vr_overlay.setOverlayTransformTrackedDeviceRelative(
                    handle, hmd_idx, pose.to_matrix34(),
                )
                self._entries[module_id] = _OverlayEntry(
                    handle=handle, width_m=pose.width_m,
                )
                log.info("Created VR overlay '%s' (handle=%s)", key, handle)
                return True
            except Exception as exc:  # pragma: no cover - runtime
                log.warning(
                    "Failed to create VR overlay for %s: %s", module_id, exc,
                )
                return False

    def destroy_overlay(self, module_id: str) -> None:
        if not self.available:
            return
        with self._lock:
            entry = self._entries.pop(module_id, None)
            if entry is None:
                return
            try:
                self._vr_overlay.destroyOverlay(entry.handle)
            except Exception as exc:  # pragma: no cover - runtime
                log.debug("destroyOverlay(%s) failed: %s", module_id, exc)

    # ----- Frame upload ------------------------------------------------

    def upload(self, module_id: str, image: QImage) -> bool:
        """Upload a QImage frame to the overlay for ``module_id``.

        Converts to RGBA8888 (byte-order RGBA on all platforms) before
        handing the buffer to ``setOverlayRaw``. Returns ``False`` if VR
        isn't available, the overlay doesn't exist yet, or the upload
        fails.
        """
        if not self.available:
            return False
        entry = self._entries.get(module_id)
        if entry is None:
            return False

        # Local import keeps this module importable in headless contexts
        # (e.g. CLI capture) where PySide6 might be absent.
        from PySide6.QtGui import QImage

        if image.isNull() or image.width() <= 0 or image.height() <= 0:
            return False

        # OpenVR's setOverlayRaw expects a tightly packed RGBA byte buffer.
        # Qt's Format_RGBA8888 has byte order R,G,B,A regardless of CPU
        # endianness, which is what we want.
        if image.format() != QImage.Format.Format_RGBA8888:
            image = image.convertToFormat(QImage.Format.Format_RGBA8888)

        w, h = image.width(), image.height()
        # Strip per-row padding if Qt aligned scanlines.
        expected_stride = w * 4
        if image.bytesPerLine() != expected_stride:
            # Force a packed copy.
            image = image.copy()
        buf = bytes(image.constBits())

        with self._lock:
            try:
                self._vr_overlay.setOverlayRaw(entry.handle, buf, w, h, 4)
                entry.last_size = (w, h)
                if not entry.visible:
                    self._vr_overlay.showOverlay(entry.handle)
                    entry.visible = True
                return True
            except Exception as exc:  # pragma: no cover - runtime
                log.debug("setOverlayRaw(%s) failed: %s", module_id, exc)
                return False

    # ----- Visibility --------------------------------------------------

    def set_visible(self, module_id: str, visible: bool) -> None:
        if not self.available:
            return
        entry = self._entries.get(module_id)
        if entry is None:
            return
        with self._lock:
            try:
                if visible and not entry.visible:
                    self._vr_overlay.showOverlay(entry.handle)
                elif not visible and entry.visible:
                    self._vr_overlay.hideOverlay(entry.handle)
                entry.visible = visible
            except Exception as exc:  # pragma: no cover - runtime
                log.debug("set_visible(%s) failed: %s", module_id, exc)

    # ----- Runtime introspection --------------------------------------

    def runtime_status(self) -> VRRuntimeStatus:
        """Return live evidence that our VR layer is wired up.

        Best-effort: every probe is wrapped so a missing pyopenvr API
        on older versions never breaks the call. The result is meant
        for the UI status label, never for control-flow decisions.
        """
        if not self.available or self._openvr is None:
            return VRRuntimeStatus()

        hmd_connected = False
        hmd_model: str | None = None
        scene_pid: int | None = None
        scene_name: str | None = None

        with self._lock:
            try:
                vrsys = self._openvr.VRSystem()
                # k_unTrackedDeviceIndex_Hmd == 0
                hmd_connected = bool(vrsys.isTrackedDeviceConnected(0))
                if hmd_connected:
                    try:
                        prop = self._openvr.Prop_TrackingSystemName_String
                        hmd_model = vrsys.getStringTrackedDeviceProperty(
                            0, prop,
                        )
                        if isinstance(hmd_model, bytes):
                            hmd_model = hmd_model.decode(
                                "utf-8", errors="replace",
                            )
                        hmd_model = (hmd_model or "").strip() or None
                    except Exception:  # pragma: no cover - runtime
                        hmd_model = None
            except Exception:  # pragma: no cover - runtime
                pass

            try:
                vrapps = self._openvr.VRApplications()
                pid = int(vrapps.getCurrentSceneProcessId() or 0)
                if pid > 0:
                    scene_pid = pid
                    scene_name = _process_name_for_pid(pid)
            except Exception:  # pragma: no cover - runtime
                pass

        is_lfs = bool(
            scene_name and scene_name.lower() in {"lfs.exe", "lfs"}
        )
        return VRRuntimeStatus(
            hmd_connected=hmd_connected,
            hmd_model=hmd_model,
            scene_app_pid=scene_pid,
            scene_app_name=scene_name,
            scene_app_is_lfs=is_lfs,
        )

    # ----- Shutdown ----------------------------------------------------

    def shutdown(self) -> None:
        if not self._initialized:
            return
        with self._lock:
            for entry in list(self._entries.values()):
                with contextlib.suppress(Exception):  # pragma: no cover
                    self._vr_overlay.destroyOverlay(entry.handle)
            self._entries.clear()
            try:
                if self._openvr is not None:
                    self._openvr.shutdown()
            except Exception:  # pragma: no cover - runtime
                pass
            self._initialized = False
            self._vr_overlay = None
            log.info("OpenVR overlay sink shut down")


__all__ = ["OpenVROverlaySink", "OverlayPose", "VRRuntimeStatus"]
