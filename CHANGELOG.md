# Changelog

Notable changes to LFS Race Engineer. This project follows semantic
versioning; dates are UTC.

## [0.5.2] — 2026-07-03

### Changed

- **VR overlays stay fixed in the cockpit.** Panels are anchored in
  SteamVR *seated* space (`setOverlayTransformAbsolute` with
  `TrackingUniverseSeated`) rather than tracking the headset, so they
  hold their place when you look around. Recenter the VR view (SteamVR
  or LFS) to set where they sit.
- The VR status label under *Mirror overlays to VR* refreshes live and
  reports the connected HMD, whether LFS currently owns the VR scene
  (`LFS scene detected`), and the display device configured in LFS's
  `cfg.txt`.
- The VR mirror copies each frame once — straight from the `QImage`
  memoryview into the OpenVR pixel buffer — instead of twice per panel
  on every tick.

### Fixed

- LFS VR-headset detection reads the current `G3D_OPTIONS` line in
  `cfg.txt` (display-device and VR-system fields), while still
  recognising the legacy `OpenVR Mode` / `Oculus Mode` lines. Installs
  configured for an OpenVR or Oculus headset are identified correctly
  and shown in the VR status label.

## [0.5.1] — 2026-07-02

### Fixed

- **VR mirror now renders inside the headset.** The SteamVR / OpenVR
  overlay path handed plain Python objects (a tuple and `bytes`) to
  pyopenvr calls that marshal through `ctypes.byref`, so every frame
  failed silently — the overlay appeared on the monitor but never in
  VR. The HMD-relative transform (`HmdMatrix34_t`) and the pixel buffer
  are now proper ctypes objects. The bug was runtime-agnostic, so this
  fixes Windows Mixed Reality (e.g. Samsung Odyssey+), Oculus and Valve
  Index alike.
- The capture subprocess is now stopped when the Studio window closes,
  so it no longer keeps the OutSim / OutGauge / InSim ports open or
  writes CSVs after you exit.
- The Live, Race-dashboard and Track-map docks stop their internal
  timers on close, avoiding stray callbacks during shutdown.
- `IS_CON` (car-to-car contact) now uses the correct 16-byte layout.
- `IS_MAL` (allowed-mods) parsing clamps the mod count to the packet
  length, so a truncated/malformed packet can no longer synthesise
  bogus mod IDs.
- RAF import reports a clear error on a malformed/undersized block
  header instead of crashing with a raw `struct` error.
- The Track-map SMX / KNW overlay now honours the configured LFS
  install folder instead of assuming `C:\LFS`.
- The lateral-grip (μ) aero fit no longer drops the fastest sample from
  its last speed bin.

### Changed

- Released builds bundle the VR mirror reliably: `openvr` is now part of
  the `[build]` extra (Windows).
- The Track-map dock reuses the shared racing-line CSV parser
  (`racing_line_loader`) instead of a private copy.
- Documentation reviewed end-to-end to match the current architecture;
  the user manual now lists every live-overlay module (added the
  speed-delta and pit-limiter entries).

### Removed

- Staged live-overlay modules that were never wired into the UI
  (mini-map, gap compass, TC/ABS slip LED).
