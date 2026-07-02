# Changelog

Notable changes to LFS Race Engineer. This project follows semantic
versioning; dates are UTC.

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
