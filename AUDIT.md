# Repo Audit — LFS Race Engineer

Tracking document for the audit of 2026-05-21. Tick items as they are completed.
Status legend: `[ ]` pending · `[~]` in progress · `[x]` done · `[-]` rejected/out of scope.

Baseline metrics (2026-05-21, post-rollout):
- Total `src/` Python files: **115** (was 95 pre-audit — sub-package splits)
- Total `src/` LOC: **29 550** (was 32 920 — dead-code removal + de-duplication)
- Bundle size: **386 MB** · Installer: **151.1 MB** · Version: **0.4.4** (locked)
- Test suite: **307 passed, 15 skipped** (was 280 pre-audit; +27 from T1/T2)

---

## Quick wins (≤30 min each, ordered by ratio impact/risk)

- [x] **QW1 — `print()` → `logging` in library code** (Low risk, ~10 min) _done 2026-05-21_
  - Only real runtime `print` in `telemetry/` library was `protocol/insim.py` L674 (stderr reconnect message); replaced with existing `_LOG.warning(...)`.
  - `insim.py` L12 and `traffic.py` L13 are docstring examples (not executable); left as-is.
  - `cli.py` / `lfs_config.py` prints kept: they are user-facing CLI output (correct usage).

- [x] **QW2 — Unit-conversion constants** (Low risk, ~10 min) — DONE 2026-05-21
  - Added `SPEED_MS_TO_KMH`, `SPEED_KMH_TO_MS`, `SPEED_MS_TO_MPH`, `TEMP_K_TO_C_OFFSET`, `PRESSURE_PA_TO_BAR`, `PRESSURE_PA_TO_PSI`, `TORQUE_NM_TO_LBFT` to `constants.py`.
  - Migrated 7 `* 3.6` call sites in `cli.py`, `telemetry/lap.py`, `telemetry/live_publisher.py` (3), `telemetry/track/knw.py` (2) to `SPEED_MS_TO_KMH`.
  - 280/280 tests passing.

- [x] **QW3 — Centralize asset/config path lookups in `app_paths.py`** (Low risk, ~20 min) — DONE 2026-05-21
  - Created `src/lfs_telemetry/app_paths.py` with `candidate_search_roots`, `candidate_asset_dirs`, `candidate_doc_roots`, `candidate_racing_lines_dirs`, `find_racing_line_csv`, `manual_doc_path`, `mod_database_path`, `cars_json_path`, `car_info_bin_dirs`. (Kept distinct from `lfs_paths.py` which is about the user's LFS install dir.)
  - Migrated `studio/widgets/manual_dialog.py`, `studio/widgets/track_map_dock.py::_racing_line_dirs`, `telemetry/track/loader.py::candidate_racing_lines_dirs`, `telemetry/mods.py::mod_database_path`, `telemetry/observables.py::_asset_search_dirs` to delegate to `app_paths`.
  - 280/280 tests passing.

- [x] **QW4 — Extract Spanish i18n from `channels.py`** (Low risk, ~30 min) — DONE 2026-05-21
  - Created `src/lfs_telemetry/telemetry/i18n_es.py` (36 KB) with all ES tables: `_GROUP_ES_FALLBACK`, `_LABEL_ES_FALLBACK`, `_DESCRIPTION_ES_FALLBACK`, `_INTERP_BY_COLUMN_ES`, `_INTERP_BY_SUFFIX_ES`, `_INTERP_BY_PATTERN_ES`, `_INTERP_BY_GROUP_ES`, `_FOCUS_BY_COLUMN_ES`.
  - `channels.py` now imports them from `i18n_es`. File shrank from ~2010 to ~1340 lines (53.9 KB).
  - 280/280 tests passing.

- [x] **QW5 — Centralize colour palette** (Low risk, ~20 min) — DONE 2026-05-21
  - Extended `src/lfs_telemetry/studio/theme.py` with semantic colours: `STATUS_ERROR_COLOR`, `LED_IDLE_COLOR`, `LED_ERROR_COLOR`, `LED_OK_COLOR`, `PROXIMITY_RED/YELLOW/WHITE/FAR`, `COMPARE_OUTLINE_COLOR`.
  - Migrated `live_modules.proximity_color()`, `capture_tab` LED colours (init + status update), `dampers_tab` compare-overlay pen, `channels_dock` error-status QSS. `charts_dock.py` already takes a colour parameter — no hardcoded literal.
  - 280/280 tests passing.

- [x] **QW6 — Tighten ruff rules** (Low risk, ~30 min) — DONE 2026-05-21
  - Extended `[tool.ruff.lint] select` to `["E", "F", "I", "UP", "B", "SIM", "C4", "PIE", "RUF", "ERA", "ASYNC"]`.
  - Added `[tool.ruff.format]` section (double quotes, space indent, LF endings).
  - Project-wide ignores for `RUF001/002/003` (TNFR uses Greek letters ν, Δ + Spanish i18n) and `ERA001` (annotated physics derivations look like commented-out code).
  - Per-file ignores for `tools/**` and `scripts/**` (dev utilities use one-line semicolons / single-letter vars on purpose).
  - Applied 71 auto + unsafe-fixes (pairwise, contextlib.suppress, dict() literals, sorted __all__, etc.). Manual fixes in `channels.py` (E402 import order after QW4), `live_modules.py` (RUF059 → `_` prefix), `openvr_overlay.py` (B008 default + SIM105), `cli.py` (ASYNC240 noqa for stop-file polling).
  - `ruff check .` reports **All checks passed!**. 280/280 tests passing.

- [x] **QW7 — Minimal CI workflow** (Low risk, ~20 min) — DONE 2026-05-21
  - Created `.github/workflows/test.yml`. Matrix: Ubuntu × py3.11/3.12/3.13 + Windows × py3.13 smoke. Installs Linux Qt system libs (libgl1, libegl1, libxkbcommon0, libxcb-cursor0, libdbus, libfontconfig1) so PySide6 can spin up in `QT_QPA_PLATFORM=offscreen` mode. Steps: `pip install -e ".[dev]"` → `ruff check .` → `pytest -q`.

- [x] **QW8 — Header comments on build scripts** (Low risk, ~10 min) — DONE 2026-05-21
  - `scripts/build_app.ps1` already had a header. Added "purpose + when to use" headers to `scripts/build_app_simple.ps1` (minimal PyInstaller wrapper, no validation/installer) and `scripts/build_installer.ps1` (Inno Setup compile of existing bundle). Each header cross-references the other build scripts.

---

## Module-size hotspots (split candidates)

- [x] **MH1 — Split `studio/widgets/live_modules.py` (2445 LOC)** · Effort L · Risk Med
  - Target sub-package `src/lfs_telemetry/studio/widgets/modules/`:
    - `_base.py` — `_LiveModuleWindow` base (~220 LOC)
    - `simple.py` — Position / FuelPct / FuelLapsRemaining / Speed
    - `inputs.py` — Gear / Rpm / pedals
    - `gaps.py` — gap ahead/behind
    - `session.py` — `SessionInfoWindow` (527 LOC alone)
    - `diagnostics.py` — TyreRisk / Flags / PitLimiter / TcAbs / GMeter
    - `driving_aids.py` — DeltaBar / SteeringWheel / BrakesWheel
    - `traffic.py` — Radar
  - Keep `live_modules.py` as a thin re-export shim for back-compat.
  - 2026-05-21: Done. Converted to package `studio/widgets/live_modules/` with `__init__.py` 83 LOC re-export shim and submodules: `_base.py` 355 LOC (base + helpers + `_LabeledValueWindow`), `simple.py` 71, `inputs.py` 187, `gaps.py` 79, `tyre_risk.py` 381, `session.py` 551, `diagnostics.py` 309, `compass_map.py` 244, `radar.py` 176, `delta_bar.py` 237. Public class names re-exported so `from lfs_telemetry.studio.widgets.live_modules import XxxWindow` still works. 280 tests pass, ruff clean.

- [x] **MH2 — Split `telemetry/channels.py` (2012 LOC)** · Effort M · Risk Low
  - After QW4 (i18n_es extraction → ~1400 LOC).
  - Further extract `_interpretation_for_lang` + `_focus_notes_for` (~960 LOC) to `telemetry/channel_interpretations.py`.
  - Keep `channels.py` focused on registry + dataclass + public API.
  - 2026-05-21: Done. `channels.py` 381 LOC (header + `ChannelInfo` + `_BASE` + `_build_registry` + public API); `i18n_es.py` 801 LOC (ES labels/descriptions/groups + ES interp tables + `_FOCUS_BY_COLUMN_ES`); `channel_interpretations.py` 883 LOC (EN focus/interp tables + ES focus sub-tables + `_interpretation_for`, `_interpretation_for_lang`, `_focus_notes_for`). 280 tests pass, ruff clean.

- [x] **MH3 — Split `cli.py` (1317 LOC)** · Effort M · Risk Med
  - Target sub-package `src/lfs_telemetry/cli/`:
    - `__init__.py` — `main()` parser + dispatch
    - `_common.py` — `_request_stop`, `_add_lfs_flags`, `_harden_std_streams`, `_ResilientTextStream`
    - `capture.py` — `_cmd_capture` (~614 LOC)
    - `calibrate.py` — `_cmd_calibrate`
    - `reslice.py` — `_cmd_reslice`
    - `raf_import.py` — `_cmd_raf_import`
  - Ensure entry-point `lfs-telemetry = lfs_telemetry.cli:main` still resolves.
  - 2026-05-21: Done. Sub-package layout: `__init__.py` 178 LOC (parser + dispatch + re-exports `_ResilientTextStream`, `main`, `_add_lfs_flags`, `_harden_std_streams`, `_request_stop`), `_state.py` 13 LOC (`STOP_REQUESTED`, `CAPTURE_LOOP`, `CAPTURE_TASK`), `_common.py` 109 LOC, `capture.py` 844 LOC, `calibrate.py` 75 LOC, `reslice.py` 39 LOC, `raf_import.py` 114 LOC. Entry-point `lfs-telemetry = lfs_telemetry.cli:main` resolves. 280 tests pass, ruff clean.
- [x] **MH4 — Split `telemetry/protocol/packets.py` (1788 LOC)** · Effort L · Risk Med
  - Conservative split: extracted OutSim sections (basic + extended) into `telemetry/protocol/packets_outsim.py`. Re-exported from `packets.py` for back-compat.
  - 2026-05-21: Done. `packets_outsim.py` 271 LOC (OutSimPacket, OutSimPack2, OutSimWheel, `outsim2_size`, OSO_* constants, OUTSIM_* size constants). `packets.py` reduced 1788 → 1557 LOC. OutGauge/InSim retained inside `packets.py` due to shared `_cstr`/`decode_car_id` decoders — splitting further would have required extracting a shared helpers module with little maintainability gain. 280 tests pass, ruff clean.
  - `packets_outsim.py`, `packets_outgauge.py`, `packets_insim.py`.
  - Keep `packets.py` as re-export shim.

---

## Tests & coverage gaps

- [x] **T1 — `tests/studio/test_live_modules.py`** — 2026-05-21: Added with 22 parametrized `WINDOW_CLASSES` smoke tests + 1 coverage-guard test. Constructs each public overlay (Position, FuelPct, FuelLapsRemaining, Speed, Gear, Rpm, Throttle, Brake, Clutch, GapAhead, GapBehind, TyreRisk, SessionInfo, Flags, PitLimiter, TcAbs, GMeter, GapCompass, MiniMap, Radar, DeltaBar, SpeedDeltaBar) against a populated `LiveDataSource`, calls `render_to_image()`, and asserts a non-null `QImage`. 23/23 pass.
- [x] **T2 — `tests/studio/test_track_map_dock.py`** — 2026-05-21: Added with 4 smoke tests (construct-empty, overlay toggle state, opacity slot with clamp coverage, show/hide without crash). Complements existing `test_track_replay.py` (7 replay-transport tests). 4/4 pass.
- [x] **T3 — `tests/test_protocol_packets.py`** — Already satisfied by existing `tests/test_packets.py` (21 tests covering round-trip for `OutSimPacket`, `OutGaugePacket`, `OutSimPack2` full + partial opts, IS_STA, IS_NPL, IS_LAP, IS_NCN, IS_CNL, IS_SLC, IS_PLA, IS_MAL plus `decode_car_id` stock/mod/empty/unknown cases). No new file needed.

---

## Tooling & CI gaps

- [x] **TG0 — Dependency declarations audited** — `pyproject.toml` now correctly declares `[studio,dev,vr,build,scripts]`; no missing or unused runtime deps. (2026-05-21)
- [x] **TG1 — GitHub Actions** — Closed 2026-05-21. Fully covered by QW7: `.github/workflows/ci.yml` runs ruff + pytest on push/PR (Python 3.13).
- [x] **TG2 — Pre-commit** — 2026-05-21: Added `.pre-commit-config.yaml` with `pre-commit-hooks` v5.0.0 (trailing-whitespace, end-of-file-fixer, check-yaml, check-toml, check-merge-conflict, mixed-line-ending --fix=lf) and `ruff-pre-commit` v0.7.4 (ruff --fix + ruff-format).
- [x] **TG3 — Type checking** — 2026-05-21: Added `[tool.mypy]` section to `pyproject.toml` with `python_version = "3.13"`, `ignore_missing_imports = true`, `disallow_untyped_defs = false`, `warn_unused_ignores = true`, `warn_redundant_casts = true`, `check_untyped_defs = false`, `follow_imports = "silent"`, and `exclude = ["build/", "dist/", "installer/Output/", "tools/", "scripts/", "tests/"]`. Advisory only; stricter knobs per-module as coverage improves.
- [x] **TG4 — Ruff hardening** — Closed 2026-05-21. Fully covered by QW6: ruff `select` now includes `E,F,I,UP,B,SIM,C4,PIE,RUF,ERA,ASYNC` with calibrated `ignore` (`E501,RUF001-003,ERA001`) and per-file ignores for `tools/**` and `scripts/**`.

---

## Code-quality nits

- [x] **CQ1 — `except Exception` review** — 2026-05-21: Narrowed 10 sites:
  - `studio/widgets/track_map_dock.py` L317 (`TrackMap.from_lap` → `ValueError, KeyError, AttributeError`), L340 (lap t/d caching → `KeyError, ValueError, AttributeError, IndexError`), L442 (lap summary access → `AttributeError, KeyError, TypeError`), L916 (KNW/PTH load → `OSError, ValueError, KeyError, AttributeError`).
  - `studio/widgets/track_elevation_dock.py` L521 (banking profile → `ValueError, IndexError, AttributeError`), L527 (apex visibility → `ValueError, IndexError`), L538 (barrier offsets → `ValueError, IndexError, AttributeError`).
  - `telemetry/track/pin.py` L109 (PIN parsing → `OSError, ValueError, struct.error`).
  - `telemetry/track/enrich.py` L55 (PIN cache load → `OSError, ValueError`), L385 (PTH profile load → `OSError, ValueError`).
  - Each narrowed `except` clause carries a one-line rationale comment. 307 tests pass, ruff clean.
- [x] **CQ2 — TODO/FIXME/HACK sweep** — 2026-05-21: Grep across `src/**/*.py` produced 11 substring hits, all false-positives (the literal substrings "todo"/"xxx" appear inside Spanish UI text like "Deseleccionar todos", code identifiers like "autoDownsample", comment phrases like "InSim : TINY_xxx with no ReqI", and the markdown documentation about importing `XxxWindow`). No actionable `TODO`/`FIXME`/`HACK` markers exist in the codebase.
- [x] **CQ3 — Typing pass on `channels.py` private helpers** — 2026-05-21: Verified. `channels.py` public surface (`_build_registry`, `channel_info`, `channels_by_group`, `ChannelInfo.tooltip_html`) is fully typed with `dict[str, ChannelInfo]`, `list[str] | None`, etc. After MH2, the private helpers `_interpretation_for_lang`, `_focus_notes_for`, `_interpretation_for` moved to `channel_interpretations.py` and all carry positional-arg types + return-type annotations (`str`, `tuple[str, str]`). No untyped helpers remain.

---

## Scripts / tools housekeeping

- [x] **SH1 — Consolidate build scripts** — 2026-05-21: Reviewed. `scripts/build_app.ps1` (197 LOC, full pipeline with validation/cleanup/installer) and `scripts/build_app_simple.ps1` (83 LOC, fast iteration for manual smoke-tests) serve genuinely different purposes — both already carry clear top-of-file docstrings explaining when to use each, and `README.md` (Scripts and tools section) documents the split between `scripts/` (build/runtime helpers) and `tools/` (low-level binary-format research). No code change needed; keeping both intentionally.
- [x] **SH2 — Deleted `scripts/_dep_audit.py`** — 2026-05-21: Removed throwaway audit script.

---

## Out of scope / explicitly rejected

- [-] **Qt Linguist `.ts` migration** — current fallback-dict approach is lean enough for ES/EN; revisit only at ≥10 languages.
- [-] **Generic packet codec (msgpack/pydantic)** — LFS packets are too irregular; manual dataclasses are the right tool.
- [-] **Single `PathResolver` class** — incremental helpers in `lfs_paths.py` are simpler and equally testable.
- [-] **Merge `scripts/` and `tools/` directories** — separation of concerns is valuable; just document the intent.
- [-] **Split `channels.py` registry by channel group** — the registry is one logical schema; only i18n / interpretations should leave.

---

## Estimated total effort

~20–25 h for the full audit. Quick wins (QW1-QW8) alone are ~2.5 h and unlock most of the centralization value with Low risk.
