# LFS Race Engineer — User manual

Quick but complete guide to using **LFS Race Engineer** from the
installed application. You do not need Python or a terminal: everything
is done from the graphical interface.

---

## 1. Installation

1. Download `lfs-race-engineer-setup-x.y.z.exe`.
2. Run it and follow the wizard (you can install for the current user
   or system-wide). Takes a few seconds.
3. Launch **LFS Race Engineer** from the Start menu.

The application is self-contained: it does not install Python or any
extra libraries.

---

## 2. First launch: configure LFS (mandatory)

**Before capturing anything you must tell LFS to emit telemetry.**
LFS does not do it by default. The app can configure it
automatically:

1. **Close LFS** (if it is open, it will overwrite the changes on
   exit).
2. In LFS Race Engineer, open **Tools → Configure LFS…**
3. The dialog tries to autodetect your LFS install folder (e.g.
   `C:\LFS`). If it cannot find it, click **Browse…** and pick it
   manually (the folder that contains `LFS.exe` and `cfg.txt`).
4. Click **Patch cfg.txt automatically**.
5. The app:
   - makes a backup as `cfg.txt.bak` (only the first time);
   - inserts/updates the **OutSim**, **OutGauge** and port entries
     (loopback `127.0.0.1`) in `cfg.txt`;
   - shows a confirmation message with the backup path.
6. Close the dialog.

### Ports used

| Service | Port | Protocol | Purpose |
| --- | --- | --- | --- |
| OutSim   | 30000 | UDP | chassis telemetry (position, velocities, accelerations, per-wheel data when `OutSim Opts = 1ff`) |
| OutGauge | 30001 | UDP | car dashboard (revs, gear, fuel, temperatures, lights) |
| InSim    | 29999 | TCP | race events (laps, splits, positions, flags, pit) |

Everything is on `localhost` (nothing is exposed externally, no
firewall changes).

### InSim: final step inside LFS

InSim **cannot be enabled from `cfg.txt`** (LFS treats it as
invalid and shows a red warning). There are two equivalent ways to
turn it on, both inside LFS:

* **Recommended shortcut**: start LFS with `/insim=29999` appended
  to the shortcut parameters, or create a shortcut to
  `LFS.exe /insim=29999`. Always active.
* **Per session**: with LFS open, press <kbd>T</kbd> (chat) and type
  `/insim 29999`, then Enter.

Without InSim, the telemetry and stint tabs still work, but
**Overlay**, **Sectors** and comparison plots require line-cross
events.

---

## 3. UI overview

```
┌──────────────────────────────────────────────────────────────────┐
│ File   View   Tools   Help                       │  Language    │
├──────────────────────────────────────────────────────────────────┤
│ Captures  │                                       │  Channels    │
│ (laps)    │     Center tabs:                      │              │
│           │                                       │              │
│ Track     │   Telemetry · Dampers · Sectors ·     │  Race        │
│ map       │   Stint · Capture · Overlay           │  dashboard   │
│           │                                       │              │
│ Elevation │                                       │              │
└──────────────────────────────────────────────────────────────────┘
                              Status bar (cursor, context)
```

* **Side panels (docks)**: draggable; you can hide them from the
  **View** menu or by closing their X. **View → Reset Layout**
  restores the default layout.
* **Language**: **View → Language → English / Spanish**. Switches
  live and is remembered.
* Window geometry, channel selection and active workspace are
  persisted in `~/.lfs-telemetry/studio.json`.

### Keyboard shortcuts

| Shortcut | Action |
| --- | --- |
| <kbd>Ctrl</kbd>+<kbd>O</kbd> | Open workspace folder |
| <kbd>F5</kbd> | Refresh the captures list |
| <kbd>F1</kbd> | **Help → Channel guide…** (what each channel measures) |
| <kbd>Ctrl</kbd>+<kbd>Q</kbd> | Quit |

The **Help → User manual…** entry opens this very document from
inside the application (in the active UI language).

---

## 4. Workspace and lap selection (**Captures** panel)

The left panel lists every CSV capture in the active folder, with
car, track, lap count and best time.

> **Important — the workspace must be a folder where your Windows
> user has read and write permissions.** The app stores capture
> CSVs, sliced laps and the analysis cache there. Folders such as
> `C:\Program Files\...`, `C:\Windows\...` or network drives without
> permissions will typically fail. If you see errors like *"Access
> denied"*, *"Permission denied"*, or **Start** stops immediately,
> **switch the workspace to a folder where you have access** (for
> example `Documents\LFS-Telemetry`, `Desktop\stints` or any folder
> inside your user profile) via **File → Open Workspace…**
> (<kbd>Ctrl</kbd>+<kbd>O</kbd>).

* **Change workspace**: drag a folder onto the panel, or use
  **File → Open Workspace…** (<kbd>Ctrl</kbd>+<kbd>O</kbd>).
* **Text filter**: type in the top box to narrow by filename, car
  or track.
* **Load a lap**: double-click the row.
* **Compare multiple laps at once** (overlay in Telemetry, Dampers,
  Sectors, Stint):
  - <kbd>Ctrl</kbd>+click — add/remove individual laps from the
    selection.
  - <kbd>Shift</kbd>+click — select a contiguous range.
  - <kbd>Ctrl</kbd>+<kbd>A</kbd> — select everything visible.
* **Refresh** the list after a new capture: <kbd>F5</kbd>.

The lap marked as **reference** (usually the best in the set) is
used for the delta-time in Telemetry and for the live delta in
Overlay.

---

## 5. Center tabs

### 5.1. Telemetry — multi-channel viewer

MoTeC-style viewer on top of `pyqtgraph`. Each **channel selected**
in the right-hand **Channels** panel occupies a horizontal lane
with its own Y scale, all synchronised on the X axis.

The toolbar contains:

* **X-axis:** two radio buttons **Distance** / **Time** (the choice
  applies to every lane).
* **Export PNG…** — exports the full plot stack as an image.
* **Export CSV…** — exports the visible channels, time-aligned, as
  CSV.
* Side caption with the number of loaded laps.

Below that, a **legend** shows a colour pill per lap (the first one
is the **reference** and is tagged with ` (ref)`).

Chart interaction:

* **Zoom**: mouse wheel = X zoom; right-drag = X pan; double-click
  = vertical autoscale of the lane.
* **Cursor**: move the mouse over any lane; a vertical line marks
  the same position on every other lane and the status bar shows
  the instantaneous values.
* **Delta vs reference**: with several laps selected, an extra
  `Δt vs ref [s]` lane appears automatically.
* **Decimation**: each lane is reduced to ≤ 4 000 points by
  min-max-per-bucket to preserve peaks without choking the render.

#### Available channels (**Channels** panel)

Grouped tree, with a checkbox per channel and a text filter. Groups
(defined in `lfs_telemetry.telemetry.channels`):

| Group | Contents |
| --- | --- |
| **Driver** | Driver inputs: throttle, brake, clutch, handbrake, steering and FFB torque. |
| **Vehicle** | Basic car state: speed, position and lap timing (OutGauge). |
| **Engine** | Power-train: RPM, gear, fuel and engine/oil temperatures (OutGauge). |
| **Chassis** | OutSim chassis dynamics: 3-axis acceleration (m/s²), 3-axis angular velocity (rad/s) and Euler attitude angles. |
| **Suspension** | Per-corner suspension state: travel, vertical load, damper velocity and wheel steer angle. |
| **Tyre** | Per-corner tyre behaviour: slip ratio, slip angle, long./lat. forces, internal temperature and contact flags. |
| **Aids** | OutGauge ShowLights bitfield: TC, ABS, pit limiter, turn signals and warning lights. |
| **Derived** | Magnitudes computed by Studio from the raw OutSim/OutGauge channels (understeer index, weight transfer, friction use, …). |
| **Lap** | Lap-relative distances and indices used to align traces. |
| **Track** | Static track geometry sampled at the current racing-line node: curvature, radius, slope, width and lateral offset of the car. |
| **Context** | Session: car, track, weather, wind (used to filter and group captures). |

> Press <kbd>F1</kbd> to open the **Channel guide** from the
> **Help** menu: every channel with its unit and a short
> explanation.

### 5.2. Dampers — damper histograms

**One damper-velocity histogram per wheel** (FL, FR, RL, RR)
computed on the **first selected lap**. The two dashed vertical
lines mark the **low/high-speed boundary**, configurable in the
toolbar via the **Low-speed boundary** spin box (defaults to
**±25 mm/s**, the convention used by MoTeC, AIM RaceStudio and
Cosworth Pi).

Under each histogram, four key metrics are summarised on one line:
`Reb avg`, `Hi-reb %`, `Bump avg`, `Hi-bump %`.

If you tick **two laps**, the second one is overlaid as a dashed
white step outline and the summary switches to compare mode (A vs B,
with Δ mm/s and Δ %).

Use it to spot:

* left/right asymmetries,
* excess time in high speed (aggressive kerbs, axle bounce),
* compression/rebound split imbalance between laps or setups.

### 5.3. Sectors — splits

Two elements over the selected laps:

* a **header summary** with the split source (`InSim splits` when
  InSim events are present, `uniform ×N` when they are not and the
  lap time is distributed by distance) and the stint **theoretical
  best** time (sum of the best individual sector times).
* a **"Sector times" bar chart** grouped per lap, with one colour per
  sector and `Lap #` on the X axis.

### 5.4. Stint — multi-lap analysis

Lap-by-lap evolution view. At the top, a **stint summary** (best,
average, theoretical best, pace drop, total fuel and laps left, G
peaks and per-wheel temperature trend); below it, a stack of
**seven plots** with `Lap #` on the X axis:

1. **Lap times** (s) — per-lap bars plus the average (dashed line).
2. **Fuel** (%) — % consumed per lap and % left at lap end.
3. **Tyre temp end-of-lap** (°C) — one line per wheel.
4. **Peak vertical load (suspension)** (kN) — peak per wheel.
5. **Friction use p95 (circle saturation)** — 95th-percentile use of
   the friction circle.
6. **Grip index (per wheel)** (%) — per-wheel grip index.
7. **Damper work — RMS shaft speed** — damper effort.

Selecting several CSV files (a whole stint) shows how the car
evolves over the session.

### 5.5. Capture — telemetry recording

This is where you drive the subprocess that captures LFS live.

Form:

| Field | Meaning |
| --- | --- |
| **Filename stem** | Prefix for the CSVs saved into the workspace. |
| **InSim host / port** | Normally `127.0.0.1` / `29999`. |
| **OutSim port** | Normally `30000`. |
| **OutGauge port** | Normally `30001`. |
| **Overlay only (no CSV recording)** | If you tick it, the app connects to LFS and feeds the **Overlay** live, **without writing any CSV** in the workspace. Handy to use the HUD in a free session without leaving recordings. |

Buttons:

* **Start** — launches the capture. The status LED changes:
  - **grey** = idle,
  - **amber** = waiting for LFS / car still in pits,
  - **green** = InSim connected and samples flowing.
* **Stop** — clean shutdown (flushes buffers and closes files).

The **embedded log** shows the capture messages (laps closed,
flags, dropped packets, etc.).

Above the form, a label shows the **active workspace** (the folder
where CSVs will be written) and below the buttons a counter
**Laps recorded: N** increments with every completed lap.

Laps are split automatically at start/finish crossings and show up
in the **Captures** panel without needing <kbd>F5</kbd>.

### 5.6. Overlay — live race-engineer HUD

The **Overlay** tab is not a single panel: it is a **manager of
independent floating windows**. Each module you enable opens as a
**frameless, always-on-top** window that you can place anywhere over
LFS. All modules are fed by a JSON snapshot
(`<workspace>/_overlay/live.json`) refreshed at ~10 Hz by the
capture process.

**Behaviour shared by every window:**

* Drag the **body** to move it.
* Drag the **bottom-right corner** to resize it.
* **Right-click** the window to reset its size to the default.
* Each module remembers its **position, size and opacity** between
  sessions (stored in `QSettings`).
* **Opacity** is per module (20–100 %) and is set from the right
  column of the module list in the tab.
* **Deselect all** — a button at the top of the module list hides
  every overlay window in one click; useful when the screen is
  cluttered or when switching between configurations (race vs
  hot-lap vs setup work).

**Available modules** (in the order they appear in the list):

| Module | What it shows |
|---|---|
| **Radar** | 360° radar with surrounding cars (blue = ahead, red = behind). Scale and colour thresholds are configurable. |
| **G-meter (friction circle)** | Friction circle with instantaneous longitudinal and lateral acceleration. Full scale configurable in g. |
| **Delta bar vs personal best** | Horizontal bar with the delta against your personal best lap (green = gaining, red = losing). Full scale configurable in ms. |
| **Session info (dynamic)** | Session summary: current lap, last lap, best lap, session time, etc. In detailed mode the window auto-resizes to fit the full live standings table (every classified driver). Optional compact mode. |
| **Grip (per wheel)** | Grip/risk indicator per wheel (4 segments), useful to spot grip loss or overheating. |
| **Gap to driver ahead** | Time gap to the car ahead (decoded from InSim). Robust against disconnected/DNF cars in the position table, stationary pit/spectator cars, and lap-mismatch wrap artifacts. |
| **Gap to driver behind** | Time gap to the car behind. Same robustness as Gap to ahead. |
| **Gear (big digit)** | Current gear as a large digit. |
| **RPM bar** | RPM bar with configurable redline. |
| **Speed (km/h)** | Current speed in km/h. |
| **Fuel %** | Remaining fuel as a percentage. |
| **Fuel laps remaining** | Laps remaining with the current fuel at the average consumption observed during the session. |
| **Flags (BLUE / YELLOW)** | Blue and yellow flag indicator decoded from InSim. |

**Configuration panels** (below the module list):

* **Radar** — scale (m), Red / Yellow / White distance thresholds (m)
  used to colour surrounding cars by proximity.
* **Delta bar** — full scale ± (ms), default ±2000 ms.
* **RPM** — redline (rpm), default 8000.
* **G-meter** — full scale (g), default 2.0 g.
* **Session overlay compact** — show session info in condensed form.
* **Borderless / windowed-fullscreen compat** — use regular top-most
  windows to improve overlay visibility when LFS runs in windowed
  or borderless mode.

> **Important — LFS exclusive fullscreen**: Windows cannot draw any
> overlay (ours, RTSS, Discord, Steam, etc.) on top of a DirectX
> exclusive-fullscreen game. If overlays are invisible while LFS is
> in fullscreen, open `LFS\cfg.txt` and set `Full screen window 1`
> (LFS's borderless windowed mode), or use a regular windowed mode.
> The **VR mirror** (see below) is the only path that bypasses this
> limit, because SteamVR has its own compositor that runs above the
> game's swap chain.

### 5.7. VR mirror (SteamVR / OpenVR)

The Live tab includes a **VR** group with a single checkbox:

* **Mirror overlays to VR (SteamVR / OpenVR)** — when enabled, every
  visible overlay module is also rendered as an `IVROverlay` panel
  anchored to the HMD. The same Qt widget is the source of truth:
  the desktop window and the VR panel show identical content. There
  is no second look-and-feel to configure.

**Why VR works where exclusive fullscreen doesn't**

SteamVR has its own scene compositor that runs above any DirectX /
Vulkan swap chain. Pushing a texture to `IVROverlay` makes SteamVR
draw it on top of whatever the game shows in the headset. This
works for LFS in any windowing mode, including exclusive fullscreen,
and it works on any OpenVR-compatible runtime (Valve Index, HTC Vive,
Windows Mixed Reality via OpenVR, Oculus headsets via OpenComposite,
Meta Quest with Steam Link, etc.).

**Requirements**

* SteamVR (or another OpenVR-compatible runtime) installed and
  running before you tick the checkbox.
* The `openvr` Python package. Already shipped inside the Windows
  installer; if you run from source, install with
  `pip install lfs-race-engineer[vr]`.
* The HMD has to be tracking (not in standby). The default pose
  places overlays roughly 1.5 m in front of the headset.

**Behaviour**

* Toggle is a no-op if SteamVR isn't running or the `openvr` module
  is missing — the checkbox bounces back to **off** and the status
  label below it shows the reason (e.g. `VR mirror unavailable: ...`).
* While enabled, a 30 Hz timer reads each visible overlay module,
  renders it off-screen to a transparent `QImage`, converts to
  `RGBA8888`, and uploads it via `IVROverlay.SetOverlayRaw`. There
  is no extra CPU/GPU cost when the toggle is off.
* Hiding a module (unchecking it in the modules list) also hides
  the corresponding VR overlay on the next tick.
* Closing the Live tab or the app shuts down all VR overlays
  cleanly and releases the OpenVR session.

**Default panel layout**

Overlays are arranged in a soft arc 1.5 m from the headset, slightly
below eye level so they don't sit on the apex of corners. Each
overlay is ~40 cm wide in world units. Per-module pose customisation
(move/scale individual panels in the headset) is on the roadmap;
today the defaults are tuned to be readable without occluding the
racing line.

**Troubleshooting**

* *Checkbox keeps unchecking itself* — read the status label. The
  most common cause is SteamVR not running.
* *Panels are visible but blank* — the source module hasn't received
  any telemetry yet. Start a session in LFS or load a replay.
* *Overlay desktop window also visible in flat monitor* — that's
  intentional. Both targets share the same Qt widget; you can move
  the desktop window off-screen if you only want the VR panel.

> **Cars supported by Overlay**: stock LFS cars and verified mods
> (those that have a record in `config/cars.json`, in the bundled
> `car_info.bin` files, or footprints under `assets/source/mods/`).
> For unknown cars, capture and the other tabs still work, but the
> widgets that depend on vehicle-specific data (Fuel %, Fuel laps
> remaining, gear indicator scaling) may show neutral values.

---

## 6. Additional side panels

* **Track map** (left) — averaged racing line of the active lap,
  start/finish marker, cursor synchronised with Telemetry, and the
  **bundled reference** of the track in grey if available
  (`racing_lines/<TRACK>_racing.csv`).
* **Elevation** (left) — altitude profile z(s) with banking bands
  and surface classification from the `.smx`.
* **Race dashboard** (right) — a tile board with the race data in
  large format: `Position`, `Lap`, `Current lap`, `Last lap`,
  `Best lap`, `Predicted`, `Δ vs best`, `SPB`, `Avg (stint)`,
  `Gap ahead`, `Gap behind`, `Fuel`, `Fuel laps left`, `Speed`,
  `Gear`, and a **standings** table underneath.

---

## 7. Tools menu

* **Configure LFS…** — the dialog described in section 2 (patches
  `cfg.txt` with the OutSim/OutGauge settings).

(The bundled racing line for a track is loaded automatically from
`racing_lines/<TRACK>_racing.csv` if it exists; there is no manual
loader dialog.)

## 8. File menu

* **Open Workspace…** (<kbd>Ctrl</kbd>+<kbd>O</kbd>) — pick the
  folder where CSVs are saved/read.
* **Refresh Captures** (<kbd>F5</kbd>) — reload the list.
* **Clear Lap Cache** — wipes the parquet cache on disk that speeds
  up reloads (useful if a lap is corrupted or you want to recompute
  the derived channels).
* **Import RAF…** — import an LFS **Replay Analyser File** (`.raf`).
  The app walks the RAF, splits it into laps (using the *index
  distance* track-ruler wrap to detect start/finish crossings) and
  writes one CSV per lap into `<workspace>/<name>_raf_laps/`. After
  import the laps appear in **Captures** and can be loaded and
  compared like any other capture. This is the only official path to
  analyse another driver's telemetry from their replay (`.mpr`/`.spr`):
  open the replay in LFS, press **Analyse** to generate the `.raf`,
  then import it here.
* **Quit** (<kbd>Ctrl</kbd>+<kbd>Q</kbd>) — close the application.

## 8 bis. Help menu

* **User manual…** — open this manual from inside the app, in the
  active UI language.
* **Channel guide…** (<kbd>F1</kbd>) — telemetry guide: what each
  channel measures and how to read it.
* **About** — version information.

---

## 9. Recommended workflow

1. **Configure LFS once** (Tools → Configure LFS) and enable InSim
   on the shortcut with `/insim=29999`.
2. **Create a workspace folder** per season or per car, and open it
   with <kbd>Ctrl</kbd>+<kbd>O</kbd>.
3. In **Capture**, press **Start**, enter the track in LFS, drive
   your laps, pit, press **Stop**.
4. In the **Captures** dock, select your best lap and add the ones
   you want to compare with <kbd>Ctrl</kbd>+click. Tick the
   channels in the **Channels** panel and use them in the
   **Telemetry** tab.
5. For race sessions without recording, tick **Overlay only** in
   Capture and keep the **Overlay** tab visible next to LFS.

---

## 10. Quick troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Capture LED stays grey/amber | InSim not active in LFS | In LFS press <kbd>T</kbd> and type `/insim 29999`, or restart LFS with `/insim=29999`. |
| No per-wheel temperatures/loads | LFS is in `OutSim Mode 1` (legacy 64 B mode) | Re-run Tools → Configure LFS and apply the patch (keeps `OutSim Mode 2` and `OutSim Opts 1ff`). |
| Patch says "invalid LFS folder" | The folder does not contain `LFS.exe` | Use **Browse…** and pick the right folder. |
| Overlay shows no fuel range / wear | Car is not in the supported list | Capture works normally; the other panels work. To add it manually, run the calibration (advanced use). |
| LFS overwrote my changes | LFS was open when patching | Close LFS and run Tools → Configure LFS again. |
| I want my original `cfg.txt` back | The patch left `cfg.txt.bak` | Rename it back to `cfg.txt` with LFS closed. |
| **Start** stops instantly or throws *"Access/Permission denied"* | The workspace is in a folder where your user has no write permission (Program Files, Windows, a protected network drive…) | **File → Open Workspace…** and pick a folder where you have permission, e.g. inside `Documents` or `Desktop`. |
