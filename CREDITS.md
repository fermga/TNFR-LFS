# Credits & Acknowledgements

LFS Race Engineer would not exist without the prior work of the Live
for Speed community. While the code in this repository is original and
licensed under MIT (see `LICENSE` / `pyproject.toml`), the project drew
inspiration from the projects listed below. Each is credited to its
author with its upstream URL and licence so that anyone reusing parts
of this codebase can trace the lineage.

If you are an author listed here and would like the wording, link or
attribution adjusted (or removed), please open an issue on
<https://github.com/fermga/LFS-Race-Engineer/issues> and it will be
addressed promptly.

## Inspirations

### LFSTelemetry — Cyril Bissey

OutSim/OutGauge wire-format parsing and telemetry capture patterns
were studied while learning the LFS UDP packet layouts.

* Author: **Cyril Bissey**
* Upstream: <https://github.com/CyrilBissey/LFSTelemetry>
* Licence: **MIT** © 2024 Cyril Bissey

### helicorsa — Jens Lohmann

Proximity / situational-awareness HUD ideas (the "radar" style display
of nearby cars and the per-quadrant alert thresholds) were inspired by
helicorsa, which is an Assetto Corsa app rather than an LFS tool but
solves the same problem space we tackle for LFS.

* Author: **Jens Lohmann (jenslohmann)**
* Upstream: <https://github.com/jenslohmann/helicorsa>
* Licence: **MIT**

### Detect&Monitor — KingOfIce ("Gum Garage Tool")

The "Detect" half of *Detect&Monitor* (rear-view radar / proximity
warnings driven by InSim + OutGauge) and several of its UX choices
(toggle commands like `/o dm_toggle_radar`, configurable alert
distances) influenced the way our own InSim-driven overlays surface
gap and proximity information.

* Author: **KingOfIce**
* Upstream / discussion: Live for Speed forums
  <https://www.lfs.net/forum> (search "Detect&Monitor")
* Licence: proprietary — *Copyright KingOfIce*. **No source code or
  binaries from Detect&Monitor are redistributed in this repository**;
  it is credited here for inspiration only.

### Live for Speed track geometry (`.smx`, `.pth`)

Track geometry is parsed from files produced by Live for Speed itself.
We rely on the documented `.pth` / `.smx` / `.knw` formats to compute
per-segment racing-line nodes and width corridors.

* Source: **Live for Speed** by Scawen Roberts, Eric Bailey and
  Victor van Vlaardingen.
* Upstream: <https://www.lfs.net/>
* Licence: proprietary. Live for Speed track files are not
  redistributed by this project — users must own a copy of LFS to
  obtain them; the `assets/tracks/*.tif` top-down map images shipped
  alongside this repository are screenshots of the official LFS map
  view used as background overlays for the racing-line view.

## Notes on history

Earlier revisions of this repository included copies of some of the
above projects under `assets/` while the codebase was being
prototyped. Those copies have been removed from `HEAD`, and any blobs
that lacked a redistribution licence have been purged from the git
history.

If a third-party project you authored still appears in any tag,
release artifact or commit and you would like it removed, please file
an issue and the relevant tag/release will be revised.
