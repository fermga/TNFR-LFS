# Preset workflow

The `docs/presets/` folder contains ready-made threshold targets for
popular combinations such as BL1 × XFG, AS3 × FZR and KY2R × FXR. Each
TOML file describes per-phase objectives (`target_delta_nfr`, slip
windows and the target `nu_f`) that can be copied into a working
profile. 【F:docs/presets/BL1_XFG.toml†L1-L15】【F:docs/presets/AS3_FZR.toml†L1-L15】【F:docs/presets/KY2R_FXR.toml†L1-L15】

To create a bespoke preset:

1. Duplicate the closest preset file and adjust the per-phase targets.
2. Drop the customised file into your project (e.g. alongside
   `profiles.toml`) and load it with :class:`tnfr_lfs.io.profiles.ProfileManager`.
3. When you call :meth:`ProfileManager.resolve`, the manager now pulls the
   base :class:`tnfr_lfs.recommender.rules.ThresholdProfile` for the
   selected car/track combination before applying your overrides. This
   preserves any track-level `phase_weights` and exposes them via the
   returned :class:`tnfr_lfs.io.profiles.ProfileSnapshot`. 【F:tnfr_lfs/io/profiles.py†L140-L176】

The CLI/HUD automatically uses the resolved `phase_weights` when
segmenting microsectors, so presets immediately influence the
recommendation flow. 【F:tnfr_lfs/cli/tnfr_lfs_cli.py†L1754-L1767】
