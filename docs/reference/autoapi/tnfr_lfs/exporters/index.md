# `tnfr_lfs.exporters` package
Exporter registry for TNFR × LFS outputs.

## Submodules
- [`tnfr_lfs.exporters.lfs_native`](lfs_native/index.md)
- [`tnfr_lfs.exporters.report_extended`](report_extended/index.md)
- [`tnfr_lfs.exporters.setup_plan`](setup_plan/index.md)

## Classes
### `Exporter` (Protocol)
Exporter callable protocol.

## Functions
- `json_exporter(results: Dict[str, Any]) -> str`
- `csv_exporter(results: Dict[str, Any]) -> str`
- `markdown_exporter(results: Dict[str, Any] | SetupPlan) -> str`
  - Render a setup plan as a Markdown table with rationales.
- `lfs_notes_exporter(results: Dict[str, Any] | SetupPlan) -> str`
  - Render TNFR adjustments as F11/F12 instructions for Live for Speed.
- `resolve_car_prefix(car_model: str) -> str`
- `normalise_set_output_name(name: str, car_model: str) -> str`
- `lfs_set_exporter(results: Dict[str, Any] | SetupPlan) -> str`
  - Persist a setup plan using the textual LFS ``.set`` representation.
- `build_coherence_map_payload(results: Mapping[str, Any]) -> Dict[str, Any]`
- `build_operator_trajectories_payload(results: Mapping[str, Any]) -> Dict[str, Any]`
- `build_delta_bifurcation_payload(results: Mapping[str, Any]) -> Dict[str, Any]`
- `render_coherence_map(payload: Mapping[str, Any], fmt: str = 'json') -> str`
- `render_operator_trajectories(payload: Mapping[str, Any], fmt: str = 'json') -> str`
- `render_delta_bifurcation(payload: Mapping[str, Any], fmt: str = 'json') -> str`
- `coherence_map_exporter(results: Mapping[str, Any]) -> str`
- `operator_trajectory_exporter(results: Mapping[str, Any]) -> str`
- `delta_bifurcation_exporter(results: Mapping[str, Any]) -> str`

## Attributes
- `CAR_MODEL_PREFIXES = {'UF1': 'UF1', 'XFG': 'XFG', 'XRG': 'XRG', 'LX4': 'LX4', 'LX6': 'LX6', 'RB4': 'RB4', 'FXO': 'FXO', 'XRT': 'XRT', 'RAC': 'RAC', 'FZ5': 'FZ5', 'MRT': 'MRT', 'FBM': 'FBM', 'FOX': 'FOX', 'FO8': 'FO8', 'BF1': 'BF1', 'XFR': 'XFR', 'UFR': 'UFR', 'FXR': 'FXR', 'XRR': 'XRR', 'FZR': 'FZR', 'gt_fzr': 'FZR', 'gt_xrr': 'XRR', 'formula_fo8': 'FO8', 'formula_fox': 'FOX'}`
- `INVALID_NAME_CHARS = set('\\/:*?"<>|')`
- `REPORT_ARTIFACT_FORMATS = ('json', 'markdown', 'visual')`
- `exporters_registry = {'json': json_exporter, 'csv': csv_exporter, 'markdown': markdown_exporter, 'set': lfs_set_exporter, 'lfs-notes': lfs_notes_exporter, 'html_ext': html_exporter, 'coherence-map': coherence_map_exporter, 'operator-trajectory': operator_trajectory_exporter, 'delta-bifurcation': delta_bifurcation_exporter}`

