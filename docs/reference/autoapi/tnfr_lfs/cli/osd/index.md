# `tnfr_lfs.cli.osd` module
Runtime HUD controller for the on-screen display (OSD).

## Classes
### `ActivePhase`
Resolved microsector/phase context for the current record.

### `MacroStatus`
State exposed in the “Apply” HUD page.

### `HUDPager`
Track the active HUD page and react to button clicks.

#### Methods
- `pages(self) -> Tuple[str, ...]`
- `update(self, pages: Sequence[str]) -> None`
- `current(self) -> str`
- `index(self) -> int`
- `advance(self) -> None`
- `handle_event(self, event: Optional[ButtonEvent], layout: ButtonLayout) -> bool`

### `TelemetryHUD`
Maintain rolling telemetry and derive HUD snapshots.

#### Methods
- `append(self, record: TelemetryRecord) -> None`
- `pages(self) -> Tuple[str, str, str, str]`
- `plan(self) -> Optional[SetupPlan]`
- `update_macro_status(self, status: 'MacroStatus') -> None`

### `OSDController`
High-level orchestrator that drives the HUD overlay.

#### Methods
- `run(self) -> str`
- `set_menu_open(self, open_state: bool) -> None`

## Attributes
- `logger = logging.getLogger(__name__)`
- `PAYLOAD_LIMIT = OverlayManager.MAX_BUTTON_TEXT - 1`
- `DEFAULT_UPDATE_RATE = 6.0`
- `DEFAULT_PLAN_INTERVAL = 5.0`
- `MOTOR_LATENCY_TARGETS_MS = {'entry': 250.0, 'apex': 180.0, 'exit': 220.0}`
- `NODE_AXIS_LABELS = {'yaw': 'Yaw', 'roll': 'Roll', 'pitch': 'Pitch'}`
- `MODAL_AXIS_SUMMARY_LABELS = {'yaw': 'Yaw', 'roll': 'Roll', 'pitch': 'Suspension'}`
- `SPARKLINE_BLOCKS = DEFAULT_SPARKLINE_BLOCKS`
- `SPARKLINE_MIDPOINT_BLOCK = SPARKLINE_BLOCKS[len(SPARKLINE_BLOCKS) // 2]`
- `HUD_PHASE_LABELS = {'entry1': 'Entry 1', 'entry2': 'Entry 2', 'apex3a': 'Apex 3A', 'apex3b': 'Apex 3B', 'exit4': 'Exit 4'}`
- `PACKING_HIGH_SPEED_THRESHOLD = 35.0`
- `PACKING_AR_THRESHOLD = 1.25`
- `PACKING_ASYMMETRY_GAP = 0.35`

