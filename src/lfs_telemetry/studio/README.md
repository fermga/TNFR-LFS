# `lfs_telemetry.studio` — Race Engineer Studio

Overlay PySide6 para el ingeniero de carrera. Frameless, oscuro,
diseñado para superponerse al cliente de LFS sin robarle foco. Toda la
lógica de telemetría vive en `lfs_telemetry.telemetry`; Studio solo
orquesta la UI, los modelos Qt y el ciclo de captura.

* Punto de entrada: `lfs-race-engineer` o `python -m lfs_telemetry.studio`.
* Dependencias específicas: `PySide6`, `pyqtgraph` (extra `[studio]`).
* Estado persistente: `workspace_state.py` (selección de captura,
  pesos de splits, geometría de docks).

---

## 1. Bootstrap

```
__main__.py  →  app.main()
                  │
                  ├── QApplication
                  ├── theme.apply(app)           # paleta oscura
                  ├── workspace_state.load()
                  ├── signals.Bus  (singleton)
                  ├── MainWindow  ← main_window.py
                  └── QTimer.singleShot(0, restore_last_capture)
```

* `app.py` configura locale, fuentes monoespaciadas y conecta los
  timers globales (refresco Live, polling del fichero stop).
* `theme.py` aplica el QSS canónico y la paleta dark-in-game.
* `signals.py` define un *bus* central de `Signal`s tipados que
  desacopla los docks: `captureSelected(Path)`, `lapChanged(int)`,
  `liveSnapshotUpdated(dict)`, `channelToggled(str, bool)`…
* `workspace_state.py` lee/escribe `~/.lfs-telemetry/studio.json` con
  la última captura abierta, splits y dock layout.

---

## 2. `MainWindow` y layout de docks

`main_window.MainWindow(QMainWindow)` registra cinco docks fijos
alrededor de un widget central tabulado:

| Posición | Dock | Widget |
| --- | --- | --- |
| Left | Captures | `widgets/captures_dock.py` |
| Left | Track map | `widgets/track_map_dock.py` |
| Left | Elevation | `widgets/track_elevation_dock.py` |
| Right | Channels | `widgets/channels_dock.py` |
| Right | Race dashboard | `widgets/race_dashboard_dock.py` |

El widget central es `widgets/center_tabs.CenterTabs(QTabWidget)`, con
las pestañas en este orden:

1. **Telemetry** — `widgets/charts_dock.py` + `charts/multi_chart.py`.
2. **Dampers** — `widgets/dampers_tab.py`.
3. **Sectors** — `widgets/sectors_tab.py`.
4. **Stint** — `widgets/stint_tab.py`.
5. **Capture** — `widgets/capture_tab.py` (controla la subprocess CLI).
6. **Overlay** — `widgets/live_tab.py` (consume `live_publisher` JSON).

### Cobertura de coches

* La **captura de telemetría y todas las pestañas basadas en CSV**
  (Telemetry, Dampers, Sectors, Stint) **funcionan con cualquier
  coche de LFS** — stock, mods o vehículos desconocidos — porque
  sólo dependen del wire-format OutSim/OutGauge/InSim, idéntico
  para todos.
* La **pestaña Overlay (HUD en vivo) está soportada para los coches
  propios de LFS y los mods verificados**. El HUD usa metadatos por
  coche (masa, depósito, gearing, compuestos de neumático, setup
  por defecto) leídos de `config/cars.json`, los `car_info.bin`
  empaquetados en `assets/source/cars/*.bin` y los footprints
  curados en `assets/source/mods/*.json`. Si el coche no aparece en
  ninguna fuente, la captura sigue funcionando, pero los widgets
  que requieren contexto específico (rango de combustible, lap
  predicho con consumo, desgaste de neumáticos, escalado del
  indicador de marcha) muestran valores neutros o quedan en blanco
  hasta que se registre una calibración con `lfs-telemetry
  calibrate`.

Los módulos `setup_tab.py` y `setup_editor_tab.py` siguen en el
repositorio como utilidad para futuras ampliaciones, pero **no están
cableados en `CenterTabs`** en la build actual.

El menú *Tools* abre `widgets/lfs_config_dialog.py` para parchear
`cfg.txt`. El módulo `widgets/racing_line_loader.py` sigue en el
repositorio porque lo usan internamente los renderers de mini-mapa y
brújula (compass/map), pero **ya no aparece como entrada de menú** en
la build actual.

---

## 3. Modelos (`studio/models/`)

* `captures_model.CapturesModel` — `QAbstractTableModel` que muestra
  el resultado de `telemetry.catalog.discover_captures(workspace)`;
  cada fila es un `CaptureInfo` (track, car, duración, número de
  vueltas, ruta).
* `channels_model.ChannelsModel` — árbol jerárquico
  `Group → ChannelInfo`; soporta toggles para mostrar/ocultar canales
  en el `MultiChart`.
* `lap_loader.LapLoader` — `QObject` con slot asíncrono que carga
  `LapTelemetry` / `StintTelemetry` desde disco en un `QThreadPool`,
  reportando progreso vía `signals.Bus`.

---

## 4. Charts (`studio/charts/`)

* `multi_chart.MultiChart` — viewer multi-canal estilo MoTeC sobre
  `pyqtgraph`. Carriles apilados con eje X compartido (distancia o
  tiempo), cursor sincronizado, overlay de delta vs vuelta de
  referencia y leyenda toggleable.
* `lap_arrays.lap_to_arrays(lap)` — adapta `LapTelemetry` a buffers
  contiguos por canal listos para `pyqtgraph`.
* `decimate.decimate(xy, max_points=4000)` — diezma a un techo de
  puntos por carril usando un algoritmo *min-max-per-bucket* que
  preserva picos.

---

## 5. Widgets clave

| Widget | Rol |
| --- | --- |
| `captures_dock.py` | Browser del workspace, drag-drop, click selecciona vuelta. |
| `channels_dock.py` | Browser jerárquico de canales y filtro por texto. |
| `charts_dock.py` | Hosting del `MultiChart` con barra de control (canal de referencia, eje X, modo delta). |
| `track_map_dock.py` | Renderiza `TrackMap.from_laps(...)` + racing line de `racing_lines/<TRACK>_racing.csv`. |
| `track_elevation_dock.py` | Perfil z(s) con bandas de surface y peralte (`geom3d`). |
| `race_dashboard_dock.py` | Splits, predicted lap, gap, traffic, fuel, lap counter. |
| `stint_tab.py` | Tabla `StintTelemetry.per_lap` + trend lines. |
| `sectors_tab.py` | Splits por sector + best / theoretical-best. |
| `dampers_tab.py` | Histogramas HS/LS por rueda (`damper_histogram`). |
| `capture_tab.py` | Botones Start/Stop sobre `app.capture_runner`. Formulario (stem, host/puerto InSim, puertos OutSim/OutGauge), checkbox **"Overlay only (no CSV recording)"** que pasa `write_csv=False` al runner (CLI `--no-csv`), LED de estado InSim (gris=idle / ámbar=esperando / verde=conectado), log embebido y contador de vueltas. |
| `live_tab.py` | Race-engineer overlay live a partir del snapshot JSON. |
| `live_modules/` | Sub-paquete con los módulos componibles del Live tab (`_base`, `simple`, `inputs`, `gaps`, `session`, `diagnostics`, `tyre_risk`, `compass_map`, `radar`, `delta_bar`). El antiguo `live_modules.py` está reemplazado por este paquete; las clases públicas se re-exportan desde `live_modules/__init__.py` para mantener compatibilidad. |
| `live_data_source.py` | Lector y watcher del fichero JSON publicado por `live_publisher`. |
| `lfs_config_dialog.py` | Diálogo *Configure LFS…* (selección de carpeta + patch automático). |
| `racing_line_loader.py` | Cargador/parser de `racing_lines/<TRACK>_racing.csv`. Lo usan internamente `track_map_dock` y los renderers compass/mini-map; no es un diálogo modal expuesto al usuario. |
| `_format.py` | Formateadores compartidos: `format_finite`, `format_signed_finite`, `format_lap_time_s`, `format_lap_time_ms`, `format_clock_ms`, `format_signed_delta_s`, `format_signed_delta_ms`, `format_gap_seconds`, `format_gap_meters` y constante `EMDASH = "—"`. Toda la UI debe usar estas funciones (no recrearlas localmente). |

---

## 6. Capture runner (`app/`)

El paquete `lfs_telemetry.app` aísla el ciclo de vida de la
subprocess de captura:

* `capture_runner.CaptureRunner` — lanza `lfs-telemetry capture` con
  los flags configurados desde el *Capture* tab, vigila el fichero
  stop, redirige stderr/stdout a la UI y emite progreso. Acepta
  `write_csv: bool = True`; cuando es `False`, el runner crea un
  directorio compartido `<workspace>/_overlay/` para `live.json` y
  el sentinel (no una carpeta de sesión por captura), reenvía
  `--no-csv` al subproceso y, por tanto, no se escribe ningún CSV.
* `state.CaptureState` — dataclass espejo del estado actual
  (running / paused / stopped, tiempo transcurrido, número de
  muestras, last error).

Studio detiene la subprocess enviando `CTRL_BREAK_EVENT` (manejado en
`cli/__init__.py`) para que el bucle asyncio se cierre limpiamente y los
buffers se descarguen al CSV.

---

## 7. Live overlay

El flujo en sesión real:

```
LFS  ─►  lfs-telemetry capture --live-file live.json [--no-csv]
                 │
                 ▼
       live_publisher.write_snapshot_atomic(...)   ~10 Hz
                 │
                 ▼
 widgets/live_data_source.LiveDataSource           polling
                 │
                 ▼
 widgets/live_tab.LiveTab  +  widgets/live_modules.*
```

Cuando el usuario marca **"Overlay only (no CSV recording)"** en la
pestaña Capture, el runner pasa `--no-csv` al subproceso: el
`live.json` sigue refrescándose a ~10 Hz y la pestaña Overlay
funciona normalmente, pero no se buferean muestras en memoria ni se
escriben CSV per-lap o agregados. Útil para usar el HUD sin
dejar registros de telemetría en el workspace.

Módulos Live disponibles (sub-paquete `widgets/live_modules/`):

* Radar 360° de tráfico con coches relativos.
* Strip de delta-time vs vuelta de referencia.
* Predicted lap (`SplitPredictor.predicted_lap_ms`) y gap a best.
* Fuel range (`FuelTracker.range_laps`).
* Mini track map con cursor.
* Bandera / pit window / penalizaciones (decodificadas vía
  `protocol.packets.penalty_name`).

---

## 8. Estado persistente

`workspace_state.WorkspaceState` se guarda en
`~/.lfs-telemetry/studio.json`:

* última captura abierta,
* ruta del workspace,
* selección de canales visibles,
* pesos y mapping de splits a sectores,
* geometría/visibilidad de docks (`saveState()` / `restoreState()` de
  Qt serializados como base64).

Recargas en frío restauran exactamente la sesión anterior, lo que
permite usar Studio como overlay siempre-presente.

---

## 9. Convenciones

* **No tocar lógica de telemetría desde widgets** — toda mutación de
  datos pasa por las clases de `lfs_telemetry.telemetry`. Los docks
  son consumidores.
* **Hilo principal** — todo update de UI ocurre en el hilo principal
  de Qt. Las cargas pesadas (CSV → DataFrame enriquecido) se hacen
  en `QThreadPool` desde `LapLoader` y se entregan vía `Signal`.
* **Formateo** — usar siempre `widgets/_format.py`; no concatenar
  strings de tiempo a mano. Esto garantiza coherencia en em-dashes,
  signos y precisión.
* **Bus de señales** — los docks no se conocen entre sí; se comunican
  por `signals.Bus`. Para añadir un nuevo widget, declarar las
  señales necesarias allí.
* **Theming** — los QSS están en `theme.py`; usar tokens (colores y
  fuentes definidos como constantes) en vez de literales.
* **Tests** — `tests/studio/test_smoke.py` está ignorado por defecto
  porque requiere display server; ejecutarlo manualmente con
  `QT_QPA_PLATFORM=offscreen` durante desarrollo.
