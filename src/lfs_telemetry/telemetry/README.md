# `lfs_telemetry.telemetry` — referencia exhaustiva

Núcleo de captura, persistencia y análisis. **No depende de ningún
framework de UI**; las clases públicas exponen `numpy`/`pandas` y
pueden usarse desde Studio, notebooks, pipelines de ML o controladores
clásicos.

* Esquema CSV: `SCHEMA_VERSION = "1.1"` (preámbulo
  `# lfs-telemetry telemetry schema=1.1`).
* OutSim extendido: `OSO_ALL = 0x1ff` → `OutSimPack2` (280 B, con
  rueda × 4 e índice de pista).
* Frecuencia nominal de muestreo: 100 Hz.

---

## 0. Configurar LFS

Sin telemetría activada en `cfg.txt`, las APIs `LiveTelemetry` /
`InSimClient` nunca recibirán paquetes. Hay tres rutas:

* `python -m lfs_telemetry.lfs_config "C:\path\to\LFS"`
* Studio: *Tools → Configure LFS…*
* Edición manual (ver `README.md` raíz).

`lfs_config.patch_cfg(lfs_dir)` inserta/actualiza las líneas
necesarias respetando los demás valores y creando un backup `.bak`.

---

## 1. Arquitectura

```
                ┌─ protocol.packets ─┐   ┌─ protocol.insim ──┐
   UDP 30000 ──►│  OutSimPacket /    │   │ InSimClient (TCP) │◄─ TCP 29999
   UDP 30001 ──►│  OutSimPack2 /     │   │  RaceContext,     │
                │  OutGaugePacket    │   │  PitStopRecord    │
                └──────────┬─────────┘   └─────────┬─────────┘
                           ▼                       ▼
                ┌──────────────────────────────────────────┐
                │   live.LiveTelemetry  (asyncio fuse)     │
                │   → TelemetrySample (slots dataclass)    │
                └────────────────────┬─────────────────────┘
                                     ▼
   ┌────────────────────────┬───────────────────────┬────────────────────┐
   │ replay.write_csv_      │  lap_summary          │ live_publisher     │
   │ replay / read_csv_*    │  (LapRecord stream)   │ (JSON snapshot)    │
   └──────────┬─────────────┴───────────────────────┴────────────────────┘
              ▼
   ┌─────────────────────────────────────────────────────────────────────┐
   │ lap.LapTelemetry  ─►  stint.StintTelemetry  ─►  comparison.Lap*     │
   │                       sectors / fuel_tracker / predict / traffic    │
   │                       derived.enrich_dataframe / damper_histogram   │
   │                       observables.observe_window / calibrate.*      │
   └─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Layout del paquete

| Módulo | Responsabilidad |
| --- | --- |
| `live` | Loop asyncio que escucha OutSim + OutGauge + InSim y emite `TelemetrySample`. |
| `replay` | Serializa/deserializa el stream a CSV 1.1 con metadatos en cabecera. |
| `lap` | Vista por vuelta sobre un CSV (`LapTelemetry`). |
| `stint` | Agregador multi-vuelta (`StintTelemetry`). |
| `comparison` | `LapComparison` alineado por distancia + delta tiempo. |
| `sectors` | Tipo `Sector` y utilidades de splits. |
| `lap_slicer` | Slicer canónico basado en `current_lap_dist_m`. |
| `lap_summary` | `LapRecord` proveniente de InSim `IS_LAP/SPX/HLV/OBH`. |
| `lap_cache` | Cache en parquet de DataFrames enriquecidos. |
| `predict` | `SplitPredictor` (sector personal + vuelta proyectada). |
| `traffic` | `TrafficSnapshot` de coches cercanos. |
| `fuel_tracker` | Estimación online de consumo y autonomía. |
| `node_delta` | Delta nodo a nodo contra una vuelta de referencia. |
| `track_map` | `TrackMap` promediado a partir de varias vueltas. |
| `channels` | Registro `ChannelInfo` de ~123 columnas. |
| `catalog` | Descubrimiento e inspección de CSVs en un workspace. |
| `derived` | `enrich_dataframe`: deriva ~30 columnas (chassis, neumático, FFB, geometría…). |
| `damper_histogram` | Histogramas de amortiguadores HS/LS por rueda. |
| `observables` | `CarSpec` + observación estructural por muestra/ventana. |
| `calibrate` | Estimadores de μ y masa a partir de telemetría cruda. |
| `car_calibration` | `CarSpecStore` (calibraciones persistentes) + `RestCalibrator`. |
| `car_info_bin` | Parser de `car_info.bin` de LFS (setup por defecto). |
| `live_publisher` | Snapshot JSON consumido por el dock Live de Studio. |
| `heading` | Helpers de proyección a marco local. |
| `protocol/` | Parsers de bajo nivel (OutSim, OutGauge, InSim). |
| `track/` | Parsers `.pth`, `.smx`, `.pin`, `.knw` + enriquecido y racing line. |

`telemetry/__init__.py` reexporta toda la API pública listada en
`__all__`; basta `from lfs_telemetry.telemetry import ...`.

---

## 3. Captura en vivo (`live.py`)

```python
from lfs_telemetry.telemetry import LiveTelemetry

async def main():
    live = LiveTelemetry(
        outsim_port=30000,
        outgauge_port=30001,
        outsim_opts=0x1ff,        # OSO_ALL → OutSimPack2 (280 B)
        insim_host="127.0.0.1",   # opcional, habilita splits/race_context
        insim_port=29999,
    )
    async with live.session():
        async for sample in live.stream():   # TelemetrySample
            print(sample.t_capture_s, sample.speed_ms, sample.rpm)
```

`TelemetrySample` es un *slots dataclass* (~70 campos) con tiempos de
captura, pose, velocidades, aceleraciones, controles del piloto,
combustible, marcha, RPM, contexto de carrera, datos por rueda y un
`current_lap_dist_m` ya proyectado sobre la pista. Internamente
`_PendingByTime` empareja paquetes OutSim + OutGauge por tiempo de
simulación; `_outsim2_to_basic` degrada un `OutSimPack2` a campos
básicos cuando el paquete extendido no está disponible.

Para uso desde la CLI, `lfs-telemetry capture` envuelve `LiveTelemetry`
y escribe el CSV; el manejo de `CTRL_BREAK_EVENT` (SIGBREAK / SIGINT)
en `cli.py` permite que Studio detenga la subprocess con su botón
*Stop*.

---

## 4. Esquema CSV 1.1 (`replay.py`)

```
# lfs-telemetry telemetry schema=1.1
# created_utc=2024-…
# car=FOX track=BL1 mass_kg=… mu_lat=… …
t_capture_s,t_sim_s,pos_x,…,current_lap_dist_m,indexed_distance_m,…
```

* `write_csv_replay(samples, path, metadata=…)` — escribe el preámbulo
  y el cuerpo, normalizando tipos.
* `read_csv_replay(path)` → iterable de `TelemetrySample` reconstruidos
  por `_row_to_outsim2` (restaura `current_lap_dist_m` e
  `indexed_distance_m`).
* `read_csv_dataframe(path, enrich=True)` → `pandas.DataFrame` con
  metadatos en `.attrs` y, opcionalmente, derivadas inyectadas por
  `derived.enrich_dataframe`.
* `detect_schema_version(path)` → `"1.0"` o `"1.1"`; rechaza ficheros
  sin preámbulo.

Round-trip de muestras es lossless: cualquier `TelemetrySample`
escrito por `write_csv_replay` se reconstruye por `read_csv_replay`
sin pérdida.

---

## 5. `LapTelemetry`

```python
from lfs_telemetry.telemetry import LapTelemetry

lap = LapTelemetry.from_csv("captures/BL1_lap03.csv")
df  = lap.dataframe()        # pandas.DataFrame enriquecido (cached_property)
lap.duration_s               # tiempo total
lap.distance_m               # longitud recorrida
lap.average_speed_ms
lap.car                      # CarSpec resuelto vía _resolve_car
lap.track                    # identificador corto (BL1, KY3R, …)
lap.metadata                 # dict de cabecera del CSV
```

`LapTelemetry` no usa `__slots__` para permitir `@cached_property`
sobre el DataFrame y métricas derivadas. `_resolve_car` busca primero
en `CarSpecStore` (calibración del usuario) y cae a `config/cars.json`.

---

## 6. `StintTelemetry`

Agregador multi-vuelta con tres constructores:

```python
from lfs_telemetry.telemetry import StintTelemetry

stint = StintTelemetry.from_dir("captures/session_01/")
stint = StintTelemetry.from_csvs(["lap01.csv", "lap02.csv", ...])
stint = StintTelemetry.from_laps([lap1, lap2, ...])
```

Vistas:

* `stint.per_lap` → `DataFrame` (una fila por vuelta).
* `stint.trends(window=3)` → medias móviles.
* `stint.average_lap_time(mode="stint"|"clean"|"total"|"rolling")`.
* `stint.race_start_lap_indices` / `stint.clean_lap_indices` —
  máscaras computadas a partir de banderas y validez.
* `stint.fuel_usage` (`cached_property`) — consumo medio por vuelta.

Gestión de vueltas inválidas:

* `stint.mark_lap_invalid(idx, reason)`.
* `stint.mark_invalid_from_records(lap_records)` — usa una secuencia
  de `LapRecord` para descartar automáticamente vueltas con `HLV`
  reportado por LFS.

### 6.1 Helpers Detect & Monitor

`StintTelemetry` integra el conjunto de heurísticos D&M (`assets/Detect&Monitor/`):

* Detección de salida de pits (primer cruce de línea con tiempo).
* Filtros de in-lap / out-lap.
* Marcado de pit-window y splash-and-go.

### 6.2 `lap_slicer`

Slicer canónico basado en el cruce de `current_lap_dist_m`:

```python
from lfs_telemetry.telemetry.lap_slicer import (
    slice_into_laps, find_line_crossings, write_per_lap_files, reslice_csv,
)

crossings = find_line_crossings(df, min_drop_m=100.0)
slices    = slice_into_laps(df, min_drop_m=100.0)   # list[LapSlice]
write_per_lap_files(slices, out_dir="captures/laps/", stem="stint")
reslice_csv("captures/stint.csv", out_dir="captures/laps/")
```

`reslice_csv` es el back-end del subcomando `lfs-telemetry reslice`.

### 6.3 `SplitPredictor`

```python
from lfs_telemetry.telemetry import SplitPredictor

pred = SplitPredictor()
pred.observe_split(lap=3, sector=1, t_ms=22500)
pred.observe_lap(lap=3, t_ms=89320)

pred.spb_ms                       # sector personal best (por sector)
pred.predicted_lap_ms             # vuelta proyectada por sectores
pred.delta_to_best_ms             # delta vs vuelta actual
pred.to_dict() / SplitPredictor.from_dict(d)   # serialización
```

---

## 7. `LapComparison`

```python
from lfs_telemetry.telemetry import LapComparison

cmp = LapComparison.from_laps(lap_ref, lap_chal, channels=("speed_ms", "throttle"))
cmp.distance_m            # eje común
cmp.reference[ch]         # vector reference por canal
cmp.challenger[ch]
cmp.delta_time_s          # delta-time vs distancia (s)
```

Internos: `_unwrapped_lap_arrays` resuelve la trayectoria, `_enforce_monotone`
limpia regresiones de distancia, `_lap_distance_m` calcula longitud
efectiva y `_resample_channel` realiza la interpolación lineal sobre el
eje común. `_time_at_distance` produce el delta acumulado en segundos.

---

## 8. `Sector`, `lap_sectors`, `TrackMap`

```python
from lfs_telemetry.telemetry import (
    Sector, lap_sectors, sector_times_s, insim_split_distances_m,
    TrackMap, TrackBounds,
)

sectors  = lap_sectors(lap, splits_m=insim_split_distances_m(record))
times_s  = sector_times_s(lap, sectors)

tmap     = TrackMap.from_laps([lap1, lap2, lap3])
tmap.bounds                  # TrackBounds (x_min..z_max)
xy       = tmap.xy_along_distance(d_m=750.0)
xy_grid  = tmap.xy_along_grid(num=2000)
```

`insim_split_distances_m` deriva los splits desde un `LapRecord`
proveniente de InSim; en su ausencia se aceptan splits manuales o se
generan particiones por igual distancia.

---

## 9. `TrafficSnapshot` y `LapRecord`

```python
from lfs_telemetry.telemetry import TrafficSnapshot, traffic_snapshot
snap = traffic_snapshot(local_sample, neighbours)
snap.ahead_m / snap.behind_m / snap.ahead_s / snap.behind_s

from lfs_telemetry.telemetry import (
    LapRecord, build_lap_records, dump_lap_records, load_lap_records,
)
records = build_lap_records(insim_events)
dump_lap_records(records, "session.laps.json")
records = load_lap_records("session.laps.json")
```

`LapRecord` consolida `IS_LAP` (vuelta), `IS_SPX` (split), `IS_HLV`
(handicap/validez) y `OBH` (incidentes) en una fila por vuelta con
flags de validez.

---

## 10. Canales y catálogo

```python
from lfs_telemetry.telemetry import (
    CHANNELS, ChannelInfo, channel_info, channels_by_group,
    CaptureInfo, discover_captures, inspect_capture, captures_to_dataframe,
)

CHANNELS["speed_ms"].group   # "Vehicle"
channels_by_group()["Tyre"]  # → lista de ChannelInfo
discover_captures("captures/")          # → list[CaptureInfo]
captures_to_dataframe(discover_captures("captures/"))
```

`CHANNELS` cubre los grupos `Vehicle`, `Engine`, `Driver`, `Chassis`,
`Suspension`, `Tyre`, `Aids`, `Derived`, `Lap`, `Context` (~123
columnas en total). `_build_registry` se ejecuta una vez en import y
queda cacheado.

---

## 11. Observables y calibración

```python
from lfs_telemetry.telemetry import (
    CarSpec, car_spec_for, observe_sample, observe_window,
    estimate_mu_lat, estimate_mu_long, estimate_mu_lat_curve,
    estimate_mass_kg, calibrate_spec, calibration_report,
    CarSpecStore, RestCalibrator, default_store_path,
)

spec = car_spec_for("FOX")              # del catálogo bundled o user store
obs  = observe_sample(sample, spec)     # StructuralObservation
win  = observe_window(samples, spec)    # agregado por ventana

mu_lat   = estimate_mu_lat(df)
mu_long  = estimate_mu_long(df)
mass_kg  = estimate_mass_kg(df, spec)
spec2    = calibrate_spec(df, spec)
report   = calibration_report(df, spec2)

store = CarSpecStore.load(default_store_path())
store.update("FOX", spec2)
store.save()

cal = RestCalibrator(seconds=5.0)
cal.run(live)        # captura una ventana en reposo y devuelve mass/μ
```

`CarSpec` agrupa masa, distribución, geometría, μ y gearing. El cache
interno de `observables.py` está protegido por `threading.RLock()` para
soportar acceso concurrente desde Studio y subprocesos de captura.
`_asset_search_dirs` unifica la búsqueda de `config/cars.json` y
`car_info.bin` en el árbol del proyecto y en el bundle congelado de
PyInstaller.

---

## 12. Derivadas (`derived.enrich_dataframe`)

`enrich_dataframe(df)` añade ~30 columnas calculadas a un DataFrame
producido por `read_csv_dataframe`. Conjuntos cubiertos por
sub-funciones privadas:

* `_add_chassis_dynamics` — yaw rate corregido, ángulo de deriva.
* `_add_load_transfer` — transferencias longitudinal y lateral.
* `_add_friction_circle` — uso instantáneo de μ por rueda.
* `_add_tyre_work` — trabajo (slip×carga) por rueda.
* `_add_brake_bias` — repartición real vs nominal.
* `_add_damper_velocities` — derivadas suspensión por rueda.
* `_add_dash_lights` — decodificación de luces del salpicadero.
* `_add_ffb` — fuerza estimada en volante.
* `_add_smoothness` — suavidad de mandos (jerks de freno/aceleración).
* `_add_gear_lfs` — corrección de la marcha reportada por LFS.
* `_add_track_geometry` — proyección sobre el perfil de pista
  (curvatura, radio, pendiente, peralte si hay `.smx`).

Sólo `enrich_dataframe` se exporta (`__all__`); las funciones privadas
están sujetas a cambios.

---

## 13. Subpaquete `protocol/`

### 13.1 `protocol.packets`

* `OutSimPacket` — 64 B básico (pose + velocidades + IDs).
* `OutSimPack2` — 280 B extendido (`OSO_ALL = 0x1ff`); incluye
  `OutSimWheel × 4`, índice de pista y telemetría adicional.
* `OutGaugePacket` — instrumentación (RPM, gear, fuel, dash lights, …).
* `InSimHeader`, `InSimVersion` — base TCP InSim.
* Helpers: `hlvc_name`, `decode_dash_lights`, `decode_pit_work`,
  `decode_host_flags`, `race_laps_kind`, `penalty_name`,
  `penalty_reason_name`, `cim_mode_name`, `build_isi_packet`,
  `outsim2_size`, `decode_car_id`.

### 13.2 `protocol.insim`

* `InSimClient` — cliente asyncio TCP, mantiene la sesión, gestiona
  reconexiones y emite eventos tipados.
* `RaceContext` — estado actual de sesión (track, layout, hosts,
  banderas).
* `PitStopRecord` — registro de parada en boxes (`PITSTOP`/`PSF`).

`LiveTelemetry` instancia `InSimClient` cuando se pasa `insim_host`
y enriquece cada `TelemetrySample` con el contexto vigente.

---

## 14. Subpaquete `track/`

### 14.1 `track.pth`

Parser canónico del fichero `.pth` (línea AI de LFS):

* `PthNode` (frozen slots) — posición, ancho izquierdo/derecho,
  límites duros izquierdo/derecho.
* `Path` (slots) — colección ordenada de nodos.
* `TrackProfile` — perfil normalizado de pista (longitud acumulada,
  curvatura, radio, pendiente, ancho).
* `parse_pth(path)` / `parse_pth_bytes(blob)`.
* `compute_profile(path, max_segment_m=50.0)` — densifica nodos y
  calcula curvatura/radio/pendiente.
* `list_path_files(root)`, `load_all(root)`, `summary_table(profiles)`,
  `DEFAULT_SMX_DIR`.

### 14.2 `track.smx`

Parser de mallas `.smx` (geometría 3D de pista):

* `SmxObject` (frozen slots), `SmxMesh` (slots).
* `parse_smx`, `parse_smx_bytes`, `list_smx_files`,
  `find_smx_for_track`, `iter_smx_directory`.
* `elevation_envelope(mesh, profile)` → z mín/máx a lo largo de s.
* `cross_section_at(mesh, station)` → corte transversal.

### 14.3 `track.pin`

Bounding boxes y metadatos del entorno:

* `PinInfo`, `parse_pin`, `parse_pin_bytes`, `list_pin_files`,
  `load_all`, `find_env_for_xy`.

### 14.4 `track.knw`

Conocimiento AI per-coche (línea ideal aprendida por LFS):

* `KnwSegment`, `KnwInfo`, `parse_knw_bytes`, `_split_layout_car`,
  `DEFAULT_KNW_DIR`, `load_for(track, car)`.

### 14.5 `track.geom3d`

Geometría 3D derivada del mesh:

* `classify_surface(mesh)` — tarmac / kerb / grass / sand / wall.
* `compute_banking_profile(mesh, profile)` — peralte a lo largo de s.
* `surface_distribution_along(mesh, profile)` — % por tipo de
  superficie por estación.
* `kerb_mask_along(mesh, profile)`.
* `CheckpointGeom`, `extract_checkpoint_geometry`.
* `CorridorHeightmap` — heightmap del corredor de pista.

### 14.6 `track.enrich`

Inyecta información geométrica en un DataFrame:

* `TrackIndex` — índice precomputado pista → ficheros.
* `enrich_dataframe(df, profile)` — añade columnas de geometría (no
  confundir con `telemetry.derived.enrich_dataframe`, que añade
  derivadas físicas).
* `enrich_csv(in_path, out_path)` — versión orientada a fichero.
* `TrackSegment`, `segment_track(profile, straight_radius_m, min_segment_m)`
  → segmentación recta/curva.
* `assign_segment(df, segments)`.
* `TrackMatch`, `detect_track(samples_or_df)` — heurístico de
  detección a partir de coordenadas y/o contexto InSim.

### 14.7 `track.racing_line`

```python
from lfs_telemetry.telemetry.track.racing_line import (
    compute_edges, compute_target_speed,
    compute_geometric_line, compute_knw_line, RacingLine,
)

edges = compute_edges(profile)
speed = compute_target_speed(profile, mu_lat=1.20, mu_long=1.10,
                             v_cap_ms=88.0, mu_lat_aero_k=0.0)
line  = compute_geometric_line(profile, segments, edge_margin_m=0.3)
# o, si hay datos KNW:
line  = compute_knw_line(profile, knw_info, edge_margin_m=0.3)
```

`RacingLine` (slots dataclass) almacena la línea elegida más la curva
de velocidad objetivo. `scripts/racing_line_view.py --all` regenera
`racing_lines/<TRACK>_racing.csv` + `.png` para las ~95 variantes de
LFS (94 OK, 1 vacía, 0 errores en la última ejecución) usando estas
funciones.

### 14.8 `track.loader`

```python
from lfs_telemetry.telemetry.track.loader import (
    candidate_racing_lines_dirs, find_racing_line_csv,
    TrackGeometry, load_track_geometry, cached_track_geometry,
)

paths = candidate_racing_lines_dirs()         # config + workspace + frozen
csv   = find_racing_line_csv("BL1")
geom  = cached_track_geometry("BL1")          # TrackGeometry
```

`cached_track_geometry` mantiene resultados en memoria por nombre de
pista para que el dock Track map y el Live tab no relean el `.pth` /
`.smx` en cada repintado.

---

## 15. Cache de vueltas, fuel, deltas y dampers

* `lap_cache` — `cache_dir()`, `CacheKey`, `load(key)`, `save(key, df)`,
  `clear()` — parquet en `%LOCALAPPDATA%\lfs-telemetry\cache\`.
* `fuel_tracker.FuelTracker` — `observe(sample)`, `liters_per_lap`,
  `range_laps`, `range_km`.
* `node_delta.NodeDeltaTracker` — delta nodo a nodo vs referencia,
  útil para overlays live.
* `damper_histogram` — `damper_histogram(df)` →
  `{wheel: DamperHistogram(low_speed, high_speed, comp, reb)}`.

---

## 16. `live_publisher`, `heading`, `car_info_bin`

```python
from lfs_telemetry.telemetry.live_publisher import (
    RadarCar, build_radar_cars, build_snapshot, write_snapshot_atomic,
)
snap = build_snapshot(local_sample, neighbours, predictor, fuel_tracker, lap_record)
write_snapshot_atomic("C:/.../live.json", snap)

from lfs_telemetry.telemetry.heading import project_to_local
xy_local = project_to_local(global_xy, origin, heading)

from lfs_telemetry.telemetry.car_info_bin import parse_car_info_bin
info = parse_car_info_bin("path/to/car_info.bin")     # CarInfoBin
info.wheels       # tuple[CarInfoWheel, ...]
info.gear_ratios  # ...
```

`write_snapshot_atomic` escribe a un temporal y hace rename atómico
para que el dock Live de Studio no lea estados parciales.

---

## 17. Patrones de uso

### Cargar un stint y comparar vueltas

```python
from lfs_telemetry.telemetry import StintTelemetry, LapComparison

stint = StintTelemetry.from_dir("captures/session_01/")
best  = stint.laps[stint.per_lap["time_s"].idxmin()]
last  = stint.laps[-1]
cmp   = LapComparison.from_laps(best, last, channels=("speed_ms", "throttle"))
```

### Pipeline completo desde live a CSV anotado

```python
import asyncio
from lfs_telemetry.telemetry import LiveTelemetry, write_csv_replay

async def run():
    live = LiveTelemetry(insim_host="127.0.0.1")
    samples = []
    async with live.session():
        async for s in live.stream():
            samples.append(s)
            if len(samples) > 36_000:   # ~6 min @ 100 Hz
                break
    write_csv_replay(samples, "captures/session.csv",
                     metadata={"driver": "F", "track": "BL1"})

asyncio.run(run())
```

### Calibración interactiva

```python
import asyncio
from lfs_telemetry.telemetry import LiveTelemetry, RestCalibrator, CarSpecStore

async def cal():
    live = LiveTelemetry(insim_host="127.0.0.1")
    async with live.session():
        spec = await RestCalibrator(seconds=5.0).run(live)
    store = CarSpecStore.load()
    store.update("FOX", spec)
    store.save()

asyncio.run(cal())
```

---

## 18. Convenciones y notas de implementación

* **Concurrencia**: las caches mutables en `observables.py` están
  protegidas por `threading.RLock()`; el resto del paquete es
  funcional o estrictamente single-threaded por instancia.
* **Versionado de esquema**: cambios incompatibles bumpean
  `SCHEMA_VERSION`. `detect_schema_version` debe tratarse como la
  única fuente de verdad al leer ficheros ajenos.
* **`__all__`**: los módulos `channels.py`, `live.py`, `derived.py`
  declaran `__all__` explícito; tratarlos como contrato público
  estable.
* **Inmutabilidad**: dataclasses de protocolo (`PthNode`, `SmxObject`,
  `SmxMesh`, `RacingLine`) usan `slots=True` (y `frozen=True` cuando
  procede) para minimizar overhead y permitir su uso seguro como
  claves o en estructuras concurrentes.
* **Encoding**: todas las lecturas/escrituras de CSV usan UTF-8; los
  scripts asumen `PYTHONIOENCODING=utf-8` en consolas Windows.
