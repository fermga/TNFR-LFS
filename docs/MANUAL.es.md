# LFS Race Engineer — Manual de uso

Guía rápida y completa para usar **LFS Race Engineer** desde la
aplicación instalada. No necesitas Python ni consola: todo se hace
desde la interfaz gráfica.

---

## 1. Instalación

1. Descarga `lfs-race-engineer-setup-x.y.z.exe`.
2. Ejecútalo y sigue el asistente (puedes instalar para tu usuario o
   para toda la máquina). Tarda unos segundos.
3. Lanza **LFS Race Engineer** desde el menú Inicio.

La aplicación se autocontiene: no instala Python ni librerías
adicionales.

---

## 2. Primer arranque: configurar LFS (paso obligatorio)

**Antes de capturar nada hay que decirle a LFS que envíe telemetría.**
LFS no lo hace por defecto. La app puede configurarlo automáticamente:

1. **Cierra LFS** (si está abierto, sobrescribirá los cambios al salir).
2. En LFS Race Engineer, ve a **Tools → Configure LFS…**
3. El diálogo intenta autodetectar la carpeta de instalación de LFS
   (p. ej. `C:\LFS`). Si no la encuentra, pulsa **Browse…** y
   selecciónala manualmente (la carpeta que contiene `LFS.exe` y
   `cfg.txt`).
4. Pulsa **Patch cfg.txt automatically**.
5. La app:
   - hace una copia de seguridad como `cfg.txt.bak` (sólo la primera
     vez);
   - inserta/actualiza las entradas de **OutSim**, **OutGauge** y los
     puertos (loopback `127.0.0.1`) en `cfg.txt`;
   - muestra un mensaje de confirmación con la ruta del backup.
6. Cierra el diálogo.

### Puertos usados

| Servicio | Puerto | Protocolo | Función |
| --- | --- | --- | --- |
| OutSim   | 30000 | UDP | telemetría del chasis (posición, velocidades, aceleraciones, ruedas si `OutSim Opts = 1ff`) |
| OutGauge | 30001 | UDP | panel del coche (revs, marcha, combustible, temperaturas, luces) |
| InSim    | 29999 | TCP | eventos de carrera (vueltas, splits, posiciones, banderas, pit) |

Todo en `localhost` (no abre nada al exterior, no toca el firewall).

### InSim: paso final dentro de LFS

InSim **no se activa desde `cfg.txt`** (LFS lo trata como inválido y
muestra un aviso rojo si lo añades ahí). Hay dos formas equivalentes
de activarlo, ambas dentro de LFS:

* **Atajo recomendado**: arranca LFS añadiendo `/insim=29999` a los
  parámetros del acceso directo, o crea un acceso directo a
  `LFS.exe /insim=29999`. Queda activo siempre.
* **Cada sesión**: con LFS abierto, pulsa <kbd>T</kbd> (chat) y
  escribe `/insim 29999`, luego Enter.

Sin InSim funcionan las pestañas de telemetría y stint, pero el
**Overlay**, los **Sectors** y las gráficas comparativas necesitan
los eventos de cruce de línea.

---

## 3. Vista general de la interfaz

```
┌──────────────────────────────────────────────────────────────────┐
│ File   View (… → Language)   Tools   Help                        │
├──────────────────────────────────────────────────────────────────┤
│ Captures  │                                       │  Channels    │
│ (vueltas) │     Pestañas centrales:               │  (canales)   │
│           │                                       │              │
│ Track     │   Telemetry · Dampers · Sectors ·     │  Race        │
│ map       │   Stint · Capture · Overlay           │  dashboard   │
│           │                                       │              │
│ Elevation │                                       │              │
└──────────────────────────────────────────────────────────────────┘
                              Status bar (cursor, contexto)
```

* **Paneles laterales (docks)**: arrastrables; se pueden ocultar
  desde el menú **View** o cerrando su X. **View → Reset Layout**
  vuelve a la disposición de fábrica.
* **Idioma**: **View → Language → English / Español**. Cambia en
  caliente y se recuerda.
* La geometría de ventanas, la selección de canales y el workspace
  activo se persisten en `~/.lfs-telemetry/studio.json`.

### Atajos de teclado

| Atajo | Acción |
| --- | --- |
| <kbd>Ctrl</kbd>+<kbd>O</kbd> | Abrir carpeta de trabajo (workspace) |
| <kbd>F5</kbd> | Recargar la lista de capturas |
| <kbd>F1</kbd> | **Help → Channel guide…** (qué mide cada canal) |
| <kbd>Ctrl</kbd>+<kbd>Q</kbd> | Salir |

El menú **Help** también incluye **About** con la información de
versión. Este manual se distribuye junto a la aplicación en
`docs/MANUAL.<lang>.md`; ábrelo desde el explorador de ficheros,
desde el repositorio o desde la página de release de GitHub — la
aplicación no incluye un lector interno para él.

---

## 4. Workspace y selección de vueltas (panel **Captures**)

El panel izquierdo lista todas las capturas CSV de la carpeta activa,
con coche, circuito, número de vueltas y mejor tiempo.

> **Importante — el workspace debe ser una carpeta a la que tu
> usuario de Windows tenga permisos de lectura y escritura.** La
> aplicación guarda ahí los CSV de captura, las vueltas troceadas y
> la caché de análisis. Carpetas como `C:\Program Files\...`,
> `C:\Windows\...` o unidades de red sin permisos típicamente
> fallarán. Si ves errores tipo *"Access denied"*, *"Permission
> denied"* o que **Start** se detiene al instante, **cambia el
> workspace a una carpeta donde sí tengas acceso** (por ejemplo
> `Documentos\LFS-Telemetry`, `Escritorio\stints` o una carpeta
> dentro de tu perfil de usuario) con **File → Open Workspace…**
> (<kbd>Ctrl</kbd>+<kbd>O</kbd>).

* **Cambiar workspace**: arrastra una carpeta sobre el panel, o usa
  **File → Open Workspace…** (<kbd>Ctrl</kbd>+<kbd>O</kbd>).
* **Filtro de texto**: escribe en la caja superior para acotar por
  nombre de fichero, coche o circuito.
* **Cargar una vuelta**: doble clic sobre la fila.
* **Comparar varias vueltas a la vez** (overlay en Telemetry, Dampers,
  Sectors, Stint):
  - <kbd>Ctrl</kbd>+clic — añade/quita vueltas sueltas a la selección.
  - <kbd>Shift</kbd>+clic — selecciona un rango contiguo.
  - <kbd>Ctrl</kbd>+<kbd>A</kbd> — selecciona todo el listado visible.
* **Refrescar** la lista tras una captura nueva: <kbd>F5</kbd>.

La vuelta marcada como **referencia** (la mejor del set, normalmente)
se usa para el delta-tiempo en Telemetry y para el delta vivo en
Overlay.

---

## 5. Pestañas centrales

### 5.1. Telemetry — viewer multicanal

Visor estilo MoTeC sobre `pyqtgraph`. Cada **canal seleccionado** en
el panel derecho **Channels** ocupa una lane horizontal con su propia
escala Y, todas sincronizadas por el eje X.

La barra superior contiene:

* **X-axis:** dos radio buttons **Distance** / **Time** (la elección
  se aplica a todas las lanes).
* **Export PNG…** — exporta la pila completa de gráficas como
  imagen.
* **Export CSV…** — exporta los canales visibles, alineados, como
  CSV.
* **Caption** lateral con el número de vueltas cargadas.

Debajo, una **leyenda** muestra una pastilla de color por vuelta
(la primera es la **referencia** y se marca con ` (ref)`).

Interacción con las gráficas:

* **Zoom**: rueda del ratón = zoom en X; arrastrar con botón derecho
  = pan en X; doble clic = autoescala vertical de la lane.
* **Cursor**: mueve el ratón sobre cualquier lane; una línea vertical
  marca la misma posición en todas las demás y la barra de estado
  muestra los valores instantáneos.
* **Delta vs referencia**: con varias vueltas seleccionadas, una
  lane extra `Δt vs ref [s]` aparece automáticamente.
* **Decimación**: cada lane se reduce a ≤ 4 000 puntos por
  min-max-por-bucket para preservar los picos sin colgar el render.

#### Canales disponibles (panel **Channels**)

Árbol agrupado, con checkbox por canal y filtro de texto. Grupos
(definidos en `lfs_telemetry.telemetry.channels`):

| Grupo | Qué contiene |
| --- | --- |
| **Driver** | Entradas del piloto: gas, freno, embrague, freno de mano, dirección y par de FFB. |
| **Vehicle** | Estado básico del coche: velocidad, posición y tiempos de vuelta (OutGauge). |
| **Engine** | Tren motriz: RPM, marcha, combustible y temperaturas de motor/aceite (OutGauge). |
| **Chassis** | Dinámica OutSim del chasis: aceleración en 3 ejes (m/s²), velocidad angular en 3 ejes (rad/s) y ángulos Euler. |
| **Suspension** | Estado por esquina de la suspensión: recorrido, carga vertical, velocidad del amortiguador y ángulo de dirección de la rueda. |
| **Tyre** | Por esquina del neumático: slip ratio, ángulo de deriva, fuerzas long./lat., temperatura interna y banderas de contacto. |
| **Aids** | Bitfield ShowLights de OutGauge: estado de TC, ABS, limitador de pit, intermitentes y testigos. |
| **Derived** | Magnitudes calculadas por Studio a partir de los canales brutos (índice de subviraje, transferencias de peso, uso de fricción, etc.). |
| **Lap** | Distancias e índices relativos a la vuelta, usados para alinear trazas. |
| **Track** | Geometría estática de pista muestreada en el nodo actual (curvatura, radio, pendiente, ancho, offset lateral del coche). |
| **Context** | Sesión: coche, pista, clima, viento (para filtrar y agrupar capturas). |

> Pulsa <kbd>F1</kbd> para abrir la **Channel guide** desde el menú
> **Help**: cada canal con su unidad y una explicación breve.

### 5.2. Dampers — histogramas de amortiguadores

**Un histograma de velocidad de amortiguador por rueda** (FL, FR,
RL, RR) calculado sobre la **primera vuelta seleccionada**. Las dos
líneas verticales discontinuas marcan la **frontera baja/alta
velocidad**, configurable en la barra superior con el spinbox
**Low-speed boundary** (por defecto **±25 mm/s**, la convención usada
por MoTeC, AIM RaceStudio y Cosworth Pi).

Bajo cada histograma se muestran en una sola línea las cuatro
métricas clave: `Reb avg`, `Hi-reb %`, `Bump avg`, `Hi-bump %`.

Si seleccionas **dos vueltas**, la segunda se superpone como
contorno escalonado discontinuo blanco y el resumen pasa al modo
comparación (A vs B, con Δ mm/s y Δ %).

Sirve para detectar:

* asimetrías izquierda/derecha,
* exceso de tiempo en alta velocidad (bordillos agresivos, rebote
  del eje),
* desbalance de reparto compresión/rebound entre vueltas o setups.

### 5.3. Sectors — splits

Dos elementos sobre las vueltas seleccionadas:

* **Resumen** en cabecera con el origen de los splits
  (`InSim splits` cuando hay eventos de InSim, `uniform ×N` cuando
  no los hay y se reparten por distancia) y el tiempo
  **theoretical best** del stint (suma de los mejores sectores
  individuales).
* **Bar chart "Sector times"** agrupado por vuelta, con un color por
  sector y `Lap #` en el eje X.

### 5.4. Stint — análisis multivuelta

Vista de evolución vuelta a vuelta. Encima un **resumen del stint**
(mejor, media, vuelta teórica, caída de ritmo, combustible total y
vueltas restantes, picos de G y tendencia de temperatura por rueda)
y debajo una pila de **siete gráficas** con el eje X en `Lap #`:

1. **Lap times** (s) — barras por vuelta y media (línea de trazos).
2. **Fuel** (%) — % usado por vuelta y % restante al final.
3. **Tyre temp end-of-lap** (°C) — una línea por rueda.
4. **Peak vertical load (suspension)** (kN) — pico por rueda.
5. **Friction use p95 (circle saturation)** — uso del círculo de
   fricción en el percentil 95.
6. **Grip index (per wheel)** (%) — índice de agarre por rueda.
7. **Damper work — RMS shaft speed** — trabajo de los amortiguadores.

Seleccionando varios CSV (todo un stint) se ve el comportamiento a
lo largo de la sesión.

### 5.5. Capture — grabación de telemetría

Aquí se controla el subproceso que captura LFS en vivo.

Formulario:

| Campo | Significado |
| --- | --- |
| **Filename stem** | Prefijo de los CSV que se guardan en el workspace. |
| **InSim host / port** | Normalmente `127.0.0.1` / `29999`. |
| **OutSim port** | Normalmente `30000`. |
| **OutGauge port** | Normalmente `30001`. |
| **Overlay only (no CSV recording)** | Si lo marcas, la app conecta con LFS y alimenta el **Overlay** en tiempo real **sin escribir ningún CSV** en el workspace. Útil para usar el HUD en sesión libre sin generar registros. |

Botones:

* **Start** — lanza la captura. El LED de estado cambia:
  - **gris** = parado,
  - **ámbar** = esperando que LFS responda / coche aún en boxes,
  - **verde** = InSim conectado y muestras llegando.
* **Stop** — detiene de forma limpia (vacía buffers y cierra ficheros).

El **log embebido** muestra los mensajes de la captura (vueltas
cerradas, banderas, pérdidas de paquetes, etc.).

Encima del formulario, una etiqueta muestra el **workspace activo**
(la carpeta donde se escribirán los CSV) y debajo de los botones un
contador **Laps recorded: N** se incrementa con cada vuelta
completa.

Las vueltas se separan automáticamente al cruzar la línea de meta y
aparecen en el panel **Captures** sin necesidad de pulsar
<kbd>F5</kbd>.

### 5.6. Overlay — HUD del race engineer en vivo

La pestaña **Overlay** no es un panel único: es un **gestor de
ventanas flotantes** independientes. Cada módulo que actives se abre
como una ventana **sin marco, siempre encima** de LFS, que puedes
colocar donde quieras. Se alimentan de un snapshot JSON
(`<workspace>/_overlay/live.json`) refrescado a ~10 Hz por el
proceso de captura.

**Comportamiento común de todas las ventanas:**

* Arrastra el **cuerpo** para moverlas.
* Arrastra la **esquina inferior derecha** para redimensionarlas.
* **Clic derecho** sobre la ventana para resetear su tamaño al
  valor por defecto.
* Cada módulo recuerda su **posición, tamaño y opacidad** entre
  sesiones (almacenado en `QSettings`).
* La **opacidad** se ajusta por módulo (20–100 %) desde la columna
  derecha de la lista de la pestaña.
* **Deseleccionar todos** — un botón al inicio de la lista de
  módulos cierra todas las ventanas del overlay con un solo clic;
  útil cuando la pantalla está saturada o cuando cambias entre
  configuraciones (carrera vs hot-lap vs trabajo de setup).

**Módulos disponibles** (en el orden en que aparecen en la lista):

| Módulo | Qué muestra |
|---|---|
| **Radar** | Radar 360° con los coches alrededor (azul = delante, rojo = detrás). Escala y umbrales de color configurables. |
| **G-meter (friction circle)** | Círculo de fricción con la aceleración longitudinal y lateral instantánea. Escala configurable en g. |
| **Delta bar vs personal best** | Barra horizontal con el delta frente a tu mejor vuelta personal (verde = ganando, rojo = perdiendo). Fondo de escala configurable en ms. |
| **Speed delta vs PB (same track point)** | Diferencia de velocidad instantánea frente a tu mejor vuelta personal en el mismo punto de la pista (verde = más rápido, rojo = más lento). |
| **Session info (dynamic)** | Resumen de la sesión: vuelta actual, última vuelta, mejor vuelta, tiempo de sesión, etc. En modo detallado la ventana se redimensiona automáticamente para mostrar toda la tabla de clasificación en vivo (todos los pilotos clasificados). Modo compacto opcional. |
| **Grip (per wheel)** | Indicador de agarre/riesgo por rueda (4 segmentos), útil para detectar pérdida de grip o sobrecalentamiento. |
| **Gap to driver ahead** | Tiempo hasta el coche que va por delante (decodificado desde InSim). Robusto frente a coches desconectados/DNF en la tabla de posiciones, coches parados en pits/espectadores y artefactos de wrap por desfase de vuelta. |
| **Gap to driver behind** | Tiempo hasta el coche que va por detrás. Misma robustez que Gap to ahead. |
| **Gear (big digit)** | Marcha actual en dígito grande. |
| **RPM bar** | Barra de revoluciones con redline configurable. |
| **Speed (km/h)** | Velocidad actual en km/h. |
| **Fuel %** | Porcentaje de combustible restante. |
| **Fuel laps remaining** | Vueltas restantes con el combustible actual al consumo medio observado en la sesión. |
| **Flags (BLUE / YELLOW)** | Indicador de banderas azul y amarilla decodificadas desde InSim. |
| **Pit limiter (flashing + speed delta)** | Banda parpadeante mientras el limitador de pit-lane está activo, más la velocidad actual frente al límite de pit configurado (por defecto 80 km/h). |

**Paneles de configuración** (debajo de la lista de módulos):

* **Radar** — escala (m), umbrales Rojo / Amarillo / Blanco (m) para
  el coloreado de coches según distancia.
* **Delta bar** — fondo de escala ± (ms), por defecto ±2000 ms.
* **RPM** — redline (rpm), por defecto 8000.
* **G-meter** — fondo de escala (g), por defecto 2.0 g.
* **Session overlay compact** — muestra la información de sesión
  en formato condensado.
* **Borderless / windowed-fullscreen compat** — usa ventanas
  top-most normales para mejorar la visibilidad del overlay cuando
  LFS corre en modo ventana o borderless.

> **Importante — LFS en pantalla completa exclusiva**: Windows no
> permite a ningún overlay (el nuestro, RTSS, Discord, Steam, etc.)
> dibujarse encima de un juego DirectX en *exclusive fullscreen*.
> Si los overlays no se ven con LFS en pantalla completa, abre
> `LFS\cfg.txt` y pon `Full screen window 1` (modo borderless
> nativo de LFS), o usa modo ventana normal. La única vía que
> sortea esta limitación es el **espejo VR** (ver más abajo),
> porque SteamVR tiene su propio compositor por encima del swap
> chain del juego.

### 5.7. Espejo VR (SteamVR / OpenVR)

La pestaña Live incluye un grupo **VR** con una sola casilla:

* **Mirror overlays to VR (SteamVR / OpenVR)** — al activarla, cada
  módulo de overlay visible se dibuja también como un panel
  `IVROverlay` anclado al casco. El widget Qt es la única fuente de
  verdad: la ventana de escritorio y el panel VR muestran contenido
  idéntico. No hay un segundo *look & feel* que configurar.

**Por qué VR funciona donde fullscreen exclusivo no**

SteamVR tiene su propio compositor de escena que corre por encima
de cualquier swap chain DirectX / Vulkan. Subir una textura a
`IVROverlay` hace que SteamVR la pinte sobre lo que sea que muestre
el juego en el casco. Esto funciona con LFS en cualquier modo de
ventana, incluido pantalla completa exclusiva, y con cualquier
runtime compatible con OpenVR (Valve Index, HTC Vive, Windows Mixed
Reality vía OpenVR, cascos Oculus vía OpenComposite, Meta Quest con
Steam Link, etc.).

**Requisitos**

* SteamVR (u otro runtime compatible con OpenVR) instalado y en
  ejecución antes de marcar la casilla.
* El paquete Python `openvr`. Ya empaquetado dentro del instalador
  de Windows; si ejecutas desde código, instala con
  `pip install lfs-race-engineer[vr]`.
* El casco debe estar trackeando (no en standby). La pose por
  defecto coloca los overlays a ~1.5 m frente al casco.

**Comportamiento**

* La casilla es un *no-op* si SteamVR no está corriendo o si falta
  el módulo `openvr` — vuelve sola a **off** y la etiqueta de estado
  bajo ella muestra el motivo (p. ej. `VR mirror unavailable: ...`).
* Mientras esté activa, un timer a 30 Hz lee cada módulo de overlay
  visible, lo renderiza off-screen en un `QImage` transparente, lo
  convierte a `RGBA8888` y lo sube vía `IVROverlay.SetOverlayRaw`.
  No hay coste extra de CPU/GPU cuando la casilla está apagada.
* Ocultar un módulo (desmarcándolo en la lista de módulos) también
  oculta el overlay VR correspondiente en el siguiente tick.
* Cerrar la pestaña Live o la app apaga todos los overlays VR
  limpiamente y libera la sesión OpenVR.

**Disposición por defecto de los paneles**

Los overlays se distribuyen en un arco suave a 1.5 m del casco,
ligeramente por debajo del nivel de los ojos para no sentarse sobre
el ápice de las curvas. Cada overlay mide ~40 cm de ancho en
unidades de mundo. La personalización de pose por módulo
(mover/escalar paneles individuales dentro del casco) está en la
hoja de ruta; hoy los valores por defecto están calibrados para
leerse sin ocultar la traza.

**Resolución de problemas**

* *La casilla se desmarca sola* — lee la etiqueta de estado. La
  causa habitual es que SteamVR no está corriendo.
* *Los paneles están visibles pero en blanco* — el módulo de origen
  todavía no ha recibido telemetría. Inicia una sesión en LFS o
  carga una repetición.
* *La ventana de escritorio también se ve en el monitor* — es
  intencionado. Ambos destinos comparten el mismo widget Qt; puedes
  mover la ventana de escritorio fuera de pantalla si solo quieres
  el panel VR.

> **Coches soportados en Overlay**: coches propios de LFS y mods
> verificados (los que tienen ficha en `config/cars.json`,
> `car_info.bin` empaquetados o footprints en `assets/source/mods/`).
> Para coches no reconocidos, la captura y el resto de pestañas
> siguen funcionando, pero los widgets que dependen de datos
> específicos del vehículo (Fuel %, Fuel laps remaining, escala del
> indicador de marcha) pueden mostrar valores neutros.

---

## 6. Paneles laterales adicionales

* **Track map** (izquierda) — línea de carrera promediada de la
  vuelta activa, marca de inicio/fin, cursor sincronizado con
  Telemetry, y la **referencia bundled** del circuito en gris si
  existe (`racing_lines/<TRACK>_racing.csv`).
* **Elevation** (izquierda) — perfil de altitud z(s) con bandas de
  peralte y clasificación de superficie del `.smx`.
* **Race dashboard** (derecha) — panel con los datos de carrera en
  formato grande: `Position`, `Lap`, `Current lap`, `Last lap`,
  `Best lap`, `Predicted`, `Δ vs best`, `SPB`, `Avg (stint)`,
  `Gap ahead`, `Gap behind`, `Fuel`, `Fuel laps left`, `Speed`,
  `Gear`, y una tabla de **standings** debajo.

---

## 7. Menú Tools

* **Configure LFS…** — el diálogo descrito en la sección 2 (parchea
  `cfg.txt` con los ajustes de OutSim/OutGauge).

(La línea de carrera bundled del circuito se carga automáticamente
desde `racing_lines/<TRACK>_racing.csv` si existe; no hay diálogo
manual.)

## 8. Menú File

* **Open Workspace…** (<kbd>Ctrl</kbd>+<kbd>O</kbd>) — elige la
  carpeta donde se guardan/leen los CSV.
* **Refresh Captures** (<kbd>F5</kbd>) — recarga la lista.
* **Clear Lap Cache** — borra la cache parquet en disco que acelera
  las recargas (útil si una vuelta se corrompe o si quieres recalcular
  los canales derivados).
* **Import RAF…** — importa un **Replay Analyser File** (`.raf`) de
  LFS. La aplicación recorre el RAF, lo parte por vueltas (detectando
  el cruce de meta con el *index distance*) y escribe un CSV por
  vuelta dentro de `<workspace>/<nombre>_raf_laps/`. Tras la
  importación los CSV aparecen en **Captures** y se pueden cargar y
  comparar como cualquier otra vuelta. Es la única vía oficial para
  analizar la telemetría de otro piloto a partir de su replay
  (`.mpr`/`.spr`): abre el replay en LFS, pulsa el botón **Analyse**
  para generar el `.raf`, y luego impórtalo aquí.
* **Quit** (<kbd>Ctrl</kbd>+<kbd>Q</kbd>) — cierra la aplicación.

## 8 bis. Menú Help

* **Channel guide…** (<kbd>F1</kbd>) — guía de telemetría: qué mide
  cada canal y cómo interpretarlo.
* **About** — información de la versión.

---

## 9. Recomendaciones de uso

1. **Configura LFS una sola vez** (Tools → Configure LFS) y activa
   InSim en el acceso directo con `/insim=29999`.
2. **Crea una carpeta de workspace** por temporada o por coche, y
   ábrela con <kbd>Ctrl</kbd>+<kbd>O</kbd>.
3. En **Capture**, pulsa **Start**, entra a pista en LFS, gira tus
   vueltas, sal a boxes y pulsa **Stop**.
4. En **Captures** dock, selecciona tu mejor vuelta y con
   <kbd>Ctrl</kbd>+clic añade las que quieras comparar. Activa los
   canales en el panel **Channels** y úsalos en la pestaña
   **Telemetry**.
5. Para sesiones de carrera sin grabación, marca **Overlay only** en
   Capture y deja la pestaña **Overlay** visible junto a LFS.

---

## 10. Solución de problemas rápida

| Síntoma | Causa probable | Solución |
| --- | --- | --- |
| LED de Capture sigue gris/ámbar | InSim no está activo en LFS | En LFS, pulsa <kbd>T</kbd> y escribe `/insim 29999`, o reinicia LFS con `/insim=29999`. |
| No hay temperaturas/cargas por rueda | LFS está en `OutSim Mode 1` (modo legacy de 64 B) | Vuelve a Tools → Configure LFS y aplica el patch (deja `OutSim Mode 2` y `OutSim Opts 1ff`). |
| Patch dice "LFS folder inválido" | La carpeta no contiene `LFS.exe` | Usa **Browse…** y elige la carpeta correcta. |
| El Overlay no muestra fuel range / desgaste | Coche no incluido en la lista de soportados | Captura normalmente; los demás paneles funcionan. Para añadirlo manualmente, ejecuta la calibración (uso avanzado). |
| LFS sobrescribió mis cambios | LFS estaba abierto al patchear | Cierra LFS y vuelve a aplicar Tools → Configure LFS. |
| Quiero el `cfg.txt` original | El patch dejó `cfg.txt.bak` | Renómbralo a `cfg.txt` con LFS cerrado. |
| **Start** se queda parado al instante o salta *"Access/Permission denied"* al grabar | El workspace está en una carpeta sin permisos de escritura para tu usuario (Program Files, Windows, unidad de red protegida…) | **File → Open Workspace…** y elige una carpeta donde tengas permisos, por ejemplo dentro de `Documentos` o `Escritorio`. |