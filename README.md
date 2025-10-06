# TNFR × LFS Toolkit

TNFR × LFS es el toolkit de análisis y automatización que operacionaliza la **Teoría de la Naturaleza Fractal-Resonante (TNFR)**. Mientras la teoría describe los fundamentos conceptuales (símbolos, glifos y métricas ΔNFR/ΔSi), el paquete suministra implementaciones reproducibles para instrumentar dichas ideas en flujos de trabajo de telemetría.

## Quickstart

### Requisitos previos
- Python 3.9 o superior instalado y disponible en el PATH.
- Un dataset de telemetría en `data/` (el repositorio incluye `data/BL1_XFG_baseline.csv`).

### Ejecución
Ejecuta el flujo end-to-end simulado con:

```bash
make quickstart
```

El objetivo invoca `examples/quickstart.sh`, que genera artefactos en `examples/out/` a partir del dataset y produce informes JSON y Markdown listos para inspeccionar.

### HUD en directo

Para mostrar el HUD ΔNFR dentro de Live for Speed habilita los broadcasters en el simulador (`/outsim 1 127.0.0.1 4123`, `/outgauge 1 127.0.0.1 3000` y `/insim 29999`) y ejecuta:

```bash
tnfr-lfs osd --host 127.0.0.1 --outsim-port 4123 --outgauge-port 3000 --insim-port 29999
```

El panel alterna tres páginas (estado de curva/fase, contribuciones nodales y plan de setup) que caben en un botón `IS_BTN` de 40×16. Puedes moverlo con `--layout-left/--layout-top` si otro mod ocupa la misma zona.

### Telemetría requerida

Todas las métricas e indicadores TNFR (`ΔNFR`, `ΔNFR_lat`, `ν_f`, `C(t)` y derivados) se calculan exclusivamente a partir de la telemetría que exponen OutSim y OutGauge en Live for Speed. Asegúrate de activar ambos broadcasters y de ampliar el bloque de OutSim con `OutSim Opts ff` en `cfg.txt` (o mediante `/outsim Opts ff` seguido de `/outsim 1 …`) para incluir ID de jugador, entradas de piloto y el paquete de ruedas que contiene fuerzas, cargas y deflexiones que consume el fusionador de telemetría del toolkit.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】 Los `slip_ratio` y `slip_angle` globales que emite TNFR × LFS provienen directamente de esos canales por rueda: se promedian las lecturas decodificadas de OutSim (ponderadas por carga cuando está disponible) y solo se recurre al modelo cinemático previo cuando algún paquete llega sin lecturas válidas.【F:tnfr_lfs/acquisition/fusion.py†L111-L175】【F:tnfr_lfs/acquisition/fusion.py†L242-L291】

Los indicadores de set-up (como el Contact Patch Health Index y los ajustes ΔP/Δcamber del operador térmico) se alimentan únicamente de esa telemetría nativa: utilizan los deslizamientos, fuerzas y cargas por rueda que expone OutSim, junto con el estado del motor y de los pedales que transmite OutGauge. No se aplican heurísticas externas ni datos sintéticos, lo que garantiza que cualquier recomendación deriva directamente de lo que transmite Live for Speed en tiempo real.【F:tnfr_lfs/core/metrics.py†L200-L284】【F:tnfr_lfs/core/operators.py†L731-L830】

#### Señales clave por métrica

- **ΔNFR / ΔNFR_lat** – combina las cargas verticales, aceleraciones longitudinales y laterales, fuerzas de rueda y deflexiones de suspensión derivadas del paquete de ruedas OutSim con el régimen del motor, posición de pedal y luces ABS/TC de OutGauge para contextualizar el bloqueo y la referencia longitudinal.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】
- **ν_f (frecuencia natural)** – requiere el reparto de carga, las ratios/ángulos de deslizamiento y la velocidad/yaw rate procedentes de OutSim junto a la señal de estilo del piloto (`throttle`, `gear`) para ajustar la categoría y ventana espectral de cada nodo.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L648-L710】
- **C(t) (coherencia estructural)** – se deriva de la distribución nodal de ΔNFR y de las bandas de ν_f, por lo que depende de los mismos campos de OutSim, de los coeficientes de adherencia `mu_eff_*` calculados a partir de las aceleraciones laterales/longitudinales y de las banderas ABS/TC que llegan por OutGauge.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】【F:tnfr_lfs/core/coherence.py†L65-L125】
- **Ventilación / fade de freno** – solo se evalúan cuando OutGauge transmite temperaturas reales de pinza y disco. Si el broadcaster envía los valores de marcador (`0 °C`) TNFR × LFS mostrará “sin datos” para evitar diagnósticos falsamente optimistas.【F:tnfr_lfs/acquisition/fusion.py†L566-L610】【F:tnfr_lfs/core/metrics.py†L860-L982】

## Documentación

La documentación completa se encuentra en el directorio `docs/` y se publica con MkDocs.

- [`docs/DESIGN.md`](docs/DESIGN.md): resumen textual del manual operativo TNFR. El PDF `TNFR.pdf` debe tratarse como artefacto generado; cualquier actualización canónica debe realizarse en Markdown para conservar diffs legibles.
- [`docs/index.md`](docs/index.md): página de inicio de la documentación del toolkit TNFR × LFS.
- [`docs/api_reference.md`](docs/api_reference.md): referencia de la API.
- [`docs/cli.md`](docs/cli.md): guía de la interfaz de línea de comandos.
- [`docs/setup_equivalences.md`](docs/setup_equivalences.md): tabla de
  correspondencias que enlaza cada métrica TNFR (`ΔNFR_lat`, `ν_f`, `C(t)`) con
  ajustes concretos de setup para preparar sesiones con coherencia.

### Plantilla de configuración del CLI

El archivo `tnfr-lfs.toml` ubicado en la raíz del repositorio proporciona una plantilla lista para usar con los valores por defecto documentados en [`docs/cli.md`](docs/cli.md). Copia este archivo en `~/.config/tnfr-lfs.toml` o en el directorio de trabajo de tus sesiones para personalizar los puertos OutSim/OutGauge/InSim, el coche/pista iniciales y los límites de ΔNFR por fase antes de invocar el CLI.

Desde esta versión el CLI también puede resolver un *pack* TNFR × LFS completo (metadatos de coches y perfiles objetivo) apuntando `paths.pack_root` al directorio que contiene `config/global.toml`, `data/cars` y `data/profiles`. La opción global `--pack-root` permite cambiar el pack en una ejecución concreta sin modificar el fichero TOML. Los resultados de `analyze`, `suggest` y `write-set` incluyen ahora las secciones `car` y `tnfr_targets`, derivadas de ese pack, para que los exports JSON reflejen el contexto TNFR activo.

### Subcomandos `pareto` y `compare`

| Subcomando | Propósito | Exportadores recomendados | Ejemplo |
| --- | --- | --- | --- |
| `pareto` | Barrer el espacio de decisión alrededor del plan generado y devolver el frente de Pareto con los candidatos dominantes. | `markdown` para inspección rápida o `html_ext` para un informe navegable con tablas, histogramas y sugerencias del playbook. | `tnfr-lfs pareto stint.jsonl --car-model generic_gt --radius 2 --export html_ext > pareto.html` |
| `compare` | Comparar dos stints o configuraciones (A/B) agregando métricas por vuelta y resaltando la variante ganadora para la métrica elegida. | `markdown` para notas de ingeniería o `html_ext` cuando se necesita un dashboard con gráficos de vueltas y mensajes de sesión. | `tnfr-lfs compare baseline.jsonl variante.jsonl --metric sense_index --export html_ext > abtest.html` |

Ambos subcomandos aceptan `--export html_ext`, que reutiliza el exportador HTML extendido para incrustar el frente de Pareto, resúmenes A/B y las sugerencias del playbook en un único archivo portable.【F:tnfr_lfs/cli/pareto.py†L23-L117】【F:tnfr_lfs/cli/compare.py†L28-L135】【F:tnfr_lfs/exporters/__init__.py†L1183-L1193】

## Branding y relación con la teoría

- **TNFR** se refiere únicamente a la teoría original: el marco conceptual que define los EPI, las métricas ΔNFR/ΔSi y los principios resonantes.
- **TNFR × LFS** identifica el toolkit de software que implementa y automatiza esos principios. La denominación utiliza el símbolo "×" para enfatizar que la teoría TNFR se combina con la metodología Load Flow Synthesis (LFS) dentro del paquete.
- El CLI (`tnfr-lfs`), los módulos de Python (`tnfr_lfs`) y las rutas de configuración (`tnfr-lfs.toml`) conservan el guion medio por compatibilidad retroactiva, pero forman parte del toolkit TNFR × LFS.

## Desarrollo

Consulta `pyproject.toml` para conocer los requisitos y scripts de desarrollo y
``docs/DEVELOPMENT.md`` para el flujo completo de verificación (``ruff``,
``mypy --strict`` y ``pytest``).
