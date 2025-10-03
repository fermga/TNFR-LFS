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

## Documentación

La documentación completa se encuentra en el directorio `docs/` y se publica con MkDocs.

- [`docs/DESIGN.md`](docs/DESIGN.md): resumen textual del manual operativo TNFR. El PDF `TNFR.pdf` debe tratarse como artefacto generado; cualquier actualización canónica debe realizarse en Markdown para conservar diffs legibles.
- [`docs/index.md`](docs/index.md): página de inicio de la documentación del toolkit TNFR × LFS.
- [`docs/api_reference.md`](docs/api_reference.md): referencia de la API.
- [`docs/cli.md`](docs/cli.md): guía de la interfaz de línea de comandos.

### Plantilla de configuración del CLI

El archivo `tnfr-lfs.toml` ubicado en la raíz del repositorio proporciona una plantilla lista para usar con los valores por defecto documentados en [`docs/cli.md`](docs/cli.md). Copia este archivo en `~/.config/tnfr-lfs.toml` o en el directorio de trabajo de tus sesiones para personalizar los puertos OutSim/OutGauge/InSim, el coche/pista iniciales y los límites de ΔNFR por fase antes de invocar el CLI.

## Branding y relación con la teoría

- **TNFR** se refiere únicamente a la teoría original: el marco conceptual que define los EPI, las métricas ΔNFR/ΔSi y los principios resonantes.
- **TNFR × LFS** identifica el toolkit de software que implementa y automatiza esos principios. La denominación utiliza el símbolo "×" para enfatizar que la teoría TNFR se combina con la metodología Load Flow Synthesis (LFS) dentro del paquete.
- El CLI (`tnfr-lfs`), los módulos de Python (`tnfr_lfs`) y las rutas de configuración (`tnfr-lfs.toml`) conservan el guion medio por compatibilidad retroactiva, pero forman parte del toolkit TNFR × LFS.

## Desarrollo

Consulta `pyproject.toml` para conocer los requisitos y scripts de desarrollo y
``docs/DEVELOPMENT.md`` para el flujo completo de verificación (``ruff``,
``mypy --strict`` y ``pytest``).
