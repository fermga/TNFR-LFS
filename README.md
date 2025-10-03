# TNFR × LFS Toolkit

TNFR × LFS es el toolkit de análisis y automatización que operacionaliza la **Teoría de la Naturaleza Fractal-Resonante (TNFR)**. Mientras la teoría describe los fundamentos conceptuales (símbolos, glifos y métricas ΔNFR/ΔSi), el paquete suministra implementaciones reproducibles para instrumentar dichas ideas en flujos de trabajo de telemetría.

## Documentación

La documentación completa se encuentra en el directorio `docs/` y se publica con MkDocs.

- [`docs/DESIGN.md`](docs/DESIGN.md): resumen textual del manual operativo TNFR. El PDF `TNFR.pdf` debe tratarse como artefacto generado; cualquier actualización canónica debe realizarse en Markdown para conservar diffs legibles.
- [`docs/index.md`](docs/index.md): página de inicio de la documentación del toolkit TNFR × LFS.
- [`docs/api_reference.md`](docs/api_reference.md): referencia de la API.
- [`docs/cli.md`](docs/cli.md): guía de la interfaz de línea de comandos.

## Branding y relación con la teoría

- **TNFR** se refiere únicamente a la teoría original: el marco conceptual que define los EPI, las métricas ΔNFR/ΔSi y los principios resonantes.
- **TNFR × LFS** identifica el toolkit de software que implementa y automatiza esos principios. La denominación utiliza el símbolo "×" para enfatizar que la teoría TNFR se combina con la metodología Load Flow Synthesis (LFS) dentro del paquete.
- El CLI (`tnfr-lfs`), los módulos de Python (`tnfr_lfs`) y las rutas de configuración (`tnfr-lfs.toml`) conservan el guion medio por compatibilidad retroactiva, pero forman parte del toolkit TNFR × LFS.

## Desarrollo

Consulta `pyproject.toml` para conocer los requisitos y scripts de desarrollo y
``docs/DEVELOPMENT.md`` para el flujo completo de verificación (``ruff``,
``mypy --strict`` y ``pytest``).
