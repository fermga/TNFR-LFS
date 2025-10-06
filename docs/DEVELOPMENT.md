# Desarrollo y verificación

Este repositorio utiliza herramientas de linting, tipado y pruebas para mantener
el toolkit TNFR × LFS en un estado saludable. Tras clonar el proyecto instala
las dependencias de desarrollo (esto incluye `numpy>=1.24,<2.0` y
`pandas>=1.5,<3.0`, requeridos por las pruebas y los pipelines de análisis):

```bash
pip install .[dev]
```

## Flujo de verificación

Ejecuta los siguientes comandos antes de enviar un cambio. Cada herramienta está
integrada en la configuración de CI y en GitHub Actions:

```bash
ruff check .
mypy --strict
pytest
```

- ``ruff`` aplica las reglas de estilo y limpieza del código definidas en
  ``pyproject.toml``.
- ``mypy --strict`` analiza ``tnfr_lfs`` y el resto de rutas añadidas al
  proyecto (por ejemplo ``typing_targets``). Si necesitas pasar las rutas de
  forma explícita puedes ejecutar ``mypy --strict tnfr_lfs typing_targets``;
  ten en cuenta que el análisis estricto sobre todo ``tnfr_lfs`` tarda algo más
  que el conjunto reducido anterior.
- ``pytest`` ejecuta la batería de pruebas unitaria e integración, incluida la
  verificación de los datasets de ejemplo ``data/test1.raf`` y ``data/test1.zip``.

## Dataset de referencia

El flujo de quickstart y los tests de integración dependen de la captura
``data/test1.raf`` y del bundle ``data/test1.zip`` exportado desde Replay
Analyzer. Ambos artefactos recogen telemetría real de Live for Speed y sirven
como datasets de referencia para los flujos de ingesta binaria, las pruebas de
regresión y los tutoriales del CLI.
