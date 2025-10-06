# Desarrollo y verificación

Este repositorio utiliza herramientas de linting, tipado y pruebas para mantener
el toolkit TNFR × LFS en un estado saludable. Tras clonar el proyecto instala
las dependencias de desarrollo:

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
- ``mypy --strict`` verifica los objetivos tipados incluidos en ``typing_targets``.
- ``pytest`` ejecuta la batería de pruebas unitaria e integración, incluida la
  verificación de los datasets de ejemplo ``data/BL1_XFG_baseline.csv`` y el
  RAF ``data/test1.raf``.

## Dataset de referencia

El flujo de quickstart y los tests de integración dependen del archivo
``data/BL1_XFG_baseline.csv``. El dataset contiene 17 muestras sintetizadas para
la combinación BL1/XFG y se utiliza tanto por ``examples/quickstart.sh`` como
por las pruebas de regresión. Para reproducir escenarios con telemetría real,
el repositorio añade además ``data/test1.raf`` (captura RAF de Live for Speed)
que sirve de referencia para la ingesta binaria y los tutoriales del CLI.
