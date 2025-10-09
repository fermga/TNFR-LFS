# Flujo de releases con semantic-release

Este proyecto utiliza [`python-semantic-release`](https://python-semantic-release.readthedocs.io/) para automatizar la generación de versiones siguiendo la convención de commits. El pipeline calcula la próxima versión semántica, actualiza el changelog y publica los artefactos cuando se fusionan cambios en la rama `main` o se crean tags que sigan el formato `vMAJOR.MINOR.PATCH`.

## Convención de commits

Las versiones se determinan a partir de los mensajes de commit empleando el esquema **Conventional Commits**. Los prefijos más comunes son:

- `feat:` incrementa la versión **minor**.
- `fix:` incrementa la versión **patch**.
- `perf:` y otros tipos compatibles con mejoras también cuentan como **patch**.
- Cualquier commit que incluya `BREAKING CHANGE:` en el cuerpo o utilice el sufijo `!` en el tipo dispara un incremento **major**.

Se recomienda describir los cambios de forma concisa en la primera línea y añadir contexto adicional en el cuerpo del commit cuando sea necesario.

## Proceso de publicación

1. Asegúrate de que las ramas de trabajo incluyan commits con la convención anterior y que las pruebas automáticas pasen antes de fusionar en `main`.
2. Cuando los cambios llegan a `main`, el workflow **Release** ejecuta `python-semantic-release` con permisos para escribir tags y para publicar en PyPI.
3. La herramienta determina la nueva versión comparando el historial desde el último tag `v*`, actualiza/crea `CHANGELOG.md`, etiqueta el commit con la nueva versión y publica los artefactos construidos en el registro configurado.
4. Si se requiere un release manual (por ejemplo, para reconstruir una versión existente), crea un tag `vMAJOR.MINOR.PATCH` apuntando al commit deseado y súbelo al repositorio remoto. El workflow detectará el tag y repetirá el proceso de publicación.

## Resolución de problemas

- Si el workflow no genera una nueva versión, verifica que los commits fusionados desde el último release contengan prefijos compatibles con la convención.
- Confirma que el secreto `PYPI_API_TOKEN` esté configurado en el repositorio para permitir la publicación automática.
- Para escenarios en los que se necesiten versiones preliminares, considera utilizar ramas dedicadas y ajustar la configuración de semantic-release para manejar pre-releases.

Siguiendo estos pasos, las nuevas versiones se gestionan de manera consistente y predecible a partir del historial de commits.
