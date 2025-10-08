# Inventario de localización en pruebas

Este inventario agrupa las cadenas en español detectadas en `tests/` (via `rg --no-heading --line-number "[áéíóúñÁÉÍÓÚ¿¡]" tests`) y documenta las fuentes en producción que deben reflejarse en las pruebas. Los grupos se ordenan por criticidad: primero los fixtures compartidos y, a continuación, los módulos con mayor cantidad de aserciones dependientes de texto.

## 1. Fixtures compartidos
- `tests/conftest.py`: valores de ejemplo como `"Validación"` y `"Tracción progresiva"` se utilizan para poblar cargas sintéticas y no provienen de módulos de producción. 【F:tests/conftest.py†L286-L358】
- `tests/test_core_operators.py`: el helper `_build_goal` establece `description=f"Meta sintética para {actual_phase}"` únicamente para los escenarios de prueba. 【F:tests/test_core_operators.py†L189-L221】

## 2. CLI y workflows (`tnfr_lfs/cli/workflows.py`)
- Las pruebas del CLI validan mensajes como `"OutSim respondió desde {host}:{port}"` o `"InSim respondió con versión 9"`, ya implementados en el helper `_udp_ping` y en `_insim_handshake`. 【F:tests/test_cli.py†L1102-L1124】【F:tnfr_lfs/cli/workflows.py†L941-L1017】
- El resumen térmico incorpora `"- Presión (bar): sin datos (OutGauge omitió el bloque de neumáticos)"`, coincidiendo con la lógica de `_build_summary`. 【F:tests/test_cli.py†L704-L728】【F:tnfr_lfs/cli/workflows.py†L1801-L1819】
- El informe avanzado incluye secciones en español como `"## Disonancia útil"`, confirmadas en `_summarise_advanced_metrics`. 【F:tests/test_exporters.py†L438-L456】【F:tnfr_lfs/cli/workflows.py†L1743-L1778】
- **Checklist producción ↔ pruebas** (ya alineado):
  - [x] `{label} responded from {addr}` → `{label} respondió desde {addr}`
  - [x] `- Pressure (bar): no data …` → `- Presión (bar): sin datos …`
  - [x] `## Useful dissonance` → `## Disonancia útil`

## 3. Exportadores de setup (`tnfr_lfs/exporters/__init__.py`)
- Los tests esperan encabezados como `"**Comparación A/B**"`, `"**Contribución SCI**"` o tablas con `"| Cambio | Δ | Acción |"`, ya traducidos en `markdown_exporter` y `lfs_notes_exporter`. 【F:tests/test_exporters.py†L140-L257】【F:tnfr_lfs/exporters/__init__.py†L172-L347】【F:tnfr_lfs/exporters/__init__.py†L391-L438】
- **Checklist producción ↔ pruebas** (ya alineado):
  - [x] `**A/B comparison**` → `**Comparación A/B**`
  - [x] `| Change | Δ | Action |` → `| Cambio | Δ | Acción |`
  - [x] `Quick TNFR notes` → `Instrucciones rápidas TNFR`

## 4. Motor de recomendaciones (`tnfr_lfs/recommender/rules.py`)
Las aserciones de `tests/test_recommender.py` requieren traducir varios mensajes y racionales aún en inglés:

- `High-speed microsector {index}: {action}` → _Objetivo_: mensaje que conserve "Alta velocidad" y detalle la acción (`increase/reduce rear wing angle`) con vocabulario como "alerón" y "carga". 【F:tests/test_recommender.py†L931-L964】【F:tnfr_lfs/recommender/rules.py†L1410-L1470】
- `High-speed microsector {index}: increase front wing angle` → _Objetivo_: referencia explícita a "alerón delantero" en la acción. 【F:tests/test_recommender.py†L965-L996】【F:tnfr_lfs/recommender/rules.py†L1495-L1534】
- `Reinforce {direction} load ({manual_reference})` → _Objetivo_: racionales con "carga delantera/carga trasera". 【F:tests/test_recommender.py†L931-L996】【F:tnfr_lfs/recommender/rules.py†L1463-L1471】【F:tnfr_lfs/recommender/rules.py†L1523-L1531】
- `Transmission operator: smooth apex→exit shifts in microsector {index}` y `Transmission metrics highlight losses during the apex→exit transition: …` → _Objetivo_: mensajes y racionales que incluyan "transmisión" y describan la transición apex→salida en español. 【F:tests/test_recommender.py†L1607-L1641】【F:tnfr_lfs/recommender/rules.py†L3389-L3405】
- `Stiffen compression` (varias ramas de `SuspensionVelocityRule`) → _Objetivo_: instrucciones con "compresión" para amortiguadores. 【F:tests/test_recommender.py†L2008-L2033】【F:tnfr_lfs/recommender/rules.py†L1085-L1097】

**Checklist producción ↔ pruebas (localización completada):**
- [x] `High-speed microsector {index}: increase rear wing angle` → `Microsector de alta velocidad {index}: aumenta alerón trasero`
- [x] `High-speed microsector {index}: reduce rear wing angle / reinforce front load` → `Microsector de alta velocidad {index}: reduce alerón trasero / refuerza carga delantera`
- [x] `High-speed microsector {index}: increase front wing angle` → `Microsector de alta velocidad {index}: aumenta alerón delantero`
- [x] `Transmission operator: smooth apex→exit shifts in microsector {index}` → `Operador de transmisión: suaviza los cambios apex→salida en el microsector {index}`
- [x] `Transmission metrics highlight losses during the apex→exit transition: …` → `Los indicadores de transmisión evidencian pérdidas durante la transición apex→salida: …`
- [x] `Stiffen compression` → `Refuerza compresión`

## 5. Exportador HTML extendido y perfil de sesión
- `html_exporter` aún utiliza encabezados en inglés como `"<h2>A/B comparison</h2>"`, `"<h2>Session profile</h2>"`, `"<h2>Playbook suggestions</h2>"` y `"<h2>Data sources and estimates</h2>"`. 【F:tnfr_lfs/exporters/report_extended.py†L250-L319】
- `format_session_messages` genera frases `"Session profile weights → …"` y `"Session profile hints → …"`. 【F:tnfr_lfs/session.py†L70-L87】
- **Checklist producción ↔ pruebas (pendiente de localización):**
  - [ ] `<h2>A/B comparison</h2>` → `<h2>Comparación A/B</h2>`
  - [ ] `<h2>Session profile</h2>` → `<h2>Perfil de sesión</h2>`
  - [ ] `<h2>Playbook suggestions</h2>` → `<h2>Sugerencias de playbook</h2>`
  - [ ] `<h2>Data sources and estimates</h2>` → `<h2>Fuentes de datos y estimaciones</h2>`
  - [ ] `Session profile weights → …` → `Perfil de sesión · pesos → …`
  - [ ] `Session profile hints → …` → `Perfil de sesión · pistas → …`

## 6. HUD OSD (`tnfr_lfs/cli/osd.py`)
- El paginador de HUD ofrece textos localizados (`"Esperando telemetría…"`, `"ΔNFR nodal en espera"`, etc.) usados por `tests/test_cli_osd.py`. 【F:tests/test_cli_osd.py†L35-L90】【F:tnfr_lfs/cli/osd.py†L126-L152】
- La sección de nodos escribe `"ΔNFR nodal · Líder → …"` y las advertencias incluyen `"Abre el menú de boxes (F12)"` y `"Detén el coche antes de aplicar"`, coincidiendo con las expectativas de las pruebas. 【F:tests/test_cli_osd.py†L321-L366】【F:tnfr_lfs/cli/osd.py†L1499-L1516】【F:tnfr_lfs/cli/osd.py†L2240-L2264】

## 7. Otros módulos
- `tests/test_playbook_rules.py` emplea cadenas cortas como `"Más soporte"` dentro de playbooks sintéticos; el recurso embebido ya contiene descripciones detalladas en español (`tnfr_lfs/data/playbooks/tnfr_playbook.toml`). 【F:tests/test_playbook_rules.py†L10-L78】【F:tnfr_lfs/data/playbooks/tnfr_playbook.toml†L1-L24】
- `tests/test_robustness.py` etiqueta laps como `"Clasificación"` o `"Vuelta 2"` dentro de datos de ejemplo sin dependencias directas en producción. 【F:tests/test_robustness.py†L23-L44】

---
Este documento sirve como hoja de ruta para las fases de localización: los grupos ya traducidos se marcan como completados y los pendientes incluyen el checklist `<texto actual> → <texto objetivo>` solicitado para seguimiento incremental.
