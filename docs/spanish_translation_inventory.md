# Localization inventory for tests

This inventory groups the Spanish strings detected in `tests/` (via
`rg --no-heading --line-number "[áéíóúñÁÉÍÓÚ¿¡]" tests`) and documents the
production sources that the tests should mirror. The groups are ordered by
criticality: shared fixtures first, followed by the modules with the highest
number of text-dependent assertions.

## 1. Shared fixtures
- `tests/conftest.py`: example values such as `"Validación"` and
  `"Tracción progresiva"` populate synthetic payloads and do not originate from
  production modules. 【F:tests/conftest.py†L286-L358】
- `tests/test_core_operators.py`: the `_build_goal` helper sets
  `description=f"Meta sintética para {actual_phase}"` exclusively for test
  scenarios. 【F:tests/test_core_operators.py†L189-L221】

## 2. CLI and workflows (`tnfr_lfs/cli/workflows.py`)
- The CLI tests validate messages such as `"OutSim respondió desde {host}:{port}"`
  or `"InSim respondió con versión 9"`, already implemented in the `_udp_ping`
  helper and in `_insim_handshake`. 【F:tests/test_cli.py†L1102-L1124】
  【F:tnfr_lfs/cli/workflows.py†L941-L1017】
- The thermal summary includes `"- Pressure (bar): no data (OutGauge omitted the
  tyre block)"`, matching the `_build_summary` logic.
  【F:tests/test_cli.py†L704-L728】【F:tnfr_lfs/cli/workflows.py†L1801-L1819】
- The advanced report contains Spanish headings such as `"## Disonancia útil"`,
  confirmed in `_summarise_advanced_metrics`.
  【F:tests/test_exporters.py†L438-L456】【F:tnfr_lfs/cli/workflows.py†L1743-L1778】
- **Production ↔ tests checklist** (already aligned):
  - [x] `{label} responded from {addr}`
  - [x] `- Pressure (bar): no data …`
  - [x] `## Useful dissonance`

## 3. Setup exporters (`tnfr_lfs/exporters/__init__.py`)
- The tests expect headings such as `"**Comparación A/B**"`,
  `"**Contribución SCI**"`, and tables with `"| Cambio | Δ | Acción |"`, already
  translated in `markdown_exporter` and `lfs_notes_exporter`.
  【F:tests/test_exporters.py†L140-L257】
  【F:tnfr_lfs/exporters/__init__.py†L172-L347】
  【F:tnfr_lfs/exporters/__init__.py†L391-L438】
- **Production ↔ tests checklist** (already aligned):
  - [x] `**A/B comparison**` → `**Comparación A/B**`
  - [x] `| Change | Δ | Action |` → `| Cambio | Δ | Acción |`
  - [x] `Quick TNFR notes` → `Instrucciones rápidas TNFR`

## 4. Recommendation engine (`tnfr_lfs/recommender/rules.py`)
The assertions in `tests/test_recommender.py` require translating several
messages and rationales that are still in English:

- `High-speed microsector {index}: {action}` → _Goal_: keep "Alta velocidad" and
  describe the action (`increase/reduce rear wing angle`) with vocabulary such
  as "alerón" and "carga".【F:tests/test_recommender.py†L931-L964】
  【F:tnfr_lfs/recommender/rules.py†L1410-L1470】
- `High-speed microsector {index}: increase front wing angle` → _Goal_: include
  an explicit reference to "alerón delantero" in the action.
  【F:tests/test_recommender.py†L965-L996】
  【F:tnfr_lfs/recommender/rules.py†L1495-L1534】
- `Reinforce {direction} load ({manual_reference})` → _Goal_: rationales that
  mention "carga delantera/carga trasera".
  【F:tests/test_recommender.py†L931-L996】
  【F:tnfr_lfs/recommender/rules.py†L1463-L1471】
  【F:tnfr_lfs/recommender/rules.py†L1523-L1531】
- `Transmission operator: smooth apex→exit shifts in microsector {index}` and
  `Transmission metrics highlight losses during the apex→exit transition: …` →
  _Goal_: messages and rationales that include "transmisión" and describe the
  apex→exit transition in Spanish.
  【F:tests/test_recommender.py†L1607-L1641】
  【F:tnfr_lfs/recommender/rules.py†L3389-L3405】
- `Stiffen compression` (several branches of `SuspensionVelocityRule`) → _Goal_:
  instructions that reference "compresión" for the dampers.
  【F:tests/test_recommender.py†L2008-L2033】
  【F:tnfr_lfs/recommender/rules.py†L1085-L1097】

**Production ↔ tests checklist (localization completed):**
- [x] `High-speed microsector {index}: increase rear wing angle` → `Microsector
  de alta velocidad {index}: aumenta alerón trasero`
- [x] `High-speed microsector {index}: reduce rear wing angle / reinforce front
  load` → `Microsector de alta velocidad {index}: reduce alerón trasero /
  refuerza carga delantera`
- [x] `High-speed microsector {index}: increase front wing angle` → `Microsector
  de alta velocidad {index}: aumenta alerón delantero`
- [x] `Transmission operator: smooth apex→exit shifts in microsector {index}` →
  `Operador de transmisión: suaviza los cambios apex→salida en el microsector
  {index}`
- [x] `Transmission metrics highlight losses during the apex→exit transition:
  …` → `Los indicadores de transmisión evidencian pérdidas durante la transición
  apex→salida: …`
- [x] `Stiffen compression` → `Refuerza compresión`

## 5. Extended HTML exporter and session profile
- `html_exporter` still uses English headings such as `"<h2>A/B comparison</h2>"`,
  `"<h2>Session profile</h2>"`, `"<h2>Playbook suggestions</h2>"`, and
  `"<h2>Data sources and estimates</h2>"`.
  【F:tnfr_lfs/exporters/report_extended.py†L250-L319】
- `format_session_messages` generates phrases `"Session profile weights → …"`
  and `"Session profile hints → …"`.
  【F:tnfr_lfs/session.py†L70-L87】
- **Production ↔ tests checklist (localization pending):**
  - [ ] `<h2>A/B comparison</h2>` → `<h2>Comparación A/B</h2>`
  - [ ] `<h2>Session profile</h2>` → `<h2>Perfil de sesión</h2>`
  - [ ] `<h2>Playbook suggestions</h2>` → `<h2>Sugerencias de playbook</h2>`
  - [ ] `<h2>Data sources and estimates</h2>` → `<h2>Fuentes de datos y estimaciones</h2>`
  - [ ] `Session profile weights → …` → `Perfil de sesión · pesos → …`
  - [ ] `Session profile hints → …` → `Perfil de sesión · pistas → …`

## 6. HUD OSD (`tnfr_lfs/cli/osd.py`)
- The HUD paginator offers localized texts (`"Esperando telemetría…"`,
  `"ΔNFR nodal en espera"`, etc.) that `tests/test_cli_osd.py` consumes.
  【F:tests/test_cli_osd.py†L35-L90】【F:tnfr_lfs/cli/osd.py†L126-L152】
- The node section writes `"ΔNFR nodal · Líder → …"` and the warnings include
  `"Abre el menú de boxes (F12)"` and `"Detén el coche antes de aplicar"`, in
  line with the test expectations.
  【F:tests/test_cli_osd.py†L321-L366】
  【F:tnfr_lfs/cli/osd.py†L1499-L1516】
  【F:tnfr_lfs/cli/osd.py†L2240-L2264】

## 7. Other modules
- `tests/test_playbook_rules.py` uses short strings such as `"Más soporte"`
  inside synthetic playbooks; the embedded resource already provides detailed
  descriptions in Spanish (`tnfr_lfs/data/playbooks/tnfr_playbook.toml`).
  【F:tests/test_playbook_rules.py†L10-L78】
  【F:tnfr_lfs/data/playbooks/tnfr_playbook.toml†L1-L24】
- `tests/test_robustness.py` labels laps as `"Clasificación"` or `"Vuelta 2"`
  inside example data with no direct production dependencies.
  【F:tests/test_robustness.py†L23-L44】

---

This document acts as a roadmap for the localization phases: the completed
groups are marked as done and the pending items include the requested checklist
`<current text> → <target text>` for incremental tracking.
