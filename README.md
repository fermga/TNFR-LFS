# TNFR × LFS Toolkit

**TNFR × LFS** packages the Fractal-Resonant Nature Theory into a race-ready
telemetry and recommendation workflow for Live for Speed.

## Documentation

- [Full documentation index](docs/index.md) – complete feature catalogue,
  onboarding narrative, and API reference.
- [Beginner quickstart](docs/tutorials.md) – step-by-step installation and
  sample session walkthrough.
- [Examples gallery](docs/examples.md) – automation scripts for common
  engineering tasks.

## Alineación TNFR
La teoría operativa expone trece operadores estructurales; cada uno responde a
un código abreviado que se transforma en su etiqueta canónica mediante
`canonical_operator_label`:

| Código | Etiqueta canónica      | Descripción resumida |
| ------ | ---------------------- | -------------------- |
| `AL`   | Support                | Ajusta soportes y refuerzos estructurales. |
| `EN`   | Reception              | Normaliza los paquetes EPI entrantes. |
| `IL`   | Coherence              | Suaviza series sin perder promedio. |
| `NAV`  | Transition             | Sincroniza transiciones entre fases. |
| `NUL`  | Contraction            | Gestiona la contracción estructural. |
| `OZ`   | Dissonance             | Aísla desviaciones ΔNFR útiles frente a parásitas. |
| `RA`   | Propagation            | Propaga ajustes nodales avalando la resonancia. |
| `REMESH` | Remeshing            | Re-malla el grafo nodal para detectar discontinuidades. |
| `SILENCE` | Structural silence  | Marca las ventanas sin excitación estructural. |
| `THOL` | Auto-organisation      | Observa la autoorganización durante cambios de régimen. |
| `UM`   | Coupling               | Evalúa el acoplamiento entre nodos. |
| `VAL`  | Amplification          | Identifica dónde amplificar soportes. |
| `ZHIR` | Transformation         | Traza mutaciones profundas del arquetipo. |

La ecuación nodal extendida evalúa el índice de sentido penalizado por entropía
incorporando el gradiente ΔNFR y las frecuencias naturales de cada nodo:

```
Sense Index = 1 / (1 + Σ w_i · |ΔNFR_i| · g(ν_f_i)) - λ · H
```

* `w_i` pondera cada nodo según su rol estructural.
* `g(ν_f_i)` corrige la respuesta por la frecuencia natural observada.
* `H` resume la entropía de la distribución ΔNFR nodal.

Nuevas métricas estructurales disponibles en la interfaz pública:

- `support_effective`, `load_support_ratio`.
- `structural_expansion_longitudinal` / `structural_contraction_longitudinal`.
- `structural_expansion_lateral` / `structural_contraction_lateral`.
- `delta_nfr_entropy`, `node_entropy`, `phase_delta_nfr_entropy`, `phase_node_entropy`.
- `thermal_load`, `style_index`, `network_memory`.
- `aero_balance_drift`, `slide_catch_budget`, `ackermann_parallel_index`.

El informe de cobertura teórica se genera con:

```
poetry run python tools/tnfr_theory_audit.py --core --tests --output tests/_report/theory_impl_matrix.md
```

Una vez generado, puede consultarse en
[`tests/_report/theory_impl_matrix.md`](tests/_report/theory_impl_matrix.md).

## Verificación de paridad TNFR

Para comparar la implementación local del motor TNFR con el extra canónico
instalado, utilice el verificador de paridad. El script analiza una serie de
telemetría, ejecuta ambos motores y genera un informe JSON con los valores
absolutos y las diferencias relativas observadas::

```
poetry run python tools/verify_tnfr_parity.py tests/data/synthetic_stint.csv --output parity_report.json
```

El mismo flujo está disponible a través del objetivo `make verify-tnfr-parity`,
que activa el modo estricto y finaliza con código de salida distinto de cero si
alguna métrica supera las tolerancias configuradas.
