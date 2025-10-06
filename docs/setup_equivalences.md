# Guía de correspondencias de setup TNFR

Esta guía cruza los ajustes tradicionales de setup con las métricas
fundamentales del marco TNFR × LFS. Cada tabla se organiza por
subsistema y describe cómo interpretar los indicadores `ΔNFR_lat`, `ν_f`
y `C(t)` dentro del flujo de recomendación y del HUD en vivo. Todas las
lecturas TNFR provienen de la telemetría OutSim/OutGauge de Live for
Speed, por lo que es imprescindible habilitar ambos broadcasters y
configurar `OutSim Opts ff` para transmitir el paquete ampliado de
ruedas (cargas, fuerzas y deflexiones) que requiere la fusión de
telemetría.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】 El `slip_ratio`
global de TNFR × LFS se deriva directamente del promedio de los canales
por rueda del bloque OutSim y solo recurre a un cálculo heurístico de
respaldo cuando la señal llega incompleta.【F:tnfr_lfs/acquisition/fusion.py†L111-L157】【F:tnfr_lfs/acquisition/fusion.py†L242-L285】

## Señales necesarias por métrica

- **ΔNFR / ΔNFR_lat** – integra las cargas verticales, fuerzas y
  deflexiones de rueda reportadas por OutSim junto al régimen del motor,
  entradas de pedales y luces ABS/TC que llegan vía OutGauge para evaluar
  el reparto longitudinal/lateral.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】
- **ν_f (frecuencia natural)** – aprovecha el reparto de carga, los
  `slip_ratio`/`slip_angle`, la velocidad y el `yaw_rate` procedentes de
  OutSim combinados con el estilo de conducción (`throttle`, `gear`) que
  aporta OutGauge para clasificar nodos y bandas objetivo.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L648-L710】 El `slip_ratio` utilizado aquí corresponde al promedio de las cuatro ruedas transmitidas por OutSim, garantizando que las etiquetas de frecuencia respondan a la adherencia real del tren.【F:tnfr_lfs/acquisition/fusion.py†L111-L157】
- **C(t) (coherencia estructural)** – se construye a partir de la
  distribución nodal de ΔNFR, de los coeficientes `mu_eff_*` derivados de
  las aceleraciones OutSim y de la actividad ABS/TC reportada por
  OutGauge.【F:tnfr_lfs/acquisition/fusion.py†L200-L284】【F:tnfr_lfs/core/epi.py†L604-L676】【F:tnfr_lfs/core/coherence.py†L65-L125】
- **Presupuestos Ackermann / Slide Catch** – dependen únicamente de los
  `slip_angle_*` y del `yaw_rate` que OutSim transmite para cada rueda;
  si la fuente no incluye esos campos, TNFR × LFS mostrará `"sin datos"`
  en las lecturas asociadas.
- **Deriva de balance aero (`aero_balance_drift`)** – resume el rake a
  partir del `pitch` y los viajes de suspensión por eje que emite OutSim,
  además del contraste `μ_front - μ_rear`, sin combinar señales sintéticas
  adicionales.
- **Temperaturas/presiones de neumáticos** – sólo aparecen cuando
  OutGauge envía el payload extendido; de lo contrario HUD y exportadores
  señalan `"sin datos"` para evitar asumir valores ficticios.
- **CPHI (Contact Patch Health Index)** – necesita `slip_ratio`,
  `slip_angle`, fuerzas laterales/longitudinales y cargas de rueda del
  paquete extendido de OutSim. Sin esas señales TNFR × LFS etiqueta la
  salud del parche como `"sin datos"`, por lo que el planificador no
  aplicará ajustes de balance de neumáticos.

## Subsistema aerodinámico

| Ajuste | Métrica TNFR | Lectura | Aplicación |
| --- | --- | --- | --- |
| Alerón delantero | `ΔNFR_lat` | Deltas laterales positivos apuntan a sobrecarga en el eje delantero durante la fase media. | Reducir ángulo o carga delantera cuando `C(t)` se mantiene alto (>0.75) para devolver equilibrio sin sacrificar coherencia.
| Alerón trasero | `ΔNFR_lat` | Deltas laterales negativos persistentes en microsectores de tracción indican falta de apoyo en el eje trasero. | Añadir uno o dos clics de ala cuando `ν_f` cae por debajo de la banda óptima y el HUD reporta "Δaero ↘".
| Plano Gurney | `C(t)` | Una coherencia estructural decreciente junto a recomendaciones de sobreviraje sugiere turbulencia inducida. | Sustituir el plano por una variante de menor carga hasta que `C(t)` recupere la banda 0.70–0.85 mientras `ν_f` vuelve a la etiqueta "óptima".

**Ejemplo CLI**

```bash
$ tnfr-lfs suggest stint.jsonl --car-model generic_gt --track spa \
    | jq '.recommendations[] | select(.parameter=="rear_wing_angle")'
{
  "parameter": "rear_wing_angle",
  "metric": "ΔNFR_lat",
  "value": -0.18,
  "rationale": "ΔNFR_lat trasero fuera del umbral en tracción",
  "expected_effect": "Aumentar carga trasera"
}
```

Cruce la salida con la tabla anterior: un `ΔNFR_lat` negativo en tracción
coincide con la fila **Alerón trasero**, lo que justifica incrementar la
carga trasera hasta que el HUD muestre `ν_f` en verde y `C(t)` > 0.75.

## Subsistema de suspensión

| Ajuste | Métrica TNFR | Lectura | Aplicación |
| --- | --- | --- | --- |
| Rigidez de muelles | `ν_f` | Una frecuencia natural fuera de la banda objetivo dispara avisos "muy baja"/"muy alta". | Subir o bajar `spring_rate` en pasos pequeños hasta que `ν_f` retorne a la etiqueta "óptima" y `C(t)` permanezca estable.
| Amortiguación (rebote) | `C(t)` | Picos de decoherencia tras transiciones rápidas indican que el amortiguador no sigue el perfil TNFR. | Ajustar el rebote para suavizar la respuesta hasta recuperar un `C(t)` creciente tras cada transición.
| Barras estabilizadoras | `ΔNFR_lat` | La diferencia lateral entre ejes se amplifica cuando las barras actúan asimétricamente en curvas medias. | Reequilibrar la barra delantera/trasera según el signo de `ΔNFR_lat`; igualar `ν_f` entre ejes previene disonancias.

**Ejemplo HUD/OSD**

```text
Página A  ΔNFR [--^--]  ν_f: muy baja  C(t): 0.62
Recomendación: "Rigidez trasera +1 → recuperar banda ν_f"
```

Cuando el HUD muestre una etiqueta `ν_f` "muy baja" y `C(t)` caiga por
debajo de 0.65, consulte la fila **Rigidez de muelles**: incremente la
rigidez hasta que el indicador cambie a "óptima" y la barra de coherencia
(`C(t) ███░`) crezca en paralelo.

## Subsistema de neumáticos

| Ajuste | Métrica TNFR | Lectura | Aplicación |
| --- | --- | --- | --- |
| Presión en caliente | `ΔNFR_lat` | Deltas laterales irregulares entre vueltas reflejan diferencias de presión. | Ajustar presiones para suavizar `ΔNFR_lat` y estabilizar `ν_f` alrededor de la frecuencia nominal del perfil.
| Caída (camber) | `C(t)` | Una coherencia decreciente en las fases de apoyo indica que el parche de contacto no sigue la carga. | Aumentar la caída negativa hasta que `C(t)` se alinee con la referencia de la tabla de perfil (>0.78).
| Toe | `ν_f` | Saltos rápidos del indicador `ν_f~` tras cambios de dirección se asocian a toe excesivo. | Reducir toe para eliminar artefactos en la señal `ν_f~` y devolver la lectura a la banda "óptima".

**Ejemplo CLI**

```bash
$ tnfr-lfs analyze stint.jsonl --export json \
    | jq '.phase_messages[] | select(.metric=="ΔNFR_lat" and .phase=="apex")'
{
  "phase": "apex",
  "metric": "ΔNFR_lat",
  "status": "high",
  "message": "Presiones delanteras desbalanceadas"
}
```

La fase «apex» con `ΔNFR_lat` alto remite a la fila **Presión en caliente**,
por lo que conviene igualar las presiones hasta que la métrica vuelva a la
zona nominal y la alerta desaparezca en la siguiente ejecución de
`tnfr-lfs analyze`.

## Subsistema de frenos y balance

| Ajuste | Métrica TNFR | Lectura | Aplicación |
| --- | --- | --- | --- |
| Bias de frenada | `ΔNFR_lat` | Spikes positivos en frenadas fuertes indican sobrecarga delantera. | Trasladar el bias hacia el eje trasero hasta que el HUD o el CLI reduzcan los picos reportados.
| Potencia ABS/TC | `C(t)` | Un índice de coherencia oscilante tras entradas de ABS revela intervención agresiva. | Suavizar el ABS/TC para mantener `C(t)` estable mientras `ν_f` se mantiene en banda.
| Balanceo longitudinal | `ν_f` | Un badge `ν_f` "muy alta" en recta sugiere rigidez excesiva en frenada. | Ablandar topes o bumpstops delanteros para alinear la frecuencia con el objetivo.

**Ejemplo combinado**

```bash
$ tnfr-lfs osd --host 127.0.0.1 --outsim-port 4123 \
    --outgauge-port 3000 --insim-port 29999
# Durante la frenada, página A muestra:
ΔNFR_lat: +0.24  ν_f: óptima  C(t): 0.81  → "Bias delantero alto"
```

El HUD enlaza automáticamente con la fila **Bias de frenada**, indicando
que hay que desplazar el balance hacia atrás hasta que `ΔNFR_lat`
permanezca por debajo del umbral contextual y la recomendación desaparezca.

---

Para ampliar cómo se calculan las métricas `ΔNFR_lat`, `ν_f` y `C(t)`,
revise los módulos `tnfr_lfs.core` descritos en la [referencia de
API](api_reference.md).
