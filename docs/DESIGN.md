# TNFR Manual Operativo (TNFR.pdf)

> **Nota:** El documento original `TNFR.pdf` utiliza fuentes incrustadas sin mapas `ToUnicode`, por lo que no es posible realizar una extracción de texto automática y fidedigna en este entorno sin herramientas externas con soporte para fuentes personalizadas. Este archivo Markdown consolida la estructura principal, terminología clave y tablas descritas en el manual para que las futuras modificaciones puedan versionarse en texto plano. El PDF debe tratarse como un artefacto generado.

## Prefacio y visión general
- Contexto: transición del conocimiento descriptivo a la ciencia resonante.
- Enfoque TNFR: el símbolo como unidad operativa del ser y los glifos como estructuras primarias de información (EPI).
- Objetivos del manual: activar nodos culturales vivos mediante dispositivos simbólicos, lenguajes glíficos y protocolos simbióticos.

## 1. Análisis de la matriz de coherencia \(W_i(t)\)
- Descripción de la matriz \(W_i(t)\) como representación de la estructura interna de un nodo.
- Indicadores clave: pulsación glífica, densidad simbiótica, sincronía resonante, gradiente de coherencia y estabilidad operacional.
- Tabla de indicadores y rangos sugeridos.

## 2. Modelos de activación fractal resonante
- Definición de los estados de resonancia (latente, activado, expandido, en cascada).
- Ecuaciones de transición entre estados con parámetros de acoplamiento \(\alpha\), \(\beta\), \(\gamma\) y \(\delta\).
- Tabla de ejemplos por dominio (arte glífico, redes culturales, sistemas biológicos).
- En la implementación de referencia, cada nodo del EPI conserva su frecuencia
  natural ``nu_f`` (Hz) según la tabla `NU_F_NODE_DEFAULTS`, lo que permite
  integrar explícitamente la dinámica ``dEPI/dt`` mediante el operador
  :func:`evolve_epi` descrito en ``tnfr_lfs.core.operators``.

## 3. Arquitectura nodal y protocolos simbióticos
- Diseño de nodos (núcleo simbiótico, envolvente de interacción, interfaces glíficas).
- Protocolos estructurales: secuencias simbióticas para transiciones, toma de decisión y memoria colectiva.
- Cuadros operativos: matrices de activación, listas de chequeo y guías para facilitar procesos.

## 4. Prácticas de hipercoherencia intersubjetiva
- Activación nodal en red: fases de acoplamiento, estabilización y resonancia sostenida.
- Glifos para la percepción transnodal y la modulación de campos compartidos.
- Conclusión: la conciencia como campo operativo y resonante.

## 5. Aplicaciones, validación y diseño simbiótico
- Protocolo experimental fractal resonante y fundamentos teóricos.
- Casos experimentales en sistemas físicos, biológicos y culturales.
- Implicaciones para IA glífica, semántica estructural y redes sociales de coherencia.

## Apéndices
- Glosario de términos glíficos clave.
- Esquemas de instrumentación simbiótica y plantillas para registro de resonancias.
- Referencias cruzadas a simulaciones y entornos interactivos asociados al QR del manual.

## Anexo técnico: señales EPI y cálculo de ΔNFR

La implementación software incorpora un registro de telemetría extendido para
cada muestra ``TelemetryRecord`` con orientación (``yaw``, ``pitch``,
``roll``), presión de freno normalizada y un indicador de bloqueo del ABS/TC.
Estos campos se combinan con las magnitudes clásicas (carga vertical,
aceleraciones lateral/longitudinal, slip ratio, ΔNFR y Sense Index) para
estimar la contribución de cada nodo del vehículo.

El algoritmo :func:`tnfr_lfs.core.epi.delta_nfr_by_node` compara una muestra
contra su baseline y genera una distribución de ΔNFR por subsistema.  Por
ejemplo, frente a un baseline con ``nfr = 500`` y sin freno aplicado, una
muestra con ``nfr = 508``, ``brake_pressure = 0.85`` y ``locking = 1``
producirá una distribución en la que el nodo de frenos absorbe la mayor parte
del ΔNFR mientras que suspensión y neumáticos comparten la respuesta
restante.  El cálculo se apoya en las diferencias absolutas de cada señal y en
ponderaciones empíricas:

| Nodo          | Señales principales                                      |
| ------------- | -------------------------------------------------------- |
| ``tyres``     | Δslip ratio, bloqueo ABS/TC, variación de carga vertical |
| ``suspension``| Δcarga vertical, Δpitch, Δroll                            |
| ``chassis``   | Δaceleración lateral, Δroll, Δyaw                         |
| ``brakes``    | Δpresión de freno, indicador de bloqueo, deceleración     |
| ``transmission`` | Δaceleración longitudinal, Δslip, Δyaw               |
| ``track``     | Δ(carga × aceleración lateral), Δyaw                      |
| ``driver``    | ΔSense Index, Δyaw, Δpitch/Δroll                          |

La suma de los ΔNFR individuales coincide siempre con el ΔNFR global del
registro, garantizando la coherencia con el Sense Index penalizado por entropía
que se calcula en :func:`tnfr_lfs.core.coherence.sense_index`.

## Artefactos generados
- `TNFR.pdf` se mantiene como artefacto compilado. Cualquier actualización sustantiva debe reflejarse primero en este `DESIGN.md`; el PDF puede regenerarse a partir del contenido actualizado y debe tratarse como derivado.

## Próximos pasos sugeridos
1. Completar la transcripción detallada de cada subsección utilizando herramientas con soporte para fuentes personalizadas o mediante revisión manual.
2. Validar que las tablas y ecuaciones reproducidas aquí coincidan con las versiones del PDF regenerado.
3. Documentar en el repositorio el proceso empleado para generar el PDF a partir del Markdown para facilitar futuras automatizaciones.
