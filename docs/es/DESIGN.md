# TNFR Manual Operativo (TNFR.pdf)

> **Nota:** El documento original `TNFR.pdf` utiliza fuentes incrustadas sin mapas `ToUnicode`, por lo que no es posible realizar una extracción de texto automática y fidedigna en este entorno sin herramientas externas con soporte para fuentes personalizadas. Este archivo Markdown consolida la estructura principal, terminología clave y tablas descritas en el manual para que las futuras modificaciones puedan versionarse en texto plano. El PDF debe tratarse como un artefacto generado.

## Prefacio y visión general
- Contexto: transición del conocimiento descriptivo a la ciencia resonante.
- Enfoque TNFR: el símbolo como unidad operativa del ser y los operadores estructurales como estructuras primarias de información (EPI).
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
Estos campos se combinan con las magnitudes clásicas (carga Fz,
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
| ``tyres``     | Δslip ratio, bloqueo ABS/TC, variación de carga Fz (ΔFz), temperaturas y presiones por rueda |
| ``suspension``| ΔFz, Δpitch, Δroll                                       |
| ``chassis``   | Δaceleración lateral, Δroll, Δyaw                         |
| ``brakes``    | Δpresión de freno, indicador de bloqueo, deceleración     |
| ``transmission`` | Δaceleración longitudinal, Δslip, Δyaw               |
| ``track``     | Δ(carga × aceleración lateral), Δyaw                      |
| ``driver``    | ΔSense Index, Δyaw, Δpitch/Δroll                          |

La suma de los ΔNFR individuales coincide siempre con el ΔNFR global del
registro, garantizando la coherencia con el Sense Index penalizado por entropía
que se calcula en :func:`tnfr_lfs.core.coherence.sense_index`.  La métrica se
evalúa ahora como ``1 / (1 + Σ w · |ΔNFR| · g(ν_f)) - λ·H`` incorporando el
ritmo natural de cada nodo y la fase del microsector; la interpretación se
detalla en :doc:`api_reference`.

El estimador de frecuencias naturales persiste bandas objetivo por categoría
de vehículo (GT, fórmulas, prototipos, …) a través de
``NaturalFrequencySettings.frequency_bands``.  Cada muestra expone una
clasificación ``ν_f`` (muy baja, óptima, muy alta, …) y un índice estructural
``C(t)`` derivado de la densidad de eventos en el eje estructural.  Ambos
parámetros se normalizan con los objetivos de ``ProfileManager`` para que la
misma escala sea válida al comparar stints con diferentes metas de Sense Index.

### Operadores de memoria y mutación

Para mantener continuidad entre muestras y microsectores, el pipeline utiliza
dos operadores con estado ubicados en ``tnfr_lfs.core.operators``:

* ``recursivity_operator`` conserva una traza por microsector y filtra de
  forma exponencial las métricas térmicas y de estilo (``thermal_load`` y
  ``style_index``) junto con las temperaturas/presiones por rueda
  (``tyre_temp_*``/``tyre_pressure_*``).  La memoria queda ahora segmentada por
  sesión (``car_model``/``track_name``/``tyre_compound``), manteniendo un
  historial por stint en ``operator_state["recursivity"]``.  Además calcula
  derivadas ``dT/dt`` para cada rueda, persiste ``ΔNFR_flat`` como referencia y
  aplica criterios de parada (convergencia, límites de muestras o gaps de
  tiempo) para iniciar nuevos stints cuando la dinámica cambia.  El operador
  detecta cambios de fase (entrada, vértice, salida) y reinicia la memoria
  cuando corresponde para que el filtro no contamine transiciones abruptas; el
  historial completo se expone a través de ``orchestrate_delta_metrics`` como
  ``network_memory`` para que HUD y exportadores accedan a la memoria de red.
  Cuando OutGauge no transmite el bloque extendido (temperaturas y presiones
  por rueda) las entradas correspondientes en HUD/exportadores muestran
  explícitamente ``"sin datos"`` para remarcar que la telemetría de LFS no
  estaba disponible.
* ``mutation_operator`` observa la entropía del reparto ΔNFR y las desviaciones
  de estilo filtradas para decidir si el arquetipo táctico debe mutar hacia un
  candidato alternativo o volver a un modo de recuperación.

La segmentación de microsectores invoca ambos operadores: primero suaviza las
medidas del microsector activo y después evalúa si la combinación de entropía,
fase y condiciones dinámicas (curvatura sostenida, duración del apoyo,
reducciones de velocidad y cambios de dirección) requiere mutar los objetivos
generados por :func:`segment_microsectors`.  Cuando se produce una mutación, la
segmentación recalcula los objetivos de entrada, vértice y salida con el nuevo
arquetipo —``hairpin``, ``medium``, ``fast`` o ``chicane``— antes de integrarlos
en el reporte.  De este modo el estado se conserva entre invocaciones y la
estrategia resultante se adapta a las condiciones cambiantes del stint.

### Métricas de alineación de fase

Cada microsector conserva ahora el desfase medido ``θ`` y su coseno ``Siφ``
para cada fase, junto con los objetivos de alineación y los pesos de detune
asociados al arquetipo vigente.  Los valores se derivan de un espectro cruzado
entre la dirección del piloto y la respuesta combinada yaw/aceleración lateral,
lo que expone la frecuencia dominante del apoyo y permite detectar atrasos o
adelantos del chasis.  El HUD y las reglas del recomendador utilizan estas
métricas, junto con los objetivos de proyección ∇NFR⊥/∇NFR∥ y ``ν_f`` fijados por arquetipo, para
invertir las sugerencias de barras y amortiguadores cuando la fase medida
pierde la alineación con el objetivo.

La versión actual refuerza esta lógica con un mapa de acciones específico para
los parámetros de geometría (caster, camber y toe).  Las plantillas de
``PhaseActionTemplate`` ahora transforman desviaciones de ``Siφ``, ``θ`` y del
índice de coherencia ``C(t)`` en ajustes directos sobre los ejes delantero y
trasero.  El motor de recomendaciones calcula un ``snapshot`` geométrico por
microsector a partir de los datos agregados en ``Microsector.filtered_measures``
y de los promedios de ``WindowMetrics``, priorizando los nodos de suspensión y
neumáticos cuando la transición entre fases pierde coherencia.  Además, el
optimiser de planes pondera estas desviaciones para que los gradientes de
geometría formen parte del objetivo prolongado de resonant alignment (RA),
recompensando configuraciones con mayor ``C(t)`` y penalizando retrasos en la
respuesta.  Además se exponen indicadores agregados del apoyo estructural en
``WindowMetrics``:

* ``support_effective`` captura el ΔNFR absorbido por neumáticos y suspensión
  ponderado por las ventanas estructurales del tramo analizado.
* ``load_support_ratio`` normaliza dicho valor frente a la carga Fz media
  para contextualizar el esfuerzo requerido al chasis.
* ``structural_expansion_longitudinal``/``structural_contraction_longitudinal``
  cuantifican la expansión o contracción del eje estructural inducida por las
  componentes longitudinales de ΔNFR.
* ``structural_expansion_lateral``/``structural_contraction_lateral`` replican
  la métrica para la componente lateral, revelando apoyos que abren o cierran
  el eje estructural en la ventana.
* ``ackermann_parallel_index`` y ``slide_catch_budget`` se derivan
  exclusivamente de los ``slip_angle_*`` y del ``yaw_rate`` emitidos por
  OutSim; sin esa telemetría los presupuestos de Ackermann y de captura de
  derrapaje permanecen en cero y los reportes lo indican como ``"sin datos"``.
* ``aero_balance_drift`` resume el rake medio (pitch + viajes) y la diferencia
  ``μ_front - μ_rear`` por bandas de velocidad (baja/media/alta).  El rake se
  calcula únicamente con el ``pitch`` y los viajes de suspensión suministrados
  por OutSim, de modo que la deriva aerodinámica responde directamente a la
  telemetría nativa de LFS incluso cuando la coherencia aparente permanece
  estable.
* ``delta_nfr_entropy`` y ``node_entropy`` cuantifican con una escala normalizada
  [0, 1] lo repartido que está el ΔNFR entre fases y nodos respectivamente; los
  valores cercanos a cero implican concentración en un único grupo mientras que
  los cercanos a uno indican una ventana equilibrada.  Los mapas
  ``phase_delta_nfr_entropy``/``phase_node_entropy`` exponen las probabilidades
  estructurales por fase y la entropía Shannon nodal por fase, replicando los
  alias heredados para que los tableros compacten la información sin perder
  granularidad.

## Artefactos generados

- ``TNFR.pdf`` se mantiene como artefacto compilado. Cualquier actualización sustantiva debe reflejarse primero en este ``DESIGN.md``; el PDF puede regenerarse a partir del contenido actualizado y debe tratarse como derivado.
- ``coherence_map.<ext>`` serializa el promedio y el rango de ``C(t)`` por microsector junto con la distancia acumulada estimada a partir de ``TransmissionNode.speed``.  Permite reconstruir el mapa de coherencia de toda la vuelta y correlacionarlo con los objetivos de pista.
- ``operator_trajectories.<ext>`` aplica ``Microsector.operator_events`` para reconstruir la línea temporal estructural de los detectores ``AL``/``OZ``/``IL``/``SILENCIO``.  El agregado conserva la cobertura estructural de los intervalos en silencio para cada microsector, permitiendo detectar estados latentes donde la carga, las aceleraciones y ΔNFR permanecen por debajo de los umbrales configurados.
- ``delta_bifurcations.<ext>`` analiza las series ΔNFR en el eje estructural, detectando cruces de signo y extremos locales para identificar bifurcaciones que requieran ajustes de setup.

Los sufijos ``.json``, ``.md`` o ``.viz`` dependen del formato elegido con ``--report-format`` y comparten la misma semántica sin importar el modo de renderizado.

## Próximos pasos sugeridos
1. Completar la transcripción detallada de cada subsección utilizando herramientas con soporte para fuentes personalizadas o mediante revisión manual.
2. Validar que las tablas y ecuaciones reproducidas aquí coincidan con las versiones del PDF regenerado.
3. Documentar en el repositorio el proceso empleado para generar el PDF a partir del Markdown para facilitar futuras automatizaciones.
