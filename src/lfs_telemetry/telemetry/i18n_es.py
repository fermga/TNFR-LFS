"""Spanish i18n tables for channel labels, descriptions and interpretations.

Extracted from :mod:`lfs_telemetry.telemetry.channels` to keep that
module focused on the channel registry and rendering logic.
"""
from __future__ import annotations

__all__ = [
    "_DESCRIPTION_ES_FALLBACK",
    "_FOCUS_BY_COLUMN_ES",
    "_GROUP_ES_FALLBACK",
    "_INTERP_BY_COLUMN_ES",
    "_INTERP_BY_GROUP_ES",
    "_INTERP_BY_PATTERN_ES",
    "_INTERP_BY_SUFFIX_ES",
    "_LABEL_ES_FALLBACK",
]


_GROUP_ES_FALLBACK: dict[str, str] = {
    "Driver": "Piloto",
    "Vehicle": "Vehículo",
    "Engine": "Motor",
    "Chassis": "Chasis",
    "Suspension": "Suspensión",
    "Tyre": "Neumáticos",
    "Derived": "Derivados",
    "Track": "Circuito",
    "Aids": "Ayudas",
    "Context": "Contexto",
}
_LABEL_ES_FALLBACK: dict[str, str] = {
    # --- Vehicle / Engine ---
    "Speed": "Velocidad",
    "RPM": "RPM",
    "Gear": "Marcha",
    "Fuel": "Combustible",
    "Turbo": "Turbo",
    # --- Driver inputs ---
    "Throttle": "Acelerador",
    "Brake": "Freno",
    "Clutch": "Embrague",
    "Steer (raw)": "Dirección (bruta)",
    "Handbrake": "Freno de mano",
    "Steer Torque": "Par de dirección",
    # --- Chassis ---
    "Roll rate": "Tasa de balanceo",
    "Pitch rate": "Tasa de cabeceo",
    "Yaw rate": "Tasa de guiñada",
    "Heading": "Rumbo",
    "Pitch": "Cabeceo",
    "Roll": "Balanceo",
    "Long. accel": "Acel. long.",
    "Lat. accel": "Acel. lat.",
    "Vert. accel": "Acel. vertical",
    "Vel. X": "Vel. X",
    "Vel. Y": "Vel. Y",
    "Vel. Z": "Vel. Z",
    "Pos. X": "Pos. X",
    "Pos. Y": "Pos. Y",
    "Pos. Z": "Pos. Z",
    # --- Derived ---
    "Yaw rate (th)": "Tasa de guiñada (teór.)",
    "Understeer idx": "Índice de subviraje",
    "Long. transfer": "Transferencia long.",
    "Lat. transfer": "Transferencia lat.",
    "Long. transfer (th)": "Transferencia long. (teór.)",
    "Lat. transfer (th)": "Transferencia lat. (teór.)",
    "Total load": "Carga total",
    "Front load frac": "Carga delantera (frac.)",
    "Left load frac": "Carga izquierda (frac.)",
    "Diag FL/RR frac": "Diagonal FL/RR (frac.)",
    "Brake bias front": "Reparto de frenada delante",
    "FFB load": "Carga FFB",
    "Steer rate": "Velocidad de dirección",
    "Steer reversals": "Inversiones de dirección",
    "Long. accel (road)": "Acel. long. (pista)",
    "Yaw misalignment": "Desalineación de guiñada",
    # --- Track ---
    "Track node": "Nodo de pista",
    "Track s": "Distancia de pista",
    "Track elevation": "Elevación de pista",
    "Track heading": "Rumbo de pista",
    "Track curvature": "Curvatura de pista",
    "Track radius": "Radio de pista",
    "Track slope": "Pendiente de pista",
    "Track width": "Ancho de pista",
    "Lateral offset": "Offset lateral",
    "Segment id": "ID de segmento",
    # --- Aids ---
    "TC active": "TC activo",
    "ABS active": "ABS activo",
    "Handbrake on": "Freno de mano puesto",
    "Pit limiter": "Limitador de pit",
    "Oil warning": "Aviso de aceite",
    "Battery warn": "Aviso de batería",
    "Indicator L": "Intermitente I",
    "Indicator R": "Intermitente D",
    "Full beam": "Luces largas",
    "Shift light": "Luz de cambio",
    # --- Context ---
    "Wind": "Viento",
    # --- Suspension / Tyre (per wheel base labels) ---
    "Susp. travel": "Recorrido susp.",
    "Vert. load": "Carga vertical",
    "Slip ratio": "Slip ratio",
    "tan(α)": "tan(α)",
    "Lat. force": "Fuerza lateral",
    "Long. force": "Fuerza long.",
    "Wheel ω": "ω rueda",
    "Camber rel.": "Camber rel.",
    "Tyre air temp": "Temp. aire neum.",
    "Slip frac": "Frac. deslizante",
    "Touching": "En contacto",
    "Wheel steer": "Dirección rueda",
    "Friction use": "Uso de fricción",
    "Tyre work": "Trabajo del neum.",
    "Susp. speed": "Velocidad susp.",
}
_DESCRIPTION_ES_FALLBACK: dict[str, str] = {
    "Wheel-derived speed": "Velocidad derivada de ruedas",
    "Engine speed": "Régimen del motor",
    "Throttle pedal (0..1)": "Pedal de acelerador (0..1)",
    "Brake pedal (0..1)": "Pedal de freno (0..1)",
    "Steering wheel angle": "Ángulo del volante",
    "FFB steering torque": "Par de dirección FFB",
    "+forward, -brake": "+adelante, -frenada",
    "+right": "+derecha",
    "Fuel level (0..1)": "Nivel de combustible (0..1)",
}
_INTERP_BY_COLUMN_ES: dict[str, str] = {
    # -------- Vehículo / Motor (OutGauge + OutSim) --------
    "speed_ms": (
        "Velocidad longitudinal sobre el suelo (m/s) desde OutGauge."
        " Superpón dos vueltas en la misma curva: la que conserva más"
        " velocidad en el ápice suele ser la mejor trazada. Caídas"
        " bruscas indican frenada, bloqueo o pérdida de tracción; las"
        " mesetas planas largas marcan tramos limitados por aero/drag"
        " más que por agarre."
    ),
    "rpm": (
        "Régimen del motor (rev/min) desde OutGauge. Trazas planas"
        " contra el corte de revoluciones = subiste tarde de marcha"
        " (estás golpeando el limitador); caídas grandes de rpm en"
        " el cambio = relación demasiado larga o cambio fuera de la"
        " zona de par. En motores atmosféricos el punto dulce es"
        " mantenerse en el tercio superior de la curva de par."
    ),
    "gear_lfs": (
        "Marcha engranada actual (R, N, 1..N) desde OutGauge."
        " Úsala para auditar puntos de cambio y reducciones: bajar"
        " una marcha antes de soltar el freno carga el eje motriz"
        " a mitad de curva y puede desestabilizar el coche."
        " Compárala con las trazas de gas y freno."
    ),
    "fuel": (
        "Nivel de combustible como fracción de la capacidad del"
        " depósito (0..1) desde OutGauge. La pendiente es el consumo"
        " instantáneo; el módulo FuelTracker la convierte en vueltas"
        " restantes y ventanas de parada. Una subida repentina suele"
        " ser un repostaje."
    ),
    "eng_temp_c": (
        "Temperatura del refrigerante (°C) desde OutGauge. Una"
        " deriva sostenida por encima del rango normal indica"
        " entradas de aire bloqueadas, daños de carrocería o exceso"
        " de revoluciones. La mayoría de motores LFS toleran hasta"
        " ~110 °C antes de perder potencia; cifras mayores y"
        " sostenidas acortan la vida del motor."
    ),
    "oil_temp_c": (
        "Temperatura del aceite (°C). Reacciona más lenta que el"
        " refrigerante y es mejor indicador del estrés térmico de"
        " largo plazo. Rango sano en carrera: aprox. 90–125 °C según"
        " el coche."
    ),
    "oil_pressure_bar": (
        "Presión de aceite del motor (bar). Caídas bajo g lateral"
        " alto indican falta de pickup (el aceite se va al lado);"
        " presión baja persistente sugiere aceite muy caliente o"
        " desgaste."
    ),
    "turbo_bar": (
        "Presión de soplado del turbo (bar) sobre la atmosférica."
        " Traza plana en pico = wastegate regulando bien. Subidas"
        " lentas a baja rpm son lag de turbo; caídas bruscas con"
        " gas medio son blow-off o flameo de wastegate."
    ),

    # -------- Entradas del piloto (OutGauge) --------
    "throttle": (
        "Posición del pedal de gas (0..1) desde OutGauge. La forma"
        " ideal a la salida es una rampa monótona suave de 0 a 1;"
        " entradas a escalones que rebotan contra 1.0 indican gas"
        " agresivo que el eje trasero puede no aceptar. Mesetas"
        " largas en 1.0 marcan tramos a fondo; puedes medir 'tiempo"
        " a fondo' por vuelta para comparar estilos."
    ),
    "brake": (
        "Posición del pedal de freno (0..1) desde OutGauge. Una"
        " buena traza sube rápido a un pico alto (justo por debajo"
        " del bloqueo) y luego baja suave al girar (trail-braking)."
        " Liberaciones a escalones revelan modulación del pedal o"
        " ciclos de ABS; mesetas muy planas cerca de 1.0 significan"
        " que estás metiendo el freno en ABS o bloqueo."
    ),
    "clutch": (
        "Embrague (0..1). Solo relevante en salidas paradas y"
        " cambios. Una traza larga en parcial al lanzar es"
        " patinamiento de embrague (usado a propósito para gestionar"
        " el patinaje)."
    ),
    "input_steer": (
        "Ángulo del volante (rad), positivo = izquierda. Trazas"
        " suaves y en un solo arco indican entrada bien juzgada; un"
        " patrón en 'sierra' de pequeñas inversiones muestra"
        " corrección excesiva o un chasis nervioso. La amplitud pico"
        " ligada al radio de curva apunta al balance de"
        " subviraje/sobreviraje."
    ),
    "steer_torque_nm": (
        "Par en la cremallera de dirección (N·m): la reacción física"
        " del FFB que sentirías en el volante. Cambios bruscos de"
        " signo delatan snap trasero; mesetas planas largas en mitad"
        " de curva indican que las gomas delanteras están saturadas"
        " (subviraje)."
    ),
    "input_handbrake": (
        "Freno de mano (0..1). Solo se usa para giros de freno de"
        " mano, drift o rallycross. Un pico en carrera normal es"
        " casi siempre un error."
    ),

    # -------- Chasis (OutSim — IMU en ejes del vehículo) --------
    "accel_x": (
        "Aceleración longitudinal en el marco del vehículo (m/s²)."
        " Positiva = acelerando, negativa = frenando. Una buena"
        " traza de frenada muestra un escalón negativo nítido hasta"
        " el límite de fricción y se mantiene plana (deceleración"
        " constante) antes de bajar a 0 en el punto de giro."
        " Referencias: ~1 g de frenada = 9,81 m/s²; los slicks de"
        " carrera tiran de 1,5–2 g."
    ),
    "accel_y": (
        "Aceleración lateral (m/s², +izquierda). Su pico absoluto en"
        " una curva equivale al agarre lateral realmente usado."
        " Cruzada con accel_x (diagrama g–g) revela cómo llenas la"
        " elipse de fricción — cuadrantes vacíos delatan zonas donde"
        " dejas agarre sin usar."
    ),
    "accel_z": (
        "Aceleración vertical (m/s²). Los picos son baches, golpes"
        " de piano o aterrizajes tras una cresta. Valores sostenidos"
        " por encima de 1 g de compresión indican mucha carga aero"
        " o una compresión pronunciada."
    ),
    "ang_vel_z": (
        "Tasa de guiñada (rad/s) sobre el eje vertical: con qué"
        " rapidez gira el coche visto en planta. La expectativa pura"
        " de geometría es v / R (velocidad ÷ radio de curva); ver"
        " understeer_index para la desviación."
    ),
    "ang_vel_x": (
        "Tasa de balanceo (rad/s) sobre el eje longitudinal. Picos"
        " grandes en cambios rápidos de apoyo (chicanes) indican"
        " barras estabilizadoras o muelles blandos y transferencia"
        " lenta."
    ),
    "ang_vel_y": (
        "Tasa de cabeceo (rad/s) sobre el eje lateral. Pulsos"
        " positivos en frenada (clavada del morro), negativos en"
        " aceleración (cuclillas). La magnitud escala con la fuerza"
        " de freno/gas dividida por la rigidez delantera/trasera."
    ),
    "pitch": (
        "Ángulo de cabeceo del chasis (rad). Positivo en frenada"
        " (morro abajo), negativo en aceleración. El exceso de"
        " cabeceo perjudica la plataforma aerodinámica y cambia el"
        " balance mecánico."
    ),
    "roll": (
        "Ángulo de balanceo del chasis (rad). Mucho roll en curvas"
        " estables apunta a muelles blandos o ARB insuficiente;"
        " coches muy rígidos mostrarán trazas casi planas pero"
        " pueden saltar en los pianos."
    ),
    "heading": (
        "Guiñada (heading) del chasis en el marco del mundo (rad)."
        " Útil sobre todo al superponer sobre el mapa del circuito;"
        " poco informativo como traza temporal por sí solo."
    ),

    # -------- Vuelta / distancia en pista (InSim + derivados) --------
    "current_lap_dist_m": (
        "Distancia recorrida desde la línea de salida/meta (m). Se"
        " resetea a 0 cada vuelta. Úsala como eje X para alinear"
        " curvas entre vueltas y comparar directamente."
    ),
    "indexed_distance_m": (
        "Distancia a lo largo del índice de la trazada (m); no se"
        " resetea en la línea. Preferida para datasets de temporada"
        " que abarcan varios stints."
    ),

    # -------- Dinámica de chasis derivada --------
    "yaw_rate_rads": (
        "Tasa de guiñada medida (rad/s). Comparada con la teórica"
        " de coche neutro indica el balance: ver understeer_index."
    ),
    "yaw_rate_theoretical_rads": (
        "Tasa de guiñada que tendría un coche perfectamente neutro"
        " a esta velocidad y ángulo de dirección (v · tan(δ) /"
        " batalla). La diferencia con la medida es lo que están"
        " entregando realmente los ejes delantero/trasero."
    ),
    "beta_deg": (
        "Ángulo de deriva del vehículo (°) = atan(v_lateral /"
        " v_long). En un coche de calle, un máximo sano es 3–6°;"
        " un valor sostenido por encima de 8° significa que el"
        " coche está derrapando notablemente y probablemente"
        " perdiendo tiempo. Útil para detectar sobreviraje o"
        " trail-braking agresivo."
    ),
    "understeer_index": (
        "(yaw_teórica − yaw_medida) / yaw_teórica."
        " Positivo = subviraje (el delantero se va abierto),"
        " negativo = sobreviraje (el trasero rota más de lo que"
        " pide el volante). Fíjate dónde aparecen los picos:"
        " entrada, mitad o salida diagnostican problemas distintos"
        " (geometría delantera, transferencia, diferencial)."
    ),
    "transfer_long_n_real": (
        "Transferencia longitudinal de carga (N). El valor de libro"
        " es m · a_x · h_cg / batalla: positiva en aceleración (la"
        " carga va atrás), negativa en frenada (la carga va"
        " adelante). Compárala con los pesos estáticos por eje para"
        " ver la oscilación relativa."
    ),
    "transfer_lat_n_real": (
        "Transferencia lateral de carga (N). Guiada por m · a_y ·"
        " h_cg / vía. Su reparto entre eje delantero y trasero"
        " depende de la rigidez al balanceo; sesgar la rigidez al"
        " balanceo hacia delante añade transferencia delantera (más"
        " subviraje) y viceversa."
    ),
    "load_total_n": (
        "Suma de las cuatro cargas verticales (N). Debe rondar el"
        " peso del coche (m · g) más la carga aerodinámica. Bajadas"
        " breves por debajo del peso estático indican una rueda en"
        " el aire o descarga general al pasar una cresta."
    ),
    "load_front_frac": (
        "Fracción de la carga vertical total que soporta el eje"
        " delantero (0..1). Los valores estáticos en coches de calle"
        " rondan 0,55–0,60 (motor delantero) o 0,40–0,45 (motor"
        " trasero); los transitorios de frenada pueden empujarla"
        " por encima de 0,75."
    ),
    "load_left_frac": (
        "Fracción de la carga total en el lado izquierdo (0..1)."
        " Las curvas a derechas cargan el lado izquierdo; la"
        " magnitud depende del g lateral y de la altura del CG."
    ),
    "load_diag_fl_rr_frac": (
        "Carga diagonal: (FL + RR) / total. Diferencias respecto a"
        " 0,5 en curva estable indican cross-weight de chasis; en"
        " transitorio revela torsión del chasis (especialmente en"
        " pianos y crestas)."
    ),
    "brake_bias_front_real": (
        "Reparto real de frenada delantera bajo frenada"
        " (fuerza_freno_delantera / fuerza_freno_total). Si se"
        " separa del valor configurado, o el ABS está interviniendo"
        " en un eje o un set de gomas se ha bloqueado. Demasiado"
        " adelante = bloqueo delantero; demasiado atrás = bloqueo"
        " trasero y snap sobreviraje."
    ),
    "ffb_load_pct": (
        "Uso del rango de par del FFB (0..1). 1,0 sostenido = la"
        " base está clipando; baja el FFB del juego o sube la"
        " 'fuerza' de la base para que los picos quepan. El FFB"
        " clipado pierde información del tren delantero."
    ),
    "steer_rate_rads": (
        "Velocidad angular de dirección (rad/s). Picos altos son"
        " correcciones bruscas; los profesionales suelen mantener"
        " los picos por debajo de ~4 rad/s en curvas rápidas."
    ),
    "steer_reversal_rate_hz": (
        "Tasa de cambios de signo en la dirección (Hz). Valores"
        " altos sostenidos (>2 Hz) revelan 'pelearse' con el volante"
        " e inestabilidad a la salida; valores bajos y estables"
        " caracterizan una vuelta limpia y 'plantada'."
    ),
    "g_total_g": (
        "Magnitud total de aceleración en g, sqrt(ax²+ay²)/g. Es el"
        " número titular del diagrama g–g: cuánto se acerca el coche"
        " a la envolvente de agarre combinado en cada instante."
        " Valores sostenidos cerca del límite del neumático/aero"
        " indican que queda poco margen para dirección o gas."
    ),
    "susp_compression_front_avg_m": (
        "Compresión media de la suspensión delantera (m). Aproximación"
        " directa a la altura libre delantera; sube al frenar y con"
        " carga aerodinámica, baja en las crestas. La tendencia a lo"
        " largo de la vuelta delata riesgo de tocar fondo y la"
        " estabilidad de la plataforma aerodinámica."
    ),
    "susp_compression_rear_avg_m": (
        "Compresión media de la suspensión trasera (m). Misma idea"
        " que el canal delantero pero para la altura trasera; sube"
        " con el 'squat' al acelerar y con downforce, baja en"
        " crestas y en lift-off."
    ),
    "rake_compression_m": (
        "Compresión delantera menos trasera (m). Positivo ⇒ actitud"
        " 'nariz abajo' (típica en entrada / frenada fuerte);"
        " negativo ⇒ 'cola abajo' (a tope de gas, mucha carga atrás)."
        " Los coches sensibles a la aero son muy exigentes en"
        " mantener esta traza dentro de un margen estrecho durante"
        " la curva."
    ),
    "slip_angle_balance_rad": (
        "Balance cinemático: media(|α_delantero|) −"
        " media(|α_trasero|) en rad. Positivo ⇒ los delanteros"
        " deslizan más (subviraje); negativo ⇒ el tren trasero"
        " desliza más (sobreviraje). Complementa understeer_index —"
        " éste es una lectura directa del ángulo de deriva, el otro"
        " se basa en la velocidad de guiñada."
    ),
    "brake_power_w": (
        "Potencia mecánica instantánea hacia los frenos (W) mientras"
        " el pedal está pisado: |suma de fuerzas longitudinales| ×"
        " velocidad. Integrada por vuelta o por zona de frenada da"
        " un proxy fiable de calor y desgaste de pastillas, y"
        " muestra dónde hay que ajustar reparto o refrigeración."
    ),
    "throttle_reversal_rate_hz": (
        "Tasa de cambios de dirección del acelerador (Hz). Espejo de"
        " steer_reversal_rate_hz para el pedal derecho: valores"
        " altos sostenidos indican conducir a 'golpecitos' y lift,"
        " lo que descompensa el balance longitudinal y castiga"
        " neumáticos traseros. Los pilotos limpios la mantienen"
        " baja salvo en gestiones puntuales de tracción."
    ),
    "coasting": (
        "Bandera booleana: throttle < 0,05 ∧ brake < 0,05 ∧"
        " velocidad > 3 m/s. Marca las fases de rodadura libre /"
        " lift-off en media curva — útil para estudiar la técnica"
        " de sobreviraje por levantar el pie y para cuantificar"
        " cuánta vuelta se pasa sin frenar ni acelerar."
    ),
    "trail_brake_intensity": (
        "brake × |input_steer|. Sólo es distinto de cero cuando"
        " coinciden pedal y volante, es decir, en la entrada a"
        " curva con trail-brake. Picos más altos ⇒ más carga"
        " combinada en el tren delantero; compara por curva para"
        " equilibrar rotación de entrada y sobrecarga del eje"
        " delantero."
    ),
    "chassis_roll_per_lat_g_rad_per_g": (
        "Flexibilidad instantánea al balanceo: roll / (accel_y/g) en"
        " rad/g, NaN por debajo de ~0,2 g lateral. Indicador directo"
        " para barras estabilizadoras y muelles — un salto brusco"
        " de esta ratio en una fase de curva suele significar que"
        " la suspensión ha tocado tope o que la rueda interior se"
        " ha aliviado."
    ),
    "chassis_pitch_per_long_g_rad_per_g": (
        "Flexibilidad instantánea al cabeceo: pitch / (accel_x/g) en"
        " rad/g, NaN por debajo de ~0,2 g longitudinal. Cuantifica"
        " el 'dive' al frenar (ax negativo) y el 'squat' a tope de"
        " gas (ax positivo); es el canal adecuado para juzgar el"
        " amortiguamiento bump/rebound delantero y trasero."
    ),
    "accel_x_road_mps2": (
        "Aceleración longitudinal corregida por la pendiente. Aísla"
        " lo que hacen realmente frenos y motor de la ayuda o"
        " penalización de la pendiente. Úsala en lugar de accel_x"
        " para análisis de freno y motor en circuitos ondulados."
    ),
    "velocity_heading_rad": (
        "Dirección del vector velocidad en el marco del mundo (rad)."
        " Combinada con la guiñada del chasis genera beta y"
        " yaw_misalign."
    ),
    "yaw_misalign_rad": (
        "Guiñada del chasis menos guiñada de la velocidad (rad)."
        " Aproximación práctica al ángulo de deriva del coche en"
        " pista; positivo cuando el trasero se va en curva a"
        " derechas, negativo en curva a izquierdas."
    ),

    # -------- Geometría de pista (racing_lines/<TRACK>_racing.csv) --------
    "track_node": (
        "Índice del nodo más cercano de la trazada. Entero discreto;"
        " útil como clave de unión pero no como eje Y."
    ),
    "track_s_m": (
        "Distancia de arco a lo largo de la trazada (m). Monótona,"
        " ideal como eje X cuando se comparan vueltas que cruzan la"
        " línea en posiciones distintas."
    ),
    "track_z_m": (
        "Elevación de la pista en el nodo actual de la trazada (m)."
    ),
    "track_heading_rad": (
        "Heading tangente de la trazada en este nodo (rad). La"
        " diferencia entre la guiñada del chasis y este valor es"
        " esencialmente el ángulo de deriva del coche respecto a la"
        " línea ideal."
    ),
    "track_curvature_1_per_m": (
        "Curvatura con signo de la trazada κ (1/m); + = curva a"
        " izquierdas, − = a derechas, ~0 = recta. Su recíproco da"
        " el radio local de la curva."
    ),
    "track_radius_m": (
        "Radio local de curva R = 1/|κ| (m). Menor radio = curva más"
        " cerrada. Acota la velocidad máxima de paso por curva vía"
        " v = sqrt(μ · g · R) para un agarre dado."
    ),
    "track_slope_pct": (
        "Pendiente de la pista (%) a lo largo de la trazada."
        " Positivo = cuesta arriba. Se usa junto con accel_x para"
        " eliminar el efecto de la pendiente."
    ),
    "track_width_m": (
        "Ancho del corredor de conducción de la IA (m) en este"
        " nodo — el asfalto utilizable."
    ),
    "drive_left_local": (
        "Distancia de la trazada al borde izquierdo del corredor de"
        " la IA (≤0). Úsala junto con track_offset_m para ver lo"
        " cerca del borde que vas."
    ),
    "drive_right_local": (
        "Distancia de la trazada al borde derecho del corredor de"
        " la IA (≥0)."
    ),
    "limit_left_local": (
        "Distancia al límite duro de la izquierda (borde del asfalto)"
        " (≤0). Cruzarlo es oficialmente 'fuera de pista'."
    ),
    "limit_right_local": (
        "Distancia al límite duro de la derecha (borde del asfalto)"
        " (≥0)."
    ),
    "track_offset_m": (
        "Offset lateral (m) del coche respecto a la trazada"
        " (línea central). Compáralo con los canales drive/limit"
        " para ver si estás clavando ápice, abriendo o saliéndote."
    ),
    "segment_kind": (
        "Etiqueta categórica del segmento de pista actual: recta |"
        " izquierda | derecha. Útil para agrupar estadísticas por"
        " tipo de segmento."
    ),
    "segment_id": (
        "ID entero del segmento actual (recta o curva). Permite"
        " agregar métricas por curva entre vueltas y stints."
    ),

    # -------- Ayudas / testigos del salpicadero (bits ShowLights de OutGauge) --------
    "dl_tc_active": (
        "Bandera de intervención del control de tracción."
        " Activaciones frecuentes indican que estás pidiendo más"
        " agarre longitudinal del que las gomas motrices pueden"
        " dar — levanta el gas a la salida o ablanda el"
        " diferencial."
    ),
    "dl_abs_active": (
        "Bandera de intervención del ABS. Activaciones frecuentes"
        " significan que la presión de freno supera el umbral de"
        " bloqueo de esas gomas — alivia o desplaza el reparto hacia"
        " el eje que satura."
    ),
    "dl_pit_limiter": "Limitador de pit-lane activado.",
    "dl_handbrake": "Aviso de freno de mano accionado.",
    "dl_shift_light": (
        "Disparo del shift-light (motor cerca de su rpm óptima de"
        " cambio). Úsalo como ayuda de coaching contra la traza de"
        " rpm."
    ),
    "dl_oil_warn": "Aviso de aceite (presión baja o temperatura alta).",
    "dl_battery_warn": "Aviso de batería / alternador.",
    "dl_fullbeam": "Luces largas encendidas.",
    "dl_signal_l": "Intermitente izquierdo.",
    "dl_signal_r": "Intermitente derecho.",
}
_INTERP_BY_SUFFIX_ES: tuple[tuple[str, str], ...] = (
    ("_susp_deflect_m", (
        "Recorrido de suspensión en esta esquina (m). Positivo ="
        " comprimida respecto a la altura de reposo. Picos cortos"
        " son baches; compresiones sostenidas reflejan transferencia"
        " de peso estable. Mirar las cuatro esquinas a la vez revela"
        " balanceo, cabeceo y carga diagonal."
    )),
    ("_vertical_load_n", (
        "Carga vertical (N) sobre esta goma. Si toca 0, la rueda"
        " está en el aire (sin agarre). Un desequilibrio"
        " izquierda/derecha estable indica balanceo; uno"
        " delantero/trasero indica cabeceo."
    )),
    ("_slip_ratio", (
        "Slip longitudinal (adimensional). 0 = rodadura pura;"
        " positivo = tracción (la goma gira más rápido que el"
        " suelo), negativo = frenada (la goma gira más despacio)."
        " El pico de agarre longitudinal está en torno a |0,10–0,15|;"
        " valores sostenidos por encima significan que la goma está"
        " deslizando."
    )),
    ("_tan_slip_angle", (
        "tan(α), donde α es el ángulo de deriva de la goma — el"
        " ángulo entre el heading de la goma y su vector velocidad."
        " El pico de agarre lateral vive en torno a α ≈ 6–8°"
        " (tan ≈ 0,10–0,14). Más significa que la goma desliza"
        " lateralmente en vez de agarrar."
    )),
    ("_x_force_n", (
        "Fuerza lateral generada por la huella de la goma (N). El"
        " signo sigue al sentido de la curva. La saturación en una"
        " meseta plana es el máximo agarre lateral que puede"
        " entregar."
    )),
    ("_y_force_n", (
        "Fuerza longitudinal en la huella (N): positiva traccionando"
        " hacia delante, negativa frenando. Según la elipse de"
        " fricción, generar mucha fuerza lateral reduce la fuerza"
        " longitudinal máxima que puedes añadir al mismo tiempo."
    )),
    ("_ang_vel_rads", (
        "Velocidad angular de la rueda (rad/s). Multiplicada por el"
        " radio de la goma da la velocidad de rodadura; comparada"
        " con la velocidad del coche da el slip ratio."
    )),
    ("_lean_rel_road_rad", (
        "Caída (camber) de la rueda relativa al asfalto (rad). El"
        " camber dinámico — lo que realmente ve la goma — y la"
        " entrada clave para el agarre lateral de una goma"
        " camberada."
    )),
    ("_air_temp_c", (
        "Temperatura del aire interno de la goma (°C) según el"
        " modelo de LFS. Sube con la energía de entrada y es el"
        " indicador más estable del trabajo de la goma; una subida"
        " repentina señala maltrato (bloqueos, deslizamientos). Las"
        " temperaturas de carcasa y superficie no están disponibles"
        " por separado en OutSim/OutGauge."
    )),
    ("_slip_fraction", (
        "Fracción de la huella que está deslizando (0..1). Valores"
        " cerca de 1 indican que la goma está más allá de la zona"
        " lineal; muy útil para detectar 'scrubbing' en el límite."
    )),
    ("_touching", (
        "1 cuando la goma está en contacto con una superficie, 0"
        " cuando está en el aire. Úsalo para enmascarar otros"
        " canales de la goma que pierden sentido en el aire."
    )),
    ("_steer_rad", (
        "Ángulo de dirección de esta rueda (rad). Difiere del ángulo"
        " del volante por la relación de dirección y la geometría"
        " Ackermann (la rueda interior gira más que la exterior en"
        " curva cerrada)."
    )),
    ("_susp_speed_mps", (
        "Velocidad del amortiguador (m/s). Positiva = comprimiendo"
        " (bump); negativa = extendiendo (rebound). El histograma"
        " HS/LS en la pestaña Damper lo divide en régimen de alta"
        " velocidad (golpes de bache) y baja velocidad (movimiento"
        " del chasis); un coche equilibrado tiene lóbulos de baja"
        " velocidad bien llenos y colas de alta velocidad breves y"
        " contenidas."
    )),
)
_INTERP_BY_PATTERN_ES: tuple[tuple[str, str], ...] = (
    ("friction_use_", (
        "Uso de la elipse de fricción de la goma (0..1) ="
        " sqrt(Fx² + Fy²) / (μ · Fz). 1,0 significa que la huella"
        " está plenamente usada y no puedes añadir más fuerza"
        " lateral o longitudinal sin perder agarre en otro lado. Si"
        " sólo una rueda satura mientras las demás se quedan cortas,"
        " el setup está desequilibrado (anti-roll, altura, presiones,"
        " alineación)."
    )),
    ("tyre_work_w_", (
        "Potencia mecánica disipada por la goma (W) — la energía que"
        " se va a calor y desgaste. Intégrala por vuelta y reparte"
        " entre las cuatro gomas para ver cuál es la esquina más"
        " trabajada y la primera en degradarse."
    )),
)
_INTERP_BY_GROUP_ES: dict[str, str] = {
    "Vehicle": (
        "Estado básico del coche — velocidad, posición, tiempos de"
        " vuelta. Procede del stream UDP OutGauge que emite LFS."
    ),
    "Engine": (
        "Canales del tren motriz: RPM, marcha, combustible y"
        " temperaturas de motor/aceite desde OutGauge."
    ),
    "Driver": (
        "Entradas del piloto: gas, freno, embrague, freno de mano,"
        " dirección y par de FFB. El indicador más limpio del"
        " estilo de conducción."
    ),
    "Chassis": (
        "Dinámica de chasis OutSim: aceleración en 3 ejes (m/s²),"
        " velocidad angular en 3 ejes (rad/s) y ángulos de actitud"
        " Euler."
    ),
    "Suspension": (
        "Estado por esquina de la suspensión: recorrido, carga"
        " vertical, velocidad del amortiguador y ángulo de"
        " dirección de la rueda."
    ),
    "Tyre": (
        "Comportamiento por esquina de la goma: slip ratio, ángulo"
        " de deriva, fuerzas longitudinal/lateral, temperatura"
        " interna y banderas de contacto."
    ),
    "Aids": (
        "Bitfield ShowLights de OutGauge: estado de TC, ABS,"
        " limitador de pit, intermitentes y testigos de aviso."
    ),
    "Derived": (
        "Magnitudes calculadas por Studio a partir de los canales"
        " brutos de OutSim/OutGauge — índice de subviraje,"
        " transferencias de peso, uso de fricción, etc."
    ),
    "Lap": (
        "Distancias e índices relativos a la vuelta usados para"
        " alinear trazas entre vueltas."
    ),
    "Track": (
        "Geometría estática de pista muestreada en el nodo actual"
        " de la trazada: curvatura, radio, pendiente, ancho y"
        " offset lateral del coche dentro del corredor."
    ),
    "Context": (
        "Contexto de sesión: coche, pista, clima, viento. Se usa"
        " para filtrar y agrupar capturas."
    ),
}
_FOCUS_BY_COLUMN_ES: dict[str, tuple[str, str]] = {
    "speed_ms": (
        "Ingeniero: compara velocidad mínima y de salida por vuelta y"
        " sector para separar pérdidas por trazada y por potencia.",
        "Piloto: prioriza conservar velocidad de entrada sin retrasar"
        " la apertura de gas en el ápice.",
    ),
    "throttle": (
        "Ingeniero: mide suavidad de gas y tiempo a fondo; picos suelen"
        " correlacionar con falta de tracción.",
        "Piloto: construye una rampa limpia de gas en salida, evita"
        " golpecitos repetidos.",
    ),
    "brake": (
        "Ingeniero: revisa mordida inicial, velocidad de liberación y"
        " consistencia para ajustar reparto y mapa de freno.",
        "Piloto: frena fuerte al inicio y suelta de forma progresiva"
        " al empezar el giro.",
    ),
    "input_steer": (
        "Ingeniero: oscilaciones de dirección apuntan a inestabilidad"
        " o exceso de sensibilidad del tren delantero.",
        "Piloto: reduce microcorrecciones; un arco claro suele ser más"
        " rápido que varias correcciones pequeñas.",
    ),
    "steer_torque_nm": (
        "Ingeniero: vigila saturación/clipping para mantener calidad de"
        " información en el FFB.",
        "Piloto: si el volante se queda 'muerto' en ápice, pide menos"
        " ángulo y reequilibra la entrada.",
    ),
    "accel_x": (
        "Ingeniero: compara pico de deceleración y timing de liberación"
        " de freno para evaluar rendimiento de frenada.",
        "Piloto: maximiza frenada en recta y libera progresivamente al"
        " añadir dirección.",
    ),
    "accel_y": (
        "Ingeniero: usa pico de g lateral por curva para detectar dónde"
        " no se está explotando el agarre.",
        "Piloto: busca picos de g lateral repetibles sin arrastrar"
        " velocidad en mitad de curva.",
    ),
    "understeer_index": (
        "Ingeniero: localiza picos por fase de curva para decidir si"
        " tocar geometría delantera, apoyo trasero o diferencial.",
        "Piloto: picos positivos piden esperar rotación; negativos"
        " piden suavizar gas/dirección.",
    ),
    "brake_bias_front_real": (
        "Ingeniero: sigue la deriva del reparto bajo carga para validar"
        " setup y actuación del ABS.",
        "Piloto: si el eje trasero se mueve en entrada, adelanta un"
        " poco; si bloquea delante, atrasa.",
    ),
}
