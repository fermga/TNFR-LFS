"""Channel metadata registry for MoTeC-style consumers.

Maps DataFrame column names → display label, units and group, so a
visualization app can build its "channel browser" tree without
hard-coding everything.

The registry is generated once from a small declaration table plus the
per-wheel naming convention used in :mod:`.replay` and :mod:`.derived`.

Usage::

    from lfs_telemetry.telemetry.channels import CHANNELS, channel_info, ChannelInfo

    info = channel_info("speed_ms")
    info.label, info.units, info.group   # 'Speed', 'm/s', 'Vehicle'
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .protocol.packets import WHEEL_ORDER


@dataclass(frozen=True, slots=True)
class ChannelInfo:
    """Metadata for one telemetry channel."""

    column: str
    label: str
    units: str
    group: str
    description: str = ""
    interpretation: str = ""

    def tooltip_html(
        self,
        *,
        translate: Callable[[str], str] | None = None,
        language: str | None = None,
    ) -> str:
        """Rich HTML tooltip combining label, units, description and
        interpretation guidance. Safe to feed directly into Qt's
        ``ToolTipRole`` (Qt auto-detects HTML by the leading ``<``).
        """
        tr = translate or (lambda s: s)
        lang = (language or "en").lower()

        label = tr(self.label)
        if lang == "es" and label == self.label:
            label = _LABEL_ES_FALLBACK.get(self.label, self.label)
            # Handle wheel-suffixed labels like "Susp. travel FL" by
            # translating the base label and re-appending the corner code.
            if label == self.label:
                parts_lbl = self.label.rsplit(" ", 1)
                if (
                    len(parts_lbl) == 2
                    and parts_lbl[1] in ("FL", "FR", "RL", "RR")
                ):
                    base = _LABEL_ES_FALLBACK.get(parts_lbl[0], parts_lbl[0])
                    label = f"{base} {parts_lbl[1]}"
        units_value = tr(self.units)
        group = tr(self.group)
        if lang == "es" and group == self.group:
            group = _GROUP_ES_FALLBACK.get(self.group, self.group)

        description_text = self.description
        if lang == "es" and description_text:
            description_text = _DESCRIPTION_ES_FALLBACK.get(
                description_text,
                description_text,
            )
        description = tr(description_text) if description_text else ""

        interpretation_text = self.interpretation
        if lang == "es":
            interpretation_text = _interpretation_for_lang(
                self.column,
                self.group,
                language="es",
            )
        interpretation = tr(interpretation_text) if interpretation_text else ""

        engineer_focus, driver_focus = _focus_notes_for(
            self.column,
            self.group,
            language=lang,
        )
        engineer_header = tr("Engineer focus")
        driver_header = tr("Driver focus")
        how_to_read = tr("How to read it")
        if lang == "es":
            if engineer_header == "Engineer focus":
                engineer_header = "Foco ingeniero"
            if driver_header == "Driver focus":
                driver_header = "Foco piloto"
            if how_to_read == "How to read it":
                how_to_read = "Cómo leerlo"

        units = f" [{units_value}]" if units_value else ""
        parts = [
            f"<b>{label}</b>{units}",
            f"<i>{group} &middot; <code>{self.column}</code></i>",
        ]
        if description:
            parts.append(description)
        if interpretation:
            parts.append(
                "<div style='margin-top:4px;'>"
                f"<u>{how_to_read}</u>: {interpretation}"
                "</div>"
            )
        if engineer_focus:
            parts.append(
                "<div style='margin-top:4px;'>"
                f"<u>{engineer_header}</u>: {engineer_focus}"
                "</div>"
            )
        if driver_focus:
            parts.append(
                "<div style='margin-top:4px;'>"
                f"<u>{driver_header}</u>: {driver_focus}"
                "</div>"
            )
        return "<div style='max-width:380px;'>" + "<br>".join(parts) + "</div>"


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


_FOCUS_BY_COLUMN_EN: dict[str, tuple[str, str]] = {
    "speed_ms": (
        "Engineer: compare minimum-speed and corner-exit speed by lap"
        " and sector to separate line losses from power losses.",
        "Driver: focus on carrying entry speed without delaying"
        " throttle pickup at the apex.",
    ),
    "throttle": (
        "Engineer: quantify throttle smoothness and time-at-full-"
        "throttle; spikes often correlate with traction instability.",
        "Driver: build one clean throttle ramp at exit instead of"
        " repeated stabs.",
    ),
    "brake": (
        "Engineer: check initial bite, release rate, and consistency"
        " across laps to tune bias and pedal map.",
        "Driver: brake hard early, then bleed pressure smoothly into"
        " turn-in to keep front grip.",
    ),
    "input_steer": (
        "Engineer: steering oscillations indicate instability or"
        " over-sensitive front axle setup.",
        "Driver: reduce micro-corrections; one decisive arc is usually"
        " faster than multiple small fixes.",
    ),
    "steer_torque_nm": (
        "Engineer: monitor clipping/saturation to preserve FFB"
        " information quality.",
        "Driver: if steering goes numb at apex, reduce steering angle"
        " demand and re-balance entry.",
    ),
    "accel_x": (
        "Engineer: compare peak decel and brake-release timing for"
        " braking performance benchmarking.",
        "Driver: maximize straight-line decel, then release progressively"
        " as steering is added.",
    ),
    "accel_y": (
        "Engineer: use peak lateral-g by corner to identify where grip"
        " is under-used.",
        "Driver: aim to reach consistent lateral-g peaks without"
        " scrubbing mid-corner speed.",
    ),
    "understeer_index": (
        "Engineer: map peaks by corner phase to decide whether changes"
        " belong to front geometry, rear support, or diff.",
        "Driver: positive peaks suggest waiting for rotation; negative"
        " peaks ask for calmer throttle/steer timing.",
    ),
    "brake_bias_front_real": (
        "Engineer: track bias drift under load to validate setup and"
        " ABS behavior.",
        "Driver: if rear gets nervous on entry, move a touch forward;"
        " if front locks first, move rearward.",
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


def _interpretation_for_lang(
    column: str,
    group: str,
    *,
    language: str,
) -> str:
    if language != "es":
        return _interpretation_for(column, group)
    if column in _INTERP_BY_COLUMN_ES:
        return _INTERP_BY_COLUMN_ES[column]
    for suffix, text in _INTERP_BY_SUFFIX_ES:
        if column.endswith(suffix):
            return text
    for prefix, text in _INTERP_BY_PATTERN_ES:
        if column.startswith(prefix):
            return text
    if group in _INTERP_BY_GROUP_ES:
        return _INTERP_BY_GROUP_ES[group]
    # Nunca caer en texto vacío/genérico si no hay traducción puntual:
    # prioriza la explicación inglesa completa por canal.
    return _interpretation_for(column, group)


_FOCUS_BY_SUFFIX_EN: tuple[tuple[str, tuple[str, str]], ...] = (
    ("_slip_ratio", (
        "Engineer: find where each axle exceeds optimal slip under brake/"
        "traction to tune bias, diff and pedal maps.",
        "Driver: target one smooth brake release and one smooth throttle"
        " pickup to keep slip in the productive range.",
    )),
    ("_tan_slip_angle", (
        "Engineer: compare front vs rear slip-angle demand to localize"
        " understeer/oversteer by corner phase.",
        "Driver: avoid adding steering once the tyre is already sliding;"
        " prioritize rotation timing.",
    )),
    ("_vertical_load_n", (
        "Engineer: monitor load distribution across the axle to validate"
        " ARB/spring and ride-height changes.",
        "Driver: smoother pedal and steering transitions reduce abrupt"
        " load spikes and improve consistency.",
    )),
    ("_susp_speed_mps", (
        "Engineer: use HS/LS split to separate platform-control issues"
        " from curb/bump compliance issues.",
        "Driver: if the car feels bouncy after curbs, calm steering and"
        " throttle timing over rough sections.",
    )),
)


_FOCUS_BY_SUFFIX_ES: tuple[tuple[str, tuple[str, str]], ...] = (
    ("_slip_ratio", (
        "Ingeniero: identifica dónde cada eje supera el slip óptimo en"
        " frenada/tracción para ajustar reparto, diferencial y mapas.",
        "Piloto: busca una suelta de freno y una apertura de gas limpias"
        " para mantener el slip en zona productiva.",
    )),
    ("_tan_slip_angle", (
        "Ingeniero: compara demanda de deriva delante/detrás para localizar"
        " subviraje/sobreviraje por fase de curva.",
        "Piloto: evita pedir más volante cuando el neumático ya desliza;"
        " prioriza el timing de rotación.",
    )),
    ("_vertical_load_n", (
        "Ingeniero: monitoriza reparto de carga por eje para validar"
        " cambios de barras, muelles y alturas.",
        "Piloto: transiciones más suaves de pedales y volante reducen"
        " picos de carga y mejoran consistencia.",
    )),
    ("_susp_speed_mps", (
        "Ingeniero: usa el reparto HS/LS para separar problemas de"
        " plataforma de problemas de absorción de baches/pianos.",
        "Piloto: si el coche rebota tras piano, suaviza volante y gas"
        " al pasar por zonas rotas.",
    )),
)


_FOCUS_BY_PATTERN_EN: tuple[tuple[str, tuple[str, str]], ...] = (
    ("friction_use_", (
        "Engineer: identify which corner saturates first and redistribute"
        " load/grip with setup before chasing driving fixes.",
        "Driver: if one wheel saturates early, delay peak input and"
        " straighten exits before full throttle.",
    )),
    ("tyre_work_w_", (
        "Engineer: track thermal workload per corner to forecast tyre"
        " degradation and pressure evolution.",
        "Driver: reduce prolonged slides and wheelspin to keep tyre work"
        " and temperatures under control.",
    )),
)


_FOCUS_BY_PATTERN_ES: tuple[tuple[str, tuple[str, str]], ...] = (
    ("friction_use_", (
        "Ingeniero: localiza qué rueda satura primero y redistribuye"
        " carga/agarre con setup antes de corregir al piloto.",
        "Piloto: si una rueda satura pronto, retrasa el pico de mando y"
        " endereza más antes de gas a fondo.",
    )),
    ("tyre_work_w_", (
        "Ingeniero: sigue trabajo térmico por rueda para anticipar"
        " degradación y evolución de presiones.",
        "Piloto: reduce deslizamientos largos y patinaje para mantener"
        " temperatura y desgaste bajo control.",
    )),
)


_FOCUS_BY_GROUP_EN: dict[str, tuple[str, str]] = {
    "Driver": (
        "Engineer: benchmark input timing and smoothness corner-by-corner"
        " to explain lap-time deltas.",
        "Driver: prioritize repeatable traces over one-off peaks; pace"
        " comes from consistency.",
    ),
    "Engine": (
        "Engineer: optimize shift points, thermal margins and fuel usage"
        " over stint length.",
        "Driver: keep the engine in its useful band and avoid limiter"
        " time that gives no extra acceleration.",
    ),
    "Chassis": (
        "Engineer: correlate inertia channels with setup changes to isolate"
        " balance and platform issues.",
        "Driver: read where the car rotates vs where it pushes, then adapt"
        " entry speed and release timing.",
    ),
    "Suspension": (
        "Engineer: verify platform control and curb compliance per corner"
        " to tune dampers/springs/ARBs.",
        "Driver: smoother curb usage and steering timing reduces upset and"
        " preserves grip on exits.",
    ),
    "Tyre": (
        "Engineer: monitor saturation, temperature trend and force balance"
        " to protect peak grip over the stint.",
        "Driver: avoid over-driving the tyre beyond peak slip windows;"
        " smoothness preserves speed.",
    ),
    "Track": (
        "Engineer: segment performance by curvature/slope/radius to target"
        " setup and gearing where it matters.",
        "Driver: use geometry channels to commit to consistent references"
        " for turn-in, apex and exit.",
    ),
}


_FOCUS_BY_GROUP_ES: dict[str, tuple[str, str]] = {
    "Driver": (
        "Ingeniero: compara timing y suavidad de mandos curva a curva"
        " para explicar deltas de vuelta.",
        "Piloto: prioriza trazas repetibles frente a picos aislados;"
        " el ritmo llega por consistencia.",
    ),
    "Engine": (
        "Ingeniero: optimiza puntos de cambio, margen térmico y consumo"
        " durante el stint.",
        "Piloto: mantén el motor en su banda útil y evita tiempo en"
        " limitador que no acelera más.",
    ),
    "Chassis": (
        "Ingeniero: correlaciona canales inerciales con cambios de setup"
        " para aislar balance y plataforma.",
        "Piloto: identifica dónde rota y dónde empuja para ajustar"
        " velocidad de entrada y liberación.",
    ),
    "Suspension": (
        "Ingeniero: valida control de plataforma y absorción de piano por"
        " rueda para afinar amortiguación/muelles/barras.",
        "Piloto: un uso más limpio de piano y volante preserva apoyo y"
        " tracción de salida.",
    ),
    "Tyre": (
        "Ingeniero: vigila saturación, deriva térmica y equilibrio de"
        " fuerzas para sostener agarre pico en el stint.",
        "Piloto: evita sobreconducir fuera de la ventana de slip pico;"
        " la suavidad conserva neumático y ritmo.",
    ),
    "Track": (
        "Ingeniero: segmenta rendimiento por curvatura/pendiente/radio"
        " para enfocar setup y desarrollo.",
        "Piloto: usa la geometría para fijar referencias consistentes de"
        " entrada, ápice y salida.",
    ),
}


def _focus_notes_for(
    column: str,
    group: str,
    *,
    language: str,
) -> tuple[str, str]:
    if language == "es":
        if column in _FOCUS_BY_COLUMN_ES:
            return _FOCUS_BY_COLUMN_ES[column]
        for suffix, notes in _FOCUS_BY_SUFFIX_ES:
            if column.endswith(suffix):
                return notes
        for prefix, notes in _FOCUS_BY_PATTERN_ES:
            if column.startswith(prefix):
                return notes
        if group in _FOCUS_BY_GROUP_ES:
            return _FOCUS_BY_GROUP_ES[group]
        return (
            "Ingeniero: compara este canal por vuelta y por sector para"
            " separar problema de setup de problema de ejecución.",
            "Piloto: busca trazas limpias y repetibles; menor ruido suele"
            " traducirse en más confianza y ritmo.",
        )
    if column in _FOCUS_BY_COLUMN_EN:
        return _FOCUS_BY_COLUMN_EN[column]
    for suffix, notes in _FOCUS_BY_SUFFIX_EN:
        if column.endswith(suffix):
            return notes
    for prefix, notes in _FOCUS_BY_PATTERN_EN:
        if column.startswith(prefix):
            return notes
    if group in _FOCUS_BY_GROUP_EN:
        return _FOCUS_BY_GROUP_EN[group]
    return (
        "Engineer: use this channel to compare consistency by lap and"
        " sector, and to validate setup changes.",
        "Driver: track trend and repeatability; a cleaner, steadier"
        " trace usually means more pace.",
    )


# (column, label, units, group, description)
_BASE: tuple[tuple[str, str, str, str, str], ...] = (
    # --- Vehicle (chassis) ---
    ("speed_ms", "Speed", "m/s", "Vehicle", "Wheel-derived speed"),
    ("rpm", "RPM", "rpm", "Engine", "Engine speed"),
    ("gear_lfs", "Gear", "", "Engine",
     "Canonical LFS gear (-1=R, 0=N, 1..N=forward)"),
    ("fuel", "Fuel", "frac", "Engine", "Fuel level (0..1)"),
    ("turbo_bar", "Turbo", "bar", "Engine", "Turbo boost"),
    # --- Driver inputs ---
    ("throttle", "Throttle", "frac", "Driver", "Throttle pedal (0..1)"),
    ("brake", "Brake", "frac", "Driver", "Brake pedal (0..1)"),
    ("clutch", "Clutch", "frac", "Driver", "Clutch pedal (0..1)"),
    ("input_steer", "Steer (raw)", "rad", "Driver", "Steering wheel angle"),
    ("input_handbrake", "Handbrake", "frac", "Driver", ""),
    ("steer_torque_nm", "Steer Torque", "Nm", "Driver", "FFB steering torque"),
    # --- Chassis dynamics ---
    ("ang_vel_x", "Roll rate", "rad/s", "Chassis", ""),
    ("ang_vel_y", "Pitch rate", "rad/s", "Chassis", ""),
    ("ang_vel_z", "Yaw rate", "rad/s", "Chassis", ""),
    ("heading", "Heading", "rad", "Chassis", ""),
    ("pitch", "Pitch", "rad", "Chassis", ""),
    ("roll", "Roll", "rad", "Chassis", ""),
    ("accel_x", "Long. accel", "m/s²", "Chassis", "+forward, -brake"),
    ("accel_y", "Lat. accel", "m/s²", "Chassis", "+right"),
    ("accel_z", "Vert. accel", "m/s²", "Chassis", "+up"),
    ("vel_x", "Vel. X", "m/s", "Chassis", "World X velocity"),
    ("vel_y", "Vel. Y", "m/s", "Chassis", "World Y velocity"),
    ("vel_z", "Vel. Z", "m/s", "Chassis", "World Z velocity"),
    ("pos_x", "Pos. X", "m", "Chassis", "World X position"),
    ("pos_y", "Pos. Y", "m", "Chassis", "World Y position"),
    ("pos_z", "Pos. Z", "m", "Chassis", "World Z position"),
    # --- Derived (chassis dynamics) ---
    ("yaw_rate_rads", "Yaw rate", "rad/s", "Derived", "Body yaw rate (z)"),
    ("yaw_rate_theoretical_rads", "Yaw rate (th)", "rad/s", "Derived",
     "Neutral-steer yaw rate from steered angle"),
    ("understeer_index", "Understeer idx", "", "Derived", "+= understeer"),
    ("transfer_long_n_real", "Long. transfer", "N", "Derived", "Real long. load Δ"),
    ("transfer_lat_n_real", "Lat. transfer", "N", "Derived", "Real lat. load Δ"),
    ("transfer_long_n_theoretical", "Long. transfer (th)", "N", "Derived", ""),
    ("transfer_lat_n_theoretical", "Lat. transfer (th)", "N", "Derived", ""),
    ("load_total_n", "Total load", "N", "Derived", "Sum of vertical loads"),
    ("load_front_frac", "Front load frac", "", "Derived", ""),
    ("load_left_frac", "Left load frac", "", "Derived", ""),
    ("load_diag_fl_rr_frac", "Diag FL/RR frac", "", "Derived", ""),
    ("brake_bias_front_real", "Brake bias front", "frac", "Derived", "Real, when braking"),
    ("ffb_load_pct", "FFB load", "frac", "Derived", "|steer_torque|/max"),
    ("steer_rate_rads", "Steer rate", "rad/s", "Derived", "d(input_steer)/dt"),
    ("steer_reversal_rate_hz", "Steer reversals", "Hz", "Derived", ""),
    # --- Derived (combined / synergy channels) ---
    ("g_total_g", "g-force total", "g", "Derived",
     "sqrt(ax²+ay²)/g; friction-circle headline magnitude"),
    ("susp_compression_front_avg_m", "Front compression", "m", "Derived",
     "Mean front-axle suspension compression"),
    ("susp_compression_rear_avg_m", "Rear compression", "m", "Derived",
     "Mean rear-axle suspension compression"),
    ("rake_compression_m", "Rake (Δ compression)", "m", "Derived",
     "Front − rear axle compression; +ve = nose-down"),
    ("slip_angle_balance_rad", "Slip balance (F−R)", "rad", "Derived",
     "mean|α_front|−mean|α_rear|; +ve=understeer"),
    ("brake_power_w", "Brake power", "W", "Derived",
     "|F_long_total|·v while braking; brake-heat proxy"),
    ("throttle_reversal_rate_hz", "Throttle reversals", "Hz", "Derived",
     "Throttle direction changes per second"),
    ("coasting", "Coasting", "bool", "Derived",
     "throttle<0.05 ∧ brake<0.05 ∧ v>3 m/s"),
    ("trail_brake_intensity", "Trail-brake intensity", "", "Derived",
     "brake × |input_steer|; combined corner-entry load"),
    ("chassis_roll_per_lat_g_rad_per_g", "Roll compliance",
     "rad/g", "Derived",
     "Instantaneous roll/ay; ARB/spring tuning probe"),
    ("chassis_pitch_per_long_g_rad_per_g", "Pitch compliance",
     "rad/g", "Derived",
     "Instantaneous pitch/ax; brake-dive / squat probe"),
    # --- Derived (track geometry; require a racing_lines/<TRACK>_racing.csv) ---
    ("track_node", "Track node", "", "Track", "Nearest centerline node index"),
    ("track_s_m", "Track s", "m", "Track", "Arc length along centerline"),
    ("track_z_m", "Track elevation", "m", "Track", "Centerline Z at this s"),
    ("track_heading_rad", "Track heading", "rad", "Track", "Centerline tangent heading"),
    ("track_curvature_1_per_m", "Track curvature", "1/m", "Track", "+left, -right"),
    ("track_radius_m", "Track radius", "m", "Track", "1/|curvature|, capped"),
    ("track_slope_pct", "Track slope", "%", "Track", "100·dz/ds along centerline"),
    ("track_width_m", "Track width", "m", "Track", "drive_right - drive_left at node"),
    ("track_offset_m", "Lateral offset", "m", "Track", "Perp. distance to centerline"),
    ("segment_id", "Segment id", "", "Track", "Geometry segment index"),
    ("accel_x_road_mps2", "Long. accel (road)", "m/s²", "Derived",
     "accel_x with road-grade gravity removed (+engine, -brake)"),
    ("yaw_misalign_rad", "Yaw misalignment", "rad", "Derived",
     "vel-heading − track-heading; trajectory slip proxy"),
    # --- Dash lights (booleans 0/1) ---
    ("dl_tc_active", "TC active", "bool", "Aids", ""),
    ("dl_abs_active", "ABS active", "bool", "Aids", ""),
    ("dl_handbrake", "Handbrake on", "bool", "Aids", ""),
    ("dl_pit_limiter", "Pit limiter", "bool", "Aids", ""),
    ("dl_oil_warn", "Oil warning", "bool", "Aids", ""),
    ("dl_battery_warn", "Battery warn", "bool", "Aids", ""),
    ("dl_signal_l", "Indicator L", "bool", "Aids", ""),
    ("dl_signal_r", "Indicator R", "bool", "Aids", ""),
    ("dl_fullbeam", "Full beam", "bool", "Aids", ""),
    ("dl_shift_light", "Shift light", "bool", "Aids", ""),
    # --- Race context ---
    ("ctx_wind", "Wind", "m/s", "Context", "Wind speed magnitude"),
)


# Per-wheel template: (suffix, label, units, group, description).
_WHEEL_TEMPLATE: tuple[tuple[str, str, str, str, str], ...] = (
    ("susp_deflect_m", "Susp. travel", "m", "Suspension", ""),
    ("vertical_load_n", "Vert. load", "N", "Suspension", ""),
    ("slip_ratio", "Slip ratio", "", "Tyre", ""),
    ("tan_slip_angle", "tan(α)", "", "Tyre", ""),
    ("x_force_n", "Lat. force", "N", "Tyre", "Tyre lateral force"),
    ("y_force_n", "Long. force", "N", "Tyre", "Tyre longitudinal force"),
    ("ang_vel_rads", "Wheel ω", "rad/s", "Tyre", ""),
    ("lean_rel_road_rad", "Camber rel.", "rad", "Suspension", ""),
    ("air_temp_c", "Tyre air temp", "°C", "Tyre", ""),
    ("slip_fraction", "Slip frac", "", "Tyre", "Sliding contact fraction"),
    ("touching", "Touching", "bool", "Tyre", ""),
    ("steer_rad", "Wheel steer", "rad", "Suspension", ""),
)


# Per-wheel template for *derived* columns.
_WHEEL_DERIVED: tuple[tuple[str, str, str, str, str], ...] = (
    ("friction_use_{c}", "Friction use", "", "Derived", "Used / available μ"),
    ("tyre_work_w_{c}", "Tyre work", "W", "Derived", "Mech. power into tyre"),
    ("wheel_{c}_susp_speed_mps", "Susp. speed", "m/s", "Suspension",
     "Damper velocity (>0 bump, <0 rebound)"),
    ("wheel_{c}_lockup", "Lockup", "bool", "Derived",
     "Braking ∧ slip_ratio<−0.3; ABS event detector"),
)


# ---------------------------------------------------------------------------
# Interpretation guide (English, plain language with telemetry context).
#
# These strings answer "what is this plot telling me?" for someone who
# knows cars but not telemetry. Where relevant they cite the LFS source
# (OutSim/OutGauge/InSim packet fields) and standard vehicle-dynamics
# concepts (slip angle/ratio, friction ellipse, weight transfer, etc.).
#
# Resolution order:
#   1. exact column-name match,
#   2. suffix match (per-wheel channels),
#   3. prefix match (derived patterns),
#   4. group fallback,
#   5. empty string.
# ---------------------------------------------------------------------------

_INTERP_BY_COLUMN: dict[str, str] = {
    # -------- Vehicle / Engine (OutGauge + OutSim) --------
    "speed_ms": (
        "Forward ground speed (m/s) from OutGauge. Overlay two laps on "
        "the same corner: the one carrying more speed at the apex is "
        "usually the better line. Sudden drops mean braking, wheel "
        "lock, or a traction loss; long flat plateaus mark sections "
        "where you are speed-limited by aero/drag rather than grip."
    ),
    "rpm": (
        "Engine speed (revs/min) from OutGauge. Flat-top traces against "
        "the rev limit = upshifts left too late (you are hitting the "
        "limiter); large RPM drops on upshift = gear ratio too long or "
        "shift performed off the power band. For naturally-aspirated "
        "engines the sweet spot is staying inside the upper third of "
        "the torque curve."
    ),
    "gear_lfs": (
        "Currently engaged gear (R, N, 1..N) from OutGauge. Use this "
        "to audit shift points and downshifts: dropping a gear before "
        "the brakes are released loads the driven wheels mid-corner "
        "and can pitch the car. Compare with throttle/brake traces."
    ),
    "fuel": (
        "Fuel level as a fraction of tank capacity (0..1) from "
        "OutGauge. The slope is the instantaneous consumption rate; "
        "the FuelTracker module turns it into laps-remaining and pit "
        "windows. A sudden drop is usually a refuel event."
    ),
    "eng_temp_c": (
        "Coolant temperature (°C) from OutGauge. Steady creep above "
        "the normal band signals blocked radiator vents, body damage, "
        "or persistent over-revving. Most LFS engines tolerate up to "
        "~110 °C before power loss; sustained higher figures shorten "
        "engine life."
    ),
    "oil_temp_c": (
        "Oil temperature (°C). It reacts slower than coolant and is "
        "the better proxy for long-term thermal stress. Healthy race "
        "range is roughly 90–125 °C depending on the car."
    ),
    "oil_pressure_bar": (
        "Engine oil pressure (bar). Drops under high lateral g "
        "indicate oil starvation (oil sloshing away from the pickup); "
        "persistently low pressure suggests a hot, thin oil or wear."
    ),
    "turbo_bar": (
        "Turbo boost pressure (bar) above atmospheric. A flat trace "
        "at peak boost = wastegate regulating correctly. Slow ramps "
        "at low RPM are turbo lag; sudden dips mid-throttle are "
        "blow-off events or wastegate flutter."
    ),

    # -------- Driver inputs (OutGauge) --------
    "throttle": (
        "Throttle pedal position (0..1) from OutGauge. The textbook "
        "shape exits a corner as a smooth, monotonic ramp from 0 to "
        "1; step inputs that bounce off 1.0 indicate aggressive "
        "throttle that the rear axle may not handle. Long 1.0 "
        "plateaus mark full-throttle sections; you can measure "
        "‘time at full throttle’ per lap to compare driving styles."
    ),
    "brake": (
        "Brake pedal position (0..1) from OutGauge. A good braking "
        "trace ramps quickly to a high initial peak (just below "
        "lock-up), then bleeds smoothly toward zero as you turn in "
        "(trail-braking). Stair-stepped releases reveal pedal "
        "modulation or ABS cycling; very flat plateaus near 1.0 mean "
        "you are pushing the brake into ABS or wheel lock."
    ),
    "clutch": (
        "Clutch pedal (0..1). Only meaningful around standing starts "
        "and gear shifts. A long partial-depress trace on launch is "
        "clutch slipping (used deliberately to manage wheelspin)."
    ),
    "input_steer": (
        "Steering wheel angle (rad), positive = left. Smooth, "
        "single-arc traces indicate well-judged corner entry; a "
        "‘sawtooth’ pattern of small reversals shows over-correction "
        "or a nervous chassis. The peak amplitude tied to corner "
        "radius hints at understeer/oversteer balance."
    ),
    "steer_torque_nm": (
        "Steering rack torque (N·m) — the physical FFB reaction that "
        "would be felt at the wheel. Sudden sign changes denote rear "
        "snap; large flat plateaus across mid-corner mean the front "
        "tyres are saturated (understeer)."
    ),
    "input_handbrake": (
        "Handbrake input (0..1). Only used for handbrake turns, drift, "
        "or rallycross. A spike in normal racing is almost always a "
        "mistake."
    ),

    # -------- Chassis (OutSim — vehicle-axis IMU) --------
    "accel_x": (
        "Longitudinal acceleration in the vehicle frame (m/s²). "
        "Positive = accelerating, negative = braking. A clean braking "
        "trace shows a sharp negative step to the friction limit and "
        "stays flat (constant deceleration) before bleeding to 0 at "
        "turn-in. Reference figures: ~1 g braking = 9.81 m/s²; race "
        "slicks can pull 1.5–2 g."
    ),
    "accel_y": (
        "Lateral acceleration (m/s², +left). Its absolute peak in a "
        "corner equals the lateral grip you actually used. Cross-plot "
        "against accel_x (a ‘g–g diagram’) to see how well you fill "
        "the friction ellipse — missing quadrants reveal areas where "
        "you are leaving grip on the table."
    ),
    "accel_z": (
        "Vertical acceleration (m/s²). Spikes are bumps, curb hits, "
        "or landings after a crest. Sustained values above 1 g "
        "compression indicate a heavy aero load or a steep "
        "compression."
    ),
    "ang_vel_z": (
        "Yaw rate (rad/s) about the vertical axis — how fast the car "
        "rotates in plan view. The pure-geometry expectation is "
        "v / R (speed ÷ corner radius); see understeer_index for the "
        "deviation."
    ),
    "ang_vel_x": (
        "Roll rate (rad/s) about the longitudinal axis. Big spikes "
        "in fast direction changes (chicanes) indicate soft anti-roll "
        "bars or springs and slow weight transfer."
    ),
    "ang_vel_y": (
        "Pitch rate (rad/s) about the lateral axis. Positive pulses "
        "under braking (nose-dive), negative under throttle "
        "(squat). Magnitude scales with brake/throttle force divided "
        "by front/rear suspension stiffness."
    ),
    "pitch": (
        "Chassis pitch angle (rad). Positive in braking (nose down), "
        "negative under acceleration. Excessive pitch hurts "
        "aerodynamic platform and changes mechanical grip balance."
    ),
    "roll": (
        "Chassis roll angle (rad). Large roll in steady-state corners "
        "points to soft springs or insufficient ARB; very stiff cars "
        "will show almost flat traces but can skip on curbs."
    ),
    "heading": (
        "Chassis yaw (heading) angle in world frame (rad). Mostly "
        "useful when overlaid on the track map; not very informative "
        "as a time-series trace by itself."
    ),

    # -------- Lap / track distance (InSim + derived) --------
    "current_lap_dist_m": (
        "Distance travelled since the start/finish line (m). Resets "
        "to 0 each lap. Use it as an x-axis to align corners between "
        "laps for direct comparison."
    ),
    "indexed_distance_m": (
        "Distance along the racing-line index (m); does not reset at "
        "the line. Preferred for season-long datasets that span "
        "multiple stints."
    ),

    # -------- Derived chassis dynamics --------
    "yaw_rate_rads": (
        "Measured yaw rate (rad/s). Compared against the theoretical "
        "neutral-car yaw rate it tells you the balance: see "
        "understeer_index."
    ),
    "yaw_rate_theoretical_rads": (
        "Yaw rate a perfectly neutral car would have at this speed "
        "and steering angle (v · tan(δ) / wheelbase). The gap to the "
        "measured yaw rate is what the front/rear axles are actually "
        "delivering."
    ),
    "beta_deg": (
        "Vehicle side-slip angle (°) = atan(v_lateral / v_long). On "
        "a road car a healthy maximum is 3–6°; a sustained value "
        "above 8° means the car is sliding noticeably and is likely "
        "losing time. Useful to spot oversteer or aggressive trail-"
        "braking."
    ),
    "understeer_index": (
        "(yaw_theoretical − yaw_measured) / yaw_theoretical. "
        "Positive = understeer (front pushing wide), negative = "
        "oversteer (rear yawing more than the steering asks for). "
        "Note where the spikes happen: entry, mid-corner or exit "
        "diagnose different setup problems (front geometry, weight "
        "transfer, diff)."
    ),
    "transfer_long_n_real": (
        "Longitudinal weight transfer (N). The textbook value is "
        "m · a_x · h_cg / wheelbase: positive under acceleration "
        "(load shifts rearward), negative under braking (load shifts "
        "forward). Compare to the static axle weights to see relative "
        "load swing."
    ),
    "transfer_lat_n_real": (
        "Lateral weight transfer (N). Driven by m · a_y · h_cg / "
        "track. Its split between front and rear axles depends on "
        "roll stiffness; biasing roll stiffness forward adds front "
        "transfer (more understeer) and vice-versa."
    ),
    "load_total_n": (
        "Sum of all four wheel vertical loads (N). Should hover "
        "around the car’s weight (m · g) plus aero downforce. Brief "
        "dips below static weight indicate a wheel airborne or all-"
        "round unloading over a crest."
    ),
    "load_front_frac": (
        "Fraction of total vertical load carried by the front axle "
        "(0..1). Static values for road cars sit around 0.55–0.60 "
        "(front-engined) or 0.40–0.45 (rear-engined); braking "
        "transients can push it past 0.75."
    ),
    "load_left_frac": (
        "Fraction of total load on the left side of the car (0..1). "
        "Right-hand corners load the left side; magnitude depends on "
        "lateral g and CG height."
    ),
    "load_diag_fl_rr_frac": (
        "Diagonal load: (FL + RR) / total. Differences from 0.5 in "
        "steady-state cornering indicate chassis cross-weight; in "
        "transients it reveals chassis twist (especially on curbs "
        "and crests)."
    ),
    "brake_bias_front_real": (
        "Actual front brake bias under braking (front_brake_force / "
        "total_brake_force). If it diverges from the dialled-in "
        "value, either ABS is intervening on one axle or one set of "
        "tyres has locked up. Front-bias too high = front lock-up; "
        "too low = rear lock-up and snap oversteer."
    ),
    "ffb_load_pct": (
        "Force-feedback torque utilisation (0..1). 1.0 sustained = "
        "the wheelbase is clipping; raise the in-game FFB setting "
        "down or the wheelbase ‘strength’ up so the peaks fit. "
        "Clipped FFB loses front-end information."
    ),
    "steer_rate_rads": (
        "Steering angular velocity (rad/s). High peaks are abrupt "
        "corrections; pros usually keep peak rates below ~4 rad/s in "
        "fast corners."
    ),
    "steer_reversal_rate_hz": (
        "Rate of steering-input sign changes (Hz). High values "
        "(>2 Hz sustained) reveal hand-wrestling and corner exit "
        "instability; low and steady values characterise a clean, "
        "‘planted’ lap."
    ),
    "g_total_g": (
        "Total acceleration magnitude in g, sqrt(ax²+ay²)/g. The "
        "headline number on a friction-circle plot: how close the "
        "car is to the combined-grip envelope at any instant. "
        "Sustained values close to the tyre/aero limit mean little "
        "margin is left for steering or throttle."
    ),
    "susp_compression_front_avg_m": (
        "Mean of front-axle suspension compression (m). A direct "
        "ride-height proxy on the front axle; rises under braking "
        "and aero load, falls over crests. Trend across a lap "
        "exposes bottoming risk and aero-platform stability."
    ),
    "susp_compression_rear_avg_m": (
        "Mean of rear-axle suspension compression (m). Same idea as "
        "the front channel but for rear ride height; rises on power "
        "squat and high downforce, drops over crests and on lift-"
        "off."
    ),
    "rake_compression_m": (
        "Front-minus-rear axle compression (m). Positive ⇒ nose-down "
        "attitude (typical on entry / under heavy braking); negative "
        "⇒ tail-down (power-on, big aero on rear). Aero-sensitive "
        "cars are very fussy about this trace staying inside a "
        "narrow band through the corner."
    ),
    "slip_angle_balance_rad": (
        "Kinematic balance: mean(|α_front|) − mean(|α_rear|) in rad. "
        "Positive ⇒ front tyres are sliding more than the rear "
        "(understeer); negative ⇒ rear is sliding more (oversteer). "
        "Complements understeer_index — this one is a direct slip-"
        "angle reading, the other one is yaw-rate based."
    ),
    "brake_power_w": (
        "Instantaneous mechanical power going into the brakes (W) "
        "while the pedal is down: |sum of longitudinal tyre force| × "
        "speed. Integrate per lap or per braking zone to get a "
        "reliable brake-heat / pad-wear proxy and to spot zones "
        "where bias or cooling needs work."
    ),
    "throttle_reversal_rate_hz": (
        "Throttle direction-change rate (Hz). Mirror of "
        "steer_reversal_rate_hz for the right pedal: high sustained "
        "values mean pedal-tap / lift-stab driving which upsets "
        "longitudinal balance and burns rear tyres. Smooth drivers "
        "keep this low except in clear traction-management bursts."
    ),
    "coasting": (
        "Boolean flag: throttle < 0.05 ∧ brake < 0.05 ∧ speed > "
        "3 m/s. Marks mid-corner coasting / lift-off phases — useful "
        "to study lift-off oversteer technique and to quantify how "
        "much of the lap is spent neither braking nor accelerating."
    ),
    "trail_brake_intensity": (
        "brake × |input_steer|. Non-zero only when both pedal and "
        "hand are engaged, i.e. on trail-braking corner entry. "
        "Higher peaks ⇒ more combined load on the front tyres; "
        "compare per-corner to balance entry rotation against "
        "front-end overload."
    ),
    "chassis_roll_per_lat_g_rad_per_g": (
        "Instantaneous roll compliance: roll / (accel_y/g) in rad/g, "
        "NaN below ~0.2 g lateral load. Direct probe for ARB and "
        "spring choice — a sudden jump in this ratio at a given "
        "corner phase usually means the suspension just hit a "
        "bump-stop or the inside wheel went light."
    ),
    "chassis_pitch_per_long_g_rad_per_g": (
        "Instantaneous pitch compliance: pitch / (accel_x/g) in "
        "rad/g, NaN below ~0.2 g longitudinal load. Quantifies "
        "brake-dive (negative ax) and power-squat (positive ax) "
        "behaviour; the right channel for judging bump/rebound "
        "damping on the front and rear axles."
    ),
    "accel_x_road_mps2": (
        "Longitudinal acceleration corrected for road slope. "
        "Isolates what brakes/engine actually do from the help or "
        "hindrance of a slope. Use this instead of accel_x for power-"
        "and-brake analysis on undulating tracks."
    ),
    "velocity_heading_rad": (
        "Direction of the velocity vector in the world frame (rad). "
        "Combined with the chassis heading it produces beta and "
        "yaw_misalign."
    ),
    "yaw_misalign_rad": (
        "Chassis-heading minus velocity-heading (rad). A practical "
        "proxy for the vehicle slip angle on track; positive while "
        "the rear is stepping out in a right-hander, negative in a "
        "left-hander."
    ),

    # -------- Track geometry (racing_lines/<TRACK>_racing.csv) --------
    "track_node": (
        "Nearest racing-line node index. Discrete integer; useful as "
        "a join key but not as a y-axis."
    ),
    "track_s_m": (
        "Arc-length distance along the racing line (m). Monotonic, "
        "ideal for x-axis when comparing laps that re-cross start/"
        "finish at different positions."
    ),
    "track_z_m": "Track elevation at the current racing-line node (m).",
    "track_heading_rad": (
        "Tangent heading of the racing line at this node (rad). The "
        "difference between chassis heading and this is essentially "
        "the car’s slip angle relative to the ideal line."
    ),
    "track_curvature_1_per_m": (
        "Signed racing-line curvature κ (1/m); + = left-hander, − = "
        "right-hander, ~0 = straight. The reciprocal gives the local "
        "corner radius."
    ),
    "track_radius_m": (
        "Local corner radius R = 1/|κ| (m). Smaller radius = tighter "
        "corner. Sets the upper-bound cornering speed via "
        "v = sqrt(μ · g · R) for given grip."
    ),
    "track_slope_pct": (
        "Track gradient (%) along the racing line. Positive = uphill. "
        "Used together with accel_x to remove slope effects."
    ),
    "track_width_m": (
        "Width of the AI drive corridor (m) at this node — the usable "
        "racing surface."
    ),
    "drive_left_local": (
        "Distance from racing line to the left edge of the AI drive "
        "corridor (≤0). Use it together with track_offset_m to see "
        "how close to the edge you are."
    ),
    "drive_right_local": (
        "Distance from racing line to the right edge of the AI drive "
        "corridor (≥0)."
    ),
    "limit_left_local": (
        "Distance to the hard left limit (asphalt edge) (≤0). "
        "Crossing this is officially ‘off-track’."
    ),
    "limit_right_local": (
        "Distance to the hard right limit (asphalt edge) (≥0)."
    ),
    "track_offset_m": (
        "Lateral offset (m) of the car from the racing line "
        "(centre-line). Compare with the drive/limit channels to see "
        "if you are clipping the apex, running wide or going off."
    ),
    "segment_kind": (
        "Categorical tag for the current track segment: straight | "
        "left | right. Useful to group statistics by segment type."
    ),
    "segment_id": (
        "Integer ID of the current segment (straight or corner). "
        "Lets you aggregate metrics per corner across laps and "
        "stints."
    ),

    # -------- Aids / dashboard lights (OutGauge ShowLights bits) --------
    "dl_tc_active": (
        "Traction-control intervention flag. Frequent activations "
        "indicate you are demanding more longitudinal grip than the "
        "driven tyres can deliver — back off throttle on exits or "
        "soften the diff."
    ),
    "dl_abs_active": (
        "ABS intervention flag. Frequent activations mean brake "
        "pressure exceeds the locking threshold for those tyres — "
        "ease off, or shift bias toward the saturating axle."
    ),
    "dl_pit_limiter": "Pit-lane speed limiter engaged.",
    "dl_handbrake": "Handbrake engaged warning.",
    "dl_shift_light": (
        "Shift-light trigger (engine near its optimum upshift RPM). "
        "Use as a coaching aid against the rpm trace."
    ),
    "dl_oil_warn": "Oil warning (low pressure or high temperature).",
    "dl_battery_warn": "Battery / alternator warning.",
    "dl_fullbeam": "High beam on.",
    "dl_signal_l": "Left turn indicator.",
    "dl_signal_r": "Right turn indicator.",
}


_INTERP_BY_SUFFIX: tuple[tuple[str, str], ...] = (
    ("_susp_deflect_m", (
        "Suspension travel at this corner (m). Positive = compressed "
        "relative to ride height. Short spikes are bumps; sustained "
        "compression reflects steady weight transfer. Watching all "
        "four corners side-by-side reveals body roll, pitch and "
        "diagonal load."
    )),
    ("_vertical_load_n", (
        "Vertical load (N) on this tyre. If it touches 0 the tyre is "
        "airborne (no grip). A left/right imbalance in steady state "
        "indicates body roll; a front/rear imbalance indicates pitch."
    )),
    ("_slip_ratio", (
        "Longitudinal slip ratio (dimensionless). 0 = pure rolling; "
        "positive = driving (tyre spinning faster than ground), "
        "negative = braking (tyre rotating slower). Peak tyre "
        "longitudinal grip occurs around |0.10–0.15|; sustained "
        "values past that mean the tyre is sliding."
    )),
    ("_tan_slip_angle", (
        "tan(α), where α is the tyre slip angle — the angle between "
        "the tyre’s heading and its velocity vector. Peak lateral "
        "grip lives around α ≈ 6–8° (tan ≈ 0.10–0.14). Bigger means "
        "the tyre is sliding sideways rather than gripping."
    )),
    ("_x_force_n", (
        "Lateral force generated by the tyre contact patch (N). Sign "
        "tracks the cornering direction. Saturation at a flat plateau "
        "is the maximum lateral grip the tyre can provide."
    )),
    ("_y_force_n", (
        "Longitudinal force at the contact patch (N): positive when "
        "driving forward, negative when braking. Per the friction "
        "ellipse, generating large lateral force reduces the maximum "
        "longitudinal force you can add at the same time."
    )),
    ("_ang_vel_rads", (
        "Wheel angular velocity (rad/s). Multiply by tyre radius to "
        "get the rolling speed; comparing it with ground speed gives "
        "the slip ratio."
    )),
    ("_lean_rel_road_rad", (
        "Wheel camber angle relative to the road surface (rad). The "
        "dynamic camber — what the tyre actually sees — and the key "
        "input for lateral grip from a cambered tyre."
    )),
    ("_air_temp_c", (
        "Inflation-air temperature inside the tyre (°C) as modelled "
        "by LFS. It rises with energy input and is the most stable "
        "indicator of tyre work; a sudden rise signals abuse "
        "(lock-ups, slides). Carcass and surface temperatures are not "
        "available separately in OutSim/OutGauge."
    )),
    ("_slip_fraction", (
        "Fraction of the contact patch that is sliding (0..1). "
        "Values near 1 mean the tyre is past the linear region; "
        "very useful to spot ‘scrubbing’ on the limit."
    )),
    ("_touching", (
        "1 when the tyre is in contact with a surface, 0 when it is "
        "airborne. Use it to mask other tyre channels that become "
        "meaningless mid-air."
    )),
    ("_steer_rad", (
        "Steered angle of this road wheel (rad). Differs from the "
        "steering-wheel angle by the steering ratio and the "
        "Ackermann geometry (inner wheel steers more than outer in "
        "a tight corner)."
    )),
    ("_susp_speed_mps", (
        "Damper velocity (m/s). Positive = compressing (bump); "
        "negative = extending (rebound). The HS/LS histogram in the "
        "Damper tab splits this into high-speed (bump strikes) and "
        "low-speed (chassis motion) regimes; a balanced car has "
        "well-filled low-speed lobes and brief, capped high-speed "
        "tails."
    )),
)


_INTERP_BY_PATTERN: tuple[tuple[str, str], ...] = (
    ("friction_use_", (
        "Tyre friction-ellipse utilisation (0..1) = sqrt(Fx² + Fy²) / "
        "(μ · Fz). 1.0 means the contact patch is fully used and you "
        "cannot add more lateral or longitudinal force without losing "
        "grip elsewhere. If just one wheel reaches saturation while "
        "the others lag, the setup is unbalanced (anti-roll, ride "
        "height, pressures, alignment)."
    )),
    ("tyre_work_w_", (
        "Mechanical power dissipated by the tyre (W) — the energy "
        "going into heat and wear. Integrate over a lap and divide "
        "across the four tyres to see which corner is the hottest "
        "worked and likely degrading first."
    )),
)


_INTERP_BY_GROUP: dict[str, str] = {
    "Vehicle": (
        "Basic car state — speed, position, lap times. Sourced from "
        "the OutGauge UDP stream that LFS emits."
    ),
    "Engine": (
        "Powertrain channels: RPM, gear, fuel and engine/oil "
        "temperatures from OutGauge."
    ),
    "Driver": (
        "Driver inputs: throttle, brake, clutch, handbrake, steering "
        "and FFB torque. The cleanest indicator of driving style."
    ),
    "Chassis": (
        "OutSim chassis dynamics: 3-axis acceleration (m/s²), "
        "3-axis angular velocity (rad/s) and Euler attitude angles."
    ),
    "Suspension": (
        "Per-corner suspension state: travel, vertical load, damper "
        "velocity and steered wheel angle."
    ),
    "Tyre": (
        "Per-corner tyre behaviour: slip ratio, slip angle, "
        "longitudinal/lateral forces, inflation temperature and "
        "contact-patch flags."
    ),
    "Aids": (
        "OutGauge ShowLights bitfield: state of TC, ABS, pit "
        "limiter, indicators and warning lamps."
    ),
    "Derived": (
        "Quantities computed by Studio from the raw OutSim/OutGauge "
        "channels — understeer index, weight transfers, friction "
        "utilisation, etc."
    ),
    "Lap": (
        "Lap-relative distances and indices used to align traces "
        "between laps."
    ),
    "Track": (
        "Static track geometry sampled at the current racing-line "
        "node: curvature, radius, slope, width and the lateral "
        "offset of the car within the corridor."
    ),
    "Context": (
        "Session context: car, track, weather, wind. Used to filter "
        "and group captures."
    ),
}


def _interpretation_for(column: str, group: str) -> str:
    if column in _INTERP_BY_COLUMN:
        return _INTERP_BY_COLUMN[column]
    for suffix, text in _INTERP_BY_SUFFIX:
        if column.endswith(suffix):
            return text
    for prefix, text in _INTERP_BY_PATTERN:
        if column.startswith(prefix):
            return text
    return _INTERP_BY_GROUP.get(group, "")


def _build_registry() -> dict[str, ChannelInfo]:
    out: dict[str, ChannelInfo] = {}
    for col, lbl, units, group, desc in _BASE:
        out[col] = ChannelInfo(
            col, lbl, units, group, desc, _interpretation_for(col, group)
        )
    for c in WHEEL_ORDER:
        for suffix, lbl, units, group, desc in _WHEEL_TEMPLATE:
            col = f"wheel_{c}_{suffix}"
            out[col] = ChannelInfo(
                col, f"{lbl} {c}", units, group, desc,
                _interpretation_for(col, group),
            )
        for tmpl, lbl, units, group, desc in _WHEEL_DERIVED:
            col = tmpl.format(c=c)
            out[col] = ChannelInfo(
                col, f"{lbl} {c}", units, group, desc,
                _interpretation_for(col, group),
            )
    return out


CHANNELS: dict[str, ChannelInfo] = _build_registry()


def channel_info(column: str) -> ChannelInfo:
    """Return :class:`ChannelInfo` for ``column``; falls back to bare metadata."""
    info = CHANNELS.get(column)
    if info is not None:
        return info
    # Unknown column → best-effort fallback so the UI never blows up.
    return ChannelInfo(
        column=column, label=column, units="", group="Other",
        interpretation=_INTERP_BY_GROUP.get("Other", ""),
    )


def channels_by_group(columns: list[str] | None = None) -> dict[str, list[ChannelInfo]]:
    """Group known channels by ``group`` for tree-style channel browsers.

    If ``columns`` is given, restrict the listing to those (handy after
    inspecting a real DataFrame's ``df.columns``).
    """
    cols = columns if columns is not None else list(CHANNELS)
    groups: dict[str, list[ChannelInfo]] = {}
    for col in cols:
        info = channel_info(col)
        groups.setdefault(info.group, []).append(info)
    for items in groups.values():
        items.sort(key=lambda i: i.label)
    return groups


__all__ = ["ChannelInfo", "CHANNELS", "channel_info", "channels_by_group"]
