"""Help dialog: general telemetry interpretation guide.

Renders an HTML overview of every channel group plus the per-channel
interpretation table from :mod:`lfs_telemetry.telemetry.channels`. The
goal is that someone who knows nothing about telemetry can read each
plot at a glance.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ...telemetry import channels_by_group
from ..i18n import LANG_SPANISH, current_language, tr

_GROUP_ORDER: tuple[str, ...] = (
    "Driver", "Vehicle", "Engine", "Chassis",
    "Suspension", "Tyre", "Derived", "Track", "Aids", "Lap", "Context",
)

_GROUP_INTRO: dict[str, str] = {
    "Driver": (
        "What the driver is doing with the controls. These traces "
        "(throttle, brake, clutch, handbrake, steering, FFB torque) "
        "are sourced directly from the LFS OutGauge UDP stream and "
        "give the most direct read on driving style. Comparing two "
        "laps on the driver channels alone usually explains most of "
        "the lap-time delta."
    ),
    "Vehicle": (
        "Basic car state: speed, position and lap timing. Overlaying "
        "speed for two laps shows exactly where you gain or lose "
        "time; integrating the speed difference gives the cumulative "
        "Δt that the chart already plots when ‘Δt vs ref’ is enabled."
    ),
    "Engine": (
        "Powertrain channels — RPM, gear, fuel, engine/oil "
        "temperatures, turbo boost. Use them to audit shift points, "
        "monitor thermal headroom and budget fuel."
    ),
    "Chassis": (
        "OutSim chassis dynamics: 3-axis acceleration (m/s²), "
        "3-axis angular velocity (rad/s) and Euler attitude. This is "
        "where understeer/oversteer, weight transfer and platform "
        "stability are read from."
    ),
    "Suspension": (
        "Per-corner suspension state: travel, vertical load, damper "
        "velocity, dynamic camber and the steered wheel angle. The "
        "Dampers tab summarises HS/LS damper histograms (the "
        "industry-standard ~50 mm/s threshold separating chassis "
        "motions from bump strikes)."
    ),
    "Tyre": (
        "Per-corner tyre channels — slip ratio, slip angle, "
        "longitudinal/lateral forces, inflation temperature, contact "
        "patch. The friction ellipse |F|/(μ·Fz) ties them all "
        "together: a tyre at saturation cannot give more lateral "
        "without losing longitudinal grip, and vice-versa."
    ),
    "Derived": (
        "Quantities Studio computes from the raw OutSim/OutGauge "
        "channels: understeer index, longitudinal / lateral load "
        "transfers, real brake bias, friction utilisation, steering "
        "smoothness… These are usually the most diagnostic plots."
    ),
    "Track": (
        "Static racing-line geometry sampled at the current node — "
        "curvature, radius, slope, width and the lateral offset of "
        "the car within the corridor. Lets you interpret the rest "
        "of the channels in their physical context (e.g. high "
        "lateral g where the radius is tight is expected)."
    ),
    "Aids": (
        "OutGauge ShowLights bitfield — state of TC, ABS, pit "
        "limiter, shift light, turn indicators and warning lamps. "
        "Frequent TC/ABS activations mean you are demanding more "
        "longitudinal grip than the tyres can supply."
    ),
    "Lap": (
        "Lap-relative distances and indices used to align traces "
        "across multiple laps."
    ),
    "Context": (
        "Session context: car, track, weather, wind. Used as "
        "metadata to filter and compare captures."
    ),
}

_HOW_TO_READ = """
<h2>How to read Studio</h2>
<ul>
  <li><b>Hover any channel</b> in the right-hand tree, or any plot,
      to see a tooltip with what it measures and how to interpret it.</li>
  <li><b>X axis</b>: distance (recommended for comparing laps) or
      time (better for inspecting individual transients).</li>
  <li><b>Δt vs ref</b>: with two or more laps on a distance x-axis,
      the first row shows the cumulative time delta against the
      reference lap. Rising = losing time; falling = gaining.</li>
  <li><b>Friction circle / Load transfer presets</b>: one-click
      channel selections for the most common diagnostic views.</li>
  <li><b>Sectors tab</b>: splits the lap into sectors, marking your
      personal best sector and the theoretical best.</li>
  <li><b>Dampers tab</b>: high-speed / low-speed damper velocity
      histogram (cross-over near ~50 mm/s). A balanced car has
      well-filled low-speed lobes and brief, capped high-speed
      tails; if one wheel never fills the HS bin it is over-damped,
      if it only fills HS it is under-damped.</li>
  <li><b>Capture tab</b>: starts and stops the live telemetry
      pipeline (UDP for OutSim/OutGauge, TCP for InSim) and writes
      one CSV per lap into your workspace.</li>
  <li><b>Overlay tab</b>: enables individual in-game overlay windows
      (delta bar, radar, g-meter, gear, RPM, speed, fuel, gaps,
      flags). Each one is a frameless, always-on-top widget you can
      drag and resize on top of LFS.</li>
  <li><b>Race dashboard</b> dock: in live sessions you see real-time
      splits, projected lap, nearby traffic and fuel autonomy.</li>
</ul>

<h2>Reading shortcuts (driving and physics)</h2>
<ul>
  <li><b>Throttle &amp; brake</b> should ‘shake hands’ at corner
      entry/exit — released brake as throttle starts to climb. Long
      simultaneous applications are deliberate trail-braking or a
      mistake. Step inputs that bounce off 1.0 suggest aggressive
      pedal work that the tyres may not absorb.</li>
  <li><b>Steering</b>: smooth single-arc traces = well-judged turn-in.
      A sawtooth pattern of small reversals is over-correction or a
      nervous chassis (steer_reversal_rate_hz quantifies it).</li>
  <li><b>g–g diagram</b>: plot accel_x vs accel_y. A car using the
      whole friction ellipse fills the circle; empty quadrants show
      where grip is being left unused (typically braking-into-turn
      transitions).</li>
  <li><b>understeer_index</b>: +ve = front pushing wide (front tyres
      saturated), −ve = rear yawing more than the steering input asks
      (oversteer). Where the spikes appear matters: <i>entry</i> spikes
      tend to come from soft front geometry or cold front tyres;
      <i>exit</i> spikes from too-much-power-too-early or a loose
      differential setting.</li>
  <li><b>friction_use_*</b> close to 1.0 on a single wheel = the
      contact patch is saturated. If only one wheel does this while
      the others lag, the load distribution is off (anti-roll bars,
      pressures, alignment, ride height).</li>
  <li><b>Tyre slip targets</b>: peak longitudinal grip lives near
      slip_ratio |0.10–0.15|; peak lateral grip near slip angle
      6–8° (tan ≈ 0.10–0.14). Past those, the tyre is sliding more
      than gripping.</li>
  <li><b>Weight transfer</b>: transfer_long ≈ m·a_x·h_cg/wheelbase,
      transfer_lat ≈ m·a_y·h_cg/track. Both scale with CG height —
      lowering ride height is the cheapest way to reduce transfer.</li>
</ul>

<h2>Where the data comes from</h2>
<p>LFS exposes three telemetry streams that Studio listens to:</p>
<ul>
  <li><b>OutSim</b> — chassis IMU at high rate (position, velocity,
      3-axis acceleration in m/s², 3-axis angular velocity in rad/s,
      Euler angles, per-wheel info for slip ratio, slip angle,
      forces, contact and suspension travel).</li>
  <li><b>OutGauge</b> — dashboard state (RPM, speed, gear, throttle,
      brake, clutch, handbrake, steering, fuel, engine/oil
      temperatures, turbo, show-lights bitfield).</li>
  <li><b>InSim</b> — session events (lap times, splits, race
      control, chat).</li>
</ul>
<p>The required entries in <code>cfg.txt</code> are
<code>OutSim Mode 2 / Opts 1ff / Delay 1 / IP 127.0.0.1 / Port 30000</code>
and <code>OutGauge Mode 1 / Delay 1 / IP 127.0.0.1 / Port 30001</code>.
InSim is started at runtime inside LFS with <code>/insim 29999</code>
(or <code>LFS.exe /insim=29999</code>).</p>
"""


_GROUP_INTRO_ES: dict[str, str] = {
    "Driver": (
        "Lo que est\u00e1 haciendo el piloto con los mandos. Estas "
        "trazas (acelerador, freno, embrague, freno de mano, "
        "direcci\u00f3n, par de FFB) provienen directamente del "
        "stream UDP OutGauge de LFS y dan la lectura m\u00e1s "
        "directa del estilo de conducci\u00f3n. Comparar dos "
        "vueltas s\u00f3lo por los canales del piloto suele "
        "explicar la mayor parte del delta de tiempo."
    ),
    "Vehicle": (
        "Estado b\u00e1sico del coche: velocidad, posici\u00f3n y "
        "cron\u00f3metros. Superponer la velocidad de dos vueltas "
        "muestra exactamente d\u00f3nde ganas o pierdes tiempo; "
        "integrar la diferencia de velocidad da el \u0394t "
        "acumulado que el gr\u00e1fico ya dibuja al activar "
        "\u2018\u0394t vs ref\u2019."
    ),
    "Engine": (
        "Canales del grupo motriz \u2014 RPM, marcha, combustible, "
        "temperaturas de motor/aceite, presi\u00f3n de turbo. \u00datiles "
        "para auditar puntos de cambio, vigilar margen t\u00e9rmico y "
        "presupuestar combustible."
    ),
    "Chassis": (
        "Din\u00e1mica de chasis OutSim: aceleraci\u00f3n en 3 ejes "
        "(m/s\u00b2), velocidad angular en 3 ejes (rad/s) y "
        "actitud Euler. De aqu\u00ed se leen subviraje/sobreviraje, "
        "transferencia de peso y estabilidad de la plataforma."
    ),
    "Suspension": (
        "Estado de la suspensi\u00f3n por rueda: recorrido, carga "
        "vertical, velocidad de amortiguador, c\u00e1ida din\u00e1mica "
        "y \u00e1ngulo de la rueda dirigida. La pesta\u00f1a "
        "Amortiguadores resume histogramas HS/LS (umbral est\u00e1ndar "
        "~50 mm/s que separa movimientos de chasis de los impactos)."
    ),
    "Tyre": (
        "Canales por rueda \u2014 slip ratio, \u00e1ngulo de deriva, "
        "fuerzas longitudinal/lateral, temperatura de hinchado, huella "
        "de contacto. La elipse de fricci\u00f3n |F|/(\u03bc\u00b7Fz) "
        "los une: un neum\u00e1tico saturado no puede dar m\u00e1s "
        "lateral sin perder longitudinal, y viceversa."
    ),
    "Derived": (
        "Magnitudes que Studio calcula a partir de los canales "
        "OutSim/OutGauge: \u00edndice de subviraje, transferencias "
        "longitudinal/lateral, reparto real de frenada, uso de "
        "fricci\u00f3n, suavidad de direcci\u00f3n\u2026 Suelen ser "
        "los gr\u00e1ficos m\u00e1s diagn\u00f3sticos."
    ),
    "Track": (
        "Geometr\u00eda est\u00e1tica de la trazada muestreada en el "
        "nodo actual \u2014 curvatura, radio, pendiente, ancho y el "
        "desplazamiento lateral del coche dentro del corredor. Permite "
        "interpretar el resto de canales en su contexto f\u00edsico "
        "(p. ej. g lateral alta donde el radio es peque\u00f1o)."
    ),
    "Aids": (
        "Bitfield ShowLights de OutGauge \u2014 estado de TC, ABS, "
        "limitador de pit, luz de cambio, intermitentes y testigos. "
        "Activaciones frecuentes de TC/ABS indican que pides m\u00e1s "
        "agarre longitudinal del que pueden dar los neum\u00e1ticos."
    ),
    "Lap": (
        "Distancias e \u00edndices relativos a la vuelta usados para "
        "alinear trazas entre m\u00faltiples vueltas."
    ),
    "Context": (
        "Contexto de sesi\u00f3n: coche, circuito, tiempo, viento. "
        "Se usa como metadatos para filtrar y comparar capturas."
    ),
}


_HOW_TO_READ_ES = """
<h2>C\u00f3mo leer Studio</h2>
<ul>
  <li><b>Pasa el rat\u00f3n por cualquier canal</b> del \u00e1rbol de
      la derecha, o por cualquier gr\u00e1fico, para ver un tooltip
      con qu\u00e9 mide y c\u00f3mo interpretarlo.</li>
  <li><b>Eje X</b>: distancia (recomendado para comparar vueltas) o
      tiempo (mejor para inspeccionar transitorios individuales).</li>
  <li><b>\u0394t vs ref</b>: con dos o m\u00e1s vueltas y eje X en
      distancia, la primera fila muestra el delta de tiempo
      acumulado contra la vuelta de referencia. Sube = pierdes
      tiempo; baja = ganas.</li>
  <li><b>Preajustes C\u00edrculo de fricci\u00f3n / Transferencia de
      carga</b>: selecciones de canales de un clic para las vistas
      diagn\u00f3sticas m\u00e1s habituales.</li>
  <li><b>Pesta\u00f1a Sectores</b>: divide la vuelta en sectores,
      marcando tu mejor sector personal y el mejor te\u00f3rico.</li>
  <li><b>Pesta\u00f1a Amortiguadores</b>: histograma de velocidad de
      amortiguador alta/baja (cruce cerca de ~50 mm/s). Un coche
      equilibrado tiene l\u00f3bulos de baja velocidad bien rellenos
      y colas cortas en alta velocidad; si una rueda nunca llena la
      zona HS est\u00e1 sobre-amortiguada, si s\u00f3lo llena HS
      est\u00e1 infra-amortiguada.</li>
  <li><b>Pesta\u00f1a Captura</b>: arranca y detiene la canalizaci\u00f3n
      de telemetr\u00eda en vivo (UDP para OutSim/OutGauge, TCP para
      InSim) y escribe un CSV por vuelta en tu espacio de trabajo.</li>
  <li><b>Pesta\u00f1a Overlay</b>: habilita ventanas individuales de
      overlay en juego (barra de delta, radar, g-metro, marcha, RPM,
      velocidad, combustible, huecos, banderas). Cada una es un
      widget sin bordes siempre encima que puedes arrastrar y
      redimensionar sobre LFS.</li>
  <li>Dock <b>Panel de carrera</b>: en sesiones en vivo ves los
      parciales en tiempo real, la vuelta proyectada, el tr\u00e1fico
      cercano y la autonom\u00eda de combustible.</li>
</ul>

<h2>Atajos de lectura (conducci\u00f3n y f\u00edsica)</h2>
<ul>
  <li><b>Acelerador y freno</b> deber\u00edan \u2018darse la mano\u2019
      en entrada/salida de curva \u2014 freno liberado cuando el
      acelerador empieza a subir. Aplicaciones largas simult\u00e1neas
      son trail-braking deliberado o un error. Entradas en escal\u00f3n
      que rebotan en 1.0 sugieren pedaleo agresivo que los
      neum\u00e1ticos pueden no absorber.</li>
  <li><b>Direcci\u00f3n</b>: trazas suaves de un solo arco = entrada
      bien juzgada. Un patr\u00f3n en sierra de peque\u00f1as
      inversiones es sobre-correcci\u00f3n o un chasis nervioso
      (steer_reversal_rate_hz lo cuantifica).</li>
  <li><b>Diagrama g\u2013g</b>: dibuja accel_x vs accel_y. Un coche
      que usa toda la elipse de fricci\u00f3n rellena el c\u00edrculo;
      los cuadrantes vac\u00edos muestran d\u00f3nde queda agarre sin
      usar (t\u00edpicamente en transiciones de freno-a-curva).</li>
  <li><b>understeer_index</b>: +ve = el tren delantero se va de largo
      (delanteros saturados), \u2212ve = el tren trasero gira m\u00e1s
      de lo que pide la direcci\u00f3n (sobreviraje). D\u00f3nde
      aparecen los picos importa: picos en <i>entrada</i> suelen
      venir de geometr\u00eda delantera blanda o neum\u00e1ticos
      delanteros fr\u00edos; picos en <i>salida</i>, de demasiada
      potencia demasiado pronto o un diferencial suelto.</li>
  <li><b>friction_use_*</b> cerca de 1.0 en una sola rueda = la
      huella de contacto est\u00e1 saturada. Si s\u00f3lo una rueda
      lo hace mientras las dem\u00e1s se quedan atr\u00e1s, el
      reparto de carga est\u00e1 mal (barras estabilizadoras,
      presiones, alineaci\u00f3n, altura).</li>
  <li><b>Objetivos de slip</b>: el agarre longitudinal m\u00e1ximo
      vive en slip_ratio |0,10\u20130,15|; el lateral m\u00e1ximo
      cerca de un \u00e1ngulo de deriva de 6\u20138\u00b0 (tan
      \u2248 0,10\u20130,14). Pasado eso, el neum\u00e1tico desliza
      m\u00e1s de lo que agarra.</li>
  <li><b>Transferencia de peso</b>: transfer_long \u2248
      m\u00b7a_x\u00b7h_cg/batalla, transfer_lat \u2248
      m\u00b7a_y\u00b7h_cg/v\u00eda. Ambas escalan con la altura del
      CG \u2014 bajar la altura es la forma m\u00e1s barata de
      reducir la transferencia.</li>
</ul>

<h2>De d\u00f3nde vienen los datos</h2>
<p>LFS expone tres streams de telemetr\u00eda que Studio escucha:</p>
<ul>
  <li><b>OutSim</b> \u2014 IMU del chasis a alta frecuencia
      (posici\u00f3n, velocidad, aceleraci\u00f3n en 3 ejes en
      m/s\u00b2, velocidad angular en 3 ejes en rad/s, \u00e1ngulos
      Euler, info por rueda con slip ratio, \u00e1ngulo de deriva,
      fuerzas, contacto y recorrido de suspensi\u00f3n).</li>
  <li><b>OutGauge</b> \u2014 estado del salpicadero (RPM, velocidad,
      marcha, acelerador, freno, embrague, freno de mano,
      direcci\u00f3n, combustible, temperaturas de motor/aceite,
      turbo, bitfield show-lights).</li>
  <li><b>InSim</b> \u2014 eventos de sesi\u00f3n (tiempos de vuelta,
      parciales, direcci\u00f3n de carrera, chat).</li>
</ul>
<p>Las entradas necesarias en <code>cfg.txt</code> son
<code>OutSim Mode 2 / Opts 1ff / Delay 1 / IP 127.0.0.1 / Port 30000</code>
y <code>OutGauge Mode 1 / Delay 1 / IP 127.0.0.1 / Port 30001</code>.
InSim se arranca en tiempo de ejecuci\u00f3n dentro de LFS con
<code>/insim 29999</code> (o <code>LFS.exe /insim=29999</code>).</p>
"""


class HelpDialog(QDialog):
    """Modal dialog with a channel-and-interpretation reference."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Channel & telemetry guide"))
        self.resize(960, 720)

        browser = QTextBrowser(self)
        browser.setOpenExternalLinks(False)
        browser.setHtml(self._build_html())
        browser.setStyleSheet(
            "QTextBrowser { font-size: 11pt; }"
            "h1 { margin-top: 14px; }"
            "h2 { margin-top: 18px; border-bottom: 1px solid #555; }"
            "h3 { margin-top: 12px; color: #d8d8d8; }"
            "table { border-collapse: collapse; margin-top: 6px; }"
            "td, th { padding: 4px 8px; border-bottom: 1px solid #333; "
            "vertical-align: top; }"
            "th { text-align: left; }"
            "code { color: #c0c0c0; }"
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(browser, 1)
        layout.addWidget(buttons)

        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_html(self) -> str:
        groups = channels_by_group()
        is_es = current_language() == LANG_SPANISH
        intros = _GROUP_INTRO_ES if is_es else _GROUP_INTRO
        how_to = _HOW_TO_READ_ES if is_es else _HOW_TO_READ
        parts: list[str] = [
            "<h1>{title}</h1>".format(
                title=tr("Channel &amp; telemetry guide"),
            ),
            "<p>{p}</p>".format(
                p=tr(
                    "This panel explains, in plain language, how to "
                    "read every plot and what each channel measures. "
                    "No telemetry background required.",
                ),
            ),
            how_to,
            "<h2>{h}</h2>".format(h=tr("Channels by group")),
        ]
        ordered = [g for g in _GROUP_ORDER if g in groups] + [
            g for g in sorted(groups) if g not in _GROUP_ORDER
        ]
        for group in ordered:
            intro = intros.get(group, "")
            parts.append(f"<h3>{tr(group)}</h3>")
            if intro:
                parts.append(f"<p>{intro}</p>")
            parts.append(
                "<table width='100%'>"
                "<tr><th width='22%'>{ch}</th>"
                "<th width='10%'>{un}</th>"
                "<th>{hw}</th></tr>".format(
                    ch=tr("Channel"),
                    un=tr("Unit"),
                    hw=tr("What it is &amp; how to read it"),
                )
            )
            for info in groups[group]:
                desc = info.description or ""
                interp = info.interpretation or ""
                cell: list[str] = []
                if desc:
                    cell.append(desc)
                if interp:
                    cell.append(f"<i>{interp}</i>")
                cell_html = "<br>".join(cell) if cell else "&mdash;"
                units = info.units or ""
                parts.append(
                    f"<tr><td><b>{info.label}</b><br>"
                    f"<code>{info.column}</code></td>"
                    f"<td>{units}</td>"
                    f"<td>{cell_html}</td></tr>"
                )
            parts.append("</table>")
        return "\n".join(parts)


__all__ = ["HelpDialog"]
