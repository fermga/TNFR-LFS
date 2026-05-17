"""Lightweight in-process i18n for the Studio.

Avoids the round-trip through ``pyside6-lupdate`` / ``.qm`` files by
installing a custom :class:`QTranslator` whose ``translate`` method
looks up source strings in a Python dictionary.

Usage from any widget::

    from .i18n import tr
    self._btn = QPushButton(tr("Save"))

Or, equivalent inside a ``QObject`` subclass::

    self._btn = QPushButton(self.tr("Save"))

Both routes go through :class:`DictTranslator.translate` and therefore
return the localised string when the active language is Spanish, and
the original English string otherwise.
"""

from __future__ import annotations

from typing import Final

from PySide6.QtCore import (
    QCoreApplication,
    QLocale,
    QSettings,
    QTranslator,
)

from ..lfs_paths import QSETTINGS_APP as APP, QSETTINGS_ORG as ORG

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

_SETTINGS_KEY: Final[str] = "ui/language"
LANG_ENGLISH: Final[str] = "en"
LANG_SPANISH: Final[str] = "es"
AVAILABLE_LANGS: Final[tuple[str, ...]] = (LANG_ENGLISH, LANG_SPANISH)

# Module-level translator kept alive by reference so Qt doesn't drop
# it. Installed by :func:`install_translator`.
_translator: "DictTranslator | None" = None


def tr(text: str, *, context: str = "app") -> str:
    """Translate ``text`` via the global translator.

    Safe to call before a :class:`QCoreApplication` exists: it then
    returns ``text`` unchanged.
    """
    app = QCoreApplication.instance()
    if app is None:
        return text
    return QCoreApplication.translate(context, text)


def current_language() -> str:
    s = QSettings(ORG, APP)
    raw = s.value(_SETTINGS_KEY, "")
    lang = str(raw or "").lower()
    if lang in AVAILABLE_LANGS:
        return lang
    # Auto-detect from system locale.
    sys_lang = QLocale.system().name().split("_")[0].lower()
    return LANG_SPANISH if sys_lang == "es" else LANG_ENGLISH


def set_language(lang: str) -> None:
    """Persist the user's language choice. Takes effect on restart."""
    if lang not in AVAILABLE_LANGS:
        lang = LANG_ENGLISH
    QSettings(ORG, APP).setValue(_SETTINGS_KEY, lang)


def install_translator(app: QCoreApplication) -> None:
    """Install the dict-backed translator on ``app``.

    Idempotent: removes the previously installed instance first.
    """
    global _translator
    if _translator is not None:
        app.removeTranslator(_translator)
        _translator = None
    lang = current_language()
    if lang == LANG_ENGLISH:
        # No translator needed; fall through to the source strings.
        return
    _translator = DictTranslator(_DICTS.get(lang, {}))
    app.installTranslator(_translator)


# ---------------------------------------------------------------------
# QTranslator backed by a Python dict
# ---------------------------------------------------------------------


class DictTranslator(QTranslator):
    """Trivial translator that returns ``mapping[source]`` if present."""

    def __init__(self, mapping: dict[str, str]) -> None:
        super().__init__()
        self._map = mapping

    def translate(  # type: ignore[override]
        self,
        context: bytes | str,
        sourceText: bytes | str,
        disambiguation: bytes | str | None = None,
        n: int = -1,
    ) -> str:
        if isinstance(sourceText, bytes):
            key = sourceText.decode("utf-8", "replace")
        else:
            key = sourceText
        return self._map.get(key, "")


# ---------------------------------------------------------------------
# Spanish dictionary
# ---------------------------------------------------------------------
#
# Keep entries alphabetised within each block to make merge conflicts
# obvious. Every key is the *exact* English source string as it appears
# in the code (case + punctuation matter).

_ES: dict[str, str] = {
    # --- Window / app -------------------------------------------------
    "About": "Acerca de",
    "Quit": "Salir",
    "&File": "&Archivo",
    "&View": "&Ver",
    "&Tools": "&Herramientas",
    "&Help": "Ay&uda",
    "&Quit": "&Salir",
    "&Language": "&Idioma",
    "English": "English",
    "Spanish": "Castellano",
    "Restart required": "Reinicio necesario",
    "Language will change the next time you start "
    "the application.":
        "El idioma cambiar\u00e1 la pr\u00f3xima vez que inicies la "
        "aplicaci\u00f3n.",

    # --- File menu ----------------------------------------------------
    "Open Workspace\u2026": "Abrir espacio de trabajo\u2026",
    "Refresh Captures": "Refrescar capturas",
    "Clear Lap Cache": "Vaciar cach\u00e9 de vueltas",
    "Choose workspace folder": "Elige la carpeta del espacio de trabajo",
    "Lap cache cleared.": "Cach\u00e9 de vueltas vaciada.",

    # --- View menu ----------------------------------------------------
    "Reset Layout": "Restablecer disposici\u00f3n",

    # --- Tools menu ---------------------------------------------------
    "Configure LFS\u2026": "Configurar LFS\u2026",
    "Patch LFS cfg.txt with the OutSim/OutGauge/InSim settings "
    "required by lfs-telemetry.":
        "Parchea cfg.txt de LFS con los ajustes de OutSim/OutGauge/InSim "
        "necesarios para lfs-telemetry.",

    # --- Help menu ----------------------------------------------------
    "Channel guide\u2026": "Gu\u00eda de canales\u2026",
    "Open the telemetry guide: what each channel measures "
    "and how to read it, in plain language.":
        "Abre la gu\u00eda de telemetr\u00eda: qu\u00e9 mide cada canal "
        "y c\u00f3mo interpretarlo, en lenguaje claro.",

    # --- Dock titles --------------------------------------------------
    "Captures": "Capturas",
    "Telemetry": "Telemetr\u00eda",
    "Track map": "Mapa del circuito",
    "Elevation": "Altimetr\u00eda",
    "Race dashboard": "Panel de carrera",

    # --- Center tabs --------------------------------------------------
    "Dampers": "Amortiguadores",
    "Sectors": "Sectores",
    "Stint": "Stint",
    "Capture": "Captura",
    "Overlay": "Overlay",

    # --- Configure LFS dialog ----------------------------------------
    "Configure LFS for telemetry":
        "Configurar LFS para telemetr\u00eda",
    "Configure LFS": "Configurar LFS",
    "lfs-telemetry needs a few <b>OutSim</b>, <b>OutGauge</b> and "
    "<b>InSim</b> entries in your LFS <code>cfg.txt</code>. Point "
    "the app at your LFS install folder and click "
    "<i>Patch cfg.txt automatically</i>, or copy the block below "
    "into the file by hand.<br><br>"
    "<b>LFS must be closed</b> while the file is patched, "
    "otherwise it will overwrite your changes on exit.":
        "lfs-telemetry necesita unas pocas entradas de <b>OutSim</b>, "
        "<b>OutGauge</b> e <b>InSim</b> en tu <code>cfg.txt</code> de "
        "LFS. Indica la carpeta de instalaci\u00f3n de LFS y pulsa "
        "<i>Parchear cfg.txt autom\u00e1ticamente</i>, o copia el "
        "bloque inferior dentro del archivo a mano.<br><br>"
        "<b>LFS debe estar cerrado</b> durante el parcheo, de lo "
        "contrario sobrescribir\u00e1 tus cambios al salir.",
    "LFS folder:": "Carpeta de LFS:",
    "Browse\u2026": "Examinar\u2026",
    "Patch cfg.txt automatically":
        "Parchear cfg.txt autom\u00e1ticamente",
    "Copy snippet": "Copiar fragmento",
    "Manual snippet (paste at the end of cfg.txt):":
        "Fragmento manual (p\u00e9galo al final de cfg.txt):",
    "Select LFS install folder":
        "Selecciona la carpeta de instalaci\u00f3n de LFS",
    "Snippet copied to clipboard. Paste it at the end of cfg.txt.":
        "Fragmento copiado al portapapeles. P\u00e9galo al final de "
        "cfg.txt.",
    "Please choose your LFS install folder first.":
        "Selecciona primero la carpeta de instalaci\u00f3n de LFS.",
    "{path}\n\nDoes not look like an LFS install folder "
    "(no LFS.exe or cfg.txt found).":
        "{path}\n\nNo parece una carpeta de instalaci\u00f3n de LFS "
        "(no se encontr\u00f3 LFS.exe ni cfg.txt).",
    "Could not write cfg.txt:\n\n{error}\n\n"
    "Make sure LFS is closed and that the file is not "
    "read-only.":
        "No se pudo escribir cfg.txt:\n\n{error}\n\n"
        "Aseg\u00farate de que LFS est\u00e1 cerrado y de que el "
        "archivo no es de s\u00f3lo lectura.",
    "\n\nDone. Launch LFS and enter a session.":
        "\n\nListo. Inicia LFS y entra en una sesi\u00f3n.",
    "<i>No folder selected.</i>":
        "<i>Sin carpeta seleccionada.</i>",
    "Folder does not look like an LFS install: {path}":
        "La carpeta no parece una instalaci\u00f3n de LFS: {path}",
    "{cfg} does not exist yet \u2014 launch LFS once to "
    "generate cfg.txt, then quit and try again.":
        "{cfg} todav\u00eda no existe \u2014 inicia LFS una vez para "
        "generar cfg.txt, luego ci\u00e9rralo y vuelve a intentarlo.",
    "Ready: {cfg}": "Listo: {cfg}",

    # --- About dialog -------------------------------------------------
    "LFS Telemetry Studio {version}\n"
    "Native dockable analyser built on PySide6 + pyqtgraph.\n\n"
    "To stream telemetry, LFS needs OutSim/OutGauge/InSim entries "
    "in cfg.txt.\n"
    "Use \u201cTools \u2192 Configure LFS\u2026\u201d to patch "
    "them automatically or copy the snippet manually.":
        "LFS Telemetry Studio {version}\n"
        "Analizador nativo con docks construido sobre PySide6 + "
        "pyqtgraph.\n\nPara enviar telemetr\u00eda, LFS necesita "
        "entradas de OutSim/OutGauge/InSim en cfg.txt.\nUsa "
        "\u201cHerramientas \u2192 Configurar LFS\u2026\u201d para "
        "parchearlas autom\u00e1ticamente o copia el fragmento a mano.",

    # --- Dampers tab --------------------------------------------------
    "Damper velocity {corner}":
        "Velocidad amortiguador {corner}",
    "samples": "muestras",
    "<i>no data</i>": "<i>sin datos</i>",
    "Low-speed boundary: ": "Umbral de baja velocidad: ",
    "No lap loaded.": "Ninguna vuelta cargada.",
    "Loading {name}\u2026": "Cargando {name}\u2026",
    "{name} \u2014 missing damper data for: {wheels}":
        "{name} \u2014 faltan datos de amortiguador en: {wheels}",
    "{name} \u2014 low-speed \u00b1{boundary:.0f} mm/s":
        "{name} \u2014 baja velocidad \u00b1{boundary:.0f} mm/s",
    "  \u2502  compare B: {name}":
        "  \u2502  comparar B: {name}",

    # --- Sectors tab --------------------------------------------------
    "No laps selected.": "Ninguna vuelta seleccionada.",
    "Sector times": "Tiempos por sector",
    "Lap #": "Vuelta n\u00ba",
    "Loading {n} lap(s)\u2026": "Cargando {n} vuelta(s)\u2026",
    "Sectors unavailable (no usable distance/time data).":
        "Sectores no disponibles (sin datos utilizables de "
        "distancia/tiempo).",
    "InSim splits": "Parciales InSim",
    "uniform \u00d7{n}": "uniforme \u00d7{n}",
    "<b>{n}</b> lap(s) \u00b7 <b>{secs}</b> sectors ({src}) "
    "\u00b7 theoretical best <b>{best} s</b>":
        "<b>{n}</b> vuelta(s) \u00b7 <b>{secs}</b> sectores ({src}) "
        "\u00b7 mejor te\u00f3rica <b>{best} s</b>",

    # --- Stint tab ----------------------------------------------------
    "Lap times": "Tiempos por vuelta",
    "Fuel": "Combustible",
    "Tyre temp end-of-lap": "Temperatura de neum\u00e1ticos fin de vuelta",
    "Peak vertical load (suspension)":
        "Carga vertical m\u00e1xima (suspensi\u00f3n)",
    "Friction use p95 (circle saturation)":
        "Uso de fricci\u00f3n p95 (saturaci\u00f3n del c\u00edrculo)",
    "Damper work \u2014 RMS shaft speed":
        "Trabajo de amortiguadores \u2014 velocidad RMS del v\u00e1stago",
    "Stint build failed: {error}":
        "Fall\u00f3 la construcci\u00f3n del stint: {error}",
    "used / lap": "usado / vuelta",
    "remaining @ end": "restante al final",

    # --- Capture tab --------------------------------------------------
    "TCP port LFS uses for InSim. Enable it inside LFS at "
    "runtime with  /insim 29999  in the console (or launch "
    "LFS.exe with /insim=29999). InSim has no cfg.txt entry.":
        "Puerto TCP que LFS usa para InSim. Act\u00edvalo en LFS en "
        "tiempo de ejecuci\u00f3n con  /insim 29999  en la consola "
        "(o inicia LFS.exe con /insim=29999). InSim no tiene entrada "
        "en cfg.txt.",
    "Filename stem:": "Ra\u00edz del nombre de archivo:",
    "InSim host:": "Host InSim:",
    "InSim port:": "Puerto InSim:",
    "OutSim port:": "Puerto OutSim:",
    "OutGauge port:": "Puerto OutGauge:",
    "Start": "Iniciar",
    "Stop": "Detener",
    "LFS InSim status: idle": "Estado InSim de LFS: inactivo",
    "LFS InSim status: waiting for connection":
        "Estado InSim de LFS: esperando conexi\u00f3n",
    "LFS InSim status: connected": "Estado InSim de LFS: conectado",
    "Idle.": "Inactivo.",
    "Laps recorded: 0": "Vueltas grabadas: 0",
    "Laps recorded: {n}{out_tag}": "Vueltas grabadas: {n}{out_tag}",
    " (+ out-lap)": " (+ vuelta de salida)",
    "Workspace: {path}": "Espacio de trabajo: {path}",
    "Records LFS UDP telemetry. You can press Start at any "
    "time (menu, pre-race countdown, pit, or already on "
    "track): the capture waits for LFS InSim to come up and "
    "only begins recording when the car actually starts "
    "moving. Every completed lap (out-lap included) is saved "
    "when you press Stop. Enable InSim in LFS first: "
    "<code>/insim 29999</code>.":
        "Graba la telemetr\u00eda UDP de LFS. Puedes pulsar Iniciar "
        "en cualquier momento (men\u00fa, cuenta atr\u00e1s previa a "
        "la carrera, pit o ya en pista): la captura espera a que "
        "LFS InSim est\u00e9 disponible y s\u00f3lo comienza a "
        "grabar cuando el coche se mueve. Cada vuelta completada "
        "(incluida la de salida) se guarda al pulsar Detener. "
        "Activa antes InSim en LFS: <code>/insim 29999</code>.",
    "Log:": "Registro:",
    "Already running.": "Ya est\u00e1 en ejecuci\u00f3n.",
    "Start failed: {error}": "Fall\u00f3 el inicio: {error}",
    " \u2014 waiting for LFS InSim":
        " \u2014 esperando a LFS InSim",
    " \u2014 waiting for car to move":
        " \u2014 esperando a que el coche se mueva",
    "\u25cf Recording \u2192 {file}{state}":
        "\u25cf Grabando \u2192 {file}{state}",
    "\u25a0 Finished (code={code}) \u2192 {file}":
        "\u25a0 Finalizado (c\u00f3digo={code}) \u2192 {file}",

    # --- Live tab -----------------------------------------------------
    "Overlay modules \u2014 drag body to move, "
    "drag bottom-right corner to resize, right-click to "
    "reset. Position and opacity persist per module.":
        "M\u00f3dulos de overlay \u2014 arrastra el cuerpo para "
        "mover, la esquina inferior derecha para redimensionar, "
        "clic derecho para restablecer. La posici\u00f3n y la "
        "opacidad se guardan por m\u00f3dulo.",
    "Opacity for this overlay module \u2014 persisted "
    "between sessions.":
        "Opacidad de este m\u00f3dulo de overlay \u2014 se guarda "
        "entre sesiones.",
    "Scale:": "Escala:",
    "Red:": "Rojo:",
    "Yellow:": "Amarillo:",
    "White:": "Blanco:",
    "Radar": "Radar",
    "Full scale (\u00b1):": "Escala completa (\u00b1):",
    "Delta bar": "Barra de delta",
    "Redline:": "L\u00ednea roja:",
    "RPM": "RPM",
    "G-meter full scale:": "Escala completa del g-metro:",
    "G-meter": "G-metro",
    "Start a capture, then tick the modules you want. "
    "Each window is frameless, stays on top, and remembers "
    "its last position and opacity.":
        "Inicia una captura y luego marca los m\u00f3dulos que "
        "quieras. Cada ventana es sin bordes, se mantiene encima y "
        "recuerda su \u00faltima posici\u00f3n y opacidad.",

    # --- Channels dock ------------------------------------------------
    "Clear all": "Limpiar todo",
    "Defaults": "Por defecto",
    "Friction circle": "C\u00edrculo de fricci\u00f3n",
    "Show the channels needed to read a friction-circle / "
    "g-g diagram: long+lat acceleration and per-wheel "
    "\u03bc-use.":
        "Muestra los canales para leer un diagrama de c\u00edrculo "
        "de fricci\u00f3n / g-g: aceleraci\u00f3n longitudinal y "
        "lateral, y uso de \u03bc por rueda.",
    "Load transfer": "Transferencia de carga",
    "Show vertical-load per wheel plus longitudinal / "
    "lateral transfer.":
        "Muestra la carga vertical por rueda y la transferencia "
        "longitudinal / lateral.",
    "Expand": "Expandir",
    "Collapse": "Colapsar",
    "Saved channel selections. Pick one to apply it.":
        "Selecciones de canales guardadas. Elige una para aplicarla.",
    "Save\u2026": "Guardar\u2026",
    "Save the current channel selection as a named preset.":
        "Guarda la selecci\u00f3n actual de canales como un "
        "preajuste con nombre.",
    "Delete": "Eliminar",
    "Delete the selected preset.":
        "Elimina el preajuste seleccionado.",
    "Preset:": "Preajuste:",
    "(no preset)": "(sin preajuste)",
    "Maximum {n} channels at once \u2014 untick one "
    "before adding another.":
        "M\u00e1ximo {n} canales a la vez \u2014 desmarca uno "
        "antes de a\u00f1adir otro.",
    "{n} / {max} channels selected.":
        "{n} / {max} canales seleccionados.",
    "Save preset": "Guardar preajuste",
    "Tick at least one channel before saving a preset.":
        "Marca al menos un canal antes de guardar un preajuste.",
    "Save channel preset": "Guardar preajuste de canales",
    "Preset name (e.g. Qualifying, Race start, Brake "
    "balance):":
        "Nombre del preajuste (p. ej. Clasificaci\u00f3n, Salida "
        "de carrera, Reparto de frenada):",
    "Overwrite preset?": "\u00bfSobrescribir preajuste?",
    "A preset named \u2018{name}\u2019 already exists. "
    "Overwrite it?":
        "Ya existe un preajuste llamado \u2018{name}\u2019. "
        "\u00bfSobrescribirlo?",
    "Delete preset?": "\u00bfEliminar preajuste?",
    "Delete preset \u2018{name}\u2019?":
        "\u00bfEliminar el preajuste \u2018{name}\u2019?",

    # --- Captures dock ------------------------------------------------
    "Filter (file, car, track)\u2026":
        "Filtrar (archivo, coche, circuito)\u2026",
    "Refresh": "Refrescar",
    "0 captures": "0 capturas",
    "{n} captures": "{n} capturas",

    # --- Race dashboard dock ------------------------------------------
    "Position": "Posici\u00f3n",
    "Lap": "Vuelta",
    "Current lap": "Vuelta actual",
    "Last lap": "\u00daltima vuelta",
    "Best lap": "Mejor vuelta",
    "Predicted": "Estimada",
    "\u0394 vs best": "\u0394 vs mejor",
    "SPB": "SPB",
    "Avg (stint)": "Media (stint)",
    "Avg ({mode})": "Media ({mode})",
    "stint": "stint",
    "clean": "limpia",
    "total": "total",
    "Gap ahead": "Hueco delante",
    "Gap behind": "Hueco detr\u00e1s",
    "Fuel laps left": "Vueltas de combustible restantes",
    "Speed": "Velocidad",
    "Gear": "Marcha",
    "Timing": "Cron\u00f3metros",
    "Gaps to rivals": "Huecos a rivales",
    "Fuel / Drive": "Combustible / Conducci\u00f3n",
    "Waiting for capture\u2026": "Esperando captura\u2026",
    "weather {value}": "tiempo {value}",
    "race": "carrera",
    "idle": "inactivo",
    "ARMED": "ARMADO",
    "off": "apagado",
    "capture {state} \u00b7 {n} samples":
        "captura {state} \u00b7 {n} muestras",

    # --- Help dialog --------------------------------------------------
    "Channel & telemetry guide": "Gu\u00eda de canales y telemetr\u00eda",
    "Channel &amp; telemetry guide":
        "Gu\u00eda de canales y telemetr\u00eda",
    "This panel explains, in plain language, how to "
    "read every plot and what each channel measures. "
    "No telemetry background required.":
        "Este panel explica, en lenguaje claro, c\u00f3mo leer cada "
        "gr\u00e1fico y qu\u00e9 mide cada canal. No hace falta saber "
        "de telemetr\u00eda.",
    "Channels by group": "Canales por grupo",
    "Channel": "Canal",
    "Unit": "Unidad",
    "What it is &amp; how to read it":
        "Qu\u00e9 es y c\u00f3mo leerlo",
    "Driver": "Piloto",
    "Vehicle": "Veh\u00edculo",
    "Engine": "Motor",
    "Chassis": "Chasis",
    "Suspension": "Suspensi\u00f3n",
    "Tyre": "Neum\u00e1ticos",
    "Derived": "Derivados",
    "Track": "Circuito",
    "Aids": "Ayudas",
    "Context": "Contexto",
}


_DICTS: dict[str, dict[str, str]] = {LANG_SPANISH: _ES}


__all__ = [
    "AVAILABLE_LANGS",
    "LANG_ENGLISH",
    "LANG_SPANISH",
    "current_language",
    "install_translator",
    "set_language",
    "tr",
]
