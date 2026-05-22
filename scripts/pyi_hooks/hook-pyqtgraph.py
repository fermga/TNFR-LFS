# PyInstaller override hook for ``pyqtgraph``.
#
# The bundled hook (``PyInstaller/hooks/hook-pyqtgraph.py``) calls
# ``collect_submodules('pyqtgraph')`` without filtering, which causes
# the submodule walker to import every subpackage. Two of them depend
# on optional third-party libraries we never install:
#
#   * ``pyqtgraph.opengl``  -> requires ``PyOpenGL`` (we use only 2D
#     plots, the optional 3D backend is dead weight).
#   * ``pyqtgraph.jupyter`` -> requires ``jupyter_rfb`` (we ship a
#     desktop GUI, not a notebook widget).
#
# Without filtering, PyInstaller logs a noisy
# "Failed to collect submodules ... ModuleNotFoundError" for each
# missing dependency on every build. This override applies the same
# collection logic with those subtrees skipped at the source.
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

_SKIP_PREFIXES = ("pyqtgraph.opengl", "pyqtgraph.jupyter")

# ``on_error="ignore"`` suppresses the ImportError trace that
# ``pkgutil.walk_packages`` reports when a subpackage fails to import
# (e.g. ``pyqtgraph.jupyter`` requires ``jupyter_rfb``). Without it,
# PyInstaller would still log the warning even after our filter drops
# the name from the final hidden-import list.
hiddenimports = collect_submodules(
    "pyqtgraph",
    filter=lambda name: not name.startswith(_SKIP_PREFIXES),
    on_error="ignore",
)
datas = collect_data_files("pyqtgraph")
