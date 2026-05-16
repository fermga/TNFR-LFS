"""Workspace facade for the Studio.

Re-exports :class:`lfs_telemetry.app.state.WorkspaceState` (which is a
framework-neutral container — pure pandas + pickle, no Dash) so we don't
duplicate code while the legacy app and the Studio coexist.

Once the Dash viewer is removed, ``WorkspaceState`` will move down to
``lfs_telemetry.telemetry.workspace`` and this thin wrapper will be
deleted along with ``lfs_telemetry.app``.
"""

from __future__ import annotations

from ..app.state import WorkspaceState

__all__ = ["WorkspaceState"]
