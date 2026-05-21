"""Shared mutable state for the cli subcommands.

Lives here so signal handlers (set up in :mod:`._common`) and the
capture coroutine (:mod:`.capture`) can share the same flags without
import-cycles.
"""
from __future__ import annotations

import asyncio

STOP_REQUESTED: bool = False
CAPTURE_LOOP: asyncio.AbstractEventLoop | None = None
CAPTURE_TASK: asyncio.Task | None = None
