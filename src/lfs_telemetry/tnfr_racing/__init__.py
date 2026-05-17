"""TNFR Setup Advisor — isolated package.

Public exports:
    SetupAdvisor      — facade for the advisor pipeline.
    ProposedSetup     — typed result with deltas + rationale.
    AdvisorConfig     — thresholds and runtime knobs.

Internal modules wrap the canonical :mod:`tnfr` engine; nothing outside this
package (except ``studio/widgets/setup_advisor_tab.py`` and ``cli.py``) should
import from ``lfs_telemetry.tnfr_racing.*`` or ``tnfr.*`` directly.
"""
from __future__ import annotations

from .advisor import ProposedSetup, SetupAdvisor
from .config import AdvisorConfig

__all__ = ["SetupAdvisor", "ProposedSetup", "AdvisorConfig"]
