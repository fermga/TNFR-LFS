"""Base classes for TNFR × LFS plugin implementations."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Sequence

if False:  # pragma: no cover - type-checker hint without runtime import
    from tnfr_lfs.core.epi import NaturalFrequencySnapshot


@dataclass
class TNFRPlugin(ABC):
    """Abstract plugin contract for TNFR × LFS integrations.

    Implementations provide identifying metadata together with dynamic state
    derived from natural-frequency and coherence computations.  The base class
    offers lifecycle hooks so concrete plugins can react whenever the spectral
    inputs change.
    """

    identifier: str
    """Stable identifier for the plugin instance."""

    display_name: str
    """Human readable label describing the plugin."""

    version: str
    """Semantic version exposed by the plugin implementation."""

    description: str = ""
    """Optional free-form description presented in UIs or logs."""

    _nu_f_by_node: MutableMapping[str, float] = field(
        default_factory=dict, init=False, repr=False
    )
    _coherence_index: float = field(default=0.0, init=False, repr=False)

    @property
    def nu_f(self) -> Mapping[str, float]:
        """Current natural frequency mapping keyed by subsystem node."""

        return dict(self._nu_f_by_node)

    @property
    def coherence_index(self) -> float:
        """Latest coherence index associated with the plugin."""

        return self._coherence_index

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """Clear the dynamic plugin state and trigger :meth:`on_reset`."""

        self._nu_f_by_node.clear()
        self._coherence_index = 0.0
        self.on_reset()

    def on_reset(self) -> None:
        """Hook executed after :meth:`reset_state` clears plugin state."""

    # Natural-frequency handling -------------------------------------------------
    def apply_natural_frequency_snapshot(
        self, snapshot: "NaturalFrequencySnapshot"
    ) -> None:
        """Update the ν_f mapping and invoke :meth:`on_nu_f_updated`.

        Parameters
        ----------
        snapshot:
            The spectral snapshot returned by
            :class:`~tnfr_lfs.core.epi.NaturalFrequencyAnalyzer`.
        """

        self._nu_f_by_node = dict(snapshot.by_node)
        self._coherence_index = float(snapshot.coherence_index)
        self.on_nu_f_updated(snapshot)

    def on_nu_f_updated(self, snapshot: "NaturalFrequencySnapshot") -> None:
        """Hook executed after a new natural-frequency snapshot is applied."""

    # Coherence handling --------------------------------------------------------
    def apply_coherence_index(
        self, coherence_index: float, *, series: Sequence[float] | None = None
    ) -> None:
        """Persist the coherence index and invoke :meth:`on_coherence_updated`."""

        self._coherence_index = float(coherence_index)
        self.on_coherence_updated(self._coherence_index, series)

    def apply_coherence_series(self, series: Sequence[float]) -> None:
        """Store coherence information derived from an operator output."""

        coherence_index = float(series[-1]) if series else 0.0
        self.apply_coherence_index(coherence_index, series=series)

    def on_coherence_updated(
        self, coherence_index: float, series: Sequence[float] | None = None
    ) -> None:
        """Hook executed whenever the coherence state changes."""
