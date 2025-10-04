"""Future Live for Speed native exporter utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .setup_plan import SetupPlan, serialise_setup_plan


FEATURE_FLAG_LFS_NATIVE_EXPORT = bool(os.getenv("TNFR_LFS_NATIVE_EXPORT"))


@dataclass(frozen=True)
class NativeSetupVector:
    """Lightweight container for serialised setup changes.

    This mirrors the information required to convert a :class:`SetupPlan`
    into the binary/text representation consumed directly by Live for Speed.
    """

    car_model: str
    session: str | None
    decision_vector: Mapping[str, float]


def build_native_vector(plan: SetupPlan) -> NativeSetupVector:
    """Extract the decision vector from a setup plan."""

    payload = serialise_setup_plan(plan)
    return NativeSetupVector(
        car_model=str(payload.get("car_model", "")),
        session=payload.get("session"),
        decision_vector={
            change["parameter"]: float(change.get("delta", 0.0))
            for change in payload.get("changes", [])
            if change.get("parameter")
        },
    )


def encode_native_setup(plan: SetupPlan) -> bytes:
    """Encode a setup plan into the native LFS payload.

    The implementation is guarded behind ``TNFR_LFS_NATIVE_EXPORT`` to avoid
    surprising side effects while the mapping logic is still a work in
    progress.
    """

    if not FEATURE_FLAG_LFS_NATIVE_EXPORT:  # pragma: no cover - feature gated
        raise RuntimeError(
            "Native LFS exporter is disabled. Set TNFR_LFS_NATIVE_EXPORT=1 to enable."
        )

    # TODO: Map ``decision_vector`` into the native LFS binary/text format.
    # This requires mirroring the structure of ``data\setups\*.set`` files or
    # interfacing with LFS utilities to ensure compatibility.
    vector = build_native_vector(plan)
    raise NotImplementedError(
        "Serialisation of NativeSetupVector to the LFS format is pending."
    )


__all__ = [
    "FEATURE_FLAG_LFS_NATIVE_EXPORT",
    "NativeSetupVector",
    "build_native_vector",
    "encode_native_setup",
]
