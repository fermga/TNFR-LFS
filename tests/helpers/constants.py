"""Shared constants used across test modules."""

BASE_NU_F = {
    "tyres": 0.18,
    "suspension": 0.14,
    "chassis": 0.12,
    "brakes": 0.16,
    "transmission": 0.11,
    "track": 0.08,
    "driver": 0.05,
}

SUPPORTED_CAR_MODELS = [
    "XFG",
    "XRG",
    "RB4",
    "FXO",
    "FXR",
    "XRR",
    "FZR",
    "FO8",
    "BF1",
]

__all__ = ["BASE_NU_F", "SUPPORTED_CAR_MODELS"]
