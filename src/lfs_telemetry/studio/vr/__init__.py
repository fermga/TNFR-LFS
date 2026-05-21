"""VR delivery sinks for the live overlay pipeline.

Currently exposes :class:`OpenVROverlaySink`, the SteamVR-based path.
Other backends (OpenXR-native, Oculus-native via OpenComposite) can
plug in here behind the same small interface.
"""

from .openvr_overlay import OpenVROverlaySink, OverlayPose

__all__ = ["OpenVROverlaySink", "OverlayPose"]
