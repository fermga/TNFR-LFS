"""Auto-split from live_modules.py — MH1."""
from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPen,
)

from ..live_data_source import LiveDataSource
from ._base import (
    _LiveModuleWindow,
)


class TyreRiskWindow(_LiveModuleWindow):
    """Per-wheel grip panel for race/long-run management."""

    MODULE_ID = "grip"

    def __init__(
        self, source: LiveDataSource, *, opacity: float = 0.85,
    ) -> None:
        super().__init__(
            source,
            size=(280, 180),
            title="LFS Live - grip",
            opacity=opacity,
        )
        self._grip_smoothed: dict[str, float] = {
            "FL": 0.0,
            "FR": 0.0,
            "RL": 0.0,
            "RR": 0.0,
        }
        self._grip_short: dict[str, float] = {
            "FL": 1.0,
            "FR": 1.0,
            "RL": 1.0,
            "RR": 1.0,
        }
        self._grip_long: dict[str, float] = {
            "FL": 1.0,
            "FR": 1.0,
            "RL": 1.0,
            "RR": 1.0,
        }
        self._trend: dict[str, float] = {
            "FL": 0.0,
            "FR": 0.0,
            "RL": 0.0,
            "RR": 0.0,
        }
        self._state_hold: dict[str, dict[str, int]] = {
            "FL": {"S": 0, "L": 0},
            "FR": {"S": 0, "L": 0},
            "RL": {"S": 0, "L": 0},
            "RR": {"S": 0, "L": 0},
        }
        # 10 Hz snapshot updates -> alpha 0.25 is responsive
        # without flicker.
        self._ema_alpha = 0.25
        self._short_alpha = 0.35
        self._long_alpha = 0.06
        # Keep S/L LEDs lit briefly so fast events are readable.
        self._state_hold_ticks = 4

    @staticmethod
    def _wheel_states(
        row: dict[str, Any],
        *,
        speed_kmh: float,
        brake: float,
        handbrake: float,
    ) -> tuple[bool, bool]:
        """Return (sliding, locked) inferred from live tyre telemetry."""
        touching = bool(row.get("touching", False))
        load_n = float(row.get("load_n") or 0.0)
        if not touching or load_n < 50.0:
            return (False, False)

        slip_frac = abs(float(row.get("slip_frac") or 0.0))
        slip_ratio = abs(float(row.get("slip_ratio") or 0.0))
        tan_slip = abs(float(row.get("tan_slip") or 0.0))

        # S = lateral/combined sliding event.
        sliding = (
            tan_slip >= 0.10
            or (slip_frac >= 0.16 and tan_slip >= 0.06)
            or (slip_frac >= 0.22 and slip_ratio >= 0.10)
        )

        # L = wheel lock tendency while braking (heuristic, no direct
        # lock bit from this telemetry stream).
        braking = max(0.0, float(brake))
        hb = max(0.0, float(handbrake))
        locked = (
            speed_kmh >= 25.0
            and slip_ratio >= 0.22
            and tan_slip <= 0.11
            and (braking >= 0.18 or hb >= 0.20)
        )

        return (sliding, locked)

    def _on_snapshot(self, snap: dict[str, Any]) -> None:
        rows = snap.get("tyres")
        tyres = rows if isinstance(rows, list) else []
        by_corner = {
            str(r.get("corner") or "?"): r
            for r in tyres if isinstance(r, dict)
        }
        speed_kmh = float(snap.get("view_speed_kmh") or 0.0)
        brake = float(snap.get("view_brake") or 0.0)
        handbrake = float(snap.get("view_handbrake") or 0.0)
        for corner in ("FL", "FR", "RL", "RR"):
            hold = self._state_hold.setdefault(corner, {"S": 0, "L": 0})
            hold["S"] = max(0, int(hold.get("S", 0)) - 1)
            hold["L"] = max(0, int(hold.get("L", 0)) - 1)

            row = by_corner.get(corner)
            raw_risk = (
                self._risk_level(row)
                if isinstance(row, dict) else 0.0
            )
            raw_grip = max(0.0, min(1.0, 1.0 - raw_risk))

            # Demand weighting: trend is only meaningful when tyres are
            # actually loaded (cornering/braking/traction).
            demand = 0.0
            if isinstance(row, dict):
                load_n = float(row.get("load_n") or 0.0)
                fx_n = float(row.get("fx_n") or 0.0)
                fy_n = float(row.get("fy_n") or 0.0)
                if load_n > 50.0:
                    demand_ratio = math.hypot(fx_n, fy_n) / load_n
                    demand = max(0.0, min(1.0, (demand_ratio - 1.5) / 7.5))

                sliding_now, locked_now = self._wheel_states(
                    row,
                    speed_kmh=speed_kmh,
                    brake=brake,
                    handbrake=handbrake,
                )
                if sliding_now:
                    hold["S"] = self._state_hold_ticks
                if locked_now:
                    hold["L"] = self._state_hold_ticks

            prev_s = self._grip_short.get(corner, raw_grip)
            prev_l = self._grip_long.get(corner, raw_grip)
            self._grip_short[corner] = (
                prev_s + self._short_alpha * (raw_grip - prev_s)
            )
            self._grip_long[corner] = (
                prev_l + self._long_alpha * (raw_grip - prev_l)
            )
            self._trend[corner] = (
                (self._grip_short[corner] - self._grip_long[corner]) * demand
            )

            prev = self._grip_smoothed.get(corner, raw_grip)
            a = self._ema_alpha
            blended = max(
                0.0,
                min(
                    1.0,
                    0.70 * self._grip_short[corner]
                    + 0.30 * self._grip_long[corner],
                ),
            )
            self._grip_smoothed[corner] = prev + a * (blended - prev)
        self._snap = snap
        self.update()

    @staticmethod
    def _risk_level(row: dict[str, Any]) -> float:
        touching = bool(row.get("touching", False))
        load_n = float(row.get("load_n") or 0.0)
        if not touching or load_n < 50.0:
            return 0.0
        slip_frac = abs(float(row.get("slip_frac") or 0.0))
        slip_ratio = abs(float(row.get("slip_ratio") or 0.0))
        tan_slip = abs(float(row.get("tan_slip") or 0.0))
        fx_n = float(row.get("fx_n") or 0.0)
        fy_n = float(row.get("fy_n") or 0.0)
        temp_c = float(row.get("temp_c") or 0.0)

        # Core degradation proxies.
        slip_frac_term = min(1.0, slip_frac / 0.25)
        slip_ratio_term = min(1.0, slip_ratio / 0.18)
        slip_angle_term = min(1.0, tan_slip / 0.14)

        # Demand/saturation proxy from tyre force magnitude over load.
        # Values around ~1 g should not imply degradation by themselves.
        force_ratio = math.hypot(fx_n, fy_n) / max(1.0, load_n)
        sat_term = min(1.0, max(0.0, (force_ratio - 4.0) / 8.0))

        # Temperature contributes but with lower weight (slow/indirect).
        temp_term = min(1.0, max(0.0, (temp_c - 102.0) / 30.0))

        score = (
            0.34 * slip_frac_term
            + 0.26 * slip_ratio_term
            + 0.18 * slip_angle_term
            + 0.14 * sat_term
            + 0.08 * temp_term
        )
        return score

    @staticmethod
    def _grip_color(grip: float, *, touching: bool) -> QColor:
        if not touching:
            return QColor(120, 120, 130)
        if grip >= 0.65:
            return QColor(120, 230, 140)
        if grip >= 0.35:
            return QColor(255, 220, 60)
        return QColor(255, 90, 90)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)

        rows = self._snap.get("tyres")
        tyres = rows if isinstance(rows, list) else []
        by_corner = {
            str(r.get("corner") or "?"): r
            for r in tyres if isinstance(r, dict)
        }
        order = ("FL", "FR", "RL", "RR")

        p.setPen(QPen(QColor(150, 150, 160)))
        p.setFont(self._font(10, QFont.Weight.Normal))
        p.drawText(
            QRectF(10, 6, self.width() - 20, 18),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            "GRIP",
        )

        if not by_corner:
            p.setPen(QPen(QColor(180, 180, 190)))
            p.setFont(
                QFont("Consolas", self._scale_pt(10), QFont.Weight.Normal)
            )
            p.drawText(
                QRectF(10, 30, self.width() - 20, self.height() - 40),
                int(Qt.AlignmentFlag.AlignCenter),
                "no wheel telemetry",
            )
            return

        card_w = (self.width() - 30) / 2.0
        card_h = (self.height() - 44) / 2.0
        lowest_corner = "--"
        lowest_grip = 1e9

        for idx, corner in enumerate(order):
            row = by_corner.get(corner) or {}
            grip = self._grip_smoothed.get(corner, 0.0)
            col = self._grip_color(
                grip,
                touching=bool(row.get("touching", False)),
            )
            if grip < lowest_grip:
                lowest_grip = grip
                lowest_corner = corner

            r = idx // 2
            c = idx % 2
            x = 10 + c * (card_w + 10)
            y = 26 + r * (card_h + 8)
            rect = QRectF(x, y, card_w, card_h)

            p.setPen(QPen(QColor(60, 60, 70), 1))
            p.setBrush(QColor(24, 24, 30, 220))
            p.drawRoundedRect(rect, 6, 6)

            side = min(rect.width(), rect.height()) - 22
            cx = rect.left() + rect.width() / 2.0
            cy = rect.top() + rect.height() * 0.47
            circle = QRectF(cx - side / 2.0, cy - side / 2.0, side, side)
            p.setPen(QPen(QColor(30, 30, 36), 2))
            p.setBrush(QColor(34, 34, 40))
            p.drawEllipse(circle)
            p.setPen(QPen(QColor(20, 20, 24), 1))
            p.setBrush(col)
            p.drawEllipse(circle.adjusted(2, 2, -2, -2))

            p.setPen(QPen(QColor(170, 170, 180)))
            p.setFont(self._font(8, QFont.Weight.Normal))
            p.drawText(
                QRectF(
                    rect.left() + 6,
                    rect.top() + 2,
                    rect.width() - 12,
                    12,
                ),
                int(
                    Qt.AlignmentFlag.AlignLeft
                    | Qt.AlignmentFlag.AlignVCenter
                ),
                corner,
            )

            # S/L realtime state badges (Sliding / Locked).
            hold = self._state_hold.get(corner, {})
            led_h = max(10.0, min(14.0, rect.height() * 0.24))
            led_w = led_h + 2.0
            y_led = rect.top() + 3.0
            x_l = rect.right() - 6.0 - led_w
            x_s = x_l - 3.0 - led_w

            for ch, x_led, active_col, key in (
                ("S", x_s, QColor(255, 180, 70), "S"),
                ("L", x_l, QColor(255, 95, 95), "L"),
            ):
                active = int(hold.get(key, 0)) > 0
                led_rect = QRectF(x_led, y_led, led_w, led_h)
                p.setPen(QPen(QColor(28, 28, 32), 1))
                p.setBrush(active_col if active else QColor(48, 48, 56))
                p.drawRoundedRect(led_rect, 3, 3)
                p.setPen(QPen(QColor(20, 20, 24) if active
                              else QColor(155, 155, 165)))
                p.setFont(QFont("Consolas", self._scale_pt(7),
                                QFont.Weight.Bold))
                p.drawText(
                    led_rect,
                    int(Qt.AlignmentFlag.AlignCenter),
                    ch,
                )

            pct = round(grip * 100.0)
            p.setPen(QPen(QColor(15, 15, 18)))
            p.setFont(QFont("Consolas", self._scale_pt(11), QFont.Weight.Bold))
            p.drawText(
                circle,
                int(Qt.AlignmentFlag.AlignCenter),
                f"{pct}%",
            )

            temp_c = row.get("temp_c")
            temp_txt = f"{float(temp_c):.0f}°C" if temp_c is not None else "--"
            tr = self._trend.get(corner, 0.0)
            if tr > 0.01:
                trend_txt = "↗"
            elif tr < -0.01:
                trend_txt = "↘"
            else:
                trend_txt = "→"

            p.setPen(QPen(QColor(190, 190, 205)))
            p.setFont(
                QFont("Consolas", self._scale_pt(9), QFont.Weight.Normal)
            )
            p.drawText(
                QRectF(
                    rect.left() + 6,
                    rect.bottom() - 16,
                    rect.width() - 12,
                    12,
                ),
                int(Qt.AlignmentFlag.AlignCenter),
                f"T {temp_txt}  {trend_txt}",
            )

        p.setPen(QPen(QColor(140, 140, 150)))
        p.setFont(self._font(8, QFont.Weight.Normal))
        p.drawText(
            QRectF(10, self.height() - 16, self.width() - 20, 12),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            f"low: {lowest_corner}",
        )


