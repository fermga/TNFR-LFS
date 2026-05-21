"""Auto-split from live_modules.py — MH1."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPen,
)

from ..live_data_source import LiveDataSource
from ._base import (
    _fmt_clock,
    _fmt_delta,
    _fmt_gap,
    _LiveModuleWindow,
)


class SessionInfoWindow(_LiveModuleWindow):
    """Compact dynamic session summary for the floating overlay."""

    MODULE_ID = "session_info"

    def __init__(
        self, source: LiveDataSource, *, opacity: float = 0.85,
    ) -> None:
        super().__init__(
            source,
            size=(360, 200),
            title="LFS Live - session info",
            opacity=opacity,
        )
        self._compact = self._load_compact_mode(default=False)

    def _load_compact_mode(self, *, default: bool) -> bool:
        raw = self._settings().value(self._settings_key("compact"), None)
        if raw is None:
            return default
        if isinstance(raw, bool):
            return raw
        txt = str(raw).strip().lower()
        return txt in {"1", "true", "yes", "on"}

    def _save_compact_mode(self) -> None:
        self._settings().setValue(
            self._settings_key("compact"), bool(self._compact),
        )

    def set_compact_mode(self, on: bool) -> None:
        self._compact = bool(on)
        self._save_compact_mode()
        # Re-trigger sizing logic so switching to detailed mode grows
        # the window immediately (without waiting for the next 10 Hz
        # snapshot) and switching back to compact restores the small
        # default footprint.
        if self._compact:
            self.resize(self.width(), self._DETAILED_MIN_H)
        else:
            self._on_snapshot(self._snap)
        self.update()

    def compact_mode(self) -> bool:
        return bool(self._compact)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.set_compact_mode(not self._compact)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # Layout constants for the detailed leaderboard. Minimum/maximum
    # window heights so the auto-resize on snapshot is bounded; the
    # actual per-element sizing is computed proportionally inside
    # :meth:`paintEvent` from ``self.width()``/``self.height()`` and
    # the number of standings rows, so the overlay reflows to whatever
    # footprint the user gives it (you no longer need a huge window
    # to read the leaderboard, it auto-shrinks fonts to fit).
    _DETAILED_MIN_H = 200
    _DETAILED_MAX_H = 720
    # Heuristic minimum pixels per leaderboard row before we'd rather
    # grow the window (used by :meth:`_on_snapshot`).
    _ROW_HINT_PX = 14
    # Pixels reserved at the top for header (SESSION / POS / times /
    # AHEAD-BEHIND / "LEADERBOARD" label) when sizing the window.
    _HEADER_HINT_PX = 110
    _FOOTER_HINT_PX = 24

    def _on_snapshot(self, snap: dict[str, Any]) -> None:
        # Resize the window so every classified driver fits when in
        # detailed mode. Compact mode keeps its fixed size so users
        # who pin it as a small HUD don't see it grow unexpectedly.
        # If the user shrank the window manually below the hint, we
        # honour their choice — :meth:`paintEvent` will scale fonts
        # down to fit anyway.
        if not self._compact:
            standings = snap.get("standings")
            n = len(standings) if isinstance(standings, list) else 0
            needed = (
                self._HEADER_HINT_PX
                + max(1, n) * self._ROW_HINT_PX
                + self._FOOTER_HINT_PX
            )
            needed = max(self._DETAILED_MIN_H,
                         min(self._DETAILED_MAX_H, needed))
            # Only grow, never override a smaller user-chosen size.
            if needed > self.height():
                self.resize(self.width(), needed)
        super()._on_snapshot(snap)

    def _mode_text(self) -> str:
        mode = str(self._snap.get("session_mode") or "practice")
        if mode == "race":
            return "RACE"
        if mode == "qualifying":
            return "QUALIFYING"
        return "PRACTICE"

    @staticmethod
    def _fit_font_pt(
        lines: list[str],
        max_w: int,
        max_h: int,
        *,
        family: str = "Consolas",
        weight: QFont.Weight = QFont.Weight.Normal,
        min_pt: int = 6,
        max_pt: int = 22,
    ) -> tuple[QFont, QFontMetrics, int]:
        """Pick the largest font that fits ``lines`` in ``max_w`` x ``max_h``.

        Returns ``(font, metrics, line_height_px)``. Falls back to
        ``min_pt`` when even that doesn't fit (caller can then elide).
        """
        if not lines:
            f = QFont(family, min_pt, weight)
            fm = QFontMetrics(f)
            return f, fm, fm.height()
        for pt in range(max_pt, min_pt - 1, -1):
            f = QFont(family, pt, weight)
            fm = QFontMetrics(f)
            lh = fm.height()
            if lh * len(lines) > max_h:
                continue
            widest = max(fm.horizontalAdvance(t) for t in lines)
            if widest <= max_w:
                return f, fm, lh
        f = QFont(family, min_pt, weight)
        fm = QFontMetrics(f)
        return f, fm, fm.height()

    @staticmethod
    def _fmt_leader_gap(row: dict[str, Any]) -> str:
        laps_down = int(row.get("laps_down") or 0)
        if laps_down >= 1:
            return f"+{laps_down}L"
        s = row.get("gap_to_leader_s")
        if isinstance(s, int | float):
            return f"+{float(s):.1f}"
        ms = row.get("gap_to_leader_ms")
        if isinstance(ms, int):
            return f"+{ms / 1000:.2f}"
        return ""

    @staticmethod
    def _fmt_interval(row: dict[str, Any]) -> str:
        s = row.get("interval_s")
        if isinstance(s, int | float):
            return f"+{float(s):.1f}"
        ms = row.get("interval_ms")
        if isinstance(ms, int):
            return f"+{ms / 1000:.2f}"
        return ""

    @staticmethod
    def _fmt_pit_tag(row: dict[str, Any]) -> str:
        # ``P`` = currently in pit lane; trailing digit = total stops
        # so far. Empty when the driver hasn't pitted and isn't in
        # the pits, so the column stays uncluttered for the leader.
        in_pit = bool(row.get("in_pit"))
        n = int(row.get("pit_stops") or 0)
        if in_pit and n > 0:
            return f"PIT{n}"
        if in_pit:
            return "PIT"
        if n > 0:
            return f"P{n}"
        return ""

    def _paint_shortcut_hint(self, p: QPainter, area: QRectF) -> None:
        p.setPen(QPen(QColor(130, 130, 140)))
        f = QFont("Segoe UI", max(7, int(area.height() * 0.7)),
                  QFont.Weight.Normal)
        p.setFont(f)
        p.drawText(
            area,
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            "double-click: compact / detailed",
        )

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card(p)

        w = max(40, self.width())
        h = max(40, self.height())
        margin = max(4, int(w * 0.025))
        inner_w = max(20, w - 2 * margin)

        mode_txt = self._mode_text()
        pos = self._snap.get("view_position")
        lap = self._snap.get("view_lap")
        cars = self._snap.get("cars") or []
        n_cars = len(cars)
        traffic = self._snap.get("traffic") or {}
        if isinstance(traffic, dict):
            n_from_traffic = traffic.get("num_cars")
            if isinstance(n_from_traffic, int) and n_from_traffic > 0:
                n_cars = n_from_traffic

        pos_txt = "--" if pos is None else (
            f"P{int(pos)}" if n_cars <= 0 else f"P{int(pos)}/{n_cars}"
        )
        lap_txt = "--" if lap is None else str(int(lap))
        ahead = _fmt_gap(traffic.get("ahead_gap_s")) if traffic else "--"
        behind = _fmt_gap(traffic.get("behind_gap_s")) if traffic else "--"

        # ---- Compact mode: 3 lines, fonts scale to widget --------------
        if self._compact:
            delta = _fmt_delta(self._snap.get("delta_vs_best_ms"))
            top_name = "--"
            standings = self._snap.get("standings")
            if isinstance(standings, list) and standings:
                top = standings[0]
                if isinstance(top, dict):
                    top_name = str(top.get("name") or "?")
            lines = [
                f"SESSION  {mode_txt}",
                f"{pos_txt}   LAP {lap_txt}",
                f"DELTA {delta}   A {ahead}   B {behind}",
                f"LEAD {top_name[:24]}",
            ]
            avail_h = max(20, h - 2 * margin)
            avail_w = inner_w
            big_font, _big_fm, big_lh = self._fit_font_pt(
                [lines[1]], avail_w, max(16, int(avail_h * 0.45)),
                family="Consolas", weight=QFont.Weight.Bold,
                min_pt=10, max_pt=32,
            )
            other = [lines[0], lines[2], lines[3]]
            small_avail_h = max(12, avail_h - big_lh)
            small_font, _small_fm, small_lh = self._fit_font_pt(
                other, avail_w, small_avail_h,
                family="Consolas", weight=QFont.Weight.Normal,
                min_pt=7, max_pt=16,
            )
            y = float(margin)
            p.setPen(QPen(QColor(150, 150, 160)))
            p.setFont(small_font)
            p.drawText(
                QRectF(margin, y, inner_w, small_lh),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                lines[0],
            )
            y += small_lh
            p.setPen(QPen(QColor(235, 235, 245)))
            p.setFont(big_font)
            p.drawText(
                QRectF(margin, y, inner_w, big_lh),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                lines[1],
            )
            y += big_lh
            p.setPen(QPen(QColor(205, 205, 215)))
            p.setFont(small_font)
            for txt in (lines[2], lines[3]):
                if y + small_lh > h - margin:
                    break
                p.drawText(
                    QRectF(margin, y, inner_w, small_lh),
                    int(
                        Qt.AlignmentFlag.AlignLeft
                        | Qt.AlignmentFlag.AlignVCenter
                    ),
                    txt,
                )
                y += small_lh
            hint_h = max(12, int(h * 0.08))
            self._paint_shortcut_hint(
                p,
                QRectF(margin, h - hint_h - 2, inner_w, hint_h),
            )
            return

        # ---- Detailed mode: header band + flexible leaderboard --------
        current = _fmt_clock(self._snap.get("current_lap_ms"))
        last = _fmt_clock(self._snap.get("last_lap_ms"))
        best = _fmt_clock(self._snap.get("best_lap_ms"))

        # Four header text lines, each on its own row band; sizes
        # derived from the band height (~6% / 14% / 8% / 8% of h, with
        # a label row reserving the same height as a small text line).
        # We then assign the rest of the vertical space to the
        # leaderboard rows.
        is_race = (
            str(self._snap.get("session_mode") or "practice") == "race"
        )

        standings_raw = self._snap.get("standings") or []
        standings = standings_raw if isinstance(standings_raw, list) else []
        n_rows = max(1, len(standings))

        hint_h = max(12, int(h * 0.06))
        footer_y = h - hint_h - 2

        # Build leaderboard rows up-front so we can size the column
        # widths to actual content (otherwise long names compress
        # everything else). Column model:
        #   [mark][pos] [name]  [info_tail]
        # info_tail in race  = "L{lap} {last_lap} {pit} {gap_leader} {int}"
        # info_tail in qual  = "BEST {best} {pit} {gap_leader} {int}"
        rendered: list[tuple[str, str, str]] = []  # (left, name, tail)
        for row in standings:
            if not isinstance(row, dict):
                continue
            try:
                rpos = int(row.get("pos", 0))
            except (TypeError, ValueError):
                rpos = 0
            mark = ">" if bool(row.get("view")) else " "
            left = f"{mark}{rpos:>2}"
            name = str(row.get("name") or "?")
            pit_tag = self._fmt_pit_tag(row)
            gap_l = self._fmt_leader_gap(row)
            interval = self._fmt_interval(row)
            if is_race:
                lap_n = int(row.get("lap", 0) or 0)
                clk = _fmt_clock(row.get("last_lap_ms"))
                parts = [f"L{lap_n:>2}", clk]
            else:
                clk = _fmt_clock(row.get("best_lap_ms"))
                parts = [f"BEST {clk}"]
            if gap_l:
                parts.append(gap_l)
            if interval:
                parts.append(interval)
            if pit_tag:
                parts.append(pit_tag)
            tail = "  ".join(parts)
            rendered.append((left, name, tail))

        # Header proportions. Each band is its own row in a virtual 7
        # row grid: 1 small (SESSION), 2 big (POS LAP), 1 small (times),
        # 1 small (ahead/behind), 1 small label (LEADERBOARD), 1 row
        # placeholder; the rest is leaderboard.
        avail_h = footer_y - margin
        # Sized fonts: big row gets ~2x small_lh.
        # Pick small font first based on the widest of the small lines.
        small_lines = [
            f"SESSION  {mode_txt}",
            f"CUR {current}   LAST {last}   BEST {best}",
            f"AHEAD {ahead}   BEHIND {behind}",
            "LEADERBOARD",
        ]
        # Tentatively allocate ~45% of height to header, rest to rows.
        header_h_target = max(60, int(avail_h * 0.45))
        # The header has 4 small lines + 1 big line (~2x small height) =
        # ~6 small-line-equivalents. Solve small_lh = header_h_target / 6.
        small_lh_target = max(10, header_h_target // 6)
        # Now pick the actual font that doesn't overflow horizontally
        # and stays close to that target line-height.
        small_font, _small_fm, small_lh = self._fit_font_pt(
            small_lines,
            inner_w,
            small_lh_target * len(small_lines) + small_lh_target,  # generous
            family="Consolas",
            weight=QFont.Weight.Normal,
            min_pt=7,
            max_pt=max(8, small_lh_target),
        )
        big_target_pt = max(small_font.pointSize() + 2,
                            int(small_font.pointSize() * 1.6))
        big_font, _big_fm, big_lh = self._fit_font_pt(
            [f"{pos_txt}   LAP {lap_txt}"],
            inner_w,
            big_target_pt * 3,
            family="Consolas",
            weight=QFont.Weight.Bold,
            min_pt=small_font.pointSize() + 1,
            max_pt=max(big_target_pt, small_font.pointSize() + 2),
        )
        header_h = small_lh * 4 + big_lh + max(2, int(small_lh * 0.3))

        # Leaderboard area.
        lb_top = margin + header_h
        lb_h = max(20, footer_y - lb_top)
        row_h_target = max(8, lb_h // n_rows)
        # Pick a monospace font where one line fits inner_w and row
        # height fits lb_h / n_rows. Use the widest rendered row as
        # the horizontal constraint, including 4-char separators.
        sample_lines = []
        for left, name, tail in rendered:
            sample_lines.append(f"{left} {name:<14} {tail}")
        if not sample_lines:
            sample_lines = ["-- no classification yet --"]
        lb_font, lb_fm, lb_lh = self._fit_font_pt(
            sample_lines,
            inner_w,
            row_h_target * len(sample_lines),
            family="Consolas",
            weight=QFont.Weight.Normal,
            min_pt=6,
            max_pt=max(7, row_h_target),
        )
        # When the leaderboard still doesn't fit horizontally at
        # min_pt we let the painter elide the name column instead of
        # truncating gaps/lap time, since position + gap are the most
        # useful pieces of info at a glance.
        # Compute available width for name column = inner_w - left_w
        # - tail_w - 2 separators.
        left_w = max(lb_fm.horizontalAdvance(left) for left, _, _ in rendered) \
            if rendered else lb_fm.horizontalAdvance("  P99")
        tail_w = max(lb_fm.horizontalAdvance(tail) for _, _, tail in rendered) \
            if rendered else 0
        sep_w = lb_fm.horizontalAdvance("  ")
        name_w = max(
            lb_fm.horizontalAdvance("AAA"),
            inner_w - left_w - tail_w - 2 * sep_w,
        )

        # ---- Now paint the header band ---------------------------------
        y = float(margin)
        # SESSION
        p.setPen(QPen(QColor(150, 150, 160)))
        p.setFont(small_font)
        p.drawText(
            QRectF(margin, y, inner_w, small_lh),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"SESSION  {mode_txt}",
        )
        y += small_lh
        # POS + LAP (big)
        p.setPen(QPen(QColor(235, 235, 245)))
        p.setFont(big_font)
        p.drawText(
            QRectF(margin, y, inner_w, big_lh),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"{pos_txt}   LAP {lap_txt}",
        )
        y += big_lh
        # Times
        p.setPen(QPen(QColor(205, 205, 215)))
        p.setFont(small_font)
        p.drawText(
            QRectF(margin, y, inner_w, small_lh),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"CUR {current}   LAST {last}   BEST {best}",
        )
        y += small_lh
        # Ahead / behind
        p.drawText(
            QRectF(margin, y, inner_w, small_lh),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"AHEAD {ahead}   BEHIND {behind}",
        )
        y += small_lh
        # LEADERBOARD label
        p.setPen(QPen(QColor(150, 150, 160)))
        p.drawText(
            QRectF(margin, y, inner_w, small_lh),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            "LEADERBOARD",
        )

        # ---- Leaderboard rows -----------------------------------------
        if not rendered:
            p.setPen(QPen(QColor(180, 180, 190)))
            p.setFont(lb_font)
            p.drawText(
                QRectF(margin, lb_top, inner_w, lb_h),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                "-- no classification yet --",
            )
        else:
            p.setFont(lb_font)
            row_y = float(lb_top)
            for left, name, tail in rendered:
                if row_y + lb_lh > footer_y:
                    break
                # Per-row colour: highlight the view car.
                if left.startswith(">"):
                    p.setPen(QPen(QColor(255, 215, 90)))
                else:
                    p.setPen(QPen(QColor(220, 220, 230)))
                # Left column
                p.drawText(
                    QRectF(margin, row_y, left_w, lb_lh),
                    int(
                        Qt.AlignmentFlag.AlignLeft
                        | Qt.AlignmentFlag.AlignVCenter
                    ),
                    left,
                )
                # Name (elided to available width)
                name_rect = QRectF(
                    margin + left_w + sep_w, row_y, name_w, lb_lh,
                )
                elided = lb_fm.elidedText(
                    name, Qt.TextElideMode.ElideRight, int(name_w),
                )
                p.drawText(
                    name_rect,
                    int(
                        Qt.AlignmentFlag.AlignLeft
                        | Qt.AlignmentFlag.AlignVCenter
                    ),
                    elided,
                )
                # Tail (right-aligned so gap/interval columns line up)
                tail_rect = QRectF(
                    margin + inner_w - tail_w, row_y, tail_w, lb_lh,
                )
                p.drawText(
                    tail_rect,
                    int(
                        Qt.AlignmentFlag.AlignLeft
                        | Qt.AlignmentFlag.AlignVCenter
                    ),
                    tail,
                )
                row_y += lb_lh

        # ---- Footer hint ----------------------------------------------
        self._paint_shortcut_hint(
            p,
            QRectF(margin, footer_y, inner_w, hint_h),
        )


# ---------------------------------------------------------------------------
# Flags + TC/ABS LED
# ---------------------------------------------------------------------------


