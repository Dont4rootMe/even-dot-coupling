"""Square drawing board for visualizing and editing planar matchings."""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QMessageBox, QSizePolicy, QWidget

from ..engine import Engine, EngineError


class BoardCanvas(QWidget):
    """Interactive square canvas that renders points and their matching."""

    def __init__(self, engine: Engine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._engine = engine
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 400)
        self._engine.add_refresh_action(self.update)

    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        board_rect = self._board_rect()
        if not board_rect.contains(event.position()):
            return

        normalized = self._screen_to_normalized(event.position(), board_rect)
        try:
            self._engine.add_point_from_click(*normalized)
        except EngineError as exc:
            QMessageBox.critical(self, "Error", str(exc))
        finally:
            event.accept()

    def paintEvent(self, event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("white"))

        board_rect = self._board_rect()
        light_pen = QPen(QColor(220, 220, 220))
        light_pen.setWidth(1)
        painter.setPen(light_pen)
        painter.drawRect(board_rect)

        data = self._engine.get_points_for_drawing()
        self._draw_segments(painter, board_rect, data.get("segments", []))
        self._draw_points(painter, board_rect, data.get("A", []), data.get("B", []))

    # ------------------------------------------------------------------
    def _board_rect(self) -> QRectF:
        """Return the maximal square centered inside the widget."""

        side = float(min(self.width(), self.height()))
        left = (self.width() - side) / 2.0
        top = (self.height() - side) / 2.0
        return QRectF(left, top, side, side)

    def _screen_to_normalized(self, pos: QPointF, rect: QRectF) -> tuple[float, float]:
        """Map widget coordinates to normalized [0, 1]^2 space."""

        x = (pos.x() - rect.left()) / rect.width()
        y = (pos.y() - rect.top()) / rect.height()
        return float(x), float(y)

    def _normalized_to_screen(self, point: tuple[float, float], rect: QRectF) -> QPointF:
        x = rect.left() + point[0] * rect.width()
        y = rect.top() + point[1] * rect.height()
        return QPointF(x, y)

    def _draw_segments(
        self,
        painter: QPainter,
        rect: QRectF,
        segments: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> None:
        if not segments:
            return

        pen = QPen(QColor(60, 120, 200))
        pen.setWidthF(2.0)
        painter.setPen(pen)
        for a, b in segments:
            pa = self._normalized_to_screen(a, rect)
            pb = self._normalized_to_screen(b, rect)
            painter.drawLine(pa, pb)

    def _draw_points(
        self,
        painter: QPainter,
        rect: QRectF,
        points_a: list[tuple[float, float]],
        points_b: list[tuple[float, float]],
    ) -> None:
        pen_a = QPen(QColor(200, 70, 70))
        pen_a.setWidthF(2.0)
        pen_b = QPen(QColor(50, 120, 60))
        pen_b.setWidthF(2.0)

        cross_half = 6.0
        circle_radius = 6.0

        painter.setPen(pen_a)
        for pt in points_a:
            center = self._normalized_to_screen(pt, rect)
            painter.drawLine(
                QPointF(center.x() - cross_half, center.y() - cross_half),
                QPointF(center.x() + cross_half, center.y() + cross_half),
            )
            painter.drawLine(
                QPointF(center.x() - cross_half, center.y() + cross_half),
                QPointF(center.x() + cross_half, center.y() - cross_half),
            )

        painter.setPen(pen_b)
        for pt in points_b:
            center = self._normalized_to_screen(pt, rect)
            painter.drawEllipse(center, circle_radius, circle_radius)
