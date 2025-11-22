"""Application entrypoint for the planar matching GUI."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QWidget

from .UI.BoardCanvas import BoardCanvas
from .UI.ControlPanel import ControlPanel
from .engine import Engine


class MainWindow(QMainWindow):
    """Main window combining control panel and drawing board."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Planar Matching")
        self._engine = Engine.instance()

        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        self._control = ControlPanel(self._engine, central)
        self._board = BoardCanvas(self._engine, central)

        layout.addWidget(self._control)
        layout.addWidget(self._board, stretch=1)
        layout.setStretch(0, 3)
        layout.setStretch(1, 7)

        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(1200, 800)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
