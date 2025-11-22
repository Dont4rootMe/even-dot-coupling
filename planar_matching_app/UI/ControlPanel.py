"""Left-side control panel for the planar matching app."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..engine import Engine, EngineError
from examples.centric_circles import get_centric_circles
from examples.randomized_circles import get_randomized_circles
from examples.two_moons import get_two_moons
from examples.swiss_roll import get_swiss_roll
from examples.triple_clusters import get_triple_clusters
from examples.geometric_composition import get_spirals, get_nested_polygons, get_pinwheel


class ControlPanel(QWidget):
    """Panel with file operations, matching trigger, and active set chooser."""

    def __init__(self, engine: Engine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._engine = engine

        self._counts_label = QLabel("A: 0    B: 0", self)
        self._matching_label = QLabel("Matching: none", self)

        self._btn_a = QPushButton("A (cross)", self)
        self._btn_b = QPushButton("B (circle)", self)
        self._btn_a.setCheckable(True)
        self._btn_b.setCheckable(True)
        self._btn_a.setChecked(True)

        self._load_button = QPushButton("Load points…", self)
        self._save_button = QPushButton("Save points…", self)
        self._clear_button = QPushButton("Clear board", self)
        self._compute_button = QPushButton("Compute matching", self)
        
        # Synthetic dataset controls
        self._n_points_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._n_points_slider.setRange(15, 100)
        self._n_points_slider.setValue(100)
        self._n_points_label = QLabel("Points per class: 100", self)
        
        self._noise_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._noise_slider.setRange(0, 20)  # 0-2.0 with 0.01 precision
        self._noise_slider.setValue(5)  # 0.1 default
        self._noise_label = QLabel("Noise: 0.05", self)
        
        self._preset_group = self._create_preset_group()

        self._notice_label = QLabel(
            "Left-click on the board adds a point to the active set.\n"
            "A and B must contain the same number of points to compute a matching.",
            self,
        )
        self._notice_label.setWordWrap(True)

        self._build_layout()
        self._connect_signals()

        self._engine.add_refresh_action(self._on_engine_refresh)
        self._engine.add_reset_action(self._on_engine_refresh)
        self._engine.set_active_set(0)
        self._on_engine_refresh()

    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        active_group = QGroupBox("Active set", self)
        active_layout = QHBoxLayout()
        active_layout.setSpacing(6)
        active_layout.addWidget(self._btn_a)
        active_layout.addWidget(self._btn_b)
        active_group.setLayout(active_layout)

        layout = QVBoxLayout()
        layout.addWidget(self._counts_label)
        layout.addWidget(active_group)
        layout.addWidget(self._notice_label)
        layout.addWidget(self._compute_button)
        layout.addWidget(self._matching_label)
        layout.addSpacing(10)
        layout.addWidget(self._clear_button)
        layout.addWidget(self._load_button)
        layout.addWidget(self._save_button)
        layout.addSpacing(15)
        layout.addWidget(self._preset_group)
        layout.addWidget(self._n_points_label)
        layout.addWidget(self._n_points_slider)
        layout.addWidget(self._noise_label)
        layout.addWidget(self._noise_slider)
        layout.addStretch()
        self.setLayout(layout)

    def _connect_signals(self) -> None:
        self._btn_a.clicked.connect(lambda: self._handle_set_toggle(0))
        self._btn_b.clicked.connect(lambda: self._handle_set_toggle(1))
        self._load_button.clicked.connect(self._handle_load)
        self._save_button.clicked.connect(self._handle_save)
        self._clear_button.clicked.connect(self._handle_clear)
        self._compute_button.clicked.connect(self._handle_compute)
        self._n_points_slider.valueChanged.connect(self._update_n_points_label)
        self._noise_slider.valueChanged.connect(self._update_noise_label)

    # ------------------------------------------------------------------
    def _on_engine_refresh(self) -> None:
        count_a, count_b = self._engine.get_point_counts()
        self._counts_label.setText(f"A: {count_a}    B: {count_b}")
        if self._engine.has_valid_matching():
            self._matching_label.setText("Matching: ready")
        else:
            self._matching_label.setText("Matching: none")

    def _handle_set_toggle(self, kind: int) -> None:
        try:
            self._engine.set_active_set(kind)
        except ValueError as exc:
            QMessageBox.warning(self, "Error", str(exc))
            return

        # Keep buttons mutually exclusive manually
        if kind == 0:
            self._btn_a.setChecked(True)
            self._btn_b.setChecked(False)
        else:
            self._btn_a.setChecked(False)
            self._btn_b.setChecked(True)

    def _handle_clear(self) -> None:
        self._engine.clear_board()

    def _handle_compute(self) -> None:
        count_a, count_b = self._engine.get_point_counts()
        if count_a != count_b:
            reply = QMessageBox.question(
                self,
                "Unbalanced sets",
                (
                    f"Set sizes: A={count_a}, B={count_b}.\n"
                    "Equal sizes are required. Add missing points via KDE sampling?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    self._engine.sample_missing_points()
                except EngineError as exc:
                    QMessageBox.warning(self, "Failed to add points", str(exc))
                    return
            else:
                return

        try:
            self._engine.compute_planar_matching()
        except EngineError as exc:
            QMessageBox.warning(self, "Cannot compute", str(exc))

    def _handle_load(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open points",
            "",
            "Text files (*.txt);;All files (*)",
        )
        if not file_path:
            return
        try:
            self._engine.load_points_from_file(file_path)
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid data", str(exc))
        except EngineError as exc:
            QMessageBox.critical(self, "Load error", str(exc))

    def _handle_save(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save points",
            "points.txt",
            "Text files (*.txt);;All files (*)",
        )
        if not file_path:
            return

        path_obj = Path(file_path)
        if not path_obj.suffix:
            path_obj = path_obj.with_suffix(".txt")

        try:
            self._engine.save_points_to_file(str(path_obj))
        except EngineError as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    def _create_preset_group(self) -> QGroupBox:
        """Create group box with synthetic dataset preset buttons."""
        presets = [
            ("Concentric circles", get_centric_circles),
            ("Randomized circles", get_randomized_circles),
            ("Two moons", get_two_moons),
            ("Swiss roll", get_swiss_roll),
            ("Triple clusters", get_triple_clusters),
            ("Spirals", get_spirals),
            ("Nested polygons", get_nested_polygons),
            ("Pinwheel", get_pinwheel),
        ]
        
        group = QGroupBox("Synthetic datasets", self)
        grid = QGridLayout()
        grid.setSpacing(6)
        
        for index, (title, generator) in enumerate(presets):
            button = QPushButton(title, group)
            button.clicked.connect(lambda _checked=False, gen=generator: self._handle_preset(gen))
            row = index // 2
            column = index % 2
            grid.addWidget(button, row, column)
        
        group.setLayout(grid)
        return group

    def _handle_preset(self, generator_func) -> None:
        """Generate and load a synthetic dataset."""
        n_points = self._n_points_slider.value()
        noise = self._noise_slider.value() / 100.0
        
        try:
            points_a, points_b = generator_func(n_samples=n_points, noise=noise)
            self._engine.load_synthetic_dataset(points_a, points_b)
        except Exception as exc:
            QMessageBox.warning(self, "Generation failed", str(exc))

    def _update_n_points_label(self, value: int) -> None:
        """Update the label for the points per class slider."""
        self._n_points_label.setText(f"Points per class: {value}")

    def _update_noise_label(self, value: int) -> None:
        """Update the label for the noise slider."""
        noise_value = value / 100.0
        self._noise_label.setText(f"Noise: {noise_value:.2f}")
