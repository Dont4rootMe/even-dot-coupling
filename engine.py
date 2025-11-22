"""Core engine singleton managing points, lines, and backend computations."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, List, Optional

import numpy as np
from matplotlib.cm import get_cmap

from level_builder import convex_layers

SCENE_SIZE: int = 1000
POINT_RADIUS: int = 4
LINE_WIDTH: int = 2
SCENE_MARGIN_RATIO: float = 0.05


class EngineError(RuntimeError):
    """Raised when the engine cannot complete a requested operation."""


class Engine:
    """Singleton engine storing state, performing I/O, and computing levels."""

    _instance: Optional["Engine"] = None

    def __init__(self) -> None:
        self._points: np.ndarray = np.empty((0, 2), dtype=np.float64)
        self._lines: np.ndarray = np.empty((0, 4), dtype=np.float64)
        self._points_listeners: List[Callable[[np.ndarray], None]] = []
        self._lines_listeners: List[Callable[[np.ndarray], None]] = []
        self._status_listeners: List[Callable[[str], None]] = []
        self._layers_info_listeners: List[Callable[[dict[str, object]], None]] = []
        self._layers_info: dict[str, object] = {
            "layer_count": 0,
            "points_per_layer": [],
            "duration": 0.0,
        }

    @classmethod
    def instance(cls) -> "Engine":
        """Return the lazily created singleton instance."""

        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Listener registration
    def register_points_listener(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register a callback invoked when point data changes."""

        self._points_listeners.append(callback)
        callback(self.points())

    def register_lines_listener(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register a callback invoked when line data changes."""

        self._lines_listeners.append(callback)
        callback(self.lines())

    def register_status_listener(self, callback: Callable[[str], None]) -> None:
        """Register a callback invoked when status messages are emitted."""

        self._status_listeners.append(callback)

    def register_layers_info_listener(self, callback: Callable[[dict[str, object]], None]) -> None:
        """Register a callback invoked when layer metadata changes."""

        self._layers_info_listeners.append(callback)
        callback(self._layers_info.copy())

    # ------------------------------------------------------------------
    # Public API
    def reset(self) -> None:
        """Clear all points and lines."""

        self._update_points(np.empty((0, 2), dtype=np.float64))
        self._update_lines(np.empty((0, 4), dtype=np.float64))
        self._clear_layers_info()
        self._emit_status("Board cleared.")

    def points(self) -> np.ndarray:
        """Return a copy of the current set of points."""

        return self._points.copy()

    def lines(self) -> np.ndarray:
        """Return a copy of the current set of lines."""

        return self._lines.copy()

    def add_point(self, x: float, y: float) -> bool:
        """Add a point if it is not a near-duplicate of an existing point."""

        candidate = np.array([x, y], dtype=np.float64)
        if not np.isfinite(candidate).all():
            raise EngineError("Point coordinates must be finite numbers.")

        if self._points.size:
            distances = np.linalg.norm(self._points - candidate, axis=1)
            if np.any(distances < 1.0):
                self._emit_status("Point is too close to an existing one and was not added.")
                return False

        new_points = np.vstack([self._points, candidate])
        self._update_points(new_points)
        self._clear_lines_silent()
        self._emit_status("New point added.")
        return True

    def bulk_set_points(self, points: Iterable[Iterable[float]], fit_to_scene: bool = False) -> None:
        """Replace the entire set of points after validation and deduplication.

        Args:
            points: Iterable with point coordinates.
            fit_to_scene: When ``True`` the point set is affinely scaled to fit the
                drawing scene while keeping aspect ratio.
        """

        array = self._prepare_points_array(points)
        if array.size == 0:
            self._update_points(array)
            self._clear_lines_silent()
            self._emit_status("Points loaded: 0 points.")
            return

        if fit_to_scene:
            array = self._scale_points_to_scene(array)

        filtered = self._deduplicate_points(array)
        self._update_points(filtered)
        self._clear_lines_silent()
        self._emit_status(f"Points loaded: {len(filtered)} points.")

    def save_points(self, path: str, overwrite: bool = True) -> None:
        """Persist the current points to a text file."""

        if not path:
            raise EngineError("Invalid path for saving.")

        file_path = Path(path)
        mode = "w" if overwrite else "x"
        try:
            with file_path.open(mode, encoding="utf-8") as handle:
                for x, y in self._points:
                    handle.write(f"{x} {y}\n")
        except FileExistsError as exc:
            raise EngineError("File already exists and cannot be overwritten.") from exc
        except OSError as exc:
            raise EngineError(f"Failed to save points: {exc}.") from exc
        else:
            self._emit_status(f"Saved points: {len(self._points)}.")

    def load_points(self, path: str) -> None:
        """Load points from a text file, replacing the current set."""

        if not path:
            raise EngineError("Invalid path for loading.")

        file_path = Path(path)
        if not file_path.exists():
            raise EngineError("File not found.")

        loaded: List[List[float]] = []
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    try:
                        x_val = float(parts[0])
                        y_val = float(parts[1])
                    except ValueError:
                        continue
                    loaded.append([x_val, y_val])
        except OSError as exc:
            raise EngineError(f"Failed to load points: {exc}.") from exc

        array = np.array(loaded, dtype=np.float64) if loaded else np.empty((0, 2), dtype=np.float64)
        self.reset()
        if array.size == 0:
            self.bulk_set_points(array)
            return

        scaled = self._scale_points_to_scene(array)
        self.bulk_set_points(scaled)

    def compute_levels(self) -> np.ndarray:
        """Compute convex layer segments for the current points and update the state."""

        segments, layer_sizes, duration = self._compute_convex_layers(self._points)
        self._update_lines(segments)
        self._update_layers_info(len(layer_sizes), layer_sizes, duration)
        self._emit_status(f"Computation finished: {len(segments)} segments.")
        return segments.copy()

    def clear_lines(self) -> None:
        """Remove all lines while keeping points intact."""

        self._update_lines(np.empty((0, 4), dtype=np.float64))
        self._clear_layers_info()
        self._emit_status("Segments removed.")

    # ------------------------------------------------------------------
    # Internal helpers
    def _emit_status(self, message: str) -> None:
        for callback in self._status_listeners:
            callback(message)

    def _update_points(self, points: np.ndarray) -> None:
        self._points = points
        for callback in self._points_listeners:
            callback(self.points())

    def _update_lines(self, lines: np.ndarray) -> None:
        self._lines = lines
        for callback in self._lines_listeners:
            callback(self.lines())

    def _clear_lines_silent(self) -> None:
        if self._lines.size == 0:
            self._clear_layers_info()
            return
        self._update_lines(np.empty((0, 4), dtype=np.float64))
        self._clear_layers_info()

    def _scale_points_to_scene(self, points: np.ndarray) -> np.ndarray:
        padding = float(SCENE_SIZE) * SCENE_MARGIN_RATIO
        if padding * 2 >= SCENE_SIZE:
            raise EngineError("Invalid scene margin configuration.")

        min_x = float(np.min(points[:, 0]))
        max_x = float(np.max(points[:, 0]))
        min_y = float(np.min(points[:, 1]))
        max_y = float(np.max(points[:, 1]))

        width = max_x - min_x
        height = max_y - min_y
        usable = float(SCENE_SIZE) - 2 * padding
        if usable <= 0:
            raise EngineError("Not enough room to display points on the scene.")

        centered = np.empty_like(points, dtype=np.float64)

        if width == 0 and height == 0:
            centered[:, 0] = SCENE_SIZE / 2
            centered[:, 1] = SCENE_SIZE / 2
            return centered

        if width == 0:
            scale = usable / height if height != 0 else 1.0
            scaled_height = height * scale
            y_offset = padding + (usable - scaled_height) / 2
            centered[:, 0] = SCENE_SIZE / 2
            centered[:, 1] = (points[:, 1] - min_y) * scale + y_offset
            return centered

        if height == 0:
            scale = usable / width if width != 0 else 1.0
            scaled_width = width * scale
            x_offset = padding + (usable - scaled_width) / 2
            centered[:, 0] = (points[:, 0] - min_x) * scale + x_offset
            centered[:, 1] = SCENE_SIZE / 2
            return centered

        scale = min(usable / width, usable / height)
        scaled_width = width * scale
        scaled_height = height * scale
        x_offset = padding + (usable - scaled_width) / 2
        y_offset = padding + (usable - scaled_height) / 2

        centered[:, 0] = (points[:, 0] - min_x) * scale + x_offset
        centered[:, 1] = (points[:, 1] - min_y) * scale + y_offset
        return centered

    def _prepare_points_array(self, points: Iterable[Iterable[float]]) -> np.ndarray:
        try:
            array = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise EngineError("Cannot convert input data to a point array.") from exc

        if array.size == 0:
            return np.empty((0, 2), dtype=np.float64)

        if array.ndim != 2 or array.shape[1] != 2:
            raise EngineError("Expected an array of shape (n, 2) for points.")

        mask = np.isfinite(array).all(axis=1)
        cleaned = array[mask]
        return cleaned.astype(np.float64, copy=False)

    def _deduplicate_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        unique_points: List[np.ndarray] = []
        for point in points:
            if not unique_points:
                unique_points.append(point)
                continue
            existing = np.vstack(unique_points)
            distances = np.linalg.norm(existing - point, axis=1)
            if np.any(distances < 1.0):
                continue
            unique_points.append(point)

        return np.vstack(unique_points) if unique_points else np.empty((0, 2), dtype=np.float64)

    def _clear_layers_info(self) -> None:
        self._update_layers_info(0, [], 0.0)

    def _update_layers_info(
        self, layer_count: int, points_per_layer: List[int], duration: float
    ) -> None:
        self._layers_info = {
            "layer_count": layer_count,
            "points_per_layer": points_per_layer,
            "duration": duration,
        }
        for callback in self._layers_info_listeners:
            callback(self._layers_info.copy())

    def _compute_convex_layers(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, List[int], float]:
        """Compute line segments corresponding to convex layers."""

        start_time = perf_counter()

        if points.shape[0] < 2:
            return np.empty((0, 4), dtype=np.float64), [], 0.0

        mask = np.isfinite(points).all(axis=1)
        filtered = points[mask]
        if filtered.shape[0] < 2:
            return np.empty((0, 4), dtype=np.float64), [], 0.0

        try:
            layers = convex_layers(filtered, tol=1e-12)
        except Exception as exc:  # pragma: no cover - safety net around external code
            raise EngineError("Failed to compute convex layers for the current point set.") from exc

        valid_layers: List[np.ndarray] = []
        for hull in layers:
            hull_array = np.asarray(hull, dtype=np.float64)
            if hull_array.size == 0:
                continue
            if hull_array.ndim != 2 or hull_array.shape[1] != 2:
                raise EngineError("convex_layers returned a layer with an invalid shape.")
            if hull_array.shape[0] < 2:
                continue
            valid_layers.append(hull_array)

        if not valid_layers:
            duration = perf_counter() - start_time
            return np.empty((0, 4), dtype=np.float64), [], duration

        colormap = get_cmap("viridis", len(valid_layers))
        segments: List[np.ndarray] = []
        for layer_index, hull_array in enumerate(valid_layers):
            color = colormap(layer_index)
            r, g, b = (float(color[0]), float(color[1]), float(color[2]))

            for idx in range(hull_array.shape[0] - 1):
                p1 = hull_array[idx]
                p2 = hull_array[idx + 1]
                segments.append(
                    np.array(
                        [p1[0], p1[1], p2[0], p2[1], r, g, b],
                        dtype=np.float64,
                    )
                )

            if hull_array.shape[0] > 2:
                p_last = hull_array[-1]
                p_first = hull_array[0]
                segments.append(
                    np.array(
                        [p_last[0], p_last[1], p_first[0], p_first[1], r, g, b],
                        dtype=np.float64,
                    )
                )

        if not segments:
            duration = perf_counter() - start_time
            return np.empty((0, 4), dtype=np.float64), [], duration

        stacked = np.vstack(segments)
        if stacked.ndim != 2 or stacked.shape[1] not in (4, 7):
            raise EngineError("convex_layers returned an invalid set of segments.")
        if not np.isfinite(stacked).all():
            raise EngineError("convex_layers returned non-finite values.")
        duration = perf_counter() - start_time
        layer_sizes = [layer.shape[0] for layer in valid_layers]

        return stacked, layer_sizes, duration
