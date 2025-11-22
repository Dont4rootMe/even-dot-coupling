"""Stateful engine coordinating point storage, I/O, and matching calls."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .backend import solve_complete_matching_robust
from point_sampling import sample_auxiliary_points


class EngineError(RuntimeError):
    """Raised when the engine cannot complete a requested operation."""


SCENE_MARGIN_RATIO: float = 0.05


class Engine:
    """Holds all domain state and exposes a UI-friendly API."""

    _instance: "Engine | None" = None

    @classmethod
    def instance(cls) -> "Engine":
        """Return the singleton engine instance."""

        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._points: list[list[tuple[float, float]]] = [[], []]  # index 0 -> A, 1 -> B
        self._segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._refresh_callbacks: list[Callable[[], None]] = []
        self._reset_callbacks: list[Callable[[], None]] = []
        self._active_set: int = 0

    # ------------------------------------------------------------------
    # Callback management
    def add_refresh_action(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked when state changes."""

        self._refresh_callbacks.append(callback)
        callback()

    def add_reset_action(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked when a full reset happens."""

        self._reset_callbacks.append(callback)

    def refresh(self) -> None:
        """Notify listeners that state has changed."""

        for callback in list(self._refresh_callbacks):
            callback()

    def reset(self) -> None:
        """Clear engine state and notify reset + refresh listeners."""

        self._points = [[], []]
        self._segments = []
        for callback in list(self._reset_callbacks):
            callback()
        self.refresh()

    # ------------------------------------------------------------------
    # Public API used by UI
    def clear_board(self) -> None:
        """Remove all points and computed segments."""

        self._points = [[], []]
        self._segments = []
        self.refresh()

    def set_active_set(self, kind: int) -> None:
        """Switch the target set for subsequent clicks (0 for A, 1 for B)."""

        if kind not in (0, 1):
            raise ValueError("Active set must be 0 (A) or 1 (B).")
        self._active_set = int(kind)

    def add_point_from_click(self, x: float, y: float) -> None:
        """Add a normalized point to the active set and invalidate matching."""

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise EngineError("Coordinates must be within [0, 1].")

        target = self._points[self._active_set]
        target.append((float(x), float(y)))
        self._segments = []
        self.refresh()

    def get_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized coordinates for sets A and B."""

        arr_a = np.array(self._points[0], dtype=np.float64) if self._points[0] else np.empty((0, 2))
        arr_b = np.array(self._points[1], dtype=np.float64) if self._points[1] else np.empty((0, 2))
        return arr_a, arr_b

    def get_points_for_drawing(self) -> dict[str, list[tuple[float, float]] | list[tuple[tuple[float, float], tuple[float, float]]]]:
        """Provide a UI-friendly snapshot of current points and segments."""

        return {
            "A": list(self._points[0]),
            "B": list(self._points[1]),
            "segments": list(self._segments),
        }

    def compute_planar_matching(self) -> None:
        """Compute planar matching via backend and store resulting segments."""

        points_a, points_b = self.get_points()
        n_a, n_b = points_a.shape[0], points_b.shape[0]

        if n_a == 0 or n_b == 0 or n_a != n_b:
            raise EngineError("Sets A and B must both be non-empty and have the same size.")

        if n_a == 0 or n_b == 0 or n_a != n_b:
            raise EngineError("Sets A and B must both be non-empty and have the same size.")

        try:
            pairs = solve_complete_matching_robust(points_a, points_b)
        except Exception as exc:
            raise EngineError(f"Failed to compute matching: {exc}") from exc

        segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for idx_a, idx_b in pairs:
            try:
                pa = points_a[int(idx_a)]
                pb = points_b[int(idx_b)]
            except IndexError as exc:
                raise EngineError("Backend returned invalid indices.") from exc
            segments.append(((float(pa[0]), float(pa[1])), (float(pb[0]), float(pb[1]))))

        self._segments = segments
        self.refresh()

    def sample_missing_points(self) -> None:
        """Balance point sets by sampling new points for the smaller class."""

        count_a, count_b = self.get_point_counts()
        if count_a == count_b:
            return
        if count_a == 0 or count_b == 0:
            raise EngineError("Cannot generate points: one of the sets is empty.")

        if count_a < count_b:
            source = np.array(self._points[0], dtype=np.float64)
            target = self._points[0]
            deficit = count_b - count_a
        else:
            source = np.array(self._points[1], dtype=np.float64)
            target = self._points[1]
            deficit = count_a - count_b

        sampled = sample_auxiliary_points(source, deficit)
        # Clip to the normalized board bounds to avoid drawing outside the canvas.
        for x, y in sampled:
            target.append((float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))))

        # Invalidate old matching.
        self._segments = []
        self.refresh()

    def load_points_from_file(self, path: str) -> None:
        """Load points from disk, normalize them, and split by class label.

        The input file may contain either two columns (x y) or three columns
        (x y label). When the label column is missing, points are assigned to
        set A (label 0) by default. Coordinates are automatically rescaled to
        fit the normalized board with a small margin, so the user does not need
        to worry about the original units.
        """

        file_path = Path(path)
        if not file_path.exists():
            raise EngineError("File not found.")

        raw_records: list[tuple[float, float, int]] = []

        try:
            with file_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    try:
                        x_val = float(parts[0])
                        y_val = float(parts[1])
                        label = int(parts[2]) if len(parts) >= 3 else 0
                    except ValueError:
                        continue
                    if not (np.isfinite(x_val) and np.isfinite(y_val)):
                        continue
                    normalized_label = 0 if label not in (0, 1) else label
                    raw_records.append((x_val, y_val, normalized_label))
        except OSError as exc:
            raise EngineError(f"Failed to read file: {exc}") from exc

        if not raw_records:
            self._points = [[], []]
            self._segments = []
            self.refresh()
            return

        coords = np.array([[x, y] for x, y, _ in raw_records], dtype=np.float64)
        normalized_coords = self._normalize_coordinates(coords)

        points_a: list[tuple[float, float]] = []
        points_b: list[tuple[float, float]] = []
        for (x_norm, y_norm), (_, _, label) in zip(normalized_coords, raw_records):
            if label == 0:
                points_a.append((float(x_norm), float(y_norm)))
            else:
                points_b.append((float(x_norm), float(y_norm)))

        self._points = [points_a, points_b]
        self._segments = []
        self.refresh()

    def save_points_to_file(self, path: str) -> None:
        """Persist current normalized points."""

        file_path = Path(path)
        try:
            with file_path.open("w", encoding="utf-8") as handle:
                for x, y in self._points[0]:
                    handle.write(f"{x} {y} 0\n")
                for x, y in self._points[1]:
                    handle.write(f"{x} {y} 1\n")
        except OSError as exc:
            raise EngineError(f"Failed to save points: {exc}") from exc

    def get_point_counts(self) -> tuple[int, int]:
        """Return counts for A and B."""

        return len(self._points[0]), len(self._points[1])

    def has_valid_matching(self) -> bool:
        """Return True if a matching has been computed for the current data."""

        expected = min(len(self._points[0]), len(self._points[1]))
        return expected > 0 and len(self._segments) == expected

    def load_synthetic_dataset(
        self, 
        points_a: np.ndarray, 
        points_b: np.ndarray
    ) -> None:
        """Load pre-generated synthetic point sets.
        
        Normalizes both sets together to preserve their relative positions,
        then updates the engine state with the new points.
        
        Parameters
        ----------
        points_a : np.ndarray
            Points for set A, shape (n, 2).
        points_b : np.ndarray
            Points for set B, shape (m, 2).
        """
        if points_a.size == 0 and points_b.size == 0:
            self._points = [[], []]
            self._segments = []
            self.refresh()
            return
        
        # Normalize both sets together to preserve relative positions
        combined = np.vstack([points_a, points_b])
        normalized = self._normalize_coordinates(combined)
        
        n_a = points_a.shape[0]
        norm_a = normalized[:n_a]
        norm_b = normalized[n_a:]
        
        self._points = [
            [(float(x), float(y)) for x, y in norm_a],
            [(float(x), float(y)) for x, y in norm_b]
        ]
        self._segments = []
        self.refresh()

    # ------------------------------------------------------------------
    # Internal helpers
    def _normalize_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Scale arbitrary coordinates to the [0, 1]x[0, 1] board with margins."""

        if coords.size == 0:
            return coords

        min_x = float(np.min(coords[:, 0]))
        max_x = float(np.max(coords[:, 0]))
        min_y = float(np.min(coords[:, 1]))
        max_y = float(np.max(coords[:, 1]))

        width = max_x - min_x
        height = max_y - min_y

        margin = SCENE_MARGIN_RATIO
        usable = 1.0 - 2.0 * margin
        if usable <= 0.0:
            raise EngineError("Invalid board margin configuration.")

        if width == 0.0 and height == 0.0:
            centered = np.empty_like(coords)
            centered[:, 0] = 0.5
            centered[:, 1] = 0.5
            return centered

        if width == 0.0:
            scale = usable / height if height != 0.0 else 1.0
            scaled_x = np.full(coords.shape[0], 0.5)
            scaled_y = (coords[:, 1] - min_y) * scale
        elif height == 0.0:
            scale = usable / width if width != 0.0 else 1.0
            scaled_x = (coords[:, 0] - min_x) * scale
            scaled_y = np.full(coords.shape[0], 0.5)
        else:
            scale = min(usable / width, usable / height)
            scaled_x = (coords[:, 0] - min_x) * scale
            scaled_y = (coords[:, 1] - min_y) * scale

        final_width = width * scale if width != 0.0 else 0.0
        final_height = height * scale if height != 0.0 else 0.0
        offset_x = margin + (usable - final_width) / 2.0
        offset_y = margin + (usable - final_height) / 2.0

        normalized = np.empty_like(coords)
        normalized[:, 0] = scaled_x + offset_x
        normalized[:, 1] = scaled_y + offset_y
        return normalized
