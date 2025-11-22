"""Stateful engine coordinating point storage, I/O, and matching calls."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .backend import solve_complete_matching_robust
from point_sampling import sample_auxiliary_points


class EngineError(RuntimeError):
    """Raised when the engine cannot complete a requested operation."""


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
        """Load normalized points from disk if sets are balanced."""

        file_path = Path(path)
        if not file_path.exists():
            raise EngineError("File not found.")

        loaded_a: list[tuple[float, float]] = []
        loaded_b: list[tuple[float, float]] = []

        try:
            with file_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 3:
                        continue
                    try:
                        x_val = float(parts[0])
                        y_val = float(parts[1])
                        label = int(parts[2])
                    except ValueError:
                        continue
                    if not (0.0 <= x_val <= 1.0 and 0.0 <= y_val <= 1.0):
                        continue
                    if label == 0:
                        loaded_a.append((x_val, y_val))
                    elif label == 1:
                        loaded_b.append((x_val, y_val))
        except OSError as exc:
            raise EngineError(f"Failed to read file: {exc}") from exc

        if len(loaded_a) != len(loaded_b):
            raise ValueError("The number of A and B points in the file must match.")

        self._points = [loaded_a, loaded_b]
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
