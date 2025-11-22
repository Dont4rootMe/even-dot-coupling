from typing import Literal, Tuple, Union
import argparse
import numpy as np
from sklearn.datasets import make_blobs

Normalize = Literal["none", "unit_square", "fit_square"]


def _to_unit_square(X: np.ndarray) -> np.ndarray:
    """Independent min–max scaling of each coordinate to [0,1]^2."""
    X = np.asarray(X, dtype=np.float64)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = hi - lo
    span[span == 0.0] = 1.0
    Z = (X - lo) / span
    return Z


def _fit_to_centered_square(X: np.ndarray) -> np.ndarray:
    """Uniform scaling to centered square [-1,1]^2 with aspect ratio preservation."""
    X = np.asarray(X, dtype=np.float64)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    c = (lo + hi) / 2.0
    half_span = (hi - lo) / 2.0
    s = float(max(half_span.max(), 1e-12))
    Z = (X - c) / s
    return np.clip(Z, -1.0, 1.0)


def _equilateral_triangle_centers(side: float, angle_deg: float) -> np.ndarray:
    """
    Vertices of equilateral triangle with side `side` and centroid at (0,0),
    then rotation by `angle_deg` counterclockwise.
    Base vertices (centroid=0):
        A = (-s/2, -sqrt(3)/6 s), B = (s/2, -sqrt(3)/6 s), C = (0, sqrt(3)/3 s)
    """
    s = float(side)
    r3 = np.sqrt(3.0)
    centers = np.array(
        [
            (-s / 2.0, -r3 * s / 6.0),
            ( s / 2.0, -r3 * s / 6.0),
            ( 0.0,      r3 * s / 3.0),
        ],
        dtype=np.float64,
    )
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]], dtype=np.float64)
    return centers @ R.T  # (3,2)


def get_triple_clusters(
    n_samples: int = 100,
    noise: float = 0.4,
    side: float = 6.0,
    angle_deg: float = 0.0,
    random_state: Union[int, None] = None,
    normalize: Normalize = "fit_square",
    dtype: np.dtype = np.float64,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate three clusters with random class assignment.

    Parameters
    ----------
    n_samples : int
        Number of points per class (total will be 2*n_samples).
    noise : float
        Gaussian noise amplitude in make_blobs.
    side : float
        Triangle side length (scale of center separation).
    angle_deg : float
        Triangle rotation (in degrees) counterclockwise.
    random_state : int | None
        RNG initialization for reproducibility.
    normalize : {"none","unit_square","fit_square"}
        "none" — leave as-is; "unit_square" — min–max per axis to [0,1]^2;
        "fit_square" — uniform scaling to [-1,1]^2 (default).
    dtype : np.dtype
        Number type in output point array.
    shuffle : bool
        Whether to shuffle the sample (as in sklearn.make_blobs).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two arrays with randomly assigned points from the three clusters.
    """
    if noise <= 0.001:
        noise = 0.4
    
    centers = _equilateral_triangle_centers(side=side, angle_deg=angle_deg)

    X, _ = make_blobs(
        n_samples=2*n_samples,
        n_features=2,
        centers=centers,
        cluster_std=noise,
        shuffle=shuffle,
        random_state=random_state,
    )

    if normalize == "unit_square":
        X = _to_unit_square(X)
    elif normalize == "fit_square":
        X = _fit_to_centered_square(X)
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be one of {'none','unit_square','fit_square'}")

    X = np.ascontiguousarray(X, dtype=dtype)
    
    # Randomly assign 50% to class A, 50% to class B
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    mid = len(indices) // 2
    
    X_a = X[indices[:mid]]
    X_b = X[indices[mid:]]
    
    return X_a, X_b