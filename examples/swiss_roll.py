from typing import Literal, Tuple
import argparse
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA


Normalize2D = Literal["none", "unit_square", "fit_square"]


def _to_unit_square(X: np.ndarray) -> np.ndarray:
    """
    Independent min–max scaling of each coordinate to [0, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = hi - lo
    span[span == 0.0] = 1.0
    Z = (X - lo) / span
    return Z


def _fit_to_centered_square(X: np.ndarray) -> np.ndarray:
    """
    Uniform scaling with aspect ratio preservation to centered square [-1, 1]^2.
    Scale by the larger bbox side: X' = (X - c)/s, where c = (min+max)/2, s = max((max-min)/2, eps).
    """
    X = np.asarray(X, dtype=np.float64)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    c = (lo + hi) / 2.0
    half_span = (hi - lo) / 2.0
    s = float(max(half_span.max(), 1e-12))
    Z = (X - c) / s
    return np.clip(Z, -1.0, 1.0)


def get_swiss_roll(
    n_samples: int = 100,
    noise: float = 0.1,
    normalize_2d: Normalize2D = "fit_square",
    dtype: np.dtype = np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Swiss roll 2D projection with random class assignment.

    Parameters
    ----------
    n_samples : int
        Number of points per class (total will be 2*n_samples).
    noise : float
        Gaussian noise amplitude in make_swiss_roll.
    normalize_2d : {"none", "unit_square", "fit_square"}
        2D projection normalization:
        - "none": leave as-is (centered PCA).
        - "unit_square": scale each axis to [0, 1].
        - "fit_square": uniformly fit to [-1, 1]^2 (default).
    dtype : numpy.dtype
        Output array type.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two arrays with randomly assigned points from the swiss roll.
    """
    X3d, _ = make_swiss_roll(n_samples=2*n_samples, noise=noise)
    X3d = np.ascontiguousarray(X3d, dtype=dtype)

    # Deterministic PCA (full SVD guarantees no randomness)
    pca = PCA(n_components=2, svd_solver="full", whiten=False)
    Z2d = pca.fit_transform(X3d)  # already centered inside PCA

    if normalize_2d == "unit_square":
        Z2d = _to_unit_square(Z2d)
    elif normalize_2d == "fit_square":
        Z2d = _fit_to_centered_square(Z2d)
    elif normalize_2d == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize_2d={normalize_2d!r}")

    Z2d = np.ascontiguousarray(Z2d, dtype=dtype)
    
    # Randomly assign 50% to class A, 50% to class B
    indices = np.arange(len(Z2d))
    np.random.shuffle(indices)
    mid = len(indices) // 2
    
    X_a = Z2d[indices[:mid]]
    X_b = Z2d[indices[mid:]]

    return X_a, X_b