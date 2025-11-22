from typing import Literal, Tuple, Union
import argparse
import numpy as np
from sklearn.datasets import make_moons


Normalize = Literal["none", "unit_square", "fit_square"]


def _to_unit_square(X: np.ndarray) -> np.ndarray:
    """
    Affine transformation to [0,1]^2 (independent coordinate scaling, no aspect ratio preservation).
    """
    X = np.asarray(X, dtype=np.float64)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = hi - lo
    # protection against degeneracy
    span[span == 0.0] = 1.0
    Z = (X - lo) / span
    return Z


def _fit_to_centered_square(X: np.ndarray) -> np.ndarray:
    """
    Uniform scaling with aspect ratio preservation to a centered square [-1, 1]^2.
    Scale is chosen based on the larger bbox side: X' = (X - c) / s, where
    c = (min + max)/2, s = (max(max-min)/2, eps).
    """
    X = np.asarray(X, dtype=np.float64)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    c = (lo + hi) / 2.0
    half_span = (hi - lo) / 2.0
    s = float(max(half_span.max(), 1e-12))
    Z = (X - c) / s
    # due to numerical precision, some values may exceed 1.0 by 1e-15 — clip carefully
    Z = np.clip(Z, -1.0, 1.0)
    return Z


def get_two_moons(
    n_samples: int = 100,
    noise: float = 0.05,
    random_state: Union[int, None] = None,
    normalize: Normalize = "fit_square",
    dtype: np.dtype = np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two-moons dataset split into two classes.

    Parameters
    ---------
    n_samples : int
        Number of points per class (total will be 2*n_samples).
    noise : float
        Gaussian noise variance (as in sklearn.datasets.make_moons).
    random_state : int | None
        RNG initialization.
    normalize : {"none", "unit_square", "fit_square"}
        - "none": return as-is from make_moons (centered around (0,0), no affine transformations).
        - "unit_square": independent min-max scaling of each coordinate to [0,1]^2.
        - "fit_square": uniform scaling with aspect ratio preservation to square [-1,1]^2 (default).
    dtype : numpy.dtype
        Type of numbers for the array of points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two arrays, each shape (n_samples, 2): one moon for class A, one for class B.
    """
    X, y = make_moons(n_samples=2*n_samples, noise=noise, shuffle=True, random_state=random_state)

    if normalize == "unit_square":
        X = _to_unit_square(X)
    elif normalize == "fit_square":
        X = _fit_to_centered_square(X)
    elif normalize == "none":
        # leave as is
        pass
    else:
        raise ValueError(f"Unknown normalize='{normalize}'. Choose from 'none', 'unit_square', 'fit_square'.")

    X = np.ascontiguousarray(X, dtype=dtype)
    
    # Split by moon (label 0 -> class A, label 1 -> class B)
    X_a = X[y == 0]
    X_b = X[y == 1]

    return X_a, X_b

