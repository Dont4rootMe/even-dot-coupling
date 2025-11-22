from typing import Literal, Tuple, Union, Sequence
import argparse
import numpy as np

Normalize = Literal["none", "unit_square", "fit_square"]


def _to_unit_square(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = hi - lo
    span[span == 0.0] = 1.0
    Z = (X - lo) / span
    return Z


def _fit_to_centered_square(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    c = (lo + hi) / 2.0
    half_span = (hi - lo) / 2.0
    s = float(max(half_span.max(), 1e-12))
    Z = (X - c) / s
    return np.clip(Z, -1.0, 1.0)


def _apply_norm(X: np.ndarray, normalize: Normalize) -> np.ndarray:
    if normalize == "none":
        return np.ascontiguousarray(X, dtype=np.float64)
    if normalize == "unit_square":
        return np.ascontiguousarray(_to_unit_square(X), dtype=np.float64)
    if normalize == "fit_square":
        return np.ascontiguousarray(_fit_to_centered_square(X), dtype=np.float64)
    raise ValueError("normalize must be one of {'none','unit_square','fit_square'}")


def get_nested_polygons(
    n_samples: int = 100,
    n_layers: int = 4,
    m_vertices: int | Sequence[int] = 24,
    r_min: float = 1.0,
    r_max: float = 4.0,
    noise: float = 0.03,
    sigma_tangent: float = 0.01,
    random_state: int | None = None,
    rotate_each_layer: bool = True,
    normalize: Normalize = "fit_square",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nested regular polygons with alternating class labels by layer.
    
    Layer 0 (outermost) → class A
    Layer 1 → class B
    Layer 2 → class A
    Layer 3 → class B
    etc.

    Parameters
    ----------
    n_samples : int
        Number of points per class (total will be ~2*n_samples depending on layer count).
    n_layers : int
        Number of nested polygon layers.
    noise : float
        Standard deviation of noise.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two arrays with points from alternating layers.
    """
    
    if noise <= 0.001:
        noise = 0.3
    noise = noise / 10.0
    
    sigma_radial = noise
    
    rng = np.random.default_rng(random_state)
    
    # Calculate points per layer to achieve n_samples per class
    total_points = 2 * n_samples
    n_per_layer = total_points // n_layers

    if isinstance(m_vertices, int):
        mL = [m_vertices] * n_layers
    else:
        mL = list(m_vertices)
        assert len(mL) == n_layers

    # radii from outer to inner
    radii = np.linspace(r_max, r_min, n_layers)

    Xs_a = []
    Xs_b = []
    for ℓ in range(n_layers):
        n = int(n_per_layer)
        m = int(mL[ℓ])
        rℓ = float(radii[ℓ])
        φℓ = rng.uniform(0, 2*np.pi) if rotate_each_layer else 0.0

        # uniformly across vertices (additional uniform "shift" along arc)
        j = rng.integers(0, m, size=n)
        base_θ = (2*np.pi * j / m) + φℓ
        δt = rng.normal(0.0, sigma_tangent, size=n)
        δr = rng.normal(0.0, sigma_radial, size=n)

        θ = base_θ + δt
        r = rℓ + δr
        X = np.column_stack((r * np.cos(θ), r * np.sin(θ)))
        
        # Alternate layers between classes
        if ℓ % 2 == 0:
            Xs_a.append(X)
        else:
            Xs_b.append(X)

    X_a = np.vstack(Xs_a) if Xs_a else np.empty((0, 2))
    X_b = np.vstack(Xs_b) if Xs_b else np.empty((0, 2))
    
    X_a = _apply_norm(X_a, normalize)
    X_b = _apply_norm(X_b, normalize)
    
    return X_a, X_b


def get_pinwheel(
    n_samples: int = 100,
    n_arms: int = 5,
    radius_loc: float = 1.0,
    radius_scale: float = 0.6,
    angle_gain: float = 0.9,
    angle_noise: float = 0.10,
    noise: float = 0.02,
    random_state: int | None = None,
    normalize: Normalize = "fit_square",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pinwheel with alternating class labels by arm.
    
    Arm 0 → class A
    Arm 1 → class B
    Arm 2 → class A
    Arm 3 → class B
    etc.

    Parameters
    ----------
    n_samples : int
        Number of points per class (total will be ~2*n_samples depending on arm count).
    n_arms : int
        Number of pinwheel arms.
    noise : float
        Standard deviation of isotropic noise.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two arrays with points from alternating arms.
    """
    if noise <= 0.001:
        noise = 0.2
    noise = noise / 20.0
    
    iso_noise = noise
    
    rng = np.random.default_rng(random_state)
    
    # Calculate points per arm to achieve n_samples per class
    total_points = 2 * n_samples
    n_per = int(np.ceil(total_points / n_arms))
    
    Xs_a = []
    Xs_b = []
    for k in range(n_arms):
        φk = 2*np.pi * k / n_arms
        r = np.abs(rng.normal(radius_loc, radius_scale, size=n_per))
        ε = rng.normal(0.0, angle_noise, size=n_per)
        θ = φk + angle_gain * r + ε
        X = np.column_stack((r * np.cos(θ), r * np.sin(θ)))
        if iso_noise > 0:
            X += rng.normal(0.0, iso_noise, size=X.shape)
        
        # Alternate arms between classes
        if k % 2 == 0:
            Xs_a.append(X)
        else:
            Xs_b.append(X)
    
    X_a = np.vstack(Xs_a) if Xs_a else np.empty((0, 2))
    X_b = np.vstack(Xs_b) if Xs_b else np.empty((0, 2))
    
    # Trim to approximately n_samples each
    X_a = X_a[:n_samples] if len(X_a) > n_samples else X_a
    X_b = X_b[:n_samples] if len(X_b) > n_samples else X_b
    
    X_a = _apply_norm(X_a, normalize)
    X_b = _apply_norm(X_b, normalize)
    
    return X_a, X_b


def get_spirals(
    n_samples: int = 100,
    n_arms: int = 2,
    turns: float = 1.75,
    a: float = 0.1,
    b: float = 0.25,
    angle_jitter: float = 0.02,
    noise: float = 0.02,
    random_state: int | None = None,
    normalize: Normalize = "fit_square",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Archimedean spirals with random class assignment.
    
    r(θ)=a+bθ, θ ~ U[0, 2π·turns], for each ray offset φ_j=2π j/n_arms.
    Points are randomly assigned to class A or class B.

    Parameters
    ----------
    n_samples : int
        Number of points per class (total will be 2*n_samples).
    noise : float
        Standard deviation of radial noise.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two arrays with randomly assigned points from spirals.
    """
    
    if noise <= 0.001:
        noise = 0.2
    
    noise = noise / 5.0
    
    rng = np.random.default_rng(random_state)
    
    total_points = 2 * n_samples
    n_per_arm = int(np.ceil(total_points / n_arms))
    
    Xs = []
    for j in range(n_arms):
        θ = rng.uniform(0.0, 2*np.pi*turns, size=n_per_arm)
        θ += 2*np.pi * j / n_arms
        θ += rng.normal(0.0, angle_jitter, size=n_per_arm)
        r = a + b * θ + rng.normal(0.0, noise, size=n_per_arm)
        r = np.clip(r, 0.0, None)
        X = np.column_stack((r * np.cos(θ), r * np.sin(θ)))
        Xs.append(X)
    X = np.vstack(Xs)[:total_points]
    X = _apply_norm(X, normalize)
    
    # Randomly assign 50% to class A, 50% to class B
    indices = np.arange(len(X))
    rng.shuffle(indices)
    mid = len(indices) // 2
    
    X_a = X[indices[:mid]]
    X_b = X[indices[mid:]]
    
    return X_a, X_b