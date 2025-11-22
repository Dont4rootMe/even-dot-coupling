import math
import numpy as np


def get_centric_circles(
    n_samples: int = 100,
    noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate concentric circles split into two classes.
    
    Creates three concentric circles with class assignments:
    - Inner circle (R=0.7): all class B (1)
    - Middle circle (R=2.0): 50% class A, 50% class B
    - Outer circle (R=3.0): all class A (0)
    
    Parameters
    ----------
    n_samples : int
        Number of points per class (total will be ~2*n_samples).
    noise : float
        Gaussian noise to add to coordinates.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two arrays, each shape (n_samples, 2): points for class A and class B.
    """
    pts_a = []
    pts_b = []
    
    # Inner circle (R=0.7) - all class B
    n_inner = n_samples // 3
    ang = np.random.random(n_inner) * 2 * math.pi
    xs = 0.7 * np.cos(ang)
    ys = 0.7 * np.sin(ang)
    if noise > 0:
        xs += np.random.normal(0, noise, size=n_inner)
        ys += np.random.normal(0, noise, size=n_inner)
    pts_b.extend(list(zip(xs, ys)))
    
    # Middle circle (R=2.0) - 50/50 split
    n_middle = n_samples // 3
    n_middle_a = n_middle // 2
    n_middle_b = n_middle - n_middle_a
    
    ang_a = np.random.random(n_middle_a) * 2 * math.pi
    xs_a = 2.0 * np.cos(ang_a)
    ys_a = 2.0 * np.sin(ang_a)
    if noise > 0:
        xs_a += np.random.normal(0, noise, size=n_middle_a)
        ys_a += np.random.normal(0, noise, size=n_middle_a)
    pts_a.extend(list(zip(xs_a, ys_a)))
    
    ang_b = np.random.random(n_middle_b) * 2 * math.pi
    xs_b = 2.0 * np.cos(ang_b)
    ys_b = 2.0 * np.sin(ang_b)
    if noise > 0:
        xs_b += np.random.normal(0, noise, size=n_middle_b)
        ys_b += np.random.normal(0, noise, size=n_middle_b)
    pts_b.extend(list(zip(xs_b, ys_b)))
    
    # Outer circle (R=3.0) - all class A, fill to n_samples
    n_outer = n_samples - len(pts_a)
    ang = np.random.random(n_outer) * 2 * math.pi
    xs = 3.0 * np.cos(ang)
    ys = 3.0 * np.sin(ang)
    if noise > 0:
        xs += np.random.normal(0, noise, size=n_outer)
        ys += np.random.normal(0, noise, size=n_outer)
    pts_a.extend(list(zip(xs, ys)))
    
    # Fill class B to n_samples with outer circle points
    n_outer_b = n_samples - len(pts_b)
    if n_outer_b > 0:
        ang = np.random.random(n_outer_b) * 2 * math.pi
        xs = 3.0 * np.cos(ang)
        ys = 3.0 * np.sin(ang)
        if noise > 0:
            xs += np.random.normal(0, noise, size=n_outer_b)
            ys += np.random.normal(0, noise, size=n_outer_b)
        pts_b.extend(list(zip(xs, ys)))
    
    # Convert to numpy arrays
    arr_a = np.asarray(pts_a, dtype=np.float64)
    arr_b = np.asarray(pts_b, dtype=np.float64)
    
    return arr_a, arr_b