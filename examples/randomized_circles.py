import math
import numpy as np


def get_randomized_circles(
    n_samples: int = 100, 
    noise: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate concentric circles with radial noise split into two classes.
    
    Creates three concentric circles with Gaussian noise applied to radii:
    - Inner circle (R=1.0): all class B
    - Middle circle (R=2.0): 50% class A, 50% class B  
    - Outer circle (R=3.0): all class A
    
    Parameters
    ----------
    n_samples : int
        Number of points per class (total will be ~2*n_samples).
    noise : float
        Standard deviation of Gaussian noise applied to radii.
        Higher values create more scattered, less circular clusters.
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Two arrays, each shape (n_samples, 2): points for class A and class B.
    """
    pts_a = []
    pts_b = []
    
    # Inner circle (R=1.0) - all class B
    n_inner = n_samples // 3
    ang = np.random.random(n_inner) * 2 * math.pi
    R_ = 1.0 + noise * np.random.standard_normal(n_inner)
    xs = R_ * np.cos(ang)
    ys = R_ * np.sin(ang)
    pts_b.extend(list(zip(xs, ys)))
    
    # Middle circle (R=2.0) - 50/50 split
    n_middle = n_samples // 3
    n_middle_a = n_middle // 2
    n_middle_b = n_middle - n_middle_a
    
    ang_a = np.random.random(n_middle_a) * 2 * math.pi
    R_a = 2.0 + noise * np.random.standard_normal(n_middle_a)
    xs_a = R_a * np.cos(ang_a)
    ys_a = R_a * np.sin(ang_a)
    pts_a.extend(list(zip(xs_a, ys_a)))
    
    ang_b = np.random.random(n_middle_b) * 2 * math.pi
    R_b = 2.0 + noise * np.random.standard_normal(n_middle_b)
    xs_b = R_b * np.cos(ang_b)
    ys_b = R_b * np.sin(ang_b)
    pts_b.extend(list(zip(xs_b, ys_b)))
    
    # Outer circle (R=3.0) - all class A, fill to n_samples
    n_outer = n_samples - len(pts_a)
    ang = np.random.random(n_outer) * 2 * math.pi
    R_ = 3.0 + noise * np.random.standard_normal(n_outer)
    xs = R_ * np.cos(ang)
    ys = R_ * np.sin(ang)
    pts_a.extend(list(zip(xs, ys)))
    
    # Fill class B to n_samples with outer circle points
    n_outer_b = n_samples - len(pts_b)
    if n_outer_b > 0:
        ang = np.random.random(n_outer_b) * 2 * math.pi
        R_ = 3.0 + noise * np.random.standard_normal(n_outer_b)
        xs = R_ * np.cos(ang)
        ys = R_ * np.sin(ang)
        pts_b.extend(list(zip(xs, ys)))
    
    # Convert to numpy arrays
    arr_a = np.asarray(pts_a, dtype=np.float64)
    arr_b = np.asarray(pts_b, dtype=np.float64)
    
    return arr_a, arr_b