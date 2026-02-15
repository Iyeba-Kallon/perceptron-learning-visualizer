import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

def generate_linear_data(n_samples=100, noise=0.1, separation=1.0):
    """Generates linearly separable data."""
    X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2, 
                      cluster_std=noise, center_box=(-separation*2, separation*2), random_state=42)
    return X, y

def generate_circles_data(n_samples=100, noise=0.1, factor=0.5):
    """Generates concentric circles data."""
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
    return X, y

def generate_moons_data(n_samples=100, noise=0.1):
    """Generates moon-shaped data."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y

def generate_spiral_data(n_samples=100, noise=0.1):
    """Generates spiral data (harder non-linear classification)."""
    n = n_samples // 2
    
    def gen_spiral(delta_t, label):
        r = np.linspace(0.1, 5, n)
        t = np.linspace(delta_t, delta_t + 3, n) + np.random.normal(0, noise, n)
        x = r * np.sin(t)
        y = r * np.cos(t)
        return np.column_stack((x, y)), np.full(n, label)

    X1, y1 = gen_spiral(0, 0)
    X2, y2 = gen_spiral(np.pi, 1)
    
    X = np.concatenate((X1, X2))
    y = np.concatenate((y1, y2))
    return X, y
