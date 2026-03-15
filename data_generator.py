"""
Utilities for generating synthetic datasets for classification tasks.
Provides various distributions like linear, concentric circles, moons, and spirals.
"""
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

def generate_linear_data(n_samples=100, noise=0.1, separation=1.0):
    """
    Generate a linearly separable dataset with two clusters.
    
    Args:
        n_samples (int): Total number of data points.
        noise (float): Standard deviation of the Gaussian noise.
        separation (float): Padding factor for cluster centers.
        
    Returns:
        tuple: (Features X, Labels y)
    """
    X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2, 
                      cluster_std=noise, 
                      center_box=(-separation*2, separation*2), 
                      random_state=42)
    return X, y

def generate_circles_data(n_samples=100, noise=0.05, factor=0.5):
    """
    Generate a non-linear dataset of concentric circles.
    
    Args:
        n_samples (int): Total number of data points.
        noise (float): Gaussian noise added to the data.
        factor (float): Scale factor between inner and outer circle.
    """
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=42)
    return X, y

def generate_moons_data(n_samples=100, noise=0.1):
    """
    Generate two interleaving half circles (moons).
    Excellent for testing non-linear classification boundaries.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return X, y

def generate_spiral_data(n_samples=100, noise=0.1):
    """
    Generate a complex spiral dataset.
    This is historically a challenging task for simple networks.
    """
    n_per_spiral = n_samples // 2
    
    def gen_spiral(offset_radians, label):
        radius = np.linspace(0.1, 5, n_per_spiral)
        # Calculate angular displacement with added noise
        theta = np.linspace(offset_radians, offset_radians + 3, n_per_spiral) + \
                np.random.normal(0, noise, n_per_spiral)
        
        x = radius * np.sin(theta)
        y = radius * np.cos(theta)
        return np.column_stack((x, y)), np.full(n_per_spiral, label)

    X1, y1 = gen_spiral(0, 0)
    X2, y2 = gen_spiral(np.pi, 1)
    
    return np.concatenate((X1, X2)), np.concatenate((y1, y2))
