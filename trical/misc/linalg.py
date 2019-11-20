"""
Relevant linear algebra functions for TriCal
"""
import numpy as np


def norm(x):
    """
    Computes the norm of an array over the last axis
    
    Args:
        x (N-dim array of float): Array of interest
    
    Returns:
        (N-1)-dim array of float: Norm of the array of interest over the last axis
    """
    return np.sqrt((x ** 2).sum(-1))


def cartesian_to_spherical(x, y, z):
    r = np.hypot(np.hypot(x, y), z)
    phi = np.arctan2(np.hypot(x, y), z)
    theta = np.arctan2(y, x)
    return r, phi, theta


def spherical_to_cartesian(r, phi, theta):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z
