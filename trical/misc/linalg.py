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
