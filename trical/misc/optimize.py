"""
Relevant optimization functions for TrICal
"""
import numpy as np
from scipy import optimize as opt


def dflt_opt(ti):
    """
    Generator of default optimizer for equilibrium_position method of TrappedIons
    
    Args:
        ti (TrappedIons): TrappedIons object of interest
    
    Returns:
        func(func(1-D array) -> 1-D array): Optimizer for a function f that returns
        the minimum value of f
    """
    if dim == 1:
        x_guess = np.linspace(-(N - 1) / 2, (N - 1) / 2, N)
    else:
        x_guess = np.append(
            np.concatenate([np.zeros(N)] * (dim - 1)),
            np.linspace(-(N - 1) / 2, (N - 1) / 2, N),
        )
    return lambda f: opt.minimize(f, x_guess, method="SLSQP").x
