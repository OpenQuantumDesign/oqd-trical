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
        func(func(1-D array of float) -> 1-D array float): Optimizer for a function f 
        that returns the minimum value of f
    """
    if self.dim == 1:
        x_guess = np.linspace(-(self.N - 1) / 2, (self.N - 1) / 2, self.N)
    else:
        x_guess = np.append(
            np.concatenate([np.zeros(self.N)] * (self.dim - 1)),
            np.linspace(-(self.N - 1) / 2, (self.N - 1) / 2, self.N),
        )
    return lambda f: opt.minimize(f, x_guess, method="SLSQP").x
