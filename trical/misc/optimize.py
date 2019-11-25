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
    if ti.dim == 1:
        x_guess = np.linspace(-(ti.N - 1) / 2, (ti.N - 1) / 2, ti.N)
    else:
        x_guess = np.append(
            np.concatenate([np.zeros(ti.N)] * (ti.dim - 1)),
            np.linspace(-(ti.N - 1) / 2, (ti.N - 1) / 2, ti.N),
        )

    def _dflt_opt(f):
        res = opt.minimize(f, x_guess, method="SLSQP")
        assert res.success
        return res.x

    return _dflt_opt


def dflt_ls_opt(deg):
    shape = np.array(deg) + 1
    alpha_guess = np.zeros(np.prod(shape))

    def _dflt_ls_opt(f):
        res = opt.least_squares(f, alpha_guess)
        assert res.success
        return res.x

    return _dflt_ls_opt
