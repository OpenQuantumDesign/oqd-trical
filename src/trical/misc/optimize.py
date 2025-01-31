"""
Module containing default optimization function generators for TrICal.
"""

########################################################################################

from typing import Callable

import numpy as np

from scipy import optimize as opt

########################################################################################


def dflt_opt(ti, **kwargs):
    """
    Default optimization function generator for equilibrium_position method of TrappedIons class.

    Args:
        ti (TrappedIons): Trapped ion system of interest.

    Returns:
        (Callable): Default optimization function that finds the equilibrium position of the trapped ions system of interest via the minimization of the potential.
    """
    opt_params = {"method": "SLSQP", "options": {"maxiter": 1000}, "tol": 1e-15}
    opt_params.update(kwargs)

    if ti.dim == 1:
        x_guess = np.linspace(-(ti.N - 1) / 2, (ti.N - 1) / 2, ti.N)
    else:
        x_guess = np.append(
            np.concatenate([np.zeros(ti.N)] * (ti.dim - 1)),
            np.linspace(-(ti.N - 1) / 2, (ti.N - 1) / 2, ti.N),
        )

    def _dflt_opt(f):
        res = opt.minimize(f, x_guess, **opt_params)
        assert res.success, res.__str__()
        return res.x

    return _dflt_opt


def dflt_ls_opt(deg):
    """
    Default optimization function generator for multivariate_polyfit function.

    Args:
        deg (np.ndarray[int]): Degree of polynomial used in the fit.

    Returns:
        (Callable): Default optimization function that finds the best polynomial, of the specified degree, fit for the data .
    """

    def _dflt_ls_opt(a, b):
        res = opt.lsq_linear(a, b)
        assert res.success
        return res.x

    return _dflt_ls_opt
