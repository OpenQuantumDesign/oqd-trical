"""
Module containing relevant functions regarding polynomials for TrIcal.
"""

########################################################################################

from typing import Callable

from autograd import numpy as agnp

import numpy as np
from numpy.polynomial import polynomial as poly

########################################################################################

from .optimize import dflt_ls_opt

########################################################################################


def multivariate_polyfit(x, vals, deg, l=1, opt=dflt_ls_opt):
    """
    Fits a set of data with a multivariate polynomial.

    Args:
        x (np.ndarray[float]): Independent values.
        vals (np.ndarray[float]): Dependent value.
        deg (np.ndarray[int]): Degree of polynomial used in the fit.
        l (float): Length scale used when fitting, defaults to 1.
        opt (Callable): Generator of the appropriate optimization function for the fit.

    Returns:
        (np.ndarray[float]): Coefficients of the best fit multivariate polynomial, of the specified degree.
    """
    dim = len(deg)
    shape = np.array(deg) + 1

    a = {1: poly.polyvander, 2: poly.polyvander2d, 3: poly.polyvander3d}[dim](
        *(x / l).transpose(), deg
    )
    b = vals
    return opt(deg)(a, b).reshape(shape) / l ** np.indices(shape).sum(0)


def polyval(x, alpha):
    dim = len(alpha.shape)

    x = agnp.moveaxis(
        agnp.tile(x, agnp.concatenate((alpha.shape, [1, 1]))),
        agnp.concatenate(
            (
                agnp.arange(dim + 2, dtype=int)[-1:-3:-1],
                agnp.arange(dim + 2, dtype=int)[:-2],
            )
        ),
        agnp.arange(dim + 2, dtype=int),
    )

    idcs = [
        agnp.moveaxis(
            agnp.tile(
                agnp.arange(alpha.shape[i]),
                agnp.concatenate(
                    (
                        agnp.array(alpha.shape)[
                            agnp.delete(agnp.arange(dim, dtype=int), i)
                        ],
                        [1],
                    )
                ),
            ),
            range(dim),
            np.concatenate((agnp.delete(agnp.arange(dim, dtype=int), i), [i])),
        )
        for i in range(dim)
    ]

    v = agnp.prod(
        [x[i] ** idcs[i] for i in range(len(alpha.shape))],
        axis=0,
    )
    return (v * alpha).sum(tuple(range(1, dim + 1)))
