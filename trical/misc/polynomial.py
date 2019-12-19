"""
Module containing relevant functions regarding polynomials for TrIcal.
"""

from .optimize import dflt_ls_opt
import numpy as np
from numpy.polynomial import polynomial as poly


def multivariate_polyfit(x, vals, deg, l=1, opt=dflt_ls_opt):
    """
    Fits a set of data with a multivariate polynomial.
    
    :param x: Independent values.
    :type x: :obj:`numpy.ndarray`
    :param vals: Dependent value.
    :type vals: :obj:`numpy.ndarray`
    :param deg: Degree of polynomial used in the fit.
    :type deg: :obj:`numpy.ndarray`
    :param l: Length scale used when fitting, defaults to 1.
    :type l: :obj:`float`, optional
    :param opt: Generator of the appropriate optimization function for the fit, defaults to :obj:`trical.misc.optimize.dflt_ls_opt`.
    :type opt: :obj:`types.FunctionType`, optional
    :returns: Coefficients of the best fit multivariate polynomial, of the specified degree.
    :rtype: :obj:`numpy.ndarray`
    """
    dim = len(deg)
    shape = np.array(deg) + 1

    a = {1: poly.polyvander, 2: poly.polyvander2d, 3: poly.polyvander3d}[dim](
        *(x / l).transpose(), deg
    )
    b = vals
    return opt(deg)(a, b).reshape(shape) / l ** np.indices(shape).sum(0)


def polygrad(N, alpha):
    pass
