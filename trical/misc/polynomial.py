from .optimize import dflt_ls_opt
import numpy as np
from numpy.polynomial import polynomial as poly


def multivariate_polyfit(x, vals, deg, opt=dflt_ls_opt):
    dim = len(deg)
    shape = np.array(deg) + 1

    a = {1: poly.polyvander, 2: poly.polyvander2d, 3: poly.polyvander3d}[dim](
        *x.transpose(), deg
    )
    b = vals
    return opt(deg)(a, b).reshape(shape)


def polygrad(N, alpha):
    pass
