from .optimize import dflt_ls_opt
import numpy as np
from numpy.polynomial import polynomial as poly


def multivariate_polyfit(x, vals, deg, l=1, ls_opt=dflt_ls_opt):
    dim = len(deg)
    shape = np.array(deg) + 1

    def residuals(alpha):
        _alpha = alpha.reshape(shape)
        return vals - {1: poly.polyval, 2: poly.polyval2d, 3: poly.polyval3d}[dim](
            *(x.transpose() / l), _alpha
        )

    return ls_opt(deg)(residuals).reshape(shape) / l ** np.indices(shape).sum(0)
