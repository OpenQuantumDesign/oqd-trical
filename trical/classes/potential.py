from .. import constants as cst
from ..misc.linalg import norm
import itertools as itr
import numpy as np
from numpy.polynomial import polynomial as poly


class Potential(object):
    def __init__(self, phi, dphi, d2phi, **kwargs):
        super(Potential, self).__init__()

        self.phi = phi
        self.dphi = dphi
        self.d2phi = d2phi

        params = {"dim": 3}
        params.update(kwargs)
        self.__dict__.update(params)
        self.params = params
        pass

    def __add__(self, other):
        params = {}
        params.update(self.params)
        params.update(other.params)
        phi = lambda x: self.phi(x) + other.phi(x)
        dphi = lambda var: (lambda x: self.dphi(var)(x) + other.dphi(var)(x))
        d2phi = lambda var1, var2: (
            lambda x: self.d2phi(var1, var2)(x) + other.d2phi(var1, var2)(x)
        )
        return Potential(phi, dphi, d2phi, **params)

    def __sub__(self, other):
        params = {}
        params.update(self.params)
        params.update(other.params)
        phi = lambda x: self.phi(x) - other.phi(x)
        dphi = lambda var: (lambda x: self.dphi(var)(x) - other.dphi(var)(x))
        d2phi = lambda var1, var2: (
            lambda x: self.d2phi(var1, var2)(x) - other.d2phi(var1, var2)(x)
        )
        return Potential(phi, dphi, d2phi, **params)

    def __mul__(self, multiplier):
        phi = lambda x: self.phi(x) * multiplier
        dphi = lambda var: (lambda x: self.dphi(var)(x) * multiplier)
        d2phi = lambda var1, var2: (lambda x: self.d2phi(var1, var2)(x) * multiplier)
        return Potential(phi, dphi, d2phi, **self.params)

    def __rmul__(self, multiplier):
        return self * multiplier

    def __truediv__(self, divisor):
        phi = lambda x: self.phi(x) / divisor
        dphi = lambda var: (lambda x: self.dphi(var)(x) / divisor)
        d2phi = lambda var1, var2: (lambda x: self.d2phi(var1, var2)(x) / divisor)
        return Potential(phi, dphi, d2phi, **self.params)

    def __call__(self, x):
        return self.phi(x)

    def first_derivative(self, var):
        return self.dphi(var)

    def second_derivative(self, var1, var2):
        return self.d2phi(var1, var2)

    def gradient(self):
        def grad_phi(x):
            grad_phi_x = np.empty(self.N * self.dim)

            i = 0
            for var in itr.product(
                ["x", "y", "z"][: self.dim], np.arange(1, self.N + 1, dtype=int)
            ):
                grad_phi_x[i] = self.dphi(var)(x)
                i += 1
            return grad_phi_x

        return grad_phi

    def hessian(self):
        def hess_phi(x):
            hess_phi_x = np.empty((self.N * self.dim, self.N * self.dim))

            i = 0
            for var1 in itr.product(
                ["x", "y", "z"][: self.dim], np.arange(1, self.N + 1, dtype=int)
            ):
                j = 0
                for var2 in itr.product(
                    ["x", "y", "z"][: self.dim], np.arange(1, self.N + 1, dtype=int)
                ):
                    hess_phi_x[i, j] = self.d2phi(var1, var2)(x)
                    j += 1
                i += 1
            return hess_phi_x

        return hess_phi

    pass


class CoulombPotential(Potential):
    def __init__(self, N, **kwargs):

        params = {"dim": 3, "N": N, "q": cst.e}
        params.update(kwargs)

        super(CoulombPotential, self).__init__(
            self.__call__, self.first_derivative, self.second_derivative, **params
        )
        pass

    def __call__(self, x):
        i, j = (
            np.fromiter(itr.chain(*itr.combinations(range(self.N), 2)), dtype=int)
            .reshape(-1, 2)
            .transpose()
        )
        nxij = norm(x[i] - x[j])
        return cst.k * self.q ** 2 * (1 / nxij).sum()

    def first_derivative(self, var):
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:] if type(var) == str else var[1:][0]) - 1
        j = np.delete(np.arange(self.N, dtype=int), i)

        def dphi_dai(x):
            xia = x[i, a]
            xja = x[j, a]
            nxij = norm(x[i] - x[j])
            return cst.k * self.q ** 2 * ((xja - xia) / nxij ** 3).sum()

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:] if type(var1) == str else var1[1:][0]) - 1
        j = int(var2[1:] if type(var2) == str else var2[1:][0]) - 1
        print(i,j)

        def d2phi_daidbj(x):
            if i == j:
                k = np.delete(np.arange(self.N, dtype=int), i)
                xia = x[i, a]
                xka = x[k, a]
                xib = x[i, b]
                xkb = x[k, b]
                nxik = norm(x[i] - x[k])
                if a == b:
                    return (
                        cst.k
                        * self.q ** 2
                        * ((-1 / nxik ** 3 + 3 * (xka - xia) ** 2 / nxik ** 5)).sum()
                    )
                else:
                    return (
                        cst.k
                        * self.q ** 2
                        * (3 * (xka - xia) * (xkb - xib) / nxik ** 5).sum()
                    )
            else:
                xia = x[i, a]
                xja = x[j, a]
                xib = x[i, b]
                xjb = x[j, b]
                nxij = norm(x[i] - x[j])
                if a == b:
                    return (
                        cst.k
                        * self.q ** 2
                        * (1 / nxij ** 3 - 3 * (xja - xia) ** 2 / nxij ** 5)
                    )
                else:
                    return (
                        cst.k
                        * self.q ** 2
                        * (-3 * (xja - xia) * (xjb - xib) / nxij ** 5)
                    )

        return d2phi_daidbj

    def nondimensionalize(self, l):
        return self / (cst.k * cst.e ** 2)

    pass


class PolynomialPotential(Potential):
    def __init__(self, alpha, **kwargs):

        self.alpha = np.array(alpha)

        params = {"deg": alpha.shape, "dim": len(alpha.shape)}
        params.update(kwargs)

        super(PolynomialPotential, self).__init__(
            self.__call__, self.first_derivative, self.second_derivative, **params
        )
        pass

    def __call__(self, x):
        return {1: poly.polyval, 2: poly.polyval2d, 3: poly.polyval3d}[self.dim](
            *x.transpose(), self.alpha
        ).sum()

    def first_derivative(self, var):
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:] if type(var) == str else var[1:][0]) - 1

        beta = poly.polyder(self.alpha, axis=a)

        def dphi_dai(x):
            return {1: poly.polyval, 2: poly.polyval2d, 3: poly.polyval3d}[self.dim](
                *x[i], beta
            )

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:] if type(var1) == str else var1[1:][0]) - 1
        j = int(var2[1:] if type(var2) == str else var2[1:][0]) - 1

        beta = poly.polyder(self.alpha, axis=a)
        gamma = poly.polyder(beta, axis=b)

        def d2phi_daidbj(x):
            if i == j:
                return {1: poly.polyval, 2: poly.polyval2d, 3: poly.polyval3d}[
                    self.dim
                ](*x[i], gamma)
            else:
                return 0

        return d2phi_daidbj

    def nondimensionalize(self, l):
        alpha = (
            l ** np.indices(self.alpha.shape).sum(0)
            * self.alpha
            * (l / (cst.k * cst.e ** 2))
        )
        return PolynomialPotential(alpha)

    pass
