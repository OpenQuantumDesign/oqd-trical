from ..misc.linalg import norm
import itertools as itr
import numpy as np


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
        phi = lambda x: self.phi(x) + other.phi(x)
        dphi = lambda var: (lambda x: self.dphi(var)(x) + other.dphi(var)(x))
        d2phi = lambda var1, var2: (
            lambda x: self.d2phi(var1, var2)(x) + other.d2phi(var1, var2)(x)
        )
        return Potential(phi, dphi, d2phi)

    def __sub__(self, other):
        phi = lambda x: self.phi(x) - other.phi(x)
        dphi = lambda var: (lambda x: self.dphi(var)(x) - other.dphi(var)(x))
        d2phi = lambda var1, var2: (
            lambda x: self.d2phi(var1, var2)(x) - other.d2phi(var1, var2)(x)
        )
        return Potential(phi, dphi, d2phi)

    def __mul__(self, multiplier):
        phi = lambda x: self.phi(x) * multiplier
        dphi = lambda var: (lambda x: self.dphi(var)(x) * multiplier)
        d2phi = lambda var1, var2: (lambda x: self.d2phi(var1, var2)(x) * multiplier)
        return Potential(phi, dphi, d2phi)

    def __rmul__(self, multiplier):
        return self * multiplier

    def __truediv__(self, divisor):
        phi = lambda x: self.phi(x) / divisor
        dphi = lambda var: (lambda x: self.dphi(var)(x) / divisor)
        d2phi = lambda var1, var2: (lambda x: self.d2phi(var1, var2)(x) / divisor)
        return Potential(phi, dphi, d2phi)

    def __call__(self, x):
        return self.phi(x)

    def first_derivative(self, var):
        return self.dphi(var)

    def second_derivative(self, var1, var2):
        return self.d2phi(var1, var2)

    pass


class CoulombPotential(Potential):
    def __init__(self, N, **kwargs):

        params = {"dim": 3, "N": N}
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
        return 1 / norm(x[i] - x[j])

    def first_derivative(var):
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:])
        j = np.delete(np.arange(self.N, dtype=int), i)

        def dphi_dai(x):
            xia = x[i, a]
            xja = x[j, a]
            nxij = norm(x[i] - x[j])
            return ((xja - xia) / nxij ** 3).sum()

        return dphi_dai

    def second_derivative(var1, var2):
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:])
        j = int(var2[1:])

        def d2phi_daidbj(x):
            if i == j:
                j = np.delete(np.arange(self.N, dtype=int), i)
                xia = x[i, a]
                xja = x[j, a]
                xib = x[i, b]
                xjb = x[j, b]
                nxij = norm(x[i] - x[j])
                if a == b:
                    return ((-1 / nxij ** 3 + 3 * (xja - xia) ** 2 / nxij ** 5)).sum()
                else:
                    return (3 * (xja - xia) * (xjb - xib) / nxij ** 5).sum()
            else:
                xia = x[i, a]
                xja = x[j, a]
                xib = x[i, b]
                xjb = x[j, b]
                nxij = norm(x[i] - x[j])
                if a == b:
                    return 1 / nxij ** 3 - 3 * (xja - xia) ** 2 / nxij ** 5
                else:
                    return -3 * (xja - xia) * (xjb - xib) / nxij ** 5

        return d2phi_daidbj

    pass


class PolynomialPotential(Potential):
    def __init__(self, **kwargs):

        params = {}
        params.update(kwargs)

        super(PolynomialPotential, self).__init__(
            self.__call__, self.first_derivative, self.second_derivative, **params
        )
        pass

    def __call__(self):
        pass

    def first_derivative(var):
        pass

    def second_derivative(var1, var2):
        pass

    pass
