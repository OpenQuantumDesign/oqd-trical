import numpy as np


class Potential(object):
    def __init__(self, phi, dphi, d2phi, **kwargs):
        super(Potential, self).__init__()

        self.phi = phi
        self.dphi = dphi
        self.d2phi = d2phi

        params = {}
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

    def second_derivatice(self, var1, var2):
        return self.d2phi(var1, var2)

    pass


class CoulombPotential(Potential):
    def __init__(self, **kwargs):
        super(CoulombPotential, self).__init__()
        pass

    def __call__(self):
        pass

    pass


class PolynomialPotential(Potential):
    def __init__(self, **kwargs):
        super(PolynomialPotential, self).__init__()
        pass

    def __call__(self):
        pass

    pass
