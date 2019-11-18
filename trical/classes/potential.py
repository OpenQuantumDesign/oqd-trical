import numpy as np


class Potential(object):
    def __init__(self, **kwargs):
        super(Potential, self).__init__()
        pass

    def __add__(self):
        return Potential()

    def __sub__(self):
        return Potential()

    def __mul__(self):
        return Potential()

    def __rmul__(self):
        return Potential()

    def __truediv__(self):
        return Potential()

    def __call__(self):
        pass

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
