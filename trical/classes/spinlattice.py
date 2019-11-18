import numpy as np


class SpinLattice(object):
    def __init__(self, J):
        super(SpinLattice, self).__init__()

        self.J = J
        pass

    pass


class SimulatedSpinLattice(SpinLattice):
    def __init__(self, ic, mu, Omega, **kwargs):
        self.ic = ic
        self.mu = mu
        self.Omega = Omega

        super(SimulatedSpinLattice, self).__init__()
        pass

    pass
